"""
master.py  —  CLIP-aware MasterOrchestrator

Changes from ViT version:
  - Default model is now CLIP's vision encoder, not torchvision vit_b_16
  - Accepts an optional model_path arg for local CLIP checkpoints
  - __main__ test block uses correct CLIP dummy shape (1, 50, 768)
  - Attribute resolution via getattr fallbacks was already correct;
    confirmed and kept identical for both torchvision and CLIP blocks
  - All ARIMA / circuit-breaker / preflight logic is unchanged
"""

import time
import torch
import socket
from concurrent.futures import ThreadPoolExecutor
from transformers import CLIPModel

from distributedSystem.shared_utils import (
    MultiDeviceARIMAManager,
    get_model_metadata, get_head_weights,
    merge_n_projections, ready_for_math,
    send_msg, recv_msg,
)

# ─────────────────────────────────────────────────────────────────────────────
PREFLIGHT_PINGS    = 8
PROBE_FAIL_LATENCY = 999.0


class MasterOrchestrator:

    def __init__(
        self,
        expected_workers: list[str],
        host: str        = "0.0.0.0",
        port: int        = 29500,
        model            = None,   # pass clip.vision_model from the detector
        meta: dict       = None,   # pass CLIP_META from the detector
        model_path: str  = None,   # only used when model=None (standalone test)
    ):
        self.expected_workers = expected_workers
        self.all_devices      = ["edge"] + expected_workers
        self.arima            = MultiDeviceARIMAManager(self.all_devices)

        # ── Model ──────────────────────────────────────────────────────────
        if model is not None:
            # Caller (spvit_detector.py) passes the already-loaded vision model
            self.model = model
        else:
            # Standalone / __main__ mode: load CLIP ourselves
            src   = model_path or "openai/clip-vit-base-patch32"
            local = model_path is not None
            clip  = CLIPModel.from_pretrained(src, local_files_only=local).eval()
            self.model = clip.vision_model

        self.meta = meta or get_model_metadata(self.model)

        # ── Network ────────────────────────────────────────────────────────
        self.sockets: dict[str, socket.socket] = {}
        self._wait_for_workers(host, port)
        self.executor = ThreadPoolExecutor(max_workers=max(1, len(expected_workers)))

        # Pre-flight: real RTT samples before block 0
        self._preflight()

    # ── Connection ───────────────────────────────────────────────────────────

    def _wait_for_workers(self, host: str, port: int):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((host, port))
        srv.listen(len(self.expected_workers))
        print(f"[Master] Waiting for {len(self.expected_workers)} worker(s) on {host}:{port} …")

        while len(self.sockets) < len(self.expected_workers):
            conn, addr = srv.accept()
            msg = recv_msg(conn)
            if msg and msg[0] == "REGISTER":
                name = msg[1]
                self.sockets[name] = conn
                print(f"  Worker '{name}' connected from {addr[0]}")

        srv.close()
        print("[Master] All workers connected.\n")

    # ── Pre-flight ───────────────────────────────────────────────────────────

    def _preflight(self):
        print("=" * 60)
        print("[Master] Pre-flight RTT measurement …")

        for w_name, sock in self.sockets.items():
            samples = []
            dummy   = torch.zeros(1, 1)

            for _ in range(PREFLIGHT_PINGS):
                t0 = time.time()
                send_msg(sock, ("PING", 0, dummy, 0, 0))
                resp = recv_msg(sock)
                rtt  = time.time() - t0
                samples.append(rtt if resp is not None else PROBE_FAIL_LATENCY)

            med_rtt = sorted(samples)[len(samples) // 2]
            print(f"  {w_name}: RTTs={[f'{r:.3f}s' for r in samples]}  "
                  f"→ median {med_rtt:.3f}s")

            self.arima.prime(w_name, samples, nominal_share=0.5)

            edge_guess = 0.05
            if med_rtt / 0.5 > edge_guess * MultiDeviceARIMAManager.DROP_MULT:
                print(f"  [Pre-flight] '{w_name}' already slow → pre-tripping CB")
                self.arima.breakers[w_name].trip(reason="pre-flight RTT too high")

        print("[Master] Pre-flight complete.\n" + "=" * 60 + "\n")

    # ── Dispatch ─────────────────────────────────────────────────────────────

    def _dispatch_task(self, w_name, task_type, block_idx, x, start_idx, end_idx):
        try:
            t0  = time.time()
            send_msg(self.sockets[w_name], (task_type, block_idx, x, start_idx, end_idx))
            res = recv_msg(self.sockets[w_name])
            lat = time.time() - t0
            if res is None:
                raise ConnectionError(f"Worker '{w_name}' disconnected mid-task.")
            return res, lat
        except Exception as exc:
            print(f"\n  [ERROR] Dispatch to '{w_name}' failed: {exc}")
            return None, PROBE_FAIL_LATENCY

    # ── Inference ────────────────────────────────────────────────────────────

    def run_inference(self, x: torch.Tensor) -> torch.Tensor:
        """
        Block-by-block forward through clip.vision_model.encoder.layers.

        Attribute resolution uses getattr fallbacks so this works identically
        for both torchvision (ln_1/ln_2, self_attention, mlp[0]/mlp[3]) and
        CLIP (layer_norm1/layer_norm2, self_attn, mlp.fc1/mlp.fc2).

        x : (1, seq_length, embed_dim)  — output of vision.embeddings + pre_layernorm
        returns last hidden state of the same shape
        """
        current_state = x

        for i, block in enumerate(self.model.encoder.layers):

            raw_latency = {dev: 0.0 for dev in self.all_devices}
            share_used  = {dev: 0.0 for dev in self.all_devices}

            probing_workers: set[str] = {
                w for w in self.expected_workers
                if w in self.arima.breakers and self.arima.breakers[w].is_half_open
            }

            self.arima.update_shares()

            # ── Resolve attribute names once per block ────────────────────────
            # torchvision  → ln_1 / ln_2 / self_attention / mlp[0] / mlp[3]
            # CLIP         → layer_norm1 / layer_norm2 / self_attn / mlp.fc1 / mlp.fc2
            ln_1  = getattr(block, "ln_1",          None) or block.layer_norm1
            ln_2  = getattr(block, "ln_2",          None) or block.layer_norm2
            attn  = getattr(block, "self_attention", None) or block.self_attn
            mlp   = block.mlp
            is_seq = isinstance(mlp, torch.nn.Sequential)
            fc1   = mlp[0] if is_seq else mlp.fc1
            fc2   = mlp[3] if is_seq else mlp.fc2

            # ── ATTENTION ────────────────────────────────────────────────────
            identity = current_state
            ln_x     = ln_1(current_state)

            attn_futures = {}
            for w_name in self.expected_workers:
                h_range = self.arima.get_indices(w_name, self.meta["num_heads"])
                if len(h_range) > 0:
                    attn_futures[w_name] = self.executor.submit(
                        self._dispatch_task, w_name, "ATTN", i,
                        ln_x, h_range.start, h_range.stop,
                    )
                    share_used[w_name] += self.arima.current_shares[w_name]

            edge_h = self.arima.get_indices("edge", self.meta["num_heads"])
            share_used["edge"] += self.arima.current_shares["edge"]

            t_edge = time.time()
            if len(edge_h) > 0:
                q = attn.q_proj(ln_x)
                k = attn.k_proj(ln_x)
                v = attn.v_proj(ln_x)

                # slice heads AFTER projection
                def slice_heads(x, h_range):
                    B, S, D = x.shape
                    H = self.meta["num_heads"]
                    x = x.view(B, S, H, -1)
                    return x[:, :, h_range, :].reshape(B, S, -1)

                q_slice = slice_heads(q, edge_h)
                k_slice = slice_heads(k, edge_h)
                v_slice = slice_heads(v, edge_h)

                edge_qkv = torch.cat([q_slice, k_slice, v_slice], dim=-1)
            else:
                edge_qkv = torch.tensor([])
            raw_latency["edge"] += time.time() - t_edge

            qkv_parts = []
            for dev in self.all_devices:
                if dev == "edge":
                    qkv_parts.append(edge_qkv)
                elif dev in attn_futures:
                    res, lat = attn_futures[dev].result()
                    if res is None:
                        res = (torch.zeros_like(edge_qkv)
                               if edge_qkv.numel() > 0 else torch.tensor([]))
                    qkv_parts.append(res)
                    raw_latency[dev] += lat

            merged_qkv  = merge_n_projections(qkv_parts)
            merged_qkv += attn.in_proj_bias
            q, k, v     = torch.chunk(merged_qkv, 3, dim=-1)
            q, k, v     = (ready_for_math(t, self.meta) for t in (q, k, v))

            scale         = self.meta["head_dim"] ** -0.5
            attn_probs    = torch.nn.functional.softmax(
                (q @ k.transpose(-2, -1)) * scale, dim=-1
            )
            ctx           = (attn_probs @ v).transpose(1, 2).reshape(
                1, self.meta["seq_length"], self.meta["embed_dim"]
            )
            attn_out      = attn.out_proj(ctx)
            current_state = identity + attn_out

            # ── MLP ───────────────────────────────────────────────────────────
            identity  = current_state
            ln_x_mlp  = ln_2(current_state)

            mlp_futures = {}
            for w_name in self.expected_workers:
                n_range = self.arima.get_indices(w_name, self.meta["mlp_hidden_dim"])
                if len(n_range) > 0:
                    mlp_futures[w_name] = self.executor.submit(
                        self._dispatch_task, w_name, "MLP", i,
                        ln_x_mlp, n_range.start, n_range.stop,
                    )
                    share_used[w_name] = (
                        share_used[w_name] + self.arima.current_shares[w_name]
                    ) / 2.0

            edge_n = self.arima.get_indices("edge", self.meta["mlp_hidden_dim"])
            t_edge = time.time()
            if len(edge_n) > 0:
                w1       = fc1.weight[edge_n.start:edge_n.stop, :]
                b1       = fc1.bias[edge_n.start:edge_n.stop]
                w2       = fc2.weight[:, edge_n.start:edge_n.stop]
                edge_mlp = torch.nn.functional.gelu(ln_x_mlp @ w1.t() + b1) @ w2.t()
            else:
                edge_mlp = torch.tensor([])
            raw_latency["edge"] += time.time() - t_edge

            mlp_parts = [edge_mlp] if edge_mlp.numel() > 0 else []
            for w_name, fut in mlp_futures.items():
                res, lat = fut.result()
                if res is not None and res.numel() > 0:
                    mlp_parts.append(res)
                raw_latency[w_name] += lat

            # fc2 bias covers the full embed dimension — add once after merging
            mlp_final     = torch.sum(torch.stack(mlp_parts), dim=0) + fc2.bias
            current_state = identity + mlp_final

            # ── ARIMA bookkeeping ─────────────────────────────────────────────
            for dev in self.all_devices:
                s = share_used[dev]
                self.arima.record_block_latency(dev, raw_latency[dev], s if s > 0 else 0.0)

            for w_name in probing_workers:
                s = share_used.get(w_name, 0.0)
                if s > 0 and raw_latency[w_name] < PROBE_FAIL_LATENCY:
                    self.arima.notify_probe_result(w_name, raw_latency[w_name] / s)
                else:
                    self.arima.notify_probe_result(w_name, PROBE_FAIL_LATENCY)

            # ── Status line ───────────────────────────────────────────────────
            n_edge_h   = len(self.arima.get_indices("edge",   self.meta["num_heads"]))
            n_worker_h = sum(
                len(self.arima.get_indices(w, self.meta["num_heads"]))
                for w in self.expected_workers
            )
            cb_str     = ", ".join(
                f"{w}={self.arima.breakers[w].state}" for w in self.expected_workers
            )
            timing_str = " | ".join(
                f"{d}: {raw_latency[d]:.4f}s (share={share_used[d]:.2f})"
                for d in self.all_devices
            )
            print(
                f"Block {i:2d}  heads[edge={n_edge_h} worker={n_worker_h}]"
                f"  CB[{cb_str}]  →  {timing_str}"
            )

        return current_state

    # ── Shutdown ─────────────────────────────────────────────────────────────

    def shutdown(self):
        for w_name, sock in self.sockets.items():
            try:
                send_msg(sock, ("QUIT", 0, None, 0, 0))
                sock.close()
            except Exception:
                pass
        self.executor.shutdown(wait=False)
        print("[Master] Shut down.")


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="CLIP MasterOrchestrator standalone test")
    p.add_argument("--model_path", default=None,
                   help="Local CLIP checkpoint dir. Omit to use HuggingFace hub.")
    p.add_argument("--port",       type=int, default=29500)
    args = p.parse_args()

    orch = MasterOrchestrator(
        expected_workers=["pc_gpu"],
        host="0.0.0.0",
        port=args.port,
        model_path=args.model_path,
    )

    # CLIP ViT-B/32: seq=50 (49 patches + 1 CLS), embed=768
    dummy_input = torch.randn(1, 50, 768)

    t0     = time.time()
    output = orch.run_inference(dummy_input)
    elapsed = time.time() - t0

    print(f"\n[Master] Inference complete in {elapsed:.4f}s  "
          f"|  output shape: {tuple(output.shape)}")

    orch.shutdown()