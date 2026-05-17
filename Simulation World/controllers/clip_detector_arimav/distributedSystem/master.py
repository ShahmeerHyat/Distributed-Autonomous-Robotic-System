"""
master.py  —  CLIP-aware MasterOrchestrator (ARIMA-V scheduler)

ATTN protocol (CLIP mode):
  Master projects Q, K, V from ln_x via attn.q_proj / k_proj / v_proj.
  Each device (edge + workers) receives a head-sliced portion as
  (B, H_slice, S, head_dim) tensors.
  Workers return their attention output in the same shape.
  Master concatenates all parts on the head dimension, reshapes to
  (B, S, embed_dim), and applies attn.out_proj.

MLP protocol:
  Each device computes act_fn(x @ w1_slice.t() + b1_slice) @ w2_slice.t()
  and returns (B, S, embed_dim). Master sums all parts and adds fc2.bias once.
"""

import time
import torch
import socket
from concurrent.futures import ThreadPoolExecutor
from transformers import CLIPModel

from distributedSystem.shared_utils import (
    MultiDeviceARIMAManager,
    get_model_metadata,
    to_head_space,
    sum_mlp_parts,
    send_msg,
    recv_msg,
)

# ─────────────────────────────────────────────────────────────────────────────
PREFLIGHT_PINGS    = 8
PROBE_FAIL_LATENCY = 999.0

MODEL_PATH = r"../../../../Clip Model"


class MasterOrchestrator:

    def __init__(
        self,
        expected_workers: list,
        host: str  = "0.0.0.0",
        port: int  = 29500,
        model      = None,
        meta: dict = None,
    ):
        self.expected_workers     = expected_workers
        self.all_devices          = ["edge"] + expected_workers
        self.arima                = MultiDeviceARIMAManager(self.all_devices)
        self.last_inference_stats = {}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model is not None:
            self.model = model.to(self.device)
        else:
            clip       = CLIPModel.from_pretrained(
                MODEL_PATH, local_files_only=True
            ).eval()
            self.model = clip.vision_model.to(self.device)

        self.meta = meta or get_model_metadata(self.model)

        self.sockets: dict        = {}
        self.worker_addrs: dict   = {}
        self._preflight_rtt: dict = {}

        self._wait_for_workers(host, port)
        self.executor = ThreadPoolExecutor(max_workers=max(1, len(expected_workers)))
        self._preflight()

    @property
    def current_shares(self) -> dict:
        """Normalized allocation weights for each device (sum to 1.0)."""
        return self.arima.current_shares

    # ── Connection ──────────────────────────────────────────────────────────

    def _wait_for_workers(self, host: str, port: int):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((host, port))
        srv.listen(len(self.expected_workers))
        print(f"[Master] Waiting for {len(self.expected_workers)} worker(s) "
              f"on {host}:{port} …")

        while len(self.sockets) < len(self.expected_workers):
            conn, addr = srv.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            msg = recv_msg(conn)
            if msg and msg[0] == "REGISTER":
                name = msg[1]
                self.sockets[name]      = conn
                self.worker_addrs[name] = (addr[0], port)
                print(f"  Worker '{name}' connected from {addr[0]}")

        srv.close()
        print("[Master] All workers connected.\n")

    # ── Pre-flight ──────────────────────────────────────────────────────────

    def _preflight(self):
        print("=" * 60)
        print("[Master] Pre-flight RTT measurement …")

        for w_name in self.expected_workers:
            sock    = self.sockets[w_name]
            samples = []
            dummy   = torch.zeros(1, 1)

            for _ in range(PREFLIGHT_PINGS):
                t0 = time.perf_counter()
                send_msg(sock, ("PING", 0, dummy, 0, 0))
                resp = recv_msg(sock)
                rtt  = time.perf_counter() - t0
                samples.append(rtt if resp is not None else PROBE_FAIL_LATENCY)

            med_rtt = sorted(samples)[len(samples) // 2]
            print(f"  {w_name}: RTTs={[f'{r:.3f}s' for r in samples]}  "
                  f"→ median {med_rtt:.3f}s")
            self._preflight_rtt[w_name] = med_rtt

            self.arima.prime(w_name, samples, nominal_share=0.5)

            edge_guess = 0.05
            if med_rtt / 0.5 > edge_guess * MultiDeviceARIMAManager.DROP_MULT:
                print(f"  [Pre-flight] '{w_name}' already slow → pre-tripping CB")
                self.arima.breakers[w_name].trip(reason="pre-flight RTT too high")

        print("[Master] Pre-flight complete.\n" + "=" * 60 + "\n")

    # ── Local inference (edge-only, used by BenchmarkCollector) ─────────────

    def _local_inference(self, x: torch.Tensor) -> torch.Tensor:
        """
        Runs the full encoder forward pass locally on the edge device only.
        No tasks are dispatched to workers. Used during preflight to measure
        pure edge compute latency without network interference.
        """
        current_state = x.to(self.device)
        H  = self.meta["num_heads"]
        hd = self.meta["head_dim"]
        S  = self.meta["seq_length"]
        D  = self.meta["embed_dim"]

        for block in self.model.encoder.layers:
            ln_1 = block.layer_norm1
            ln_2 = block.layer_norm2
            attn = block.self_attn
            fc1  = block.mlp.fc1
            fc2  = block.mlp.fc2

            identity = current_state
            ln_x     = ln_1(current_state)
            q = attn.q_proj(ln_x)
            k = attn.k_proj(ln_x)
            v = attn.v_proj(ln_x)

            def reshape(t):
                return t.view(1, S, H, hd).transpose(1, 2)

            scale      = hd ** -0.5
            attn_probs = torch.softmax(
                (reshape(q) @ reshape(k).transpose(-2, -1)) * scale, dim=-1
            )
            ctx           = (attn_probs @ reshape(v)).transpose(1, 2).reshape(1, S, D)
            current_state = identity + attn.out_proj(ctx)

            identity      = current_state
            ln_x_mlp      = ln_2(current_state)
            act_fn        = block.mlp.activation_fn
            mlp_out       = act_fn(ln_x_mlp @ fc1.weight.t() + fc1.bias) @ fc2.weight.t() + fc2.bias
            current_state = identity + mlp_out

        return current_state

    # ── Dispatch ────────────────────────────────────────────────────────────

    def _dispatch_task(
        self,
        worker_name,
        task_type,
        block_idx,
        payload,
        start_idx=None,
        end_idx=None,
    ) -> tuple:
        try:
            sock = self.sockets[worker_name]
            t0   = time.perf_counter()

            if task_type == "ATTN":
                assert isinstance(payload, tuple) and len(payload) == 3
                send_msg(sock, ("ATTN", block_idx, payload))
            elif task_type == "MLP":
                send_msg(sock, ("MLP", block_idx, payload, start_idx, end_idx))
            elif task_type == "PING":
                send_msg(sock, ("PING", 0, payload, 0, 0))
            else:
                raise ValueError(f"Unknown task type: {task_type!r}")

            res        = recv_msg(sock)
            total_time = time.perf_counter() - t0

            if res is None:
                raise ConnectionError(f"Worker '{worker_name}' disconnected mid-task.")

            return res, total_time

        except Exception as exc:
            print(f"\n  [ERROR] Dispatch to '{worker_name}' failed: {exc}")
            return None, PROBE_FAIL_LATENCY

    # ── Inference ────────────────────────────────────────────────────────────

    def run_inference(self, x: torch.Tensor) -> torch.Tensor:
        """
        Block-by-block distributed forward through clip.vision_model.encoder.layers.

        x : (1, seq_length, embed_dim)  — output of vision.embeddings + pre_layrnorm
        Returns last hidden state (1, seq_length, embed_dim).
        """
        current_state = x.to(self.device)
        H  = self.meta["num_heads"]
        hd = self.meta["head_dim"]
        S  = self.meta["seq_length"]
        D  = self.meta["embed_dim"]

        _blk_attn_ms      = []
        _blk_mlp_ms       = []
        _blk_evaluator_us = []

        for i, block in enumerate(self.model.encoder.layers):
            raw_latency = {dev: 0.0 for dev in self.all_devices}
            share_used  = {dev: 0.0 for dev in self.all_devices}

            probing_workers = {
                w for w in self.expected_workers
                if w in self.arima.breakers and self.arima.breakers[w].is_half_open
            }

            _t_eval = time.perf_counter()
            self.arima.update_shares()
            _blk_evaluator_us.append((time.perf_counter() - _t_eval) * 1e6)

            ln_1 = block.layer_norm1
            ln_2 = block.layer_norm2
            attn = block.self_attn
            fc1  = block.mlp.fc1
            fc2  = block.mlp.fc2

            # ── ATTENTION ───────────────────────────────────────────────────
            _t_attn  = time.perf_counter()
            identity = current_state
            ln_x     = ln_1(current_state)

            q_full = attn.q_proj(ln_x)
            k_full = attn.k_proj(ln_x)
            v_full = attn.v_proj(ln_x)

            # Dispatch head slices to workers in parallel
            attn_futures = {}
            for w_name in self.expected_workers:
                h_range = self.arima.get_indices(w_name, H)
                if len(h_range) > 0:
                    q_s = to_head_space(q_full, h_range, H, hd)
                    k_s = to_head_space(k_full, h_range, H, hd)
                    v_s = to_head_space(v_full, h_range, H, hd)

                    attn_futures[w_name] = self.executor.submit(
                        self._dispatch_task,
                        w_name, "ATTN", i,
                        (q_s.half(), k_s.half(), v_s.half()),
                    )
                    share_used[w_name] += self.arima.current_shares[w_name]

            # Edge computes its own head slice locally while workers run
            edge_h = self.arima.get_indices("edge", H)
            share_used["edge"] += self.arima.current_shares["edge"]

            t_edge    = time.perf_counter()
            edge_attn = None
            if len(edge_h) > 0:
                q_e = to_head_space(q_full, edge_h, H, hd)
                k_e = to_head_space(k_full, edge_h, H, hd)
                v_e = to_head_space(v_full, edge_h, H, hd)

                scale      = hd ** -0.5
                attn_probs = torch.softmax(
                    (q_e @ k_e.transpose(-2, -1)) * scale, dim=-1
                )
                edge_attn  = attn_probs @ v_e
            raw_latency["edge"] += time.perf_counter() - t_edge

            # Collect and merge all head outputs in device order
            head_parts = []
            for dev in self.all_devices:
                if dev == "edge":
                    if edge_attn is not None:
                        head_parts.append(edge_attn)
                elif dev in attn_futures:
                    res, lat = attn_futures[dev].result()
                    raw_latency[dev] += lat
                    if res is not None and res.numel() > 0:
                        head_parts.append(res.to(dtype=torch.float32, device=ln_x.device))
                    else:
                        h_range = self.arima.get_indices(dev, H)
                        if len(h_range) > 0:
                            head_parts.append(
                                torch.zeros(1, len(h_range), S, hd, device=ln_x.device)
                            )

            ctx      = torch.cat(head_parts, dim=1).transpose(1, 2).reshape(1, S, D)
            attn_out = attn.out_proj(ctx)
            current_state = identity + attn_out
            _blk_attn_ms.append((time.perf_counter() - _t_attn) * 1e3)

            # ── MLP ─────────────────────────────────────────────────────────
            _t_mlp   = time.perf_counter()
            identity  = current_state
            ln_x_mlp  = ln_2(current_state)

            mlp_futures = {}
            for w_name in self.expected_workers:
                n_range = self.arima.get_indices(w_name, self.meta["mlp_hidden_dim"])
                if len(n_range) > 64:
                    mlp_futures[w_name] = self.executor.submit(
                        self._dispatch_task,
                        w_name, "MLP", i,
                        ln_x_mlp.half(), n_range.start, n_range.stop,
                    )
                    share_used[w_name] = (
                        share_used[w_name] + self.arima.current_shares[w_name]
                    ) / 2.0

            edge_n   = self.arima.get_indices("edge", self.meta["mlp_hidden_dim"])
            t_edge   = time.perf_counter()
            edge_mlp = None
            if len(edge_n) > 0:
                w1       = fc1.weight[edge_n.start:edge_n.stop, :]
                b1       = fc1.bias[edge_n.start:edge_n.stop]
                w2       = fc2.weight[:, edge_n.start:edge_n.stop]
                act_fn   = block.mlp.activation_fn
                edge_mlp = act_fn(ln_x_mlp @ w1.t() + b1) @ w2.t()
            raw_latency["edge"] += time.perf_counter() - t_edge

            mlp_parts = [edge_mlp] if edge_mlp is not None else []
            for w_name, fut in mlp_futures.items():
                res, lat = fut.result()
                raw_latency[w_name] += lat
                if res is not None and res.numel() > 0:
                    mlp_parts.append(res.float().to(ln_x_mlp.device))

            current_state = identity + sum_mlp_parts(mlp_parts) + fc2.bias
            _blk_mlp_ms.append((time.perf_counter() - _t_mlp) * 1e3)

            # ── ARIMA bookkeeping ────────────────────────────────────────────
            for dev in self.all_devices:
                s = share_used[dev]
                self.arima.record_block_latency(
                    dev, raw_latency[dev], s if s > 0 else 0.0
                )

            for w_name in probing_workers:
                s = share_used.get(w_name, 0.0)
                if s > 0 and raw_latency[w_name] < PROBE_FAIL_LATENCY:
                    self.arima.notify_probe_result(
                        w_name, raw_latency[w_name] / s
                    )
                else:
                    self.arima.notify_probe_result(w_name, PROBE_FAIL_LATENCY)

        self.last_inference_stats = {
            "per_block_attn_ms":     _blk_attn_ms,
            "per_block_mlp_ms":      _blk_mlp_ms,
            "evaluator_overhead_us": _blk_evaluator_us,
        }

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
    p.add_argument("--port", type=int, default=29500)
    args = p.parse_args()

    orch = MasterOrchestrator(
        expected_workers=["pc_gpu"],
        host="0.0.0.0",
        port=args.port,
    )

    dummy_input = torch.randn(1, 50, 768)

    t0      = time.time()
    output  = orch.run_inference(dummy_input)
    elapsed = time.time() - t0

    print(f"\n[Master] Inference complete in {elapsed:.4f}s  "
          f"|  output shape: {tuple(output.shape)}")

    orch.shutdown()
