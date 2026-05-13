"""
master_vit.py  —  ViT MasterOrchestrator (SPViT paper strategy)

Follows the paper's design:
  - Socket protocol: pickle + 6-byte big-endian length header (Connect.py)
  - Load balancing:  PartitionManager using the paper's ARIMA + FLOPs-based
                     adaptive_vit_inference_offloading algorithm (AdaptiveSplit.py)
  - No CLIP-detector components (no MultiDeviceARIMAManager, no circuit breaker,
    no torch.save, no REGISTER/PING protocol)

ATTN split:
  Master runs attn.norm + attn.to_qkv locally (one LayerNorm + one Linear).
  Slices Q, K, V by head range → sends (q_slice, k_slice, v_slice) to worker.
  Worker returns (B, H_slice, S, head_dim).
  Master cat's all parts → rearranges → applies attn.to_out.

MLP split:
  Master runs ff.net[0] (LayerNorm) locally.
  Sends (ln_x, w1_slice, b1_slice, w2_slice) to worker.
  Worker returns GELU(ln_x @ w1.t() + b1) @ w2.t() → (B, S, D).
  Master sums all parts + adds ff.net[4].bias once.
"""

import time
import socket
import torch
from concurrent.futures import ThreadPoolExecutor
from einops import rearrange

from distributedSystem.shared_utils import send_msg, recv_msg
from distributedSystem.adaptive_split import PartitionManager


class ViTMasterOrchestrator:

    def __init__(
        self,
        expected_workers: list,
        host: str        = "0.0.0.0",
        port: int        = 6688,        # same default as paper's Connect.py
        vit_config: dict = {},
        checkpoint: str  = None,
        lambda_value: float = 0.05,     # paper's load-imbalance threshold
    ):
        self.expected_workers = expected_workers
        self.all_devices      = ["edge"] + expected_workers

        # ── Build ViT model ──────────────────────────────────────────────
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from vit_model import ViT

        self.model = ViT(**vit_config)
        self.model.eval()

        if checkpoint:
            state = torch.load(checkpoint, map_location="cpu")
            self.model.load_state_dict(state)
            print(f"[Master] Loaded checkpoint: {checkpoint}")
        else:
            print("[Master] No checkpoint — random weights (verification mode).")

        # ── Partition manager (paper's adaptive split) ───────────────────
        heads   = vit_config["heads"]
        mlp_dim = vit_config["mlp_dim"]
        self.partition = PartitionManager(
            devices      = self.all_devices,
            num_heads    = heads,
            mlp_dim      = mlp_dim,
            lambda_value = lambda_value,
        )

        # Store meta needed for FLOPs calculations
        self.meta = {
            "num_heads":  heads,
            "dim_head":   vit_config.get("dim_head", vit_config["dim"] // heads),
            "mlp_dim":    mlp_dim,
            "dim":        vit_config["dim"],
            "seq_length": (vit_config["image_size"] // vit_config["patch_size"]) ** 2 + 1,
            "scale":      vit_config.get("dim_head", vit_config["dim"] // heads) ** -0.5,
        }

        # ── Network ──────────────────────────────────────────────────────
        self.sockets: dict = {}
        self._wait_for_workers(host, port)
        self.executor = ThreadPoolExecutor(max_workers=max(1, len(expected_workers)))

    # ── Connection ────────────────────────────────────────────────────────

    def _wait_for_workers(self, host: str, port: int):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((host, port))
        srv.listen(len(self.expected_workers))
        print(f"[Master] Waiting for {len(self.expected_workers)} worker(s) "
              f"on {host}:{port} …")

        while len(self.sockets) < len(self.expected_workers):
            conn, addr = srv.accept()
            # Paper's protocol: first message is the worker name
            name = recv_msg(conn)
            self.sockets[name] = conn
            print(f"  Worker '{name}' connected from {addr[0]}")

        srv.close()
        print("[Master] All workers connected.\n")

    # ── Dispatch ──────────────────────────────────────────────────────────

    def _dispatch_attn(self, worker_name, block_idx, q_s, k_s, v_s):
        """
        Send (q_slice, k_slice, v_slice) → receive (B, H_slice, S, hd) result.
        Returns (result_tensor, exec_time_on_worker, bytes_sent).
        """
        sock    = self.sockets[worker_name]
        payload = ("ATTN", block_idx, q_s, k_s, v_s)

        # Measure bytes sent (for BW estimation in adaptive split)
        import pickle
        nbytes  = len(pickle.dumps(payload))

        t0      = time.time()
        send_msg(sock, payload)
        result  = recv_msg(sock)   # (attn_out, worker_exec_time)
        rtt     = time.time() - t0

        if result is None:
            return None, rtt, nbytes
        attn_out, worker_exec_time = result
        return attn_out, worker_exec_time, nbytes

    def _dispatch_mlp(self, worker_name, block_idx, ln_x, w1, b1, w2, n_start, n_end):
        """
        Send (ln_x, w1_slice, b1_slice, w2_slice) → receive (B, S, D) result.
        Returns (result_tensor, exec_time_on_worker, bytes_sent).
        """
        sock    = self.sockets[worker_name]
        payload = ("MLP", block_idx, ln_x, w1, b1, w2, n_start, n_end)

        import pickle
        nbytes  = len(pickle.dumps(payload))

        t0      = time.time()
        send_msg(sock, payload)
        result  = recv_msg(sock)   # (mlp_out, worker_exec_time)
        rtt     = time.time() - t0

        if result is None:
            return None, rtt, nbytes
        mlp_out, worker_exec_time = result
        return mlp_out, worker_exec_time, nbytes

    # ── Inference ─────────────────────────────────────────────────────────

    def run_inference(self, img: torch.Tensor) -> torch.Tensor:
        """
        Full distributed forward pass.
        img : (1, C, H, W)
        Returns logits (1, num_classes).
        """
        m = self.model

        # ── Patch embedding → CLS → positional embedding ─────────────────
        x       = m.to_patch_embedding(img)
        b, n, _ = x.shape
        cls     = m.cls_token.expand(b, -1, -1)
        x       = torch.cat((cls, x), dim=1)
        x      += m.pos_embedding[:, :(n + 1)]

        H     = self.meta["num_heads"]
        hd    = self.meta["dim_head"]
        S     = self.meta["seq_length"]
        D     = self.meta["dim"]
        scale = self.meta["scale"]

        # ── Transformer blocks ────────────────────────────────────────────
        for i, (attn_mod, ff_mod) in enumerate(m.transformer.layers):

            exec_times = {dev: 0.0 for dev in self.all_devices}
            data_sizes = {dev: 0   for dev in self.all_devices}

            # ── ATTENTION ──────────────────────────────────────────────────
            identity = x
            ln_x     = attn_mod.norm(x)

            qkv                    = attn_mod.to_qkv(ln_x)
            q_full, k_full, v_full = qkv.chunk(3, dim=-1)
            q_full = rearrange(q_full, 'b n (h d) -> b h n d', h=H)
            k_full = rearrange(k_full, 'b n (h d) -> b h n d', h=H)
            v_full = rearrange(v_full, 'b n (h d) -> b h n d', h=H)

            # Dispatch worker head slices in parallel
            attn_futures = {}
            for w_name in self.expected_workers:
                h_range = self.partition.get_head_range(w_name)
                if len(h_range) > 0:
                    q_s = q_full[:, h_range, :, :]
                    k_s = k_full[:, h_range, :, :]
                    v_s = v_full[:, h_range, :, :]
                    attn_futures[w_name] = self.executor.submit(
                        self._dispatch_attn, w_name, i, q_s, k_s, v_s
                    )

            # Edge computes its own head slice
            edge_h = self.partition.get_head_range("edge")
            t0 = time.time()
            if len(edge_h) > 0:
                q_e = q_full[:, edge_h, :, :]
                k_e = k_full[:, edge_h, :, :]
                v_e = v_full[:, edge_h, :, :]
                attn_probs = torch.softmax(
                    (q_e @ k_e.transpose(-2, -1)) * scale, dim=-1
                )
                edge_attn = attn_probs @ v_e   # (1, H_e, S, hd)
            else:
                edge_attn = None
            exec_times["edge"] += time.time() - t0

            # Collect results (in device order so head indices stay sorted)
            head_parts = []
            for dev in self.all_devices:
                if dev == "edge":
                    if edge_attn is not None:
                        head_parts.append(edge_attn)
                elif dev in attn_futures:
                    res, wt, nb = attn_futures[dev].result()
                    exec_times[dev] += wt
                    data_sizes[dev] += nb
                    if res is not None and res.numel() > 0:
                        head_parts.append(res.to(ln_x.device))
                    else:
                        h_range = self.partition.get_head_range(dev)
                        head_parts.append(
                            torch.zeros(1, len(h_range), S, hd, device=ln_x.device)
                        )

            ctx      = torch.cat(head_parts, dim=1)           # (1, H, S, hd)
            ctx      = rearrange(ctx, 'b h n d -> b n (h d)') # (1, S, inner_dim)
            attn_out = attn_mod.to_out(ctx)
            x        = identity + attn_out

            # ── MLP ─────────────────────────────────────────────────────────
            identity = x
            ln_x_mlp = ff_mod.net[0](x)

            mlp_futures = {}
            for w_name in self.expected_workers:
                n_range = self.partition.get_neuron_range(w_name)
                if len(n_range) > 0:
                    w1 = ff_mod.net[1].weight[n_range.start:n_range.stop, :]
                    b1 = ff_mod.net[1].bias[n_range.start:n_range.stop]
                    w2 = ff_mod.net[4].weight[:, n_range.start:n_range.stop]
                    mlp_futures[w_name] = self.executor.submit(
                        self._dispatch_mlp,
                        w_name, i, ln_x_mlp, w1, b1, w2,
                        n_range.start, n_range.stop,
                    )

            edge_n = self.partition.get_neuron_range("edge")
            t0 = time.time()
            if len(edge_n) > 0:
                w1 = ff_mod.net[1].weight[edge_n.start:edge_n.stop, :]
                b1 = ff_mod.net[1].bias[edge_n.start:edge_n.stop]
                w2 = ff_mod.net[4].weight[:, edge_n.start:edge_n.stop]
                hidden   = torch.nn.functional.gelu(ln_x_mlp @ w1.t() + b1)
                edge_mlp = hidden @ w2.t()
            else:
                edge_mlp = None
            exec_times["edge"] += time.time() - t0

            mlp_parts = [edge_mlp] if edge_mlp is not None else []
            for w_name, fut in mlp_futures.items():
                res, wt, nb = fut.result()
                exec_times[w_name] += wt
                data_sizes[w_name] += nb
                if res is not None and res.numel() > 0:
                    mlp_parts.append(res.to(ln_x_mlp.device))

            non_empty = [p for p in mlp_parts if p is not None and p.numel() > 0]
            mlp_final = torch.stack(non_empty, dim=0).sum(dim=0) + ff_mod.net[4].bias
            x         = identity + mlp_final

            # ── Adaptive partition update (paper's Algorithm 1) ────────────
            self.partition.update(
                exec_times = exec_times,
                data_sizes = data_sizes,
                seq_len    = S,
                dim_head   = hd,
                embed_dim  = D,
            )

            edge_h_n   = len(self.partition.get_head_range("edge"))
            worker_h_n = sum(
                len(self.partition.get_head_range(w)) for w in self.expected_workers
            )
            t_str = "  ".join(
                f"{d}: {exec_times[d]:.4f}s" for d in self.all_devices
            )
            print(f"  block {i:2d}  "
                  f"heads[edge={edge_h_n} | workers={worker_h_n}]  {t_str}")

        # ── Final norm → pool → head ───────────────────────────────────────
        x = m.transformer.norm(x)
        x = x.mean(dim=1) if m.pool == 'mean' else x[:, 0]
        x = m.to_latent(x)
        return m.mlp_head(x)

    # ── Shutdown ──────────────────────────────────────────────────────────

    def shutdown(self):
        for w_name, sock in self.sockets.items():
            try:
                send_msg(sock, ("QUIT",))
                sock.close()
            except Exception:
                pass
        self.executor.shutdown(wait=False)
        print("[Master] Shut down.")


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--port",       type=int, default=6688)
    p.add_argument("--checkpoint", type=str, default=None)
    args = p.parse_args()

    VIT_CONFIG = dict(
        image_size=32, patch_size=4, num_classes=10,
        dim=512, depth=6, heads=8, mlp_dim=512, dim_head=64,
    )

    orch = ViTMasterOrchestrator(
        expected_workers=["laptop"],
        host="0.0.0.0",
        port=args.port,
        vit_config=VIT_CONFIG,
        checkpoint=args.checkpoint,
    )

    dummy = torch.randn(1, 3, 32, 32)
    t0     = time.time()
    logits = orch.run_inference(dummy)
    print(f"\n[Master] done in {time.time()-t0:.4f}s  "
          f"logits={tuple(logits.shape)}  class={logits.argmax(-1).item()}")

    orch.shutdown()
