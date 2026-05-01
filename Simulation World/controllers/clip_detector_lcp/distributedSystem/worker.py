"""
worker.py  —  LCP CLIP distributed inference worker

Key differences vs clip_detector version:
  - Bug fix: MLP now uses block.mlp.activation_fn (quick_gelu) instead of
    torch.nn.functional.gelu. CLIP ViT-B/32 uses quick_gelu which is a
    faster approximation; using standard gelu produced wrong activations.
  - Reconnection loop: if the master drops or restarts, the worker keeps
    retrying every --reconnect-interval seconds instead of exiting.
  - TCP_NODELAY + enlarged OS buffers via tune_socket() for lower latency.
  - fp16 wire protocol is transparent — send_msg/recv_msg in shared_utils
    handle all conversion automatically.

Usage:
    python worker.py --name pc_gpu --master-host 192.168.1.10
    python worker.py --name laptop  --master-host 192.168.1.10 --port 29500 --gpu
"""

import time
import argparse
import socket
import torch
from transformers import CLIPModel

from shared_utils import recv_msg, send_msg, get_model_metadata, tune_socket

MODEL_PATH = r"Clip Model"


class CLIPWorker:

    def __init__(self, use_gpu: bool = True):
        self.device = torch.device(
            "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        )
        print(f"[Worker] Initialising on: {self.device}")

        clip = CLIPModel.from_pretrained(
            MODEL_PATH, local_files_only=True
        ).to(self.device).eval()

        self.vision = clip.vision_model
        self.meta   = get_model_metadata(self.vision)

        print(f"[Worker] Ready — "
              f"embed={self.meta['embed_dim']}  "
              f"heads={self.meta['num_heads']}  "
              f"head_dim={self.meta['head_dim']}  "
              f"seq={self.meta['seq_length']}")

    # ── Attention ────────────────────────────────────────────────────────────

    def compute_attn_from_slices(
        self,
        block_idx: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """
        Scaled dot-product attention for a head slice.

        q, k, v : (B, H_slice, S, head_dim)  — already sliced by master.
        Returns  : (B, H_slice, S, head_dim)  — master concatenates on head dim.
        """
        with torch.no_grad():
            q = q.to(self.device)
            k = k.to(self.device)
            v = v.to(self.device)
            scale      = self.meta["head_dim"] ** -0.5
            attn_probs = torch.nn.functional.softmax(
                (q @ k.transpose(-2, -1)) * scale, dim=-1
            )
            return (attn_probs @ v).cpu()

    # ── MLP ──────────────────────────────────────────────────────────────────

    def compute_mlp_slice(
        self,
        block_idx: int,
        x: torch.Tensor,
        start_n: int,
        end_n: int,
    ) -> torch.Tensor:
        """
        Partial MLP contribution for neurons [start_n, end_n).

        Uses block.mlp.activation_fn (quick_gelu for CLIP ViT-B/32).
        Returns (B, S, embed_dim). Master sums slices and adds fc2.bias once.
        """
        if start_n >= end_n:
            return torch.tensor([])

        with torch.no_grad():
            x     = x.to(self.device)
            block = self.vision.encoder.layers[block_idx]
            mlp   = block.mlp

            w1 = mlp.fc1.weight[start_n:end_n, :]   # (slice, embed)
            b1 = mlp.fc1.bias[start_n:end_n]         # (slice,)
            w2 = mlp.fc2.weight[:, start_n:end_n]    # (embed, slice)

            # Use the model's own activation (quick_gelu, not standard gelu).
            act_fn = mlp.activation_fn
            hidden = act_fn(x @ w1.t() + b1)         # (B, S, slice)
            return (hidden @ w2.t()).cpu()            # (B, S, embed)


# ─────────────────────────────────────────────────────────────────────────────
# Task server
# ─────────────────────────────────────────────────────────────────────────────

def _serve(worker: CLIPWorker, sock: socket.socket):
    """Process tasks from the master until the connection drops."""
    while True:
        msg = recv_msg(sock)
        if msg is None:
            print("[Worker] Master closed connection.")
            return

        task = msg[0]

        if task == "QUIT":
            print("[Worker] Received QUIT.")
            return

        elif task == "PING":
            send_msg(sock, torch.tensor([1.0]))

        elif task == "ATTN":
            _, block_idx, (q, k, v) = msg
            result = worker.compute_attn_from_slices(block_idx, q, k, v)
            send_msg(sock, result)

        elif task == "MLP":
            _, block_idx, x, start_idx, end_idx = msg
            result = worker.compute_mlp_slice(block_idx, x, start_idx, end_idx)
            send_msg(sock, result)

        else:
            print(f"[Worker] Unknown task: {task!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Main client loop with reconnection
# ─────────────────────────────────────────────────────────────────────────────

def run_worker_client(
    name: str,
    master_host: str,
    port: int,
    use_gpu: bool,
    reconnect_interval: float,
):
    worker = CLIPWorker(use_gpu=use_gpu)

    while True:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            print(f"[Worker] Connecting to {master_host}:{port} …")
            sock.connect((master_host, port))
            tune_socket(sock)

            send_msg(sock, ("REGISTER", name))
            print(f"[Worker] Registered as '{name}'. Waiting for tasks …\n")

            _serve(worker, sock)

        except (ConnectionRefusedError, OSError) as e:
            print(f"[Worker] Connection failed: {e}")
        except Exception as e:
            print(f"[Worker] Unexpected error: {e}")
        finally:
            try:
                sock.close()
            except Exception:
                pass

        print(f"[Worker] Retrying in {reconnect_interval:.0f}s …")
        time.sleep(reconnect_interval)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="LCP CLIP distributed inference worker")
    p.add_argument("--name",               required=True,
                   help="Worker identifier, e.g. pc_gpu")
    p.add_argument("--master-host",        required=True,
                   help="Master IP address")
    p.add_argument("--port",              type=int,   default=29500)
    p.add_argument("--gpu",               action="store_true",
                   help="Use CUDA if available")
    p.add_argument("--reconnect-interval", type=float, default=5.0,
                   help="Seconds to wait before reconnecting after a drop (default 5)")
    args = p.parse_args()

    run_worker_client(
        name               = args.name,
        master_host        = args.master_host,
        port               = args.port,
        use_gpu            = args.gpu,
        reconnect_interval = args.reconnect_interval,
    )
