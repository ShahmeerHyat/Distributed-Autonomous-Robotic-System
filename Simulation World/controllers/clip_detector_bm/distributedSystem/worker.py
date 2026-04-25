"""
worker.py  —  CLIP distributed inference worker

ATTN protocol (CLIP mode):
  Receives (q_slice, k_slice, v_slice), each shaped (B, H_slice, S, head_dim).
  Computes scaled dot-product attention per head.
  Returns (B, H_slice, S, head_dim) — master concatenates head outputs.

MLP protocol (unchanged):
  Receives (ln_x_mlp, start_n, end_n).
  Returns gelu(x @ w1_slice.t() + b1_slice) @ w2_slice.t() — (B, S, embed_dim).
  Master sums all slices and adds fc2.bias once.
"""

import torch
import argparse
import socket
from transformers import CLIPModel

from shared_utils import recv_msg, send_msg, get_model_metadata

MODEL_PATH = r"Clip Model"


class CLIPWorker:

    def __init__(self, use_gpu: bool = True):
        self.device = torch.device(
            "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        )
        print(f"[Worker] Initialised on: {self.device}")

        clip = CLIPModel.from_pretrained(
            MODEL_PATH, local_files_only=True
        ).to(self.device).eval()

        self.vision = clip.vision_model
        self.meta   = get_model_metadata(self.vision)

        print(f"[Worker] Ready  "
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
        Compute scaled dot-product attention for a head slice.

        Parameters
        ----------
        q, k, v : (B, H_slice, S, head_dim)
            Pre-sliced and pre-shaped tensors sent by the master.

        Returns
        -------
        (B, H_slice, S, head_dim)  — master will cat these on the head dim.
        """
        with torch.no_grad():
            q = q.to(self.device)   # (B, H_slice, S, head_dim)
            k = k.to(self.device)
            v = v.to(self.device)

            scale      = self.meta["head_dim"] ** -0.5
            # (B, H_slice, S, S)
            attn_probs = torch.nn.functional.softmax(
                (q @ k.transpose(-2, -1)) * scale, dim=-1
            )
            # (B, H_slice, S, head_dim)
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
        Compute the contribution of neurons [start_n, end_n) for a given block.

        fc1 : (mlp_hidden, embed) — up-projection + GELU
        fc2 : (embed, mlp_hidden) — down-projection

        Returns (B, S, embed_dim). Master sums slices and adds fc2.bias once.
        """
        if start_n >= end_n:
            return torch.tensor([])

        with torch.no_grad():
            x     = x.to(self.device)
            block = self.vision.encoder.layers[block_idx]
            mlp   = block.mlp

            w1 = mlp.fc1.weight[start_n:end_n, :]    # (slice, embed)
            b1 = mlp.fc1.bias[start_n:end_n]          # (slice,)
            w2 = mlp.fc2.weight[:, start_n:end_n]     # (embed, slice)

            hidden = torch.nn.functional.gelu(x @ w1.t() + b1)   # (B, S, slice)
            return (hidden @ w2.t()).cpu()                         # (B, S, embed)


# ─────────────────────────────────────────────────────────────────────────────

def run_worker_client(name: str, master_ip: str, port: int, use_gpu: bool):
    worker = CLIPWorker(use_gpu=use_gpu)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print(f"[Worker] Connecting to master at {master_ip}:{port} …")
    sock.connect((master_ip, port))

    send_msg(sock, ("REGISTER", name))
    print(f"[Worker] Registered as '{name}'. Waiting for tasks …\n")

    try:
        while True:
            msg = recv_msg(sock)
            if msg is None:
                print("[Worker] Master closed connection.")
                break

            task = msg[0]

            # ── QUIT ─────────────────────────────────────────────────────
            if task == "QUIT":
                print("[Worker] Received QUIT. Shutting down.")
                break

            # ── PING ─────────────────────────────────────────────────────
            elif task == "PING":
                send_msg(sock, torch.tensor([1.0]))

            # ── ATTN ─────────────────────────────────────────────────────
            elif task == "ATTN":
                # msg = ("ATTN", block_idx, (q_slice, k_slice, v_slice))
                _, block_idx, (q, k, v) = msg

                result = worker.compute_attn_from_slices(block_idx, q, k, v)
                send_msg(sock, result)

                print(f"  ATTN  block={block_idx:2d}  "
                      f"heads={q.shape[1]}  out={tuple(result.shape)}")

            # ── MLP ──────────────────────────────────────────────────────
            elif task == "MLP":
                # msg = ("MLP", block_idx, ln_x_mlp, start_n, end_n)
                _, block_idx, x, start_idx, end_idx = msg

                result = worker.compute_mlp_slice(block_idx, x, start_idx, end_idx)
                send_msg(sock, result)

                print(f"  MLP   block={block_idx:2d}  "
                      f"neurons={start_idx}:{end_idx}  out={tuple(result.shape)}")

            else:
                print(f"[Worker] Unknown task: {task!r}")

    finally:
        sock.close()
        print("[Worker] Socket closed.")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="CLIP distributed inference worker")
    p.add_argument("--name",      required=True, help="Worker name, e.g. pc_gpu")
    p.add_argument("--master_ip", required=True, help="Master IP address")
    p.add_argument("--port",      type=int, default=29500)
    p.add_argument("--gpu",       action="store_true", help="Use CUDA if available")
    args = p.parse_args()

    run_worker_client(
        name      = args.name,
        master_ip = args.master_ip,
        port      = args.port,
        use_gpu   = args.gpu,
    )