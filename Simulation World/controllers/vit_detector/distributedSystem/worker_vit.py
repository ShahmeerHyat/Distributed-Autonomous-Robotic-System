"""
worker_vit.py  —  Stateless ViT distributed inference worker (SPViT paper strategy)

Socket protocol: pickle + 6-byte big-endian length header (paper's Connect.py).
No model loading required — the worker is purely a compute node.

ATTN task:
  Receives (q_slice, k_slice, v_slice), each (B, H_slice, S, head_dim).
  Returns (result, exec_time) where result is (B, H_slice, S, head_dim).

MLP task:
  Receives (ln_x, w1_slice, b1_slice, w2_slice).
  Returns (result, exec_time) where result is (B, S, embed_dim).
  Master sums all worker outputs and adds fc2.bias once.
"""

import time
import torch
import socket
import argparse

from shared_utils import send_msg, recv_msg


def compute_attn(q, k, v):
    """Scaled dot-product attention on a head slice."""
    scale      = q.shape[-1] ** -0.5
    attn_probs = torch.nn.functional.softmax(
        (q @ k.transpose(-2, -1)) * scale, dim=-1
    )
    return attn_probs @ v   # (B, H_slice, S, head_dim)


def compute_mlp(x, w1, b1, w2):
    """
    MLP neuron-slice contribution.
    GELU(x @ w1.t() + b1) @ w2.t()  →  (B, S, embed_dim)
    Master sums all slices and adds fc2.bias once.
    """
    hidden = torch.nn.functional.gelu(x @ w1.t() + b1)
    return hidden @ w2.t()


def run_worker(name: str, master_ip: str, port: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Worker '{name}'] Device: {device}")

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print(f"[Worker '{name}'] Connecting to {master_ip}:{port} …")
    sock.connect((master_ip, port))

    # Paper's protocol: send name as the first message
    send_msg(sock, name)
    print(f"[Worker '{name}'] Registered. Waiting for tasks …\n")

    try:
        while True:
            msg = recv_msg(sock)
            if msg is None:
                print(f"[Worker '{name}'] Master closed connection.")
                break

            task = msg[0]

            # ── QUIT ────────────────────────────────────────────────────
            if task == "QUIT":
                print(f"[Worker '{name}'] QUIT received. Shutting down.")
                break

            # ── ATTN ────────────────────────────────────────────────────
            elif task == "ATTN":
                # ("ATTN", block_idx, q_slice, k_slice, v_slice)
                _, block_idx, q, k, v = msg

                q = q.to(device)
                k = k.to(device)
                v = v.to(device)

                t0     = time.time()
                result = compute_attn(q, k, v).cpu()
                exec_t = time.time() - t0

                send_msg(sock, (result, exec_t))
                print(f"  ATTN  block={block_idx:2d}  "
                      f"heads={q.shape[1]}  {exec_t:.4f}s")

            # ── MLP ─────────────────────────────────────────────────────
            elif task == "MLP":
                # ("MLP", block_idx, ln_x, w1_slice, b1_slice, w2_slice, n_start, n_end)
                _, block_idx, ln_x, w1, b1, w2, n_start, n_end = msg

                ln_x = ln_x.to(device)
                w1   = w1.to(device)
                b1   = b1.to(device)
                w2   = w2.to(device)

                t0     = time.time()
                result = compute_mlp(ln_x, w1, b1, w2).cpu()
                exec_t = time.time() - t0

                send_msg(sock, (result, exec_t))
                print(f"  MLP   block={block_idx:2d}  "
                      f"neurons={n_start}:{n_end}  {exec_t:.4f}s")

            else:
                print(f"[Worker '{name}'] Unknown task: {task!r}")

    finally:
        sock.close()
        print(f"[Worker '{name}'] Socket closed.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="ViT distributed inference worker")
    p.add_argument("--name",      required=True, help="Worker name, e.g. laptop")
    p.add_argument("--master_ip", required=True, help="Master IP address")
    p.add_argument("--port",      type=int, default=6688)
    args = p.parse_args()

    run_worker(name=args.name, master_ip=args.master_ip, port=args.port)
