"""
worker.py  —  CLIP distributed inference worker

ATTN protocol:
  Receives ("ATTN", block_idx, (q_slice, k_slice, v_slice))
  each shaped (B, H_slice, S, head_dim).
  Returns (B, H_slice, S, head_dim).

MLP protocol:
  Receives ("MLP", block_idx, ln_x_mlp, start_n, end_n)
  Returns (B, S, embed_dim). Master sums slices and adds fc2.bias once.

PROBE protocol:
  Receives {"type": "probe"}
  Returns  {"type": "probe_ack"} immediately, no compute.
"""

import torch
import argparse
import socket
from transformers import CLIPModel

from splitInfer import MultiDeviceEvaluator
from comms import send_msg, recv_msg, probe_rtt, ping_worker, CircuitBreaker
from helper import get_model_metadata, to_head_space, sum_mlp_parts
from helper import AttentionMeta, allocate_heads, split_heads, merge_heads


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

    def compute_attn_from_slices(
        self,
        block_idx: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """
        q, k, v : (B, H_slice, S, head_dim)
        Returns  : (B, H_slice, S, head_dim)
        """
        with torch.no_grad():
            q = q.to(self.device).float()
            k = k.to(self.device).float()
            v = v.to(self.device).float()

            scale      = self.meta["head_dim"] ** -0.5
            attn_probs = torch.nn.functional.softmax(
                (q @ k.transpose(-2, -1)) * scale, dim=-1
            )
            return (attn_probs @ v).cpu().half()

    def compute_mlp_slice(
        self,
        block_idx: int,
        x: torch.Tensor,
        start_n: int,
        end_n: int,
    ) -> torch.Tensor:
        """
        Returns (B, S, embed_dim). Master sums slices and adds fc2.bias once.
        """
        if start_n >= end_n:
            return torch.tensor([])

        with torch.no_grad():
            x     = x.to(self.device).float()
            block = self.vision.encoder.layers[block_idx]
            mlp   = block.mlp

            w1     = mlp.fc1.weight[start_n:end_n, :]
            b1     = mlp.fc1.bias[start_n:end_n]
            w2     = mlp.fc2.weight[:, start_n:end_n]
            act_fn = mlp.activation_fn          # matches master — correct quick_gelu

            hidden = act_fn(x @ w1.t() + b1)   # (B, S, slice)
            return (hidden @ w2.t()).cpu().half() # (B, S, embed_dim)


def run_worker_client(name: str, master_ip: str, port: int, use_gpu: bool):
    worker = CLIPWorker(use_gpu=use_gpu)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
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

            # ── PROBE ─────────────────────────────────────────────────────
            # dict-type messages are probes — handle before tuple unpacking
            if isinstance(msg, dict):
                if msg.get("type") == "probe":
                    send_msg(sock, {"type": "probe_ack"})
                continue

            task = msg[0]

            # ── QUIT ──────────────────────────────────────────────────────
            if task == "QUIT":
                print("[Worker] Received QUIT. Shutting down.")
                break

            # ── ATTN ──────────────────────────────────────────────────────
            elif task == "ATTN":
                _, block_idx, (q, k, v) = msg
                result = worker.compute_attn_from_slices(block_idx, q, k, v)
                send_msg(sock, result)
                # print(f"  ATTN  block={block_idx:2d}  "
                #       f"heads={q.shape[1]}  out={tuple(result.shape)}")

            # ── MLP ───────────────────────────────────────────────────────
            elif task == "MLP":
                _, block_idx, x, start_idx, end_idx = msg
                result = worker.compute_mlp_slice(block_idx, x, start_idx, end_idx)
                send_msg(sock, result)
                # print(f"  MLP   block={block_idx:2d}  "
                #       f"neurons={start_idx}:{end_idx}  out={tuple(result.shape)}")

            else:
                print(f"[Worker] Unknown task: {task!r}")

    finally:
        sock.close()
        print("[Worker] Socket closed.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="CLIP distributed inference worker")
    p.add_argument("--name",      required=True, help="Worker name e.g. pc_gpu")
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