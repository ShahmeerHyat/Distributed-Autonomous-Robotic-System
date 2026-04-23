"""
worker.py  —  CLIP-aware distributed inference worker

Changes from ViT version:
  - Loads CLIP's vision encoder instead of torchvision vit_b_16
  - Uses CLIP attribute names: self_attn, mlp.fc1, mlp.fc2
  - Accepts an optional --model_path for local CLIP checkpoints
  - PING handler unchanged (just echoes back a scalar tensor)
  - All compute is identical in maths; only attribute names differ
"""

import torch
import argparse
import socket
from transformers import CLIPModel

from shared_utils import recv_msg, send_msg, get_model_metadata


# ─────────────────────────────────────────────────────────────────────────────
class CLIPWorker:

    def __init__(self, model_path: str = None, use_gpu: bool = True):
        self.device = torch.device(
            "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        )
        print(f"[Worker] Initialised on: {self.device}")

        # Load CLIP — prefer a local path if supplied, else HuggingFace hub
        clip_src = model_path or "openai/clip-vit-base-patch32"
        local    = model_path is not None
        clip     = CLIPModel.from_pretrained(
            clip_src, local_files_only=local
        ).to(self.device).eval()

        # We only need the vision encoder on the worker side
        self.vision = clip.vision_model
        self.meta   = get_model_metadata(self.vision)

        print(f"[Worker] CLIP vision encoder ready  "
              f"(embed={self.meta['embed_dim']}  "
              f"heads={self.meta['num_heads']}  "
              f"mlp_hidden={self.meta['mlp_hidden_dim']}  "
              f"seq={self.meta['seq_length']})")

    # ── Attention slice ───────────────────────────────────────────────────────
    
    def compute_attn_slice(
        self, block_idx: int, x: torch.Tensor, start_h: int, end_h: int
    ) -> torch.Tensor:

        if start_h >= end_h:
            return torch.tensor([])

        with torch.no_grad():
            x     = x.to(self.device)
            block = self.vision.encoder.layers[block_idx]

            # --- CLIP: separate projections ---
            q_w = block.self_attn.q_proj.weight
            k_w = block.self_attn.k_proj.weight
            v_w = block.self_attn.v_proj.weight

            q_b = block.self_attn.q_proj.bias
            k_b = block.self_attn.k_proj.bias
            v_b = block.self_attn.v_proj.bias

            embed_dim = self.meta["embed_dim"]
            head_dim  = self.meta["head_dim"]

            qkv_slices = []

            for proj_w, proj_b in [(q_w, q_b), (k_w, k_b), (v_w, v_b)]:
                slices = []
                for h in range(start_h, end_h):
                    start = h * head_dim
                    end   = start + head_dim
                    w_slice = proj_w[start:end, :]
                    b_slice = proj_b[start:end]
                    slices.append(x @ w_slice.t() + b_slice)

                if slices:
                    qkv_slices.append(torch.cat(slices, dim=-1))

            return torch.cat(qkv_slices, dim=-1).cpu() if qkv_slices else torch.tensor([])
    
    # ── MLP slice ─────────────────────────────────────────────────────────────

    def compute_mlp_slice(
        self, block_idx: int, x: torch.Tensor, start_n: int, end_n: int
    ) -> torch.Tensor:
        """
        Computes the contribution of neurons [start_n, end_n) of the MLP for
        the given block.  Returns a (seq, embed_dim) tensor on CPU.

        CLIP MLP layout:
            fc1  : (mlp_hidden_dim, embed_dim)   — up-projection + GELU
            fc2  : (embed_dim, mlp_hidden_dim)   — down-projection
        Note: fc2 bias is added by the master after merging all slices.
        """
        if start_n >= end_n:
            return torch.tensor([])

        with torch.no_grad():
            x     = x.to(self.device)
            block = self.vision.encoder.layers[block_idx]
            mlp   = block.mlp                               # CLIPEncoderMLP

            w1 = mlp.fc1.weight[start_n:end_n, :].to(self.device)   # (slice, embed)
            b1 = mlp.fc1.bias[start_n:end_n].to(self.device)         # (slice,)
            w2 = mlp.fc2.weight[:, start_n:end_n].to(self.device)    # (embed, slice)

            # CLIP uses GELU (same activation used in master.py edge path)
            hidden = torch.nn.functional.gelu(x @ w1.t() + b1)       # (seq, slice)
            return (hidden @ w2.t()).cpu()                             # (seq, embed)


# ─────────────────────────────────────────────────────────────────────────────

def run_worker_client(
    name: str,
    master_ip: str,
    port: int,
    use_gpu: bool,
    model_path: str = None,
):
    worker = CLIPWorker(model_path=model_path, use_gpu=use_gpu)

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

            task, block_idx, x, start_idx, end_idx = msg

            if task == "QUIT":
                print("[Worker] Received QUIT. Shutting down.")
                break

            elif task == "PING":
                # Pre-flight RTT probe — echo a minimal tensor back immediately
                send_msg(sock, torch.tensor([1.0]))

            elif task == "ATTN":
                result = worker.compute_attn_slice(block_idx, x, start_idx, end_idx)
                send_msg(sock, result)
                print(f"  ATTN  block={block_idx:2d}  "
                      f"heads={start_idx}:{end_idx}  out={tuple(result.shape)}")

            elif task == "MLP":
                result = worker.compute_mlp_slice(block_idx, x, start_idx, end_idx)
                send_msg(sock, result)
                print(f"  MLP   block={block_idx:2d}  "
                      f"neurons={start_idx}:{end_idx}  out={tuple(result.shape)}")

            else:
                print(f"[Worker] Unknown task type: {task!r}")

    finally:
        sock.close()
        print("[Worker] Socket closed.")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="CLIP distributed inference worker")
    p.add_argument("--name",       required=True,
                   help="Worker name, e.g. pc_gpu")
    p.add_argument("--master_ip",  required=True,
                   help="Master IP address")
    p.add_argument("--port",       type=int, default=29500)
    p.add_argument("--gpu",        action="store_true",
                   help="Use CUDA if available")
    p.add_argument("--model_path", default=None,
                   help="Path to local CLIP checkpoint directory. "
                        "Omit to download from HuggingFace hub.")
    args = p.parse_args()

    run_worker_client(
        name       = args.name,
        master_ip  = args.master_ip,
        port       = args.port,
        use_gpu    = args.gpu,
        model_path = args.model_path,
    )