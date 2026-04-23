import torch
import argparse
import socket
from transformers import CLIPModel

from shared_utils import recv_msg, send_msg, get_model_metadata

MODEL_PATH = r"../../../Clip Model"


class CLIPWorker:

    def __init__(self, use_gpu: bool = True):
        self.device = torch.device(
            "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        )

        print(f"[Worker] Initialised on: {self.device}")

        clip = CLIPModel.from_pretrained(
            MODEL_PATH,
            local_files_only=True
        ).to(self.device).eval()

        self.vision = clip.vision_model
        self.meta = get_model_metadata(self.vision)

        print(f"[Worker] Ready (embed={self.meta['embed_dim']}, heads={self.meta['num_heads']})")

    # ─────────────────────────────────────────────
    # ATTENTION (NEW CLIP MODE)
    # ─────────────────────────────────────────────
    def compute_attn_from_slices(self, block_idx, q, k, v):

        with torch.no_grad():
            q = q.to(self.device)
            k = k.to(self.device)
            v = v.to(self.device)

            scale = self.meta["head_dim"] ** -0.5

            attn = torch.nn.functional.softmax(
                (q @ k.transpose(-2, -1)) * scale,
                dim=-1
            )

            out = attn @ v
            return out.cpu()

    # ─────────────────────────────────────────────
    # MLP (UNCHANGED)
    # ─────────────────────────────────────────────
    def compute_mlp_slice(self, block_idx, x, start_n, end_n):

        if start_n >= end_n:
            return torch.tensor([])

        with torch.no_grad():
            x = x.to(self.device)
            block = self.vision.encoder.layers[block_idx]
            mlp = block.mlp

            w1 = mlp.fc1.weight[start_n:end_n, :]
            b1 = mlp.fc1.bias[start_n:end_n]
            w2 = mlp.fc2.weight[:, start_n:end_n]

            hidden = torch.nn.functional.gelu(x @ w1.t() + b1)
            return (hidden @ w2.t()).cpu()


# ─────────────────────────────────────────────
def run_worker_client(name, master_ip, port, use_gpu):

    worker = CLIPWorker(use_gpu)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((master_ip, port))

    send_msg(sock, ("REGISTER", name))
    print(f"[Worker] Registered as {name}")

    while True:

        msg = recv_msg(sock)
        if msg is None:
            break

        task = msg[0]

        # ─────────────────────────────
        # QUIT
        # ─────────────────────────────
        if task == "QUIT":
            break

        # ─────────────────────────────
        # PING
        # ─────────────────────────────
        if task == "PING":
            send_msg(sock, torch.tensor([1.0]))
            continue

        # ─────────────────────────────
        # ATTENTION (CLIP MODE)
        # ─────────────────────────────
        if task == "ATTN":

            _, block_idx, payload = msg
            q, k, v = payload

            result = worker.compute_attn_from_slices(
                block_idx, q, k, v
            )

            send_msg(sock, result)

            print(f"[ATTN] block={block_idx} out={tuple(result.shape)}")
            continue

        # ─────────────────────────────
        # MLP (OLD MODE)
        # ─────────────────────────────
        if task == "MLP":

            _, block_idx, x, start_idx, end_idx = msg

            result = worker.compute_mlp_slice(
                block_idx, x, start_idx, end_idx
            )

            send_msg(sock, result)

            print(f"[MLP] block={block_idx} neurons={start_idx}:{end_idx}")
            continue

        print(f"[Worker] Unknown task: {task}")


# ─────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--master_ip", required=True)
    p.add_argument("--port", type=int, default=29500)
    p.add_argument("--gpu", action="store_true")

    args = p.parse_args()

    run_worker_client(
        args.name,
        args.master_ip,
        args.port,
        args.gpu
    )