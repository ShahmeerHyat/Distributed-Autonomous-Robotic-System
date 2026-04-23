import torch
import argparse
import socket
import torchvision.models as models
from shared_utils import recv_msg, send_msg, get_model_metadata, get_head_weights


class ViTWorker:
    def __init__(self, model_name="vit_b_16", use_gpu=True):
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        print(f"Worker initialised on: {self.device}")
        self.model = getattr(models, model_name)(weights="DEFAULT").to(self.device).eval()
        self.meta  = get_model_metadata(self.model)

    def compute_attn_slice(self, block_idx: int, x: torch.Tensor, start_h: int, end_h: int):
        if start_h >= end_h:
            return torch.tensor([])
        with torch.no_grad():
            x       = x.to(self.device)
            block   = self.model.encoder.layers[block_idx]
            attn_w  = get_head_weights(
                block.self_attention.in_proj_weight,
                range(start_h, end_h),
                self.meta["embed_dim"],
                self.meta["head_dim"],
            ).to(self.device)
            return (x @ attn_w.t()).cpu()

    def compute_mlp_slice(self, block_idx: int, x: torch.Tensor, start_n: int, end_n: int):
        if start_n >= end_n:
            return torch.tensor([])
        with torch.no_grad():
            x   = x.to(self.device)
            mlp = self.model.encoder.layers[block_idx].mlp
            w1  = mlp[0].weight[start_n:end_n, :]
            b1  = mlp[0].bias[start_n:end_n]
            w2  = mlp[3].weight[:, start_n:end_n]
            h   = torch.nn.functional.gelu(x @ w1.t() + b1)
            return (h @ w2.t()).cpu()


def run_worker_client(name: str, master_ip: str, port: int, use_gpu: bool):
    worker = ViTWorker(use_gpu=use_gpu)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print(f"Connecting to master at {master_ip}:{port}…")
    sock.connect((master_ip, port))

    send_msg(sock, ("REGISTER", name))
    print(f"Registered as '{name}'. Waiting for tasks…\n")

    try:
        while True:
            msg = recv_msg(sock)
            if msg is None:
                print("Master closed connection.")
                break

            task, block_idx, x, start_idx, end_idx = msg

            if task == "QUIT":
                print("Received QUIT. Shutting down.")
                break

            elif task == "PING":
                # Pre-flight RTT probe — echo a tiny tensor back immediately
                send_msg(sock, torch.tensor([1.0]))

            elif task == "ATTN":
                result = worker.compute_attn_slice(block_idx, x, start_idx, end_idx)
                send_msg(sock, result)
                print(f"  ATTN  block={block_idx}  heads={start_idx}:{end_idx}  out={result.shape}")

            elif task == "MLP":
                result = worker.compute_mlp_slice(block_idx, x, start_idx, end_idx)
                send_msg(sock, result)
                print(f"  MLP   block={block_idx}  neurons={start_idx}:{end_idx}  out={result.shape}")

            else:
                print(f"Unknown task type: {task!r}")

    finally:
        sock.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="ViT distributed inference worker")
    p.add_argument("--name",      required=True, help="Worker name, e.g. pc_gpu")
    p.add_argument("--master_ip", required=True, help="Master IP address")
    p.add_argument("--port",      type=int, default=29500)
    p.add_argument("--gpu",       action="store_true", help="Use CUDA if available")
    args = p.parse_args()

    run_worker_client(args.name, args.master_ip, args.port, args.gpu)