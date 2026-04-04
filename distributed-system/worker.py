import os
import torch
import argparse
import socket
import torchvision.models as models
from shared_utils import recv_msg, send_msg

class ViTWorker:
    def __init__(self, model_name='vit_b_16', use_gpu=True):
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        print(f"Worker initialized on {self.device}")
        self.model = getattr(models, model_name)(weights='DEFAULT').to(self.device).eval()
        self.head_dim = 768 // 12

    def compute_attn_slice(self, block_idx, x, start_h, end_h):
        if start_h >= end_h: return torch.tensor([])
        with torch.no_grad():
            x = x.to(self.device)
            block = self.model.encoder.layers[block_idx]
            w = block.self_attention.in_proj_weight[start_h*self.head_dim : end_h*self.head_dim, :]
            b = block.self_attention.in_proj_bias[start_h*self.head_dim : end_h*self.head_dim]
            return (x @ w.t() + b).cpu()

    def compute_mlp_slice(self, block_idx, x, start_n, end_n):
        if start_n >= end_n: return torch.tensor([])
        with torch.no_grad():
            x = x.to(self.device)
            mlp = self.model.encoder.layers[block_idx].mlp
            w1, b1 = mlp[0].weight[start_n:end_n, :], mlp[0].bias[start_n:end_n]
            w2 = mlp[3].weight[:, start_n:end_n]
            h = torch.nn.functional.gelu(x @ w1.t() + b1)
            return (h @ w2.t()).cpu()

def run_worker_client(name, master_ip, port, use_gpu):
    worker_instance = ViTWorker(use_gpu=use_gpu)
    
    # Setup TCP Socket Client
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print(f"Connecting to Master at {master_ip}:{port}...")
    sock.connect((master_ip, port))
    
    # Send Registration Handshake
    send_msg(sock, ('REGISTER', name))
    print(f"Registered with Master as '{name}'. Waiting for tasks...")
    
    try:
        while True:
            # Wait for instruction payload from Master
            msg = recv_msg(sock)
            if msg is None: break
            
            task, block_idx, x, start_idx, end_idx = msg
            
            if task == 'QUIT':
                print("Shutdown signal received from Master.")
                break
            elif task == 'ATTN':
                res = worker_instance.compute_attn_slice(block_idx, x, start_idx, end_idx)
                send_msg(sock, res)
            elif task == 'MLP':
                res = worker_instance.compute_mlp_slice(block_idx, x, start_idx, end_idx)
                send_msg(sock, res)

    finally:
        sock.close()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True, help="Name of the worker (e.g., pc_gpu)")
    p.add_argument('--master_ip', required=True, help="IP Address of the Master")
    p.add_argument('--port', type=int, default=29500)
    p.add_argument('--gpu', action='store_true')
    args = p.parse_args()
    
    run_worker_client(args.name, args.master_ip, args.port, args.gpu)