import os, torch, argparse
import torch.distributed.rpc as rpc
import torchvision.models as models

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
            # ONS: Apply GELU locally before sending back
            h = torch.nn.functional.gelu(x @ w1.t() + b1)
            return (h @ w2.t()).cpu()

def run_worker(name, rank, world_size, master_ip, use_gpu):
    os.environ['MASTER_ADDR'] = master_ip
    os.environ['MASTER_PORT'] = '29500'
    rpc.init_rpc(name, rank=rank, world_size=world_size)
    print(f"RPC Node {name} is online.")
    rpc.shutdown()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True) # e.g. pc_gpu
    p.add_argument('--rank', type=int, required=True)
    p.add_argument('--master_ip', required=True)
    p.add_argument('--gpu', action='store_true')
    args = p.parse_args()
    
    # Instance must be globally accessible for RPC
    worker_instance = ViTWorker(use_gpu=args.gpu)
    run_worker(args.name, args.rank, 2, args.master_ip, args.gpu)