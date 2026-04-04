import os, time, torch
import torch.distributed.rpc as rpc
import torchvision.models as models
from shared_utils import get_device_indices, SPViT_ARIMA_Manager

class MasterOrchestrator:
    def __init__(self, worker_names):
        self.workers = worker_names
        self.all_devices = ['edge'] + worker_names
        self.arima = SPViT_ARIMA_Manager(self.all_devices)
        self.model = models.vit_b_16(weights='DEFAULT').eval()
        self.head_dim = 768 // 12

    def run_inference(self, x):
        current_state = x
        for i, block in enumerate(self.model.encoder.layers):
            shares = self.arima.update_shares()
            
            # --- PHASE 1: ATTENTION ---
            ln_x = block.ln_1(current_state)
            attn_futures = {}
            
            # Dispatch to Workers
            for w_name in self.workers:
                h_range = get_device_indices(w_name, self.all_devices, shares, 12)
                if len(h_range) > 0:
                    attn_futures[w_name] = rpc.rpc_async(w_name, worker_instance.compute_attn_slice, 
                                                        args=(i, ln_x, h_range.start, h_range.stop))

            # Edge Local Work
            edge_h = get_device_indices('edge', self.all_devices, shares, 12)
            w_start, w_end = edge_h.start * self.head_dim, edge_h.stop * self.head_dim
            edge_qkv = ln_x @ block.self_attention.in_proj_weight[w_start:w_end, :].t() + \
                       block.self_attention.in_proj_bias[w_start:w_end]

            # Collect & Merge
            qkv_parts = []
            for dev in self.all_devices:
                if dev == 'edge': qkv_parts.append(edge_qkv)
                else: 
                    start_t = time.time()
                    qkv_parts.append(attn_futures[dev].wait())
                    self.arima.record_block_latency(dev, time.time() - start_t)

            # Final Attention Math (Simplified for demo)
            merged_qkv = torch.cat(qkv_parts, dim=-1)
            q, k, v = torch.chunk(merged_qkv, 3, dim=-1)
            # (Standard Vit Attention logic here...)
            attn_out = block.self_attention.out_proj(q) # Simplified projection
            current_state = current_state + attn_out

            # --- PHASE 2: MLP (ONS) ---
            ln_x_mlp = block.ln_2(current_state)
            mlp_futures = {}
            for w_name in self.workers:
                n_range = get_device_indices(w_name, self.all_devices, shares, 3072)
                if len(n_range) > 0:
                    mlp_futures[w_name] = rpc.rpc_async(w_name, worker_instance.compute_mlp_slice,
                                                       args=(i, ln_x_mlp, n_range.start, n_range.stop))
            
            # Collect and Sum (ONS logic)
            mlp_parts = [f.wait() for f in mlp_futures.values()]
            current_state = current_state + torch.sum(torch.stack(mlp_parts), dim=0) + block.mlp[3].bias

        return current_state

# Main execution
if __name__ == "__main__":
    os.environ['MASTER_ADDR'] = '0.0.0.0'
    os.environ['MASTER_PORT'] = '29500'
    rpc.init_rpc("edge", rank=0, world_size=2)
    
    orch = MasterOrchestrator(['pc_gpu'])
    dummy_input = torch.randn(1, 197, 768)
    output = orch.run_inference(dummy_input)
    print("Inference Successful.")
    rpc.shutdown()