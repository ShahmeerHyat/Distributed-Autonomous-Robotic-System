import time, torch
import socket
import torchvision.models as models
from concurrent.futures import ThreadPoolExecutor
from shared_utils import (
    MultiDeviceARIMAManager, get_model_metadata, get_head_weights, 
    merge_n_projections, ready_for_math, send_msg, recv_msg
)

class MasterOrchestrator:
    def __init__(self, expected_workers, host='0.0.0.0', port=29500):
        self.expected_workers = expected_workers
        self.all_devices = ['edge'] + expected_workers
        self.arima = MultiDeviceARIMAManager(self.all_devices)
        self.model = models.vit_b_16(weights='DEFAULT').eval()
        self.meta = get_model_metadata(self.model)
        
        self.sockets = {}
        self._wait_for_workers(host, port)
        self.executor = ThreadPoolExecutor(max_workers=max(1, len(expected_workers)))

    def _wait_for_workers(self, host, port):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((host, port))
        server_socket.listen(len(self.expected_workers)) 
        print(f"Waiting for {len(self.expected_workers)} workers on {host}:{port}...")

        while len(self.sockets) < len(self.expected_workers):
            conn, addr = server_socket.accept()
            msg = recv_msg(conn)
            if msg and msg[0] == 'REGISTER':
                w_name = msg[1]
                self.sockets[w_name] = conn
                print(f"Worker '{w_name}' connected.")
        server_socket.close()

    def _dispatch_task(self, w_name, task_type, block_idx, x, start_idx, end_idx):
        sock = self.sockets[w_name]
        send_msg(sock, (task_type, block_idx, x, start_idx, end_idx))
        return recv_msg(sock)

    def run_inference(self, x):
        current_state = x
        
        for i, block in enumerate(self.model.encoder.layers):
            start_block_time = time.time()
            self.arima.update_shares()
            
            # --- PHASE 1: ATTENTION ---
            identity = current_state
            ln_x = block.ln_1(current_state)
            
            # 1. Dispatch Attn to Workers
            attn_futures = {}
            for w_name in self.expected_workers:
                h_range = self.arima.get_indices(w_name, self.meta["num_heads"])
                if len(h_range) > 0:
                    attn_futures[w_name] = self.executor.submit(
                        self._dispatch_task, w_name, 'ATTN', i, ln_x, h_range.start, h_range.stop
                    )

            # 2. Local Edge Attn
            edge_h = self.arima.get_indices('edge', self.meta["num_heads"])
            if len(edge_h) > 0:
                edge_attn_w = get_head_weights(
                    block.self_attention.in_proj_weight, edge_h, 
                    self.meta["embed_dim"], self.meta["head_dim"]
                )
                edge_qkv = ln_x @ edge_attn_w.t()
            else:
                edge_qkv = torch.tensor([])

            # 3. Collect & Merge (Strictly in device order)
            qkv_parts = []
            for dev in self.all_devices:
                if dev == 'edge': 
                    qkv_parts.append(edge_qkv)
                elif dev in attn_futures:
                    qkv_parts.append(attn_futures[dev].result())
            
            merged_qkv = merge_n_projections(qkv_parts)
            
            # 4. Notebook: Add Bias & Global Softmax
            merged_qkv = merged_qkv + block.self_attention.in_proj_bias
            q, k, v = torch.chunk(merged_qkv, 3, dim=-1)
            
            q = ready_for_math(q, self.meta)
            k = ready_for_math(k, self.meta)
            v = ready_for_math(v, self.meta)
            
            scaling = self.meta["head_dim"] ** -0.5
            attn_probs = torch.nn.functional.softmax((q @ k.transpose(-2, -1)) * scaling, dim=-1)
            context_layer = attn_probs @ v
            
            context_layer = context_layer.transpose(1, 2).reshape(1, self.meta["seq_length"], self.meta["embed_dim"])
            attn_output = block.self_attention.out_proj(context_layer)
            current_state = identity + attn_output

            # --- PHASE 2: MLP (ONS) ---
            identity = current_state
            ln_x_mlp = block.ln_2(current_state)
            
            # 1. Dispatch MLP
            mlp_futures = {}
            for w_name in self.expected_workers:
                n_range = self.arima.get_indices(w_name, self.meta["mlp_hidden_dim"])
                if len(n_range) > 0:
                    mlp_futures[w_name] = self.executor.submit(
                        self._dispatch_task, w_name, 'MLP', i, ln_x_mlp, n_range.start, n_range.stop
                    )
            
            # 2. Local Edge MLP
            edge_n = self.arima.get_indices('edge', self.meta["mlp_hidden_dim"])
            if len(edge_n) > 0:
                w1 = block.mlp[0].weight[edge_n.start:edge_n.stop, :]
                b1 = block.mlp[0].bias[edge_n.start:edge_n.stop]
                w2 = block.mlp[3].weight[:, edge_n.start:edge_n.stop]
                edge_mlp = torch.nn.functional.gelu(ln_x_mlp @ w1.t() + b1) @ w2.t()
            else:
                edge_mlp = torch.tensor([])

            # 3. Collect & Sum (ONS Logic)
            mlp_parts = [edge_mlp] if edge_mlp.numel() > 0 else []
            for f in mlp_futures.values():
                res = f.result()
                if res.numel() > 0: mlp_parts.append(res)
            
            mlp_final = torch.sum(torch.stack(mlp_parts), dim=0) + block.mlp[3].bias
            current_state = identity + mlp_final

            # Record total loop time for ARIMA
            block_time = time.time() - start_block_time
            for dev in self.all_devices:
                self.arima.record_block_latency(dev, block_time)
            
            print(f"Block {i} completed. Attn Heads: [Edge: {len(edge_h)} | GPU: {sum([len(self.arima.get_indices(w, self.meta['num_heads'])) for w in self.expected_workers])}]")

        return current_state

    def shutdown(self):
        for w_name, sock in self.sockets.items():
            try:
                send_msg(sock, ('QUIT', 0, None, 0, 0))
                sock.close()
            except: pass
        self.executor.shutdown()

if __name__ == "__main__":
    expected_workers = ['pc_gpu']
    orch = MasterOrchestrator(expected_workers, host='0.0.0.0', port=29500)
    dummy_input = torch.randn(1, 197, 768)
    
    start_t = time.time()
    output = orch.run_inference(dummy_input)
    print(f"Inference Successful in {time.time() - start_t:.4f} seconds.")
    orch.shutdown()