import os, time, torch
import socket
import torchvision.models as models
from concurrent.futures import ThreadPoolExecutor
from shared_utils import get_device_indices, SPViT_ARIMA_Manager, send_msg, recv_msg

class MasterOrchestrator:
    def __init__(self, expected_workers, host='0.0.0.0', port=29500):
        self.expected_workers = expected_workers
        self.worker_names = expected_workers
        self.all_devices = ['edge'] + self.worker_names
        self.arima = SPViT_ARIMA_Manager(self.all_devices)
        self.model = models.vit_b_16(weights='DEFAULT').eval()
        self.head_dim = 768 // 12
        
        # Sockets dict populated by waiting for worker connections
        self.sockets = {}
        self._wait_for_workers(host, port)
            
        self.executor = ThreadPoolExecutor(max_workers=max(1, len(self.worker_names)))

    def _wait_for_workers(self, host, port):
        """Sets up a server and waits for workers to connect and register."""
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((host, port))
        # Listen for exactly the number of expected workers
        server_socket.listen(len(self.expected_workers)) 
        
        print(f"Master listening on {host}:{port}. Waiting for {len(self.expected_workers)} workers to connect...")

        while len(self.sockets) < len(self.expected_workers):
            conn, addr = server_socket.accept()
            msg = recv_msg(conn)
            
            # The first message from a worker should be its registration handshake
            if msg and msg[0] == 'REGISTER':
                w_name = msg[1]
                if w_name in self.expected_workers:
                    self.sockets[w_name] = conn
                    print(f"Worker '{w_name}' connected from {addr}")
                else:
                    print(f"Worker '{w_name}' is not in expected list. Rejecting.")
                    conn.close()
            else:
                print(f"Unknown connection from {addr} rejected.")
                conn.close()

        print("All expected workers are connected and registered!")
        server_socket.close() # Stop listening for new connections

    def _dispatch_task(self, w_name, task_type, block_idx, x, start_idx, end_idx):
        """Thread-safe worker task dispatch."""
        sock = self.sockets[w_name]
        send_msg(sock, (task_type, block_idx, x, start_idx, end_idx))
        return recv_msg(sock)

    def run_inference(self, x):
        current_state = x
        for i, block in enumerate(self.model.encoder.layers):
            shares = self.arima.update_shares()
            
            # --- PHASE 1: ATTENTION ---
            ln_x = block.ln_1(current_state)
            attn_futures = {}
            
            for w_name in self.worker_names:
                h_range = get_device_indices(w_name, self.all_devices, shares, 12)
                if len(h_range) > 0:
                    attn_futures[w_name] = self.executor.submit(
                        self._dispatch_task, w_name, 'ATTN', i, ln_x, h_range.start, h_range.stop
                    )

            edge_h = get_device_indices('edge', self.all_devices, shares, 12)
            w_start, w_end = edge_h.start * self.head_dim, edge_h.stop * self.head_dim
            edge_qkv = ln_x @ block.self_attention.in_proj_weight[w_start:w_end, :].t() + \
                       block.self_attention.in_proj_bias[w_start:w_end]

            qkv_parts = []
            for dev in self.all_devices:
                if dev == 'edge': 
                    qkv_parts.append(edge_qkv)
                else:
                    if dev in attn_futures:
                        start_t = time.time()
                        qkv_parts.append(attn_futures[dev].result()) 
                        self.arima.record_block_latency(dev, time.time() - start_t)

            merged_qkv = torch.cat(qkv_parts, dim=-1)
            q, k, v = torch.chunk(merged_qkv, 3, dim=-1)
            attn_out = block.self_attention.out_proj(q)
            current_state = current_state + attn_out

            # --- PHASE 2: MLP (ONS) ---
            ln_x_mlp = block.ln_2(current_state)
            mlp_futures = {}
            for w_name in self.worker_names:
                n_range = get_device_indices(w_name, self.all_devices, shares, 3072)
                if len(n_range) > 0:
                    mlp_futures[w_name] = self.executor.submit(
                        self._dispatch_task, w_name, 'MLP', i, ln_x_mlp, n_range.start, n_range.stop
                    )
            
            mlp_parts = [f.result() for f in mlp_futures.values()]
            if mlp_parts:
                current_state = current_state + torch.sum(torch.stack(mlp_parts), dim=0) + block.mlp[3].bias

        return current_state

    def shutdown(self):
        for w_name, sock in self.sockets.items():
            try:
                send_msg(sock, ('QUIT', 0, None, 0, 0))
                sock.close()
            except:
                pass
        self.executor.shutdown()

if __name__ == "__main__":
    # We only need to know the names of the workers we expect. 
    # We no longer need their IPs, because THEY are connecting to US.
    expected_workers = ['pc_gpu']
    
    orch = MasterOrchestrator(expected_workers, host='0.0.0.0', port=29500)
    dummy_input = torch.randn(1, 197, 768)
    
    start_time = time.time()
    output = orch.run_inference(dummy_input)
    print(f"Inference Successful in {time.time() - start_time:.4f} seconds.")
    
    orch.shutdown()