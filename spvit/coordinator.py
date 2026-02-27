"""
SPViT Coordinator — Simulated Multi-Device Inference
=====================================================
Simulates SPViT's distributed inference on ONE computer.
Runs multiple worker processes connected via localhost sockets.

Architecture:
    Coordinator (this file)
        ├── Sends image patches to Worker 0 (heads 0-3)
        ├── Sends image patches to Worker 1 (heads 4-7)
        └── Merges partial outputs → final classification

Each worker = one simulated edge device (Pi, Manifold, etc.)
"""

import socket
import pickle
import threading
import time
import torch
import torch.nn as nn
import numpy as np
from arima_v import ARIMAVScheduler


# ─────────────────────────────────────────────
# NETWORK CONFIG
# ─────────────────────────────────────────────
COORDINATOR_HOST = '127.0.0.1'
BASE_PORT = 6700  # workers listen on 6700, 6701, 6702...


def send_tensor(sock, tensor):
    """Serialize and send a tensor over socket."""
    data = pickle.dumps(tensor)
    length = len(data).to_bytes(8, byteorder='big')
    sock.sendall(length + data)


def recv_tensor(sock):
    """Receive and deserialize a tensor from socket."""
    raw_len = _recv_exactly(sock, 8)
    if not raw_len:
        return None
    length = int.from_bytes(raw_len, byteorder='big')
    data = _recv_exactly(sock, length)
    return pickle.loads(data)


def _recv_exactly(sock, n):
    """Receive exactly n bytes."""
    buf = b''
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


# ─────────────────────────────────────────────
# WORKER DEVICE (runs in separate process/thread)
# ─────────────────────────────────────────────

class WorkerDevice:
    """
    Simulates one edge device (e.g. Raspberry Pi, Manifold).
    Listens for image tensors, runs partial ViT inference,
    sends partial output back.

    In real SPViT: this runs on actual hardware.
    In our simulation: runs as a thread on localhost.
    """

    def __init__(self, device_id, port, head_indices, embed_dim=256, num_classes=2):
        self.device_id = device_id
        self.port = port
        self.head_indices = head_indices
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.model = None  # loaded when worker starts
        self.running = False

        # Simulated compute speed (FLOPS) — varies to mimic real devices
        # Device 0 = fast (like Manifold), Device 1+ = slower (like Pi)
        self.base_flops = 1e9 / (device_id + 1)
        self.base_bw = 160e6 / (device_id + 1)  # 160 Mb/s for device 0

    def _simulate_compute_delay(self, flops_needed):
        """Simulate realistic compute time based on device speed."""
        delay = flops_needed / self.base_flops
        # Add small random noise (temperature, other processes, etc.)
        delay *= np.random.uniform(0.9, 1.1)
        time.sleep(min(delay, 0.1))  # cap at 100ms for simulation speed

    def run_partial_inference(self, x, partial_weights):
        """
        Run attention for our assigned heads only.
        x: (B, N, D) — full input token sequence
        partial_weights: dict of weight tensors for our heads
        Returns: partial output tensor (B, N, D)
        """
        import torch
        B, N, D = x.shape
        num_heads = len(self.head_indices)
        head_dim = D // 8  # assume 8 total heads

        # Extract our Q, K, V weights
        W_qkv = partial_weights['qkv_weight']   # (3 * num_heads * head_dim, D)
        W_out = partial_weights['out_weight']   # (D, num_heads * head_dim)

        partial_dim = num_heads * head_dim

        # Compute partial Q, K, V
        qkv = x @ W_qkv.T  # (B, N, 3 * partial_dim)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3)

        # Attention
        scale = head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).permute(0, 2, 1, 3).reshape(B, N, partial_dim)

        # Partial projection
        partial_out = out @ W_out.T  # (B, N, D)

        # Simulate device compute time
        flops = 4 * num_heads * N * D * D + 2 * num_heads * (N ** 2)
        self._simulate_compute_delay(flops)

        return partial_out

    def start(self):
        """Start listening for work from coordinator."""
        self.running = True
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((COORDINATOR_HOST, self.port))
        self.server.listen(1)

        print(f"[Worker {self.device_id}] Listening on port {self.port} "
              f"| Heads: {self.head_indices}")

        conn, addr = self.server.accept()
        print(f"[Worker {self.device_id}] Connected to coordinator")

        while self.running:
            try:
                payload = recv_tensor(conn)
                if payload is None:
                    break
                if payload == 'STOP':
                    break

                x = payload['x']
                weights = payload['weights']
                t_start = time.time()

                # Run partial inference
                partial_out = self.run_partial_inference(x, weights)

                t_end = time.time()
                result = {
                    'partial_out': partial_out,
                    'compute_time': t_end - t_start,
                    'device_id': self.device_id,
                }
                send_tensor(conn, result)

            except (ConnectionResetError, BrokenPipeError):
                break

        conn.close()
        self.server.close()
        print(f"[Worker {self.device_id}] Stopped")


# ─────────────────────────────────────────────
# COORDINATOR — main device (simulated Pi)
# ─────────────────────────────────────────────

class SPViTCoordinator:
    """
    Coordinator device — manages the full inference pipeline.

    1. Receives image from Webots camera
    2. Asks ARIMA-V how to split heads across devices
    3. Sends partial work to each worker via socket
    4. Collects results and merges
    5. Returns final classification
    """

    def __init__(self, model, num_devices=3, total_heads=8,
                 embed_dim=256, num_classes=2):
        self.model = model
        self.model.eval()
        self.num_devices = num_devices
        self.total_heads = total_heads
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        # ARIMA-V scheduler
        self.scheduler = ARIMAVScheduler(
            device_count=num_devices,
            lambda_threshold=0.05
        )

        # Connections to workers
        self.worker_sockets = []
        self.head_partition = None

        # Performance tracking
        self.inference_times = []

    def connect_to_workers(self):
        """Connect to all worker devices (they must be started first)."""
        print("[Coordinator] Connecting to workers...")
        for d in range(self.num_devices):
            port = BASE_PORT + d
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # Retry until worker is ready
            for attempt in range(20):
                try:
                    sock.connect((COORDINATOR_HOST, port))
                    print(f"[Coordinator] Connected to worker {d} on port {port}")
                    break
                except ConnectionRefusedError:
                    time.sleep(0.5)
            self.worker_sockets.append(sock)

    def _extract_partial_weights(self, block_idx, head_indices):
        """
        Extract Q,K,V and output projection weights for specific heads.
        Used to send only the relevant weights to each worker device.
        """
        block = self.model.blocks[block_idx]
        full_qkv_w = block.attn.qkv.weight.data   # (3*D, D)
        full_out_w = block.attn.proj.weight.data   # (D, D)

        D = self.embed_dim
        head_dim = D // self.total_heads
        num_heads = len(head_indices)

        # Extract rows corresponding to our heads from QKV weight
        # QKV is [Q_head0...Q_headN | K_head0...K_headN | V_head0...V_headN]
        q_rows = [h * head_dim + i for h in head_indices for i in range(head_dim)]
        k_rows = [D + h * head_dim + i for h in head_indices for i in range(head_dim)]
        v_rows = [2*D + h * head_dim + i for h in head_indices for i in range(head_dim)]
        all_rows = q_rows + k_rows + v_rows

        partial_qkv_w = full_qkv_w[all_rows, :]   # (3 * num_heads * head_dim, D)

        # Extract columns for output projection
        out_cols = [h * head_dim + i for h in head_indices for i in range(head_dim)]
        partial_out_w = full_out_w[:, out_cols]    # (D, num_heads * head_dim)

        return {
            'qkv_weight': partial_qkv_w.detach(),
            'out_weight': partial_out_w.detach(),
        }

    def _get_head_indices_per_device(self, head_counts):
        """Convert head counts to explicit head index lists per device."""
        head_indices = []
        start = 0
        for count in head_counts:
            head_indices.append(list(range(start, start + count)))
            start += count
        return head_indices

    def infer(self, image_tensor):
        """
        Full SPViT inference pipeline.

        Args:
            image_tensor: (1, 3, H, W) from Webots camera

        Returns:
            class_id (int): 0=no_person, 1=person
            confidence (float): 0-1
            stats (dict): timing and partition info
        """
        t_total_start = time.time()

        with torch.no_grad():
            # ── Step 1: Patch embedding (runs on coordinator) ─────────────────
            x = self.model.patch_embed(image_tensor)   # (1, N, D)
            B, N, D = x.shape

            # Add CLS token and position embeddings
            cls = self.model.cls_token.expand(B, -1, -1)
            x = torch.cat([cls, x], dim=1)
            x = self.model.dropout(x + self.model.pos_embed)

            # ── Step 2: Get partition from ARIMA-V ────────────────────────────
            # Patch height/width for FLOP calculations
            patch_h = patch_w = 32 // 4  # image_size // patch_size = 8

            head_counts, neuron_counts = self.scheduler.get_partition(
                total_heads=self.total_heads,
                total_neurons=D * 4,        # MLP hidden dim
                img_h=patch_h,
                img_w=patch_w,
                channels=D,
                total_neurons_in=D,
            )

            self.head_partition = head_counts
            head_indices_per_device = self._get_head_indices_per_device(head_counts)

            print(f"[Coordinator] Partition: {head_counts} heads per device")

            # ── Step 3: Process each Transformer block ────────────────────────
            for block_idx, block in enumerate(self.model.blocks):
                residual = x

                # LayerNorm (coordinator)
                x_norm = block.norm1(x)

                # ── Parallel attention across devices ─────────────────────────
                t_send = time.time()

                # Send work to all devices simultaneously
                partial_outs = [None] * self.num_devices
                threads = []

                def send_to_device(d, sock, x_norm, weights):
                    payload = {'x': x_norm.detach(), 'weights': weights}
                    send_tensor(sock, payload)

                def recv_from_device(d, sock):
                    result = recv_tensor(sock)
                    partial_outs[d] = result

                # If we have live workers, use them; otherwise local fallback
                if self.worker_sockets:
                    # Send to all workers in parallel
                    for d in range(self.num_devices):
                        if not head_indices_per_device[d]:
                            continue
                        weights = self._extract_partial_weights(
                            block_idx, head_indices_per_device[d]
                        )
                        t = threading.Thread(
                            target=send_to_device,
                            args=(d, self.worker_sockets[d], x_norm, weights)
                        )
                        threads.append(t)
                        t.start()
                    for t in threads:
                        t.join()

                    # Receive from all workers in parallel
                    threads = []
                    for d in range(self.num_devices):
                        if not head_indices_per_device[d]:
                            continue
                        t = threading.Thread(
                            target=recv_from_device,
                            args=(d, self.worker_sockets[d])
                        )
                        threads.append(t)
                        t.start()
                    for t in threads:
                        t.join()

                    # Merge partial outputs (sum, as per paper Eq. 5)
                    attn_out = torch.zeros_like(x_norm)
                    for d in range(self.num_devices):
                        if partial_outs[d] is not None:
                            attn_out += partial_outs[d]['partial_out']

                else:
                    # Fallback: run full attention locally (no workers connected)
                    attn_out = block.attn(x_norm)

                t_recv = time.time()
                comm_time = t_recv - t_send

                # Residual connection
                x = residual + attn_out

                # MLP (runs on coordinator for now)
                x = x + block.mlp(block.norm2(x))

                # Update ARIMA-V metrics
                for d in range(self.num_devices):
                    simulated_flops = 1e9 / (d + 1) * np.random.uniform(0.8, 1.2)
                    simulated_bw = 160e6 / (d + 1) * np.random.uniform(0.7, 1.3)
                    self.scheduler.update_metrics(d, simulated_flops, simulated_bw)

            # ── Step 4: Final classification (coordinator) ────────────────────
            x = self.model.norm(x)
            logits = self.model.head(x[:, 0])   # CLS token → class scores
            probs = logits.softmax(dim=-1)

        t_total = time.time() - t_total_start
        self.inference_times.append(t_total)

        class_id = int(probs.argmax(dim=-1).item())
        confidence = float(probs.max().item())

        stats = {
            'inference_time_ms': t_total * 1000,
            'head_partition': head_counts,
            'avg_inference_ms': np.mean(self.inference_times) * 1000
        }

        return class_id, confidence, stats

    def stop(self):
        """Send stop signal to all workers and close connections."""
        for sock in self.worker_sockets:
            try:
                send_tensor(sock, 'STOP')
                sock.close()
            except Exception:
                pass
        print("[Coordinator] Stopped all workers")
