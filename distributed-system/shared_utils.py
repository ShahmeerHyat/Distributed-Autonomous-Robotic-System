import struct
import socket
import io
import torch
import numpy as np

# --- Socket Networking Utilities ---

def send_msg(sock, msg):
    """Serializes a PyTorch object/tensor and sends it over the socket with a length header."""
    buffer = io.BytesIO()
    torch.save(msg, buffer)
    data = buffer.getvalue()
    
    # Prefix each message with a 4-byte length (unsigned integer, big-endian)
    length_header = struct.pack('>I', len(data))
    sock.sendall(length_header + data)

def recvall(sock, n):
    """Helper to ensure we receive exactly 'n' bytes."""
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data

def recv_msg(sock):
    """Reads the length header, then reads the exact payload and deserializes it."""
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    
    msglen = struct.unpack('>I', raw_msglen)[0]
    data = recvall(sock, msglen)
    
    buffer = io.BytesIO(data)
    # weights_only=False is required for custom tuples/lists containing tensors
    return torch.load(buffer, weights_only=False) 

# --- Existing Utils ---

def get_device_indices(device_id, device_list, shares, total_count):
    start_idx = 0
    for dev in device_list:
        share = shares[dev] if isinstance(shares[dev], float) else shares[dev].get('heads', shares[dev].get('neurons', 0))
        count = int(round(share * total_count))
        if share > 0: count = max(1, count)
        
        end_idx = start_idx + count
        if dev == device_id:
            return range(start_idx, min(end_idx, total_count))
        start_idx = end_idx
    return range(0, 0)

class SPViT_ARIMA_Manager:
    def __init__(self, devices):
        self.devices = devices
        self.history = {d: [] for d in devices}
        self.current_shares = {d: 1.0/len(devices) for d in devices}

    def record_block_latency(self, device_id, latency):
        self.history[device_id].append(latency)
        if len(self.history[device_id]) > 20: self.history[device_id].pop(0)

    def update_shares(self):
        preds = {d: (np.mean(self.history[d]) if self.history[d] else 0.05) for d in self.devices}
        inv_latency = {d: 1.0 / preds[d] for d in self.devices}
        total = sum(inv_latency.values())
        self.current_shares = {d: inv_latency[d] / total for d in self.devices}
        return self.current_shares