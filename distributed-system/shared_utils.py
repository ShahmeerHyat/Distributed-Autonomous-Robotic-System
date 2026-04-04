import struct
import socket
import io
import torch
import numpy as np

# --- Socket Networking Utilities ---
def send_msg(sock, msg):
    buffer = io.BytesIO()
    torch.save(msg, buffer)
    data = buffer.getvalue()
    length_header = struct.pack('>I', len(data))
    sock.sendall(length_header + data)

def recvall(sock, n):
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet: return None
        data.extend(packet)
    return data

def recv_msg(sock):
    raw_msglen = recvall(sock, 4)
    if not raw_msglen: return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    data = recvall(sock, msglen)
    buffer = io.BytesIO(data)
    return torch.load(buffer, weights_only=False)

# --- Notebook Math Utilities ---
def get_model_metadata(model):
    embed_dim = model.encoder.layers[0].self_attention.in_proj_weight.shape[1]
    seq_length = model.encoder.pos_embedding.shape[1]
    mlp_hidden_dim = model.encoder.layers[0].mlp[0].weight.shape[0]
    num_heads = model.encoder.layers[0].self_attention.num_heads
    return {
        "embed_dim": embed_dim,
        "seq_length": seq_length,
        "mlp_hidden_dim": mlp_hidden_dim,
        "num_heads": num_heads,
        "head_dim": embed_dim // num_heads
    }

def get_head_weights(full_weight, head_indices, embed_dim, head_dim):
    slices = []
    for i in [0, 1, 2]: # Q, K, V
        offset = i * embed_dim
        for h in head_indices:
            start = offset + (h * head_dim)
            end = start + head_dim
            slices.append(full_weight[start:end, :])
    if not slices: return torch.tensor([])
    return torch.cat(slices, dim=0)

def merge_n_projections(projections):
    all_q, all_k, all_v = [], [], []
    for p in projections:
        if p.numel() == 0: continue
        q, k, v = torch.chunk(p, 3, dim=-1)
        all_q.append(q)
        all_k.append(k)
        all_v.append(v)
    
    if not all_q: return torch.tensor([])
    q_final = torch.cat(all_q, dim=-1)
    k_final = torch.cat(all_k, dim=-1)
    v_final = torch.cat(all_v, dim=-1)
    return torch.cat([q_final, k_final, v_final], dim=-1)

def ready_for_math(t, meta):
    return t.view(1, meta["seq_length"], meta["num_heads"], meta["head_dim"]).transpose(1, 2)

# --- Advanced ARIMA-V Manager ---
class MultiDeviceARIMAManager:
    def __init__(self, device_ids, p=3, d=1, q=1, min_share_threshold=0.05):
        self.p, self.d, self.q = p, d, q
        self.min_share_threshold = min_share_threshold
        self.devices = device_ids
        self.history = {dev: [] for dev in device_ids}
        self.residuals = {dev: [] for dev in device_ids}
        self.last_predictions = {dev: 0.0 for dev in device_ids}
        
        initial_share = 1.0 / len(device_ids)
        self.current_shares = {dev: initial_share for dev in device_ids}

    def record_block_latency(self, device_id, latency):
        if self.last_predictions[device_id] > 0:
            error = latency - self.last_predictions[device_id]
            self.residuals[device_id].append(error)
        self.history[device_id].append(latency)
        if len(self.history[device_id]) > 20: self.history[device_id].pop(0)

    def _predict_device_latency(self, dev):
        h = self.history[dev]
        if len(h) < self.p + self.d: return np.mean(h) if h else 0.1
        diffs = [h[i] - h[i-1] for i in range(1, len(h))]
        ar_val = np.mean(diffs[-self.p:])
        ma_val = 0.1 * self.residuals[dev][-1] if self.residuals[dev] else 0
        prediction = h[-1] + ar_val + ma_val
        self.last_predictions[dev] = max(0.001, prediction)
        return self.last_predictions[dev]

    def update_shares(self):
        predictions = {dev: self._predict_device_latency(dev) for dev in self.devices}
        
        # The Edge device is our baseline. It represents the "guaranteed" local speed.
        edge_pred = predictions.get('edge', 0.1) 
        
        # If a network worker is predicted to take X times longer than the Edge device,
        # it has become a severe bottleneck. Cut it off so the system doesn't wait.
        LATENCY_BOTTLENECK_MULTIPLIER = 3.0 
        
        scores = {dev: 1.0 / pred for dev, pred in predictions.items()}
        total_score = sum(scores.values())
        
        # 1. Calculate raw mathematical shares
        raw_shares = {dev: scores[dev] / total_score for dev in self.devices}
        
        # 2. Apply Latency & Payload Thresholding
        valid_scores = {}
        for dev in self.devices:
            if dev == 'edge':
                valid_scores[dev] = scores[dev]
                continue
            
            # Check 1: Is the predicted latency absurdly high? (Network Stutter)
            if predictions[dev] > (edge_pred * LATENCY_BOTTLENECK_MULTIPLIER):
                self.current_shares[dev] = 0.0
                print(f"\n[ARIMA-V] ⚠️ Dropped '{dev}'. Predicted latency ({predictions[dev]:.3f}s) is too slow compared to local edge ({edge_pred:.3f}s).")
            
            # Check 2: Is the payload mathematically too small to justify serialization?
            elif raw_shares[dev] < self.min_share_threshold:
                self.current_shares[dev] = 0.0
                print(f"\n[ARIMA-V] ⚠️ Dropped '{dev}'. Payload share ({raw_shares[dev]*100:.1f}%) is too small to overcome TCP overhead.")
                
            else:
                valid_scores[dev] = scores[dev]
                
        # 3. Re-normalize shares among the surviving, healthy devices
        new_total = sum(valid_scores.values())
        for dev in valid_scores:
            self.current_shares[dev] = valid_scores[dev] / new_total
            
        return self.current_shares

    def get_indices(self, dev, total_items):
        start_idx = 0
        for d in self.devices:
            share = self.current_shares[d]
            count = int(round(share * total_items))
            # count = max(1, count) # Removed forced 1 to allow 0 work if device is terrible
            end_idx = start_idx + count
            if d == dev:
                return range(start_idx, min(end_idx, total_items))
            start_idx = end_idx
        return range(0, 0)