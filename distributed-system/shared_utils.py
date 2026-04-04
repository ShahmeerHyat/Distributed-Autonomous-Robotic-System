import torch
import numpy as np

def get_device_indices(device_id, device_list, shares, total_count):
    start_idx = 0
    for dev in device_list:
        # If shares is a dict of dicts (from ARIMA-V), extract the right key
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
        # History tracks [latency]
        self.history = {d: [] for d in devices}
        self.current_shares = {d: 1.0/len(devices) for d in devices}

    def record_block_latency(self, device_id, latency):
        self.history[device_id].append(latency)
        if len(self.history[device_id]) > 20: self.history[device_id].pop(0)

    def update_shares(self):
        # Inverse proportional: share = (1/latency) / sum(1/latency)
        preds = {d: (np.mean(self.history[d]) if self.history[d] else 0.05) for d in self.devices}
        inv_latency = {d: 1.0 / preds[d] for d in self.devices}
        total = sum(inv_latency.values())
        self.current_shares = {d: inv_latency[d] / total for d in self.devices}
        return self.current_shares