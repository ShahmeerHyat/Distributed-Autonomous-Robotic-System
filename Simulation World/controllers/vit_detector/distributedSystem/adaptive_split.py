"""
adaptive_split.py  —  Fixed, working implementation of the paper's
                       adaptive_vit_inference_offloading algorithm.

Source: Edge/AdaptiveSplit.py (SPViT paper).

Fixes applied vs. the original:
  - Params computed from actual tensor byte sizes (was `...` placeholder).
  - FLOPs_f defined before it is used (was referenced before assignment).
  - ARIMA.predict_next() used instead of fit() which returned self (not a number).
  - partitions_prime is actually updated with the new head/neuron counts
    (original always wrote partitions_prime = partitions without changes).
  - Greedy assignment loop uses the correct per-device predicted throughput
    (original used a single undefined `pre_flops_d` for all devices).

Algorithm (paper's Algorithm 1):
  For each minibatch:
    1. Compute device load DL_d = Params_d/BW_d + FLOPs_a_d/FLOPS_d + FLOPs_f_d/FLOPS_d
    2. Predict future FLOPs and BW with ARIMA.
    3. Compute eta = max(DL) - min(DL) - lambda.
    4. If eta > 0 (imbalanced):
         Greedily assign each attention head to the device with the lowest
         predicted load, updating the load estimate after each assignment.
         Do the same for each MLP neuron.
    5. Store the resulting head/neuron counts as the new partition.
"""

import numpy as np
from distributedSystem.arima import ARIMA


# ─────────────────────────────────────────────────────────────────────────────
# FLOPs formulas (from Edge/AdaptiveSplit.py, unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def calculate_flops_MSA(seq_len, dim_head, num_heads):
    """
    FLOPs for Multi-head Self-Attention.

    Original formula uses H*W for spatial dims and C for per-head channels.
    For ViT: H*W = seq_len (num_patches + 1), C = dim_head.
    """
    return (4 * num_heads * seq_len * dim_head ** 2
            + 2 * num_heads * seq_len ** 2)


def calculate_flops_FC(seq_len, num_neurons, embed_dim):
    """FLOPs for one FC layer slice (up-proj + down-proj contribution)."""
    return seq_len * (num_neurons + 1) * embed_dim


def compute_DL(params_bytes, bw, flops_a, flops_f, throughput):
    """
    Device load = communication time + attention compute time + MLP compute time.
    DL = Params/BW + FLOPs_a/FLOPS + FLOPs_f/FLOPS
    """
    return params_bytes / (bw + 1e-9) + (flops_a + flops_f) / (throughput + 1e-9)


def compute_eta(DL_list, lambda_value):
    return abs(max(DL_list) - min(DL_list)) - lambda_value


# ─────────────────────────────────────────────────────────────────────────────
# Partition state
# ─────────────────────────────────────────────────────────────────────────────

class PartitionManager:
    """
    Maintains which device owns which attention heads and MLP neurons.
    Exposes get_head_range(device) and get_neuron_range(device) for the master.
    Call update() after each block with the measured per-device timing.
    """

    def __init__(self, devices: list, num_heads: int, mlp_dim: int,
                 lambda_value: float = 0.05):
        self.devices      = devices
        self.num_heads    = num_heads
        self.mlp_dim      = mlp_dim
        self.lambda_value = lambda_value

        # One ARIMA model per device for FLOPs and bandwidth histories
        self.arima_flops = {d: ARIMA(p=5, d=1, q=0) for d in devices}
        self.arima_bw    = {d: ARIMA(p=5, d=1, q=0) for d in devices}
        self.seq_flops   = {d: [] for d in devices}
        self.seq_bw      = {d: [] for d in devices}

        # Start with equal partition
        self.head_counts   = self._equal_split(num_heads)
        self.neuron_counts = self._equal_split(mlp_dim)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _equal_split(self, total: int) -> dict:
        n    = len(self.devices)
        base = total // n
        counts = {d: base for d in self.devices}
        counts[self.devices[0]] += total - base * n   # give remainder to edge
        return counts

    def get_head_range(self, device: str) -> range:
        start = 0
        for d in self.devices:
            end = start + self.head_counts[d]
            if d == device:
                return range(start, end)
            start = end
        return range(0, 0)

    def get_neuron_range(self, device: str) -> range:
        start = 0
        for d in self.devices:
            end = start + self.neuron_counts[d]
            if d == device:
                return range(start, end)
            start = end
        return range(0, 0)

    # ── Paper's Algorithm 1 ────────────────────────────────────────────────────

    def update(self, exec_times: dict, data_sizes: dict,
               seq_len: int, dim_head: int, embed_dim: int):
        """
        Update the partition after one block.

        exec_times  : dict[device, float]  — measured execution time (seconds)
        data_sizes  : dict[device, int]    — bytes sent to each device this block
        seq_len     : S = num_patches + 1
        dim_head    : head dimension
        embed_dim   : D (full embedding dimension)
        """
        DL = []

        for d in self.devices:
            t      = exec_times.get(d, 1e-3)
            nbytes = data_sizes.get(d, 1)

            BW         = nbytes / (t + 1e-9)
            FLOPs_a    = calculate_flops_MSA(seq_len, dim_head, self.head_counts[d])
            FLOPs_f    = calculate_flops_FC(seq_len, self.neuron_counts[d], embed_dim)
            throughput = (FLOPs_a + FLOPs_f) / (t + 1e-9)

            self.seq_flops[d].append(throughput)
            self.seq_bw[d].append(BW)

            dl = compute_DL(nbytes, BW, FLOPs_a, FLOPs_f, throughput)
            DL.append(dl)

        # ARIMA predictions for next block
        pre_flops = {}
        for d in self.devices:
            hist = self.seq_flops[d]
            pre_flops[d] = (self.arima_flops[d].predict_next(hist)
                            if len(hist) >= 2 else (hist[-1] if hist else 1.0))

        eta = compute_eta(DL, self.lambda_value)
        if eta <= 0:
            return  # partition is balanced, keep as-is

        # ── Greedy head reassignment (paper's inner loop) ─────────────────────
        # Assign each head one-by-one to the device with lowest predicted load.
        # This rebuilds the full assignment from scratch, matching the paper's
        # "for h in range(head_count): d = argmin(DL); assign head h to d" loop.
        temp_DL        = {d: 0.0 for d in self.devices}
        new_head_counts = {d: 0   for d in self.devices}
        flops_per_head  = calculate_flops_MSA(seq_len, dim_head, 1)

        for _ in range(self.num_heads):
            d = min(temp_DL, key=temp_DL.get)
            new_head_counts[d] += 1
            temp_DL[d] += flops_per_head / pre_flops[d]

        # ── Greedy neuron reassignment ─────────────────────────────────────────
        temp_DL          = {d: 0.0 for d in self.devices}
        new_neuron_counts = {d: 0   for d in self.devices}
        flops_per_neuron  = calculate_flops_FC(seq_len, 1, embed_dim)

        for _ in range(self.mlp_dim):
            d = min(temp_DL, key=temp_DL.get)
            new_neuron_counts[d] += 1
            temp_DL[d] += flops_per_neuron / pre_flops[d]

        self.head_counts   = new_head_counts
        self.neuron_counts = new_neuron_counts
