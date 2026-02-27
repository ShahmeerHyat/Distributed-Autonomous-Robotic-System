"""
ARIMA-V: Adaptive Splitting and Offloading Algorithm
=====================================================
Cleaned-up implementation from SPViT paper + repo skeleton.
Predicts each device's compute/bandwidth capacity and
allocates attention heads + FC neurons accordingly.
"""

import numpy as np
import math


class ARIMA:
    """
    Auto-Regressive Integrated Moving Average model.
    Predicts next value from historical sequence.
    Used to predict device FLOPS and bandwidth per minibatch.
    """
    def __init__(self, p=3, d=1, q=0):
        self.p = p  # autoregressive order
        self.d = d  # differencing order
        self.q = q  # moving average order

    def _difference(self, data):
        """Apply d-th order differencing to make series stationary."""
        diff = list(data)
        for _ in range(self.d):
            diff = [diff[i] - diff[i-1] for i in range(1, len(diff))]
        return diff

    def _inverse_difference(self, original, prediction):
        """Undo differencing to get back to original scale."""
        result = prediction
        for i in range(self.d):
            result = result + original[-(i+1)]
        return result

    def predict_next(self, history):
        """
        Predict the next value given history of observations.
        Uses simple AR model on differenced series.
        Falls back to mean if not enough history.
        """
        if len(history) < self.p + self.d + 1:
            return np.mean(history) if history else 1.0

        # Difference the series
        diff = self._difference(history)

        if len(diff) < self.p:
            return self._inverse_difference(history, np.mean(diff))

        # AR(p) prediction: weighted sum of last p values
        # Simple least-squares coefficients
        recent = diff[-self.p:]
        # Equal weights for simplicity (production would use OLS)
        weights = np.array([1.0 / self.p] * self.p)
        prediction_diff = float(np.dot(weights, recent))

        # Inverse difference back to original scale
        return self._inverse_difference(history, prediction_diff)


# ─────────────────────────────────────────────────────────
# FLOP Calculators (from SPViT paper equations 6 and 7)
# ─────────────────────────────────────────────────────────

def calculate_flops_msa(height, width, channels, num_heads):
    """
    FLOPs for MSA partition (Eq. 6 in paper).
    F = 4 * H * v * w * C^2 + 2 * H * (v*w)^2
    H = heads assigned to this device
    """
    return 4 * num_heads * height * width * (channels ** 2) + \
           2 * num_heads * (height * width) ** 2


def calculate_flops_fc(height, width, num_output_neurons, num_input_neurons):
    """
    FLOPs for FC (MLP) partition (Eq. 7 in paper).
    F = v * w * (O + 1) * I
    """
    return height * width * (num_output_neurons + 1) * num_input_neurons


# ─────────────────────────────────────────────────────────
# ARIMA-V: Main Adaptive Algorithm
# ─────────────────────────────────────────────────────────

class ARIMAVScheduler:
    """
    Adaptive partition scheduler using ARIMA prediction.

    Tracks each device's:
      - FLOPS history (compute capacity)
      - BW history (bandwidth capacity)

    Then allocates heads and neurons to minimize
    the difference in total delay across all devices.
    """

    def __init__(self, device_count, lambda_threshold=0.05):
        """
        Args:
            device_count:      number of simulated devices
            lambda_threshold:  only re-split if imbalance > this (Eq. 14)
        """
        self.device_count = device_count
        self.lambda_threshold = lambda_threshold

        self.seq_flops = [[] for _ in range(device_count)]
        self.seq_bw    = [[] for _ in range(device_count)]

        self.arima_flops = [ARIMA(p=3, d=1, q=0) for _ in range(device_count)]
        self.arima_bw    = [ARIMA(p=3, d=1, q=0) for _ in range(device_count)]

        # Current partition: how many heads per device
        self.current_head_partition = None
        self.current_neuron_partition = None

    def update_metrics(self, device_id, flops, bandwidth):
        """
        Call after each minibatch with measured device performance.
        flops: operations per second achieved
        bandwidth: MB/s achieved
        """
        self.seq_flops[device_id].append(flops)
        self.seq_bw[device_id].append(bandwidth)

    def predict_capacities(self):
        """Predict next minibatch's FLOPS and BW for all devices."""
        pred_flops = []
        pred_bw = []
        for d in range(self.device_count):
            pf = self.arima_flops[d].predict_next(self.seq_flops[d])
            pb = self.arima_bw[d].predict_next(self.seq_bw[d])
            pred_flops.append(max(pf, 1e-6))  # avoid div by zero
            pred_bw.append(max(pb, 1e-6))
        return pred_flops, pred_bw

    def compute_delay(self, pred_flops, pred_bw, head_counts, neuron_counts,
                      img_h, img_w, channels, total_neurons_in, total_neurons_out,
                      params_per_device):
        """
        Compute estimated total delay per device (Eq. 13 in paper).
        DL = transmission_delay + MSA_delay + FC_delay
        """
        delays = []
        for d in range(self.device_count):
            # Transmission delay
            dt = params_per_device[d] / pred_bw[d]

            # MSA computation delay
            flops_msa = calculate_flops_msa(img_h, img_w, channels, head_counts[d])
            da = flops_msa / pred_flops[d]

            # FC computation delay
            flops_fc = calculate_flops_fc(img_h, img_w, neuron_counts[d], total_neurons_in)
            df = flops_fc / pred_flops[d]

            delays.append(dt + da + df)
        return delays

    def should_repartition(self, delays):
        """
        Check if imbalance exceeds threshold lambda (Eq. 14 in paper).
        eta = |max(DL) - min(DL)| - lambda
        Returns True if we need to re-split.
        """
        eta = abs(max(delays) - min(delays)) - self.lambda_threshold
        return eta > 0

    def allocate_heads(self, total_heads, pred_flops, pred_bw,
                       img_h, img_w, channels, params_per_device,
                       total_neurons_in, neuron_counts):
        """
        Greedy head allocation: assign each head to device with smallest current delay.
        Ensures faster devices get more heads.
        """
        # Start with transmission delays only
        dl = [params_per_device[d] / pred_bw[d] for d in range(self.device_count)]
        head_counts = [0] * self.device_count

        # Assign each head to device with minimum current delay
        for h in range(total_heads):
            d = int(np.argmin(dl))
            head_counts[d] += 1
            flops_h = calculate_flops_msa(img_h, img_w, channels, 1)
            dl[d] += flops_h / pred_flops[d]

        return head_counts

    def allocate_neurons(self, total_neurons, pred_flops, pred_bw,
                         img_h, img_w, total_neurons_in,
                         params_per_device, head_counts, channels):
        """
        Greedy neuron allocation: assign each neuron to fastest available device.
        """
        # Start DL from head allocation result
        dl = []
        for d in range(self.device_count):
            flops_msa = calculate_flops_msa(img_h, img_w, channels, head_counts[d])
            da = flops_msa / pred_flops[d]
            dt = params_per_device[d] / pred_bw[d]
            dl.append(dt + da)

        neuron_counts = [0] * self.device_count
        for n in range(total_neurons):
            d = int(np.argmin(dl))
            neuron_counts[d] += 1
            flops_n = calculate_flops_fc(img_h, img_w, 1, total_neurons_in)
            dl[d] += flops_n / pred_flops[d]

        return neuron_counts

    def get_partition(self, total_heads, total_neurons,
                      img_h, img_w, channels, total_neurons_in,
                      params_per_device=None):
        """
        Main entry point. Returns head and neuron allocation per device.

        Args:
            total_heads:       total attention heads in model
            total_neurons:     total output neurons in MLP layer
            img_h, img_w:      patch height/width
            channels:          embedding channels
            total_neurons_in:  input neurons to MLP layer
            params_per_device: estimated bytes to transmit per device

        Returns:
            head_counts   [list]: heads assigned to each device
            neuron_counts [list]: neurons assigned to each device
        """
        if params_per_device is None:
            # Default: equal transmission assumed
            params_per_device = [1000] * self.device_count

        pred_flops, pred_bw = self.predict_capacities()

        # Initial equal split if no history yet
        if not self.seq_flops[0]:
            heads_each = total_heads // self.device_count
            neurons_each = total_neurons // self.device_count
            head_counts = [heads_each] * self.device_count
            neuron_counts = [neurons_each] * self.device_count
            # Give remainder to device 0
            head_counts[0] += total_heads - sum(head_counts)
            neuron_counts[0] += total_neurons - sum(neuron_counts)
            self.current_head_partition = head_counts
            self.current_neuron_partition = neuron_counts
            return head_counts, neuron_counts

        # Check if re-partition is needed
        if self.current_head_partition is not None:
            delays = self.compute_delay(
                pred_flops, pred_bw,
                self.current_head_partition, self.current_neuron_partition,
                img_h, img_w, channels, total_neurons,
                total_neurons_in, params_per_device
            )
            if not self.should_repartition(delays):
                # Imbalance is acceptable — keep current partition
                return self.current_head_partition, self.current_neuron_partition

        # Re-partition using greedy algorithm
        head_counts = self.allocate_heads(
            total_heads, pred_flops, pred_bw,
            img_h, img_w, channels, params_per_device,
            total_neurons_in, [0] * self.device_count
        )
        neuron_counts = self.allocate_neurons(
            total_neurons, pred_flops, pred_bw,
            img_h, img_w, total_neurons_in,
            params_per_device, head_counts, channels
        )

        self.current_head_partition = head_counts
        self.current_neuron_partition = neuron_counts
        return head_counts, neuron_counts
