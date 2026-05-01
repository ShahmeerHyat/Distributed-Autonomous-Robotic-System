import numpy as np

TOTAL_STEPS = 20

TOTAL_STEPS = 100
ALPHA       = 0.25   # EMA smoothing for compute (passive)
BETA        = 0.5    # AIMD multiplicative decrease factor
DELTA       = 0.05   # AIMD additive increase step
SPIKE_RATIO = 2.0    # latency is "bad" if > SPIKE_RATIO * current EMA
PROBE_THRESHOLD = 0.2  # link score below this triggers probing

class MultiDeviceEvaluator:
    def __init__(self, devices: list):
        self.devices = {
            device: {
                "latencies"    : np.zeros(TOTAL_STEPS),
                "compute_times": np.zeros(TOTAL_STEPS),
                "latency_aimd" : 1.0,   # start neutral, not zero
                "compute_ema"  : 0.0,
                "step"         : 0
            }
            for device in devices
        }

    def _update_compute_ema(self, device: str, compute_time: float):
        d = self.devices[device]
        if d["compute_ema"] == 0.0:
            d["compute_ema"] = compute_time  # cold start
        else:
            d["compute_ema"] = (1 - ALPHA) * d["compute_ema"] + ALPHA * compute_time

    def _update_latency_aimd(self, device: str, latency: float):
        d        = self.devices[device]
        prev_ema = d["latency_aimd"]

        is_spike = (prev_ema > 0) and (latency > SPIKE_RATIO * prev_ema)

        if is_spike:
            d["latency_aimd"] = max(prev_ema * BETA, 1e-6)  # hard cut, floor at epsilon
        else:
            d["latency_aimd"] = prev_ema + DELTA             # slow additive recovery
            
    def record_step(self, device: str, latency: float, compute_time: float):
        if device not in self.devices:
            raise ValueError(f"Unknown device: {device}")

        d   = self.devices[device]
        idx = d["step"] % TOTAL_STEPS

        d["latencies"][idx]     = latency
        d["compute_times"][idx] = compute_time
        d["step"] += 1

        self._update_compute_ema(device, compute_time)
        self._update_latency_aimd(device, latency)

    def compute_score(self, device: str) -> float:
        """
        Higher score = more work should be assigned to this device.
        Compute score: inverse of EMA (faster = better), passive signal.
        Link score: AIMD value directly, aggressive signal.
        """
        d           = self.devices[device]
        compute_ema = d["compute_ema"]
        link_score  = d["latency_aimd"]

        if compute_ema <= 0:
            compute_score = 0.0
        else:
            compute_score = 1.0 / compute_ema

        # equal weights for now, tunable
        raw_score = 0.5 * compute_score + 0.5 * link_score
        return raw_score

    def get_allocation(self) -> dict[str, float]:
        """
        Returns normalized allocation weights across all devices.
        Scores sum to 1.0, ready to multiply against head count or neuron budget.
        """
        scores = {device: self.compute_score(device) for device in self.devices}
        total  = sum(scores.values())

        if total == 0:
            # all devices look dead, distribute equally
            n = len(self.devices)
            return {device: 1.0 / n for device in self.devices}

        return {device: s / total for device, s in scores.items()}

    def needs_probe(self, device: str) -> bool:
        """
        Returns True if a device's link score has dropped low enough
        that we should probe it rather than assuming it's dead.
        """
        return self.devices[device]["latency_aimd"] < PROBE_THRESHOLD

    def get_diagnostics(self, device: str) -> dict:
        """Useful for logging and your evaluation comparisons against ARIMA-V."""
        d = self.devices[device]
        return {
            "device"       : device,
            "step"         : d["step"],
            "compute_ema"  : d["compute_ema"],
            "latency_aimd" : d["latency_aimd"],
            "score"        : self.compute_score(device),
            "needs_probe"  : self.needs_probe(device),
        }
    