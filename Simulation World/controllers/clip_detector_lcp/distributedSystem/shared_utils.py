"""
shared_utils.py  —  Networking + model utilities for LCP distributed CLIP inference.

Changes vs clip_detector version:
  - send_msg / recv_msg transparently convert float32 tensors to fp16 on the wire,
    cutting bandwidth by ~50% with negligible numerical impact on cosine similarity.
  - _indices_from_shares(): standalone function for thread-safe block-level snapshots
    (avoids reading live ARIMA state mid-inference).
  - MultiDeviceARIMAManager gains add_device(), reset_device(), remove_device() for
    dynamic worker join/leave without restarting the orchestrator.
  - tune_socket(): sets TCP_NODELAY + large send/recv buffers on a socket.
"""

import struct
import io
import torch
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# fp16 wire helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_fp16(obj):
    """Recursively cast float32 tensors to fp16 for network transmission."""
    if isinstance(obj, torch.Tensor):
        return obj.half() if obj.dtype == torch.float32 else obj
    if isinstance(obj, tuple):
        return tuple(_to_fp16(x) for x in obj)
    if isinstance(obj, list):
        return [_to_fp16(x) for x in obj]
    return obj


def _to_fp32(obj):
    """Recursively cast fp16 tensors back to fp32 after network reception."""
    if isinstance(obj, torch.Tensor):
        return obj.float() if obj.dtype == torch.float16 else obj
    if isinstance(obj, tuple):
        return tuple(_to_fp32(x) for x in obj)
    if isinstance(obj, list):
        return [_to_fp32(x) for x in obj]
    return obj


# ─────────────────────────────────────────────────────────────────────────────
# Socket helpers
# ─────────────────────────────────────────────────────────────────────────────

import socket as _socket


def tune_socket(sock):
    """Disable Nagle's algorithm and widen OS buffers for lower latency."""
    sock.setsockopt(_socket.IPPROTO_TCP, _socket.TCP_NODELAY, 1)
    sock.setsockopt(_socket.SOL_SOCKET, _socket.SO_RCVBUF, 1 << 20)  # 1 MiB
    sock.setsockopt(_socket.SOL_SOCKET, _socket.SO_SNDBUF, 1 << 20)


def send_msg(sock, msg):
    """Serialize msg with fp16 tensors and send with a 4-byte length prefix."""
    msg = _to_fp16(msg)
    buf = io.BytesIO()
    torch.save(msg, buf)
    data = buf.getvalue()
    sock.sendall(struct.pack('>I', len(data)) + data)


def recvall(sock, n):
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data


def recv_msg(sock):
    """Receive a length-prefixed message and return with tensors cast to fp32."""
    raw_len = recvall(sock, 4)
    if not raw_len:
        return None
    msglen = struct.unpack('>I', raw_len)[0]
    data = recvall(sock, msglen)
    obj = torch.load(io.BytesIO(data), weights_only=False)
    return _to_fp32(obj)


# ─────────────────────────────────────────────────────────────────────────────
# Model metadata  (CLIP vision encoder only)
# ─────────────────────────────────────────────────────────────────────────────

def get_model_metadata(vision_model) -> dict:
    layer  = vision_model.encoder.layers[0]
    attn   = layer.self_attn
    embed_dim  = attn.q_proj.weight.shape[1]
    num_heads  = attn.num_heads
    mlp_hidden = layer.mlp.fc1.weight.shape[0]
    seq_length = vision_model.embeddings.position_embedding.weight.shape[0]
    return {
        "embed_dim":      embed_dim,
        "num_heads":      num_heads,
        "head_dim":       embed_dim // num_heads,
        "mlp_hidden_dim": mlp_hidden,
        "seq_length":     seq_length,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Head-space helper
# ─────────────────────────────────────────────────────────────────────────────

def to_head_space(x: torch.Tensor, h_range: range, num_heads: int, head_dim: int) -> torch.Tensor:
    """
    Reshape (B, S, embed_dim) → (B, H_slice, S, head_dim) for a head slice.
    """
    B, S, _ = x.shape
    x = x.view(B, S, num_heads, head_dim)
    x = x[:, :, h_range, :]
    return x.transpose(1, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Index helper  (thread-safe, works on snapshots)
# ─────────────────────────────────────────────────────────────────────────────

def indices_from_shares(device_order: list, shares: dict, dev: str, total: int) -> range:
    """
    Compute a device's contiguous index range from a pre-snapshotted shares dict.

    Uses cumulative rounding: each boundary is round(cumulative_share * total).
    This guarantees all counts sum exactly to `total` regardless of floating-point
    precision — individual per-device rounding can lose units (e.g. round(5.4999)
    + round(6.4999) = 5 + 6 = 11 instead of 12).
    """
    cumulative = 0.0
    start = 0
    for d in device_order:
        cumulative += shares.get(d, 0.0)
        end = round(cumulative * total)
        if d == dev:
            return range(start, min(end, total))
        start = end
    return range(0, 0)


# ─────────────────────────────────────────────────────────────────────────────
# MLP merge helper
# ─────────────────────────────────────────────────────────────────────────────

def sum_mlp_parts(parts: list) -> torch.Tensor:
    non_empty = [p for p in parts if p is not None and p.numel() > 0]
    if not non_empty:
        raise RuntimeError("sum_mlp_parts: all parts are empty.")
    return torch.stack(non_empty, dim=0).sum(dim=0)


# ─────────────────────────────────────────────────────────────────────────────
# Circuit Breaker
# ─────────────────────────────────────────────────────────────────────────────

class CircuitBreaker:
    CLOSED, OPEN, HALF_OPEN = "CLOSED", "OPEN", "HALF_OPEN"

    def __init__(self, base_cooldown: int = 4, max_cooldown: int = 32):
        self.state             = self.CLOSED
        self.base_cooldown     = base_cooldown
        self.max_cooldown      = max_cooldown
        self.blocks_remaining  = 0
        self.consecutive_trips = 0

    def trip(self, reason: str = ""):
        self.consecutive_trips += 1
        self.state = self.OPEN
        cooldown = min(
            self.base_cooldown * (2 ** (self.consecutive_trips - 1)),
            self.max_cooldown,
        )
        self.blocks_remaining = cooldown

    def tick(self) -> bool:
        if self.state == self.OPEN:
            self.blocks_remaining -= 1
            if self.blocks_remaining <= 0:
                self.state = self.HALF_OPEN
                return True
        return False

    def on_probe_success(self):
        self.state             = self.CLOSED
        self.consecutive_trips = 0

    def on_probe_failure(self, reason: str = ""):
        self.trip(reason=f"probe failed: {reason}" if reason else "probe failed")

    @property
    def is_open(self)      -> bool: return self.state == self.OPEN
    @property
    def is_half_open(self) -> bool: return self.state == self.HALF_OPEN
    @property
    def is_closed(self)    -> bool: return self.state == self.CLOSED


# ─────────────────────────────────────────────────────────────────────────────
# MultiDeviceARIMAManager
# ─────────────────────────────────────────────────────────────────────────────

class MultiDeviceARIMAManager:

    # WiFi workers include ~6ms RTT + ~15ms data transfer per block in their
    # measured latency, making them appear 8-10× slower than edge in normalized
    # terms even when healthy.  DROP_MULT=15 means a worker must be 15× slower
    # before being excluded; ADMIT_MULT=5 sets the hysteresis recovery bar.
    DROP_MULT    = 15.0
    ADMIT_MULT   = 5.0
    PROBE_SHARE  = 0.10
    EMA_ALPHA    = 0.35
    MAX_HISTORY  = 24
    MIN_WORKER_SHARE = 0.20  # floor raised: always keep workers at ≥20% so
                              # history keeps updating and WiFi workers stay active

    def __init__(
        self,
        device_ids,
        p: int = 3,
        d: int = 1,
        q: int = 1,
        min_share_threshold: float = 0.05,
    ):
        self.p, self.d, self.q   = p, d, q
        self.min_share_threshold = min_share_threshold
        self.devices             = list(device_ids)

        self.norm_history = {dev: [] for dev in self.devices}
        self.residuals    = {dev: [] for dev in self.devices}
        self.last_preds   = {dev: 0.0 for dev in self.devices}

        self.breakers: dict[str, CircuitBreaker] = {
            dev: CircuitBreaker() for dev in self.devices if dev != "edge"
        }

        initial = 1.0 / len(self.devices)
        self.current_shares = {dev: initial for dev in self.devices}

    # ── Dynamic device management ───────────────────────────────────────────

    def add_device(self, device_id: str):
        """
        Register a brand-new worker. Initialises its state with zero share;
        update_shares() will compute a real share once prime() seeds the history.
        """
        if device_id in self.devices:
            return  # Already present — caller should use reset_device instead
        self.devices.append(device_id)
        self.norm_history[device_id]   = []
        self.residuals[device_id]      = []
        self.last_preds[device_id]     = 0.0
        self.current_shares[device_id] = 0.0
        if device_id != "edge":
            self.breakers[device_id] = CircuitBreaker()

    def reset_device(self, device_id: str):
        """
        Clear a reconnecting worker's latency history and circuit-breaker state
        so ARIMA starts fresh with the new preflight measurements.
        """
        if device_id not in self.devices:
            return
        self.norm_history[device_id] = []
        self.residuals[device_id]    = []
        self.last_preds[device_id]   = 0.0
        if device_id in self.breakers:
            self.breakers[device_id] = CircuitBreaker()
        # Keep current_shares as-is; update_shares() will re-calibrate.

    def remove_device(self, device_id: str):
        """
        Permanently remove a disconnected worker and redistribute its share to edge.
        Safe to call even if device_id is not present.
        """
        if device_id == "edge" or device_id not in self.devices:
            return
        removed_share = self.current_shares.pop(device_id, 0.0)
        self.devices.remove(device_id)
        self.norm_history.pop(device_id, None)
        self.residuals.pop(device_id, None)
        self.last_preds.pop(device_id, None)
        self.breakers.pop(device_id, None)
        # Give removed share back to edge so total stays ≈ 1.0
        self.current_shares["edge"] = self.current_shares.get("edge", 0.0) + removed_share
        total = sum(self.current_shares.values())
        if total > 0:
            for dev in self.devices:
                self.current_shares[dev] /= total

    # ── Pre-flight ─────────────────────────────────────────────────────────

    def prime(self, device_id: str, rtt_samples: list, nominal_share: float = 0.5):
        """
        Seed a device's normalized latency history from preflight RTTs.
        Uses the full RTT (not RTT/2) so the estimate includes the round-trip
        network overhead that will appear in every real dispatch as well.
        """
        for rtt in rtt_samples:
            norm = rtt / nominal_share   # full RTT / share → realistic baseline
            self.norm_history[device_id].append(norm)
            if len(self.norm_history[device_id]) > self.MAX_HISTORY:
                self.norm_history[device_id].pop(0)

    def prime_task(self, device_id: str, task_latency: float, nominal_share: float = 0.5):
        """
        Seed history from an actual ATTN task dispatch (measured in master's
        _preflight_worker).  More accurate than RTT alone because it includes
        the data-transfer overhead for real tensor payloads.
        Called after prime() so task measurements overwrite the RTT-only seeds.
        """
        norm = task_latency / nominal_share
        for _ in range(self.p + self.d + 1):   # enough samples for _predict to use ARIMA path
            self.norm_history[device_id].append(norm)
            if len(self.norm_history[device_id]) > self.MAX_HISTORY:
                self.norm_history[device_id].pop(0)

    def prime_edge(self, latency_samples: list, nominal_share: float = 1.0):
        for lat in latency_samples:
            self.norm_history["edge"].append(lat / nominal_share)
            if len(self.norm_history["edge"]) > self.MAX_HISTORY:
                self.norm_history["edge"].pop(0)

    # ── Latency recording ──────────────────────────────────────────────────

    def record_block_latency(self, device_id: str, latency: float, share_used: float):
        if device_id not in self.norm_history:
            return  # Device was removed mid-block

        if device_id in self.breakers and self.breakers[device_id].is_open:
            h = self.norm_history[device_id]
            if h:
                h.append(h[-1] * 1.08)
                if len(h) > self.MAX_HISTORY:
                    h.pop(0)
            return

        if share_used <= 0.0:
            return

        norm_lat = latency / share_used

        if self.last_preds[device_id] > 0:
            err = norm_lat - self.last_preds[device_id]
            self.residuals[device_id].append(err)
            if len(self.residuals[device_id]) > self.MAX_HISTORY:
                self.residuals[device_id].pop(0)

        self.norm_history[device_id].append(norm_lat)
        if len(self.norm_history[device_id]) > self.MAX_HISTORY:
            self.norm_history[device_id].pop(0)

    # ── ARIMA prediction ───────────────────────────────────────────────────

    def _predict(self, dev: str) -> float:
        if dev not in self.norm_history:
            return 0.1
        h = self.norm_history[dev]
        if not h:
            return 0.1
        if len(h) < self.p + self.d:
            pred = float(np.mean(h))
        else:
            diffs  = [h[i] - h[i - 1] for i in range(1, len(h))]
            ar_val = float(np.mean(diffs[-self.p:]))
            ma_val = 0.1 * self.residuals[dev][-1] if self.residuals[dev] else 0.0
            pred   = h[-1] + ar_val + ma_val
        pred = max(0.001, pred)
        self.last_preds[dev] = pred
        return pred

    # ── Share update ───────────────────────────────────────────────────────

    def update_shares(self) -> dict:
        for cb in self.breakers.values():
            cb.tick()

        preds     = {dev: self._predict(dev) for dev in self.devices}
        edge_pred = preds.get("edge", 0.1)

        probe_devs  = []
        active_devs = {}

        for dev in self.devices:
            if dev == "edge":
                active_devs[dev] = 1.0 / preds[dev]
                continue

            cb   = self.breakers[dev]
            pred = preds[dev]

            if cb.is_open:
                continue
            elif cb.is_half_open:
                probe_devs.append(dev)
            else:
                # CB is CLOSED: always include the worker.
                # ARIMA gives it a score inversely proportional to predicted
                # latency — slower workers get fewer heads automatically.
                # We never trip the CB here; the circuit breaker fires ONLY
                # on actual dispatch failures (exceptions / disconnects) so
                # that WiFi overhead does not incorrectly exclude healthy workers.
                active_devs[dev] = 1.0 / pred

        probe_reserved = len(probe_devs) * self.PROBE_SHARE
        remaining      = max(0.0, 1.0 - probe_reserved)
        total_score    = sum(active_devs.values()) or 1.0

        target = {dev: 0.0 for dev in self.devices}
        for dev, score in active_devs.items():
            target[dev] = (score / total_score) * remaining
        for dev in probe_devs:
            target[dev] = self.PROBE_SHARE

        # Floor: every CLOSED worker always gets at least MIN_WORKER_SHARE.
        # This covers both the case where ARIMA assigns a tiny score (worker
        # slow but healthy) and the case where target == 0 (safety net).
        for dev in self.devices:
            if dev == "edge" or dev not in self.breakers:
                continue
            if self.breakers[dev].is_closed and target[dev] < self.MIN_WORKER_SHARE:
                deficit = self.MIN_WORKER_SHARE - target[dev]
                target[dev]    = self.MIN_WORKER_SHARE
                target["edge"] = max(0.0, target.get("edge", 0.0) - deficit)

        for dev in self.devices:
            t = target[dev]
            if t == 0.0:
                self.current_shares[dev] = 0.0
            else:
                prev = self.current_shares[dev]
                self.current_shares[dev] = self.EMA_ALPHA * t + (1.0 - self.EMA_ALPHA) * prev

        for dev in self.devices:
            if dev != "edge" and 0 < self.current_shares[dev] < self.min_share_threshold:
                if target[dev] < self.MIN_WORKER_SHARE:
                    self.current_shares[dev] = 0.0

        total = sum(self.current_shares.values())
        if total > 0:
            for dev in self.devices:
                self.current_shares[dev] /= total

        return self.current_shares

    # ── Probe feedback ─────────────────────────────────────────────────────

    def notify_probe_result(self, device_id: str, norm_latency: float):
        if device_id not in self.breakers:
            return
        cb        = self.breakers[device_id]
        edge_pred = self._predict("edge")
        if cb.is_half_open:
            if norm_latency < edge_pred * self.ADMIT_MULT:
                cb.on_probe_success()
                self.norm_history[device_id].append(norm_latency)
            else:
                cb.on_probe_failure(
                    reason=f"norm_lat {norm_latency:.3f}s >= {self.ADMIT_MULT}× edge {edge_pred:.3f}s"
                )

    # ── Index calculation (kept for compatibility) ─────────────────────────

    def get_indices(self, dev: str, total_items: int) -> range:
        return indices_from_shares(self.devices, self.current_shares, dev, total_items)
