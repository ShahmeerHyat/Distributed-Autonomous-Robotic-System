"""
shared_utils.py  —  Networking + model utilities for distributed CLIP inference.

All ViT / torchvision fallback code has been removed. This file is CLIP-only.

Key changes:
  - get_model_metadata: reads CLIP's q_proj / fc1 attributes directly; no
    torchvision getattr fallbacks.
  - Removed dangling compute_attn_from_slices free function (was using `self`
    at module level — instant NameError on import).
  - merge_n_projections / ready_for_math kept for MLP path in master.py.
  - get_head_weights removed — no longer used now that master projects Q/K/V
    via q_proj/k_proj/v_proj and slices by head index directly.

Wire protocol (replaces torch.save/pickle):
  Tensors are downcast to float16 on the wire (2 bytes/value vs 4 for float32),
  then restored to float32 on recv. This halves per-frame bandwidth.
  master.py and worker.py are completely unchanged — they see the same Python
  tuples out of recv_msg as before.

  Message frame: [4B big-endian payload length][1B type code][type payload]

  Type codes:
    0x00  REGISTER  worker→master on connect
    0x01  PING      master→worker RTT probe (carries a dummy tensor)
    0x02  ATTN      master→worker: block_idx + q, k, v head slices
    0x03  MLP       master→worker: block_idx + start_n + end_n + ln_x
    0x04  QUIT      master→worker: shutdown
    0x05  TENSOR    any→any: bare tensor (all worker replies)

  Tensor on wire: [1B dtype=0(f16)][1B ndim][ndim×4B shape uint32 BE][raw f16 bytes]
"""

import struct
import io
import torch
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Wire protocol constants
# ─────────────────────────────────────────────────────────────────────────────

_T_REGISTER = 0x00
_T_PING     = 0x01
_T_ATTN     = 0x02
_T_MLP      = 0x03
_T_QUIT     = 0x04
_T_TENSOR   = 0x05


# ─────────────────────────────────────────────────────────────────────────────
# Low-level tensor encode / decode (no pickle, no Python metadata)
# ─────────────────────────────────────────────────────────────────────────────

def _encode_tensor(buf: io.BytesIO, t: torch.Tensor) -> None:
    t16 = t.detach().half().contiguous()
    shape = t16.shape
    buf.write(struct.pack('>BB', 0, len(shape)))            # dtype_code=0 (f16), ndim
    buf.write(struct.pack(f'>{len(shape)}I', *shape))       # shape dims
    buf.write(t16.numpy().tobytes())                        # raw float16 bytes


def _decode_tensor(data: bytes, offset: int):
    """Returns (tensor_as_float32, new_offset)."""
    _dtype_code, ndim = struct.unpack_from('>BB', data, offset)
    offset += 2
    shape = struct.unpack_from(f'>{ndim}I', data, offset)
    offset += 4 * ndim
    n_elems = 1
    for d in shape:
        n_elems *= d
    arr = np.frombuffer(data, dtype=np.float16, count=n_elems, offset=offset).copy()
    t = torch.from_numpy(arr).reshape(shape).float()        # restore to float32
    return t, offset + n_elems * 2


# ─────────────────────────────────────────────────────────────────────────────
# Socket helpers
# ─────────────────────────────────────────────────────────────────────────────

def send_msg(sock, msg) -> None:
    """
    Serialize msg to a compact binary frame and send it over sock.

    Accepted forms (identical to the old torch.save-based API):
      torch.Tensor                           → bare tensor reply
      ("REGISTER", name)
      ("PING",     0, dummy_tensor, 0, 0)
      ("ATTN",     block_idx, (q, k, v))
      ("MLP",      block_idx, x, start_n, end_n)
      ("QUIT",     0, None, 0, 0)
    """
    buf = io.BytesIO()

    if isinstance(msg, torch.Tensor):
        buf.write(struct.pack('>B', _T_TENSOR))
        _encode_tensor(buf, msg)

    elif isinstance(msg, tuple):
        tag = msg[0]

        if tag == "REGISTER":
            name_b = msg[1].encode('utf-8')
            buf.write(struct.pack('>BB', _T_REGISTER, len(name_b)))
            buf.write(name_b)

        elif tag == "QUIT":
            buf.write(struct.pack('>B', _T_QUIT))

        elif tag == "PING":
            buf.write(struct.pack('>B', _T_PING))
            _encode_tensor(buf, msg[2])

        elif tag == "ATTN":
            _, block_idx, (q, k, v) = msg
            buf.write(struct.pack('>BB', _T_ATTN, block_idx))
            _encode_tensor(buf, q)
            _encode_tensor(buf, k)
            _encode_tensor(buf, v)

        elif tag == "MLP":
            _, block_idx, x, start_n, end_n = msg
            buf.write(struct.pack('>BBII', _T_MLP, block_idx, start_n, end_n))
            _encode_tensor(buf, x)

        else:
            raise ValueError(f"send_msg: unknown tag {tag!r}")

    else:
        raise TypeError(f"send_msg: unsupported type {type(msg)}")

    payload = buf.getvalue()
    sock.sendall(struct.pack('>I', len(payload)) + payload)


def recvall(sock, n: int):
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return bytes(data)


def recv_msg(sock):
    """
    Receive one message and reconstruct the same Python object the old
    torch.load would have returned — master.py and worker.py unchanged.
    """
    raw_len = recvall(sock, 4)
    if not raw_len:
        return None
    msglen = struct.unpack('>I', raw_len)[0]
    data = recvall(sock, msglen)
    if not data:
        return None

    code = data[0]
    offset = 1

    if code == _T_TENSOR:
        t, _ = _decode_tensor(data, offset)
        return t

    elif code == _T_REGISTER:
        name_len = data[offset]
        name = data[offset + 1: offset + 1 + name_len].decode('utf-8')
        return ("REGISTER", name)

    elif code == _T_QUIT:
        return ("QUIT", 0, None, 0, 0)

    elif code == _T_PING:
        t, _ = _decode_tensor(data, offset)
        return ("PING", 0, t, 0, 0)

    elif code == _T_ATTN:
        block_idx = data[offset]
        offset += 1
        q, offset = _decode_tensor(data, offset)
        k, offset = _decode_tensor(data, offset)
        v, _      = _decode_tensor(data, offset)
        return ("ATTN", block_idx, (q, k, v))

    elif code == _T_MLP:
        block_idx, start_n, end_n = struct.unpack_from('>BII', data, offset)
        offset += 9
        x, _ = _decode_tensor(data, offset)
        return ("MLP", block_idx, x, start_n, end_n)

    else:
        raise ValueError(f"recv_msg: unknown message code {code:#04x}")


# ─────────────────────────────────────────────────────────────────────────────
# Model metadata  (CLIP vision encoder only)
# ─────────────────────────────────────────────────────────────────────────────

def get_model_metadata(vision_model) -> dict:
    """
    Extracts architecture constants from clip.vision_model.

    CLIP CLIPEncoderLayer layout:
        layer.self_attn          → CLIPAttention (q_proj / k_proj / v_proj / out_proj)
        layer.mlp.fc1 / mlp.fc2 → CLIPEncoderMLP
        layer.layer_norm1/2      → LayerNorm

    Returns
    -------
    dict with keys: embed_dim, num_heads, head_dim, mlp_hidden_dim, seq_length
    """
    layer = vision_model.encoder.layers[0]
    attn  = layer.self_attn   # CLIPAttention

    # q_proj weight shape: (embed_dim, embed_dim)  — square for self-attention
    embed_dim     = attn.q_proj.weight.shape[1]
    num_heads     = attn.num_heads
    mlp_hidden    = layer.mlp.fc1.weight.shape[0]   # fc1: (mlp_hidden, embed)

    # CLIP: positional embedding lives in vision_model.embeddings
    seq_length = vision_model.embeddings.position_embedding.weight.shape[0]

    return {
        "embed_dim":      embed_dim,
        "num_heads":      num_heads,
        "head_dim":       embed_dim // num_heads,
        "mlp_hidden_dim": mlp_hidden,
        "seq_length":     seq_length,   # 50 for ViT-B/32 (49 patches + 1 CLS)
    }


# ─────────────────────────────────────────────────────────────────────────────
# Head-space helpers
# ─────────────────────────────────────────────────────────────────────────────

def to_head_space(x: torch.Tensor, h_range: range, num_heads: int, head_dim: int) -> torch.Tensor:
    """
    Slice a projected tensor into a specific head range and reshape for
    multi-head attention math.

    Parameters
    ----------
    x        : (B, S, embed_dim)   — output of q_proj / k_proj / v_proj
    h_range  : range(start, end)   — which heads to keep
    num_heads: total number of attention heads
    head_dim : embed_dim // num_heads

    Returns
    -------
    (B, len(h_range), S, head_dim)  — ready for scaled dot-product attention
    """
    B, S, _ = x.shape
    x = x.view(B, S, num_heads, head_dim)     # (B, S, H, hd)
    x = x[:, :, h_range, :]                   # (B, S, H_slice, hd)
    return x.transpose(1, 2)                  # (B, H_slice, S, hd)


# ─────────────────────────────────────────────────────────────────────────────
# MLP merge helper  (used by master after collecting partial MLP outputs)
# ─────────────────────────────────────────────────────────────────────────────

def sum_mlp_parts(parts: list) -> torch.Tensor:
    """
    Each element of `parts` is a (B, S, embed_dim) tensor representing the
    contribution of a neuron slice: gelu(x @ w1_slice.t() + b1_slice) @ w2_slice.t()

    Summing them reconstructs the full fc2 output (bias is added by caller).
    """
    non_empty = [p for p in parts if p is not None and p.numel() > 0]
    if not non_empty:
        raise RuntimeError("sum_mlp_parts: all parts are empty — no MLP work was done.")
    return torch.stack(non_empty, dim=0).sum(dim=0)


# ─────────────────────────────────────────────────────────────────────────────
# Legacy helpers kept so nothing explodes if imported elsewhere
# ─────────────────────────────────────────────────────────────────────────────

def merge_n_projections(projections):
    """Kept for backwards compatibility. Not used in the CLIP ATTN path."""
    all_q, all_k, all_v = [], [], []
    for p in projections:
        if p.numel() == 0:
            continue
        q, k, v = torch.chunk(p, 3, dim=-1)
        all_q.append(q); all_k.append(k); all_v.append(v)
    if not all_q:
        return torch.tensor([])
    return torch.cat(
        [torch.cat(all_q, -1), torch.cat(all_k, -1), torch.cat(all_v, -1)], -1
    )


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
        # tag = f" ({reason})" if reason else ""
        # print(f"  [CB] ⚡ Tripped{tag}. Cooldown = {cooldown} blocks "
        #       f"(trip #{self.consecutive_trips})")

    def tick(self) -> bool:
        if self.state == self.OPEN:
            self.blocks_remaining -= 1
            if self.blocks_remaining <= 0:
                self.state = self.HALF_OPEN
                # print("  [CB] 🔍 Cooldown elapsed → HALF_OPEN (probing next block)")
                return True
        return False

    def on_probe_success(self):
        self.state             = self.CLOSED
        self.consecutive_trips = 0
        # print("  [CB] ✅ Probe succeeded → CLOSED")

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

    # Hysteresis gap: trip at 3×, re-admit only below 1.8×.
    # The gap prevents oscillation — once dropped, the bar to return
    # is meaningfully lower than the bar that caused the drop.
    DROP_MULT   = 3.0
    ADMIT_MULT  = 1.8
    PROBE_SHARE = 0.10   # share given to HALF_OPEN device during probe
    EMA_ALPHA   = 0.35   # smoothing on healthy share transitions
    MAX_HISTORY = 24

    # Minimum share a CLOSED worker always receives so its history
    # keeps updating and the death spiral (zero share → no data →
    # stale prediction → zero share) cannot occur.
    MIN_WORKER_SHARE = 0.10

    def __init__(
        self,
        device_ids,
        p: int   = 3,
        d: int   = 1,
        q: int   = 1,
        min_share_threshold: float = 0.05,
    ):
        self.p, self.d, self.q   = p, d, q
        self.min_share_threshold = min_share_threshold
        self.devices             = device_ids

        self.norm_history = {dev: [] for dev in device_ids}
        self.residuals    = {dev: [] for dev in device_ids}
        self.last_preds   = {dev: 0.0 for dev in device_ids}

        self.breakers: dict[str, CircuitBreaker] = {
            dev: CircuitBreaker() for dev in device_ids if dev != "edge"
        }

        initial = 1.0 / len(device_ids)
        self.current_shares = {dev: initial for dev in device_ids}

    # ── Pre-flight ─────────────────────────────────────────────────────────

    def prime(self, device_id: str, rtt_samples: list, nominal_share: float = 0.5):
        """
        Seed a worker's normalized latency history from preflight RTTs.
        RTT includes network round-trip. We subtract an estimated one-way
        transmission time so the seeded value reflects compute latency only,
        making it comparable to the edge's pure-compute latency.

        estimated_compute ≈ (rtt / 2) is a conservative lower bound.
        Divide by nominal_share to normalize.
        """
        for rtt in rtt_samples:
            # Use half the RTT as a compute-only estimate (subtracts one-way
            # network transit). This prevents the worker from starting with
            # an inflated normalized latency purely due to network overhead.
            compute_estimate = rtt / 2.0
            self.norm_history[device_id].append(compute_estimate / nominal_share)

    def prime_edge(self, latency_samples: list, nominal_share: float = 1.0):
        """
        Seed the edge device's normalized latency history from measured
        local inference times. Call this after a few warm-up inference
        passes before connecting workers, so ARIMA starts with a realistic
        edge baseline rather than the default 0.1ms.
        """
        for lat in latency_samples:
            self.norm_history["edge"].append(lat / nominal_share)

    # ── Latency recording ──────────────────────────────────────────────────

    def record_block_latency(self, device_id: str, latency: float, share_used: float):
        """
        Record normalized latency (latency / share_used) for a device.

        If the device is OPEN (CB tripped), inject synthetic upward-drifting
        history so ARIMA cannot drift toward optimism without real evidence.

        If share_used is zero (worker was skipped this block), do NOT return
        early — instead record nothing but do not corrupt state. The history
        stays at its last value, which is appropriate since no new information
        arrived.
        """
        if device_id in self.breakers and self.breakers[device_id].is_open:
            h = self.norm_history[device_id]
            if h:
                h.append(h[-1] * 1.08)
                if len(h) > self.MAX_HISTORY:
                    h.pop(0)
            return

        # No work was dispatched this block — history unchanged, no update.
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
        """
        Differenced moving average predictor (ARIMA-V approximation).

        With fewer than p+d samples, falls back to the mean of available
        history. With sufficient history, computes:
            pred = last_value + mean(recent_differences) + 0.1 * last_residual

        This is a first-order drift estimator, not a full ARIMA(p,d,q) fit,
        but it captures trending latency changes (e.g., thermal throttling,
        network congestion building up) which is the core requirement.
        """
        h = self.norm_history[dev]
        if not h:
            # No history at all — return a neutral estimate
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
        """
        Compute new shares for the upcoming block.

        Key design decisions:
        - edge_pred is used raw, with no artificial multiplier.
        - CLOSED workers below DROP_MULT threshold are always active.
        - Any CLOSED worker whose computed share would fall below
          MIN_WORKER_SHARE is floored at MIN_WORKER_SHARE to prevent
          the zero-share death spiral (no share → no data → stale
          prediction → no share).
        - Drops are applied instantly; healthy transitions are EMA-smoothed.
        """
        for cb in self.breakers.values():
            cb.tick()

        preds     = {dev: self._predict(dev) for dev in self.devices}
        edge_pred = preds.get("edge", 0.1)   # raw, no artificial inflation

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

            elif pred > edge_pred * self.DROP_MULT:
                print(f"\n  [ARIMA] ⚠️  '{dev}': norm_pred={pred:.3f}s "
                      f"vs edge={edge_pred:.3f}s (>{self.DROP_MULT}×) → tripping CB")
                cb.trip(
                    reason=f"norm_pred {pred:.2f}s > "
                           f"{self.DROP_MULT}× edge {edge_pred:.2f}s"
                )

            elif pred > edge_pred * self.ADMIT_MULT:
                # Marginal: exclude this block but do not trip.
                # Still give it MIN_WORKER_SHARE so history keeps updating.
                print(f"  [ARIMA] '{dev}' marginal "
                      f"({pred:.3f}s vs edge {edge_pred:.3f}s). "
                      f"Assigning floor share to preserve history.")
                active_devs[dev] = 1.0 / pred   # small score → small share

            else:
                active_devs[dev] = 1.0 / pred

        # ── Allocate shares ──────────────────────────────────────────────
        probe_reserved = len(probe_devs) * self.PROBE_SHARE
        remaining      = max(0.0, 1.0 - probe_reserved)
        total_score    = sum(active_devs.values()) or 1.0

        target = {dev: 0.0 for dev in self.devices}
        for dev, score in active_devs.items():
            target[dev] = (score / total_score) * remaining
        for dev in probe_devs:
            target[dev] = self.PROBE_SHARE

        # ── Floor: CLOSED workers always get at least MIN_WORKER_SHARE ──
        # This breaks the death spiral. If ARIMA assigns a worker less than
        # the floor, we bring it up and reduce edge proportionally.
        for dev in self.devices:
            if dev == "edge" or dev not in self.breakers:
                continue
            if self.breakers[dev].is_closed and 0 < target[dev] < self.MIN_WORKER_SHARE:
                deficit = self.MIN_WORKER_SHARE - target[dev]
                target[dev]      = self.MIN_WORKER_SHARE
                target["edge"]   = max(0.0, target.get("edge", 0.0) - deficit)

        # ── Apply EMA on healthy transitions, instant drop to zero ───────
        for dev in self.devices:
            t = target[dev]
            if t == 0.0:
                self.current_shares[dev] = 0.0
            else:
                prev = self.current_shares[dev]
                self.current_shares[dev] = (
                    self.EMA_ALPHA * t + (1.0 - self.EMA_ALPHA) * prev
                )

        # ── Hard zero below min threshold (but floor overrides this) ─────
        for dev in self.devices:
            if dev != "edge" and 0 < self.current_shares[dev] < self.min_share_threshold:
                # Only zero out if the target was also below floor,
                # meaning the worker is genuinely assigned nothing.
                if target[dev] < self.MIN_WORKER_SHARE:
                    self.current_shares[dev] = 0.0

        # ── Re-normalise to sum to 1.0 ───────────────────────────────────
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
                    reason=f"norm_lat {norm_latency:.3f}s ≥ "
                           f"{self.ADMIT_MULT}× edge {edge_pred:.3f}s"
                )

    # ── Index calculation ──────────────────────────────────────────────────

    def get_indices(self, dev: str, total_items: int) -> range:
        start = 0
        for d in self.devices:
            count = int(round(self.current_shares[d] * total_items))
            end   = start + count
            if d == dev:
                return range(start, min(end, total_items))
            start = end
        return range(0, 0)