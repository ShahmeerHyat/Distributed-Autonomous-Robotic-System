import struct
import io
import torch
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Socket Networking Utilities
# ─────────────────────────────────────────────────────────────────────────────

def send_msg(sock, msg):
    buffer = io.BytesIO()
    torch.save(msg, buffer)
    data = buffer.getvalue()
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
    raw_len = recvall(sock, 4)
    if not raw_len:
        return None
    msglen = struct.unpack('>I', raw_len)[0]
    data = recvall(sock, msglen)
    return torch.load(io.BytesIO(data), weights_only=False)


# ─────────────────────────────────────────────────────────────────────────────
# Model Math Utilities
# ─────────────────────────────────────────────────────────────────────────────

def get_model_metadata(model):
    embed_dim   = model.encoder.layers[0].self_attention.in_proj_weight.shape[1]
    seq_length  = model.encoder.pos_embedding.shape[1]
    mlp_hidden  = model.encoder.layers[0].mlp[0].weight.shape[0]
    num_heads   = model.encoder.layers[0].self_attention.num_heads
    return {
        "embed_dim":      embed_dim,
        "seq_length":     seq_length,
        "mlp_hidden_dim": mlp_hidden,
        "num_heads":      num_heads,
        "head_dim":       embed_dim // num_heads,
    }

def get_head_weights(full_weight, head_indices, embed_dim, head_dim):
    slices = []
    for i in [0, 1, 2]:          # Q, K, V
        offset = i * embed_dim
        for h in head_indices:
            start = offset + h * head_dim
            slices.append(full_weight[start:start + head_dim, :])
    return torch.cat(slices, dim=0) if slices else torch.tensor([])

def merge_n_projections(projections):
    all_q, all_k, all_v = [], [], []
    for p in projections:
        if p.numel() == 0:
            continue
        q, k, v = torch.chunk(p, 3, dim=-1)
        all_q.append(q); all_k.append(k); all_v.append(v)
    if not all_q:
        return torch.tensor([])
    return torch.cat([torch.cat(all_q, -1), torch.cat(all_k, -1), torch.cat(all_v, -1)], -1)

def ready_for_math(t, meta):
    return t.view(1, meta["seq_length"], meta["num_heads"], meta["head_dim"]).transpose(1, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Circuit Breaker
#
# Three states:
#   CLOSED    – device is healthy, receives normal share
#   OPEN      – device is blocked after a bad event; counts down blocks
#   HALF_OPEN – cooldown expired; probe with a tiny share to re-evaluate
# ─────────────────────────────────────────────────────────────────────────────

class CircuitBreaker:
    CLOSED, OPEN, HALF_OPEN = "CLOSED", "OPEN", "HALF_OPEN"

    def __init__(self, base_cooldown: int = 4, max_cooldown: int = 32):
        """
        base_cooldown   – blocks to wait after first failure before probing
        max_cooldown    – cap on exponential backoff (in blocks)
        """
        self.state              = self.CLOSED
        self.base_cooldown      = base_cooldown
        self.max_cooldown       = max_cooldown
        self.blocks_remaining   = 0
        self.consecutive_trips  = 0

    # ── State transitions ───────────────────────────────────────────────────

    def trip(self, reason: str = ""):
        """CLOSED/HALF_OPEN → OPEN with exponential backoff."""
        self.consecutive_trips += 1
        self.state = self.OPEN
        cooldown = min(
            self.base_cooldown * (2 ** (self.consecutive_trips - 1)),
            self.max_cooldown,
        )
        self.blocks_remaining = cooldown
        tag = f" ({reason})" if reason else ""
        print(
            f"  [CB] ⚡ Tripped{tag}. "
            f"Cooldown = {cooldown} blocks (trip #{self.consecutive_trips})"
        )

    def tick(self) -> bool:
        """
        Must be called once per inference block.
        Returns True the block that OPEN → HALF_OPEN transitions, signalling
        the master to send a probe task.
        """
        if self.state == self.OPEN:
            self.blocks_remaining -= 1
            if self.blocks_remaining <= 0:
                self.state = self.HALF_OPEN
                print(f"  [CB] 🔍 Cooldown elapsed → HALF_OPEN (probing next block)")
                return True
        return False

    def on_probe_success(self):
        """HALF_OPEN → CLOSED on a good probe result."""
        self.state             = self.CLOSED
        self.consecutive_trips = 0
        print("  [CB] ✅ Probe succeeded → CLOSED")

    def on_probe_failure(self, reason: str = ""):
        """HALF_OPEN → OPEN again with longer cooldown."""
        self.trip(reason=f"probe failed: {reason}" if reason else "probe failed")

    # ── Accessors ───────────────────────────────────────────────────────────

    @property
    def is_open(self)      -> bool: return self.state == self.OPEN
    @property
    def is_half_open(self) -> bool: return self.state == self.HALF_OPEN
    @property
    def is_closed(self)    -> bool: return self.state == self.CLOSED


# ─────────────────────────────────────────────────────────────────────────────
# MultiDeviceARIMAManager  (v2 — robust)
#
# Key fixes over v1:
#
#  1. NORMALISED LATENCY  – stores (latency / share_used) so predictions are
#     workload-independent.  The old code compared raw latencies: when the
#     worker was dropped and edge took 100% of work, edge_pred grew → the
#     worker/edge ratio shrank → worker got re-admitted even though the network
#     was still bad.  Normalised latency removes this dependency.
#
#  2. CIRCUIT BREAKER     – each non-edge device has an independent CB with
#     exponential backoff.  A tripped device is frozen for N blocks before it
#     gets a small probe task; re-admission only happens if the probe latency
#     is actually better than the hysteresis threshold.
#
#  3. HYSTERESIS          – two separate thresholds:
#       DROP_MULT   (3.0×)  – trip the CB when normalised latency exceeds this
#       ADMIT_MULT  (1.8×)  – re-admit only when probe passes below this
#     The gap between the two prevents the oscillation: once dropped, the bar
#     to get back in is lower than the bar that got you kicked out.
#
#  4. SYNTHETIC HISTORY   – while a device is OPEN, we inject a gently
#     inflated copy of its last normalised sample every block.  This prevents
#     ARIMA from predicting recovery when there is no new evidence for it.
#
#  5. IMMEDIATE DROPS     – the EMA is only applied on healthy→healthy
#     transitions.  Drops are applied instantly (no EMA lag leaking share).
#
#  6. prime()             – seed history from pre-flight pings before block 0
#     so ARIMA has initial data and block 0 doesn't blindly dispatch.
# ─────────────────────────────────────────────────────────────────────────────

class MultiDeviceARIMAManager:

    DROP_MULT   = 3.0    # trip CB if norm_lat_worker > edge_norm * this
    ADMIT_MULT  = 1.8    # re-admit probe only if norm_lat_worker < edge_norm * this
    PROBE_SHARE = 0.08   # fixed share given to a HALF_OPEN device during probe
    EMA_ALPHA   = 0.35   # share smoothing on healthy transitions (0 = no smooth)
    MAX_HISTORY = 24     # rolling window length

    def __init__(
        self,
        device_ids,
        p: int   = 3,
        d: int   = 1,
        q: int   = 1,
        min_share_threshold: float = 0.05,
    ):
        self.p, self.d, self.q    = p, d, q
        self.min_share_threshold  = min_share_threshold
        self.devices              = device_ids

        # Normalised latency history per device (latency / share_used)
        self.norm_history   = {dev: [] for dev in device_ids}
        self.residuals      = {dev: [] for dev in device_ids}
        self.last_preds     = {dev: 0.0 for dev in device_ids}

        # Circuit breakers for every non-edge device
        self.breakers: dict[str, CircuitBreaker] = {
            dev: CircuitBreaker() for dev in device_ids if dev != "edge"
        }

        initial = 1.0 / len(device_ids)
        self.current_shares = {dev: initial for dev in device_ids}

    # ── Pre-flight ──────────────────────────────────────────────────────────

    def prime(self, device_id: str, rtt_samples: list[float], nominal_share: float = 0.5):
        """
        Seed the normalised history for a device from pre-flight ping RTTs.
        Call this before the first inference block to avoid the cold-start
        problem where block 0 dispatches blindly with equal shares.

        rtt_samples   – list of measured round-trip latencies from preflight pings
        nominal_share – the share to assume during normalisation (default 0.5)
        """
        for rtt in rtt_samples:
            self.norm_history[device_id].append(rtt / nominal_share)

    # ── Latency recording ───────────────────────────────────────────────────

    def record_block_latency(self, device_id: str, latency: float, share_used: float):
        """
        Call once per block per device with the observed wall-clock latency
        and the share of work that was actually dispatched.

        If the device is OPEN (CB tripped) we inject synthetic history instead
        so ARIMA cannot drift towards optimism without real evidence.
        """
        # --- OPEN device: no real data, inject synthetic penalty ---
        if device_id in self.breakers and self.breakers[device_id].is_open:
            h = self.norm_history[device_id]
            if h:
                # Slight upward drift to keep the CB tripped
                synthetic = h[-1] * 1.08
                h.append(synthetic)
                if len(h) > self.MAX_HISTORY:
                    h.pop(0)
            return

        if share_used <= 0.0:
            return   # Shouldn't happen, but guard against divide-by-zero

        norm_lat = latency / share_used

        # ARIMA residual
        if self.last_preds[device_id] > 0:
            err = norm_lat - self.last_preds[device_id]
            self.residuals[device_id].append(err)
            if len(self.residuals[device_id]) > self.MAX_HISTORY:
                self.residuals[device_id].pop(0)

        self.norm_history[device_id].append(norm_lat)
        if len(self.norm_history[device_id]) > self.MAX_HISTORY:
            self.norm_history[device_id].pop(0)

    # ── ARIMA prediction ────────────────────────────────────────────────────

    def _predict(self, dev: str) -> float:
        h = self.norm_history[dev]
        if len(h) < self.p + self.d:
            pred = np.mean(h) if h else 0.1
        else:
            diffs  = [h[i] - h[i - 1] for i in range(1, len(h))]
            ar_val = float(np.mean(diffs[-self.p:]))
            ma_val = 0.1 * self.residuals[dev][-1] if self.residuals[dev] else 0.0
            pred   = h[-1] + ar_val + ma_val

        pred = max(0.001, pred)
        self.last_preds[dev] = pred
        return pred

    # ── Share update (main entry point each block) ───────────────────────────

    def update_shares(self) -> dict[str, float]:
        """
        Compute new shares for the upcoming block.
        Returns the updated current_shares dict.
        """
        # 1. Tick all circuit breakers
        for cb in self.breakers.values():
            cb.tick()

        # 2. Predict normalised latencies
        preds     = {dev: self._predict(dev) for dev in self.devices}
        edge_pred = preds.get("edge", 0.1)

        # 3. Classify each device
        probe_devs  = []   # HALF_OPEN → tiny probe share
        active_devs = {}   # healthy device → score (1/pred)

        for dev in self.devices:
            if dev == "edge":
                active_devs[dev] = 1.0 / preds[dev]
                continue

            cb   = self.breakers[dev]
            pred = preds[dev]

            if cb.is_open:
                # Blocked — skip entirely
                continue

            elif cb.is_half_open:
                # Cooldown elapsed: give it a probe slice
                probe_devs.append(dev)

            elif pred > edge_pred * self.DROP_MULT:
                # Significantly slower per unit of work → trip CB
                print(
                    f"\n  [ARIMA-V] ⚠️  '{dev}': norm_pred={pred:.3f}s "
                    f"vs edge={edge_pred:.3f}s (>{self.DROP_MULT}×) → tripping CB"
                )
                cb.trip(reason=f"norm_pred {pred:.2f}s > {self.DROP_MULT}× edge {edge_pred:.2f}s")

            elif pred > edge_pred * self.ADMIT_MULT:
                # Marginal — exclude this block but don't trip (hysteresis gap)
                print(f"  [ARIMA-V] '{dev}' marginal ({pred:.3f}s vs edge {edge_pred:.3f}s). Skipping block.")

            else:
                # Healthy
                active_devs[dev] = 1.0 / pred

        # 4. Allocate shares
        # Reserve PROBE_SHARE per probing device; split remainder among active.
        probe_reserved = len(probe_devs) * self.PROBE_SHARE
        remaining      = max(0.0, 1.0 - probe_reserved)
        total_score    = sum(active_devs.values()) or 1.0

        target = {dev: 0.0 for dev in self.devices}
        for dev, score in active_devs.items():
            target[dev] = (score / total_score) * remaining
        for dev in probe_devs:
            target[dev] = self.PROBE_SHARE

        # 5. Apply shares:
        #    • Drops  → immediate (no EMA lag)
        #    • Active → EMA-smoothed for gradual transitions
        for dev in self.devices:
            t = target[dev]
            if t == 0.0:
                self.current_shares[dev] = 0.0          # Instant drop
            else:
                prev = self.current_shares[dev]
                self.current_shares[dev] = self.EMA_ALPHA * t + (1.0 - self.EMA_ALPHA) * prev

        # 6. Zero out shares that are below the minimum useful payload
        for dev in self.devices:
            if dev != "edge" and 0 < self.current_shares[dev] < self.min_share_threshold:
                self.current_shares[dev] = 0.0

        # 7. Re-normalise so shares sum to exactly 1.0
        total = sum(self.current_shares.values())
        if total > 0:
            for dev in self.devices:
                self.current_shares[dev] /= total

        return self.current_shares

    # ── Probe result feedback ────────────────────────────────────────────────

    def notify_probe_result(self, device_id: str, norm_latency: float):
        """
        Call after a HALF_OPEN probe task completes with its normalised latency.
        Decides whether to CLOSE the breaker or re-trip it.
        """
        if device_id not in self.breakers:
            return
        cb         = self.breakers[device_id]
        edge_pred  = self._predict("edge")

        if cb.is_half_open:
            if norm_latency < edge_pred * self.ADMIT_MULT:
                cb.on_probe_success()
                # Seed the recovered latency so ARIMA starts fresh
                self.norm_history[device_id].append(norm_latency)
            else:
                cb.on_probe_failure(
                    reason=f"norm_lat {norm_latency:.3f}s ≥ {self.ADMIT_MULT}× edge {edge_pred:.3f}s"
                )

    # ── Index calculation ────────────────────────────────────────────────────

    def get_indices(self, dev: str, total_items: int) -> range:
        start = 0
        for d in self.devices:
            count = int(round(self.current_shares[d] * total_items))
            end   = start + count
            if d == dev:
                return range(start, min(end, total_items))
            start = end
        return range(0, 0)