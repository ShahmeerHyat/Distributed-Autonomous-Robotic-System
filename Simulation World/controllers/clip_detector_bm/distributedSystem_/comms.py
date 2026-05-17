import io
import pickle
import struct
import numpy as np
import torch
import time
import socket

PROBE_PAYLOAD_SHAPE = (1, 1, 1)

# Wire-format magic prefixes (2 bytes each).
# Every payload on the socket is prefixed with one of these so recv_msg
# knows how to deserialise without any out-of-band signalling.
_MAGIC_TENSOR   = b'\xfe\xed'   # single raw tensor (fast path)
_MAGIC_COMPOUND = b'\xba\xbe'   # skeleton + N raw tensors (fast path)
_MAGIC_PICKLE   = b'\xca\xfe'   # pure-python / control message (fallback)

# Sentinel used inside skeletons to mark where a tensor was extracted.
_T = '__T__'


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, tuple):
        return tuple(_to_cpu(x) for x in obj)
    if isinstance(obj, list):
        return [_to_cpu(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    return obj


def _any_tensor(obj) -> bool:
    if isinstance(obj, torch.Tensor):
        return True
    if isinstance(obj, (tuple, list)):
        return any(_any_tensor(x) for x in obj)
    if isinstance(obj, dict):
        return any(_any_tensor(v) for v in obj.values())
    return False


def _strip(obj, bucket: list):
    """Replace tensors with (_T, idx) sentinels; collect numpy arrays."""
    if isinstance(obj, torch.Tensor):
        idx = len(bucket)
        bucket.append(obj.detach().cpu().contiguous().numpy())
        return (_T, idx)
    if isinstance(obj, tuple):
        return tuple(_strip(x, bucket) for x in obj)
    if isinstance(obj, list):
        return [_strip(x, bucket) for x in obj]
    if isinstance(obj, dict):
        return {k: _strip(v, bucket) for k, v in obj.items()}
    return obj


def _restore(obj, tensors: list):
    if isinstance(obj, tuple):
        if len(obj) == 2 and obj[0] == _T:
            return tensors[obj[1]]
        return tuple(_restore(x, tensors) for x in obj)
    if isinstance(obj, list):
        return [_restore(x, tensors) for x in obj]
    if isinstance(obj, dict):
        return {k: _restore(v, tensors) for k, v in obj.items()}
    return obj


def _encode_arr(arr: np.ndarray) -> bytes:
    """Encode a numpy array as: [ndim][dlen][shape…][dtype][data]."""
    dtype_b = arr.dtype.str.encode()
    return (
        struct.pack('>BB', arr.ndim, len(dtype_b))
        + struct.pack(f'>{arr.ndim}Q', *arr.shape)
        + dtype_b
        + arr.tobytes()
    )


def _decode_arr(data: bytes) -> torch.Tensor:
    ndim, dlen = struct.unpack('>BB', data[:2])
    shape  = struct.unpack(f'>{ndim}Q', data[2:2 + ndim * 8])
    offset = 2 + ndim * 8
    dtype  = data[offset:offset + dlen].decode()
    raw    = data[offset + dlen:]
    return torch.from_numpy(
        np.frombuffer(raw, dtype=np.dtype(dtype)).reshape(shape).copy()
    )


# ── Public API ────────────────────────────────────────────────────────────────

def send_msg(sock, msg):
    """
    Three paths, in priority order:

    1. Pure tensor  → _MAGIC_TENSOR  + raw numpy bytes   (~0.05 ms)
    2. Container with tensors → _MAGIC_COMPOUND:
         pickle skeleton (tensors stripped out, ~100 B)
         + N × raw tensor blocks                         (~0.15 ms per tensor)
    3. Pure-Python control message → _MAGIC_PICKLE + torch.save fallback
    """
    if isinstance(msg, torch.Tensor):
        arr     = msg.detach().cpu().contiguous().numpy()
        dtype_b = arr.dtype.str.encode()
        payload = (
            _MAGIC_TENSOR
            + struct.pack('>BB', arr.ndim, len(dtype_b))
            + struct.pack(f'>{arr.ndim}Q', *arr.shape)
            + dtype_b
            + arr.tobytes()
        )

    elif _any_tensor(msg):
        bucket   = []
        skeleton = _strip(msg, bucket)
        skel_b   = pickle.dumps(skeleton, protocol=4)
        parts    = [_MAGIC_COMPOUND, struct.pack('>I', len(skel_b)), skel_b]
        for arr in bucket:
            enc = _encode_arr(arr)
            parts.append(struct.pack('>I', len(enc)) + enc)
        payload = b''.join(parts)

    else:
        buf = io.BytesIO()
        torch.save(msg, buf)
        payload = _MAGIC_PICKLE + buf.getvalue()

    sock.sendall(struct.pack('>I', len(payload)) + payload)


def recvall(sock, n):
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return bytes(data)


def recv_msg(sock):
    raw_len = recvall(sock, 4)
    if not raw_len:
        return None
    data  = recvall(sock, struct.unpack('>I', raw_len)[0])
    magic = data[:2]

    if magic == _MAGIC_TENSOR:
        return _decode_arr(data[2:])

    if magic == _MAGIC_COMPOUND:
        skel_len = struct.unpack('>I', data[2:6])[0]
        skeleton = pickle.loads(data[6:6 + skel_len])
        offset   = 6 + skel_len
        tensors  = []
        while offset < len(data):
            t_len = struct.unpack('>I', data[offset:offset + 4])[0]
            offset += 4
            tensors.append(_decode_arr(data[offset:offset + t_len]))
            offset += t_len
        return _restore(skeleton, tensors)

    # _MAGIC_PICKLE — control messages (QUIT, REGISTER, probe acks, …)
    return torch.load(io.BytesIO(data[2:]), weights_only=False)


# ── RTT probing ───────────────────────────────────────────────────────────────

def probe_rtt(host: str, port: int, timeout: float = 2.0) -> float | None:
    """
    Opens a short-lived connection, sends a minimal ping tensor,
    waits for echo. Returns RTT in seconds or None if unreachable.
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect((host, port))

        payload = {"type": "probe", "tensor": torch.zeros(PROBE_PAYLOAD_SHAPE)}

        t0       = time.perf_counter()
        send_msg(sock, payload)
        response = recv_msg(sock)
        rtt      = time.perf_counter() - t0

        sock.close()
        return rtt if (response and response.get("type") == "probe_ack") else None

    except (socket.timeout, ConnectionRefusedError, OSError):
        return None


def ping_worker(sock, load: bool = False, timeout: float = 2.0) -> float | None:
    """
    Sends a lightweight ping over an existing connected socket.
    Returns RTT in seconds or None on failure.
    """
    try:
        sock.settimeout(timeout)
        payload = (
            {"type": "probe", "tensor": torch.zeros(PROBE_PAYLOAD_SHAPE)}
            if load else {"type": "probe"}
        )

        t0       = time.perf_counter()
        send_msg(sock, payload)
        response = recv_msg(sock)
        rtt      = time.perf_counter() - t0

        return rtt if (response and response.get("type") == "probe_ack") else None

    except (socket.timeout, OSError):
        return None


# ── Circuit breaker ───────────────────────────────────────────────────────────

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
        tag = f" ({reason})" if reason else ""
        print(f"  [CB] ⚡ Tripped{tag}. Cooldown = {cooldown} blocks "
              f"(trip #{self.consecutive_trips})")

    def tick(self) -> bool:
        if self.state == self.OPEN:
            self.blocks_remaining -= 1
            if self.blocks_remaining <= 0:
                self.state = self.HALF_OPEN
                print("  [CB] 🔍 Cooldown elapsed → HALF_OPEN (probing next block)")
                return True
        return False

    def on_probe_success(self):
        self.state             = self.CLOSED
        self.consecutive_trips = 0
        print("  [CB] ✅ Probe succeeded → CLOSED")

    def on_probe_failure(self, reason: str = ""):
        self.trip(reason=f"probe failed: {reason}" if reason else "probe failed")

    @property
    def is_open(self)      -> bool: return self.state == self.OPEN
    @property
    def is_half_open(self) -> bool: return self.state == self.HALF_OPEN
    @property
    def is_closed(self)    -> bool: return self.state == self.CLOSED
