import io
import struct
import numpy as np
import torch
import time
import socket

PROBE_PAYLOAD_SHAPE = (1, 1, 1)  # tiny tensor, just for RTT

# Magic prefixes to distinguish fast tensor path from pickle path.
# Both master and worker use this module, so the protocol is symmetric.
_MAGIC_TENSOR = b'\xfe\xed'
_MAGIC_PICKLE = b'\xca\xfe'


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


def send_msg(sock, msg):
    """
    Pure tensors use a raw-bytes fast path (~10× faster than pickle for small
    tensors). Everything else (tuples, dicts, control messages) uses torch.save.
    """
    if isinstance(msg, torch.Tensor):
        arr     = msg.detach().cpu().contiguous().numpy()
        dtype_b = arr.dtype.str.encode()          # e.g. b'<f2' for float16
        ndim    = arr.ndim
        payload = (
            _MAGIC_TENSOR
            + struct.pack('>BB', ndim, len(dtype_b))
            + struct.pack(f'>{ndim}Q', *arr.shape)
            + dtype_b
            + arr.tobytes()
        )
    else:
        buf = io.BytesIO()
        torch.save(_to_cpu(msg), buf)
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
    msglen = struct.unpack('>I', raw_len)[0]
    data   = recvall(sock, msglen)

    if data[:2] == _MAGIC_TENSOR:
        ndim, dlen = struct.unpack('>BB', data[2:4])
        shape      = struct.unpack(f'>{ndim}Q', data[4:4 + ndim * 8])
        offset     = 4 + ndim * 8
        dtype_str  = data[offset:offset + dlen].decode()
        raw        = data[offset + dlen:]
        arr        = np.frombuffer(raw, dtype=np.dtype(dtype_str)).reshape(shape)
        return torch.from_numpy(arr.copy())
    else:
        return torch.load(io.BytesIO(data[2:]), weights_only=False)


def probe_rtt(host: str, port: int, timeout: float = 2.0) -> float | None:
    """
    Opens a short-lived connection, sends a minimal ping tensor,
    waits for echo. Returns RTT in seconds or None if unreachable.
    Kept separate from inference socket — pure network signal, no compute noise.
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect((host, port))

        payload = {"type": "probe", "tensor": torch.zeros(PROBE_PAYLOAD_SHAPE)}

        t0 = time.perf_counter()
        send_msg(sock, payload)
        response = recv_msg(sock)
        rtt = time.perf_counter() - t0

        sock.close()

        if response and response.get("type") == "probe_ack":
            return rtt
        return None

    except (socket.timeout, ConnectionRefusedError, OSError):
        return None


def ping_worker(sock, load: bool = False, timeout: float = 2.0) -> float | None:
    """
    Sends a lightweight ping over an existing connected socket.
    Returns RTT in seconds or None on failure.
    Pure network signal — worker echoes immediately without compute.
    """
    try:
        sock.settimeout(timeout)
        payload = {"type": "probe", "tensor": torch.zeros(PROBE_PAYLOAD_SHAPE)} if load else {"type": "probe"}

        t0 = time.perf_counter()
        send_msg(sock, payload)

        response = recv_msg(sock)
        rtt = time.perf_counter() - t0

        if response and response.get("type") == "probe_ack":
            return rtt
        return None

    except (socket.timeout, OSError):
        return None


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
