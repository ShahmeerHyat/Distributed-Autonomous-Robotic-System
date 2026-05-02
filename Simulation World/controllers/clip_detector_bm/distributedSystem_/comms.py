import io
import struct
import torch
import time
import socket

PROBE_PAYLOAD_SHAPE = (1, 1, 1)  # tiny tensor, just for RTT

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
    data   = recvall(sock, msglen)
    return torch.load(io.BytesIO(data), weights_only=False)


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
    
def ping_worker(sock, load: bool = False,timeout: float = 2.0) -> float | None:
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