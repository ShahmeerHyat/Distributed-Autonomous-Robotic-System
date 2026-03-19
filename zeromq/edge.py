import os
import socket
import pickle
import struct
import sys
import time


def recv_exact(sock, n):
    data = b''
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            return None
        data += chunk
    return data


def send_object(sock, obj):
    """Send any Python object"""
    payload = pickle.dumps(obj)
    header  = struct.pack('!Q', len(payload))
    sock.sendall(header + payload)


def recv_object(sock):
    """Receive any Python object"""
    header = recv_exact(sock, 8)
    if not header:
        return None
    size    = struct.unpack('!Q', header)[0]
    payload = recv_exact(sock, size)
    return pickle.loads(payload)


# ─── Client ────────────────────────────────────────────────
def run_client(server_ip):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((server_ip, 9999))
    print(f"[CLIENT] Connected to {server_ip}")

    # Test sending different data types
    test_data = [
        "Hello from edge laptop",
        {'frame_id': 1, 'confidence': 0.92},
        [1, 2, 3, 4, 5],
        b'\x00\x01\x02\x03',         # Raw bytes
    ]

    for item in test_data:
        t0 = time.perf_counter()
        send_object(sock, item)
        response = recv_object(sock)
        rtt = (time.perf_counter() - t0) * 1000
        print(f"[CLIENT] Sent {type(item).__name__} | "
              f"RTT: {rtt:.2f}ms | Response: {response}")

    sock.close()

server_ip = os.getenv("SERVER_IP")
run_client(server_ip)