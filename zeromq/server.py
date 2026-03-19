
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


# ─── Server ────────────────────────────────────────────────
def run_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('0.0.0.0', 9999))
    server.listen(1)
    print("[SERVER] Waiting for connection...")

    conn, addr = server.accept()
    print(f"[SERVER] Connected: {addr}")

    while True:
        obj = recv_object(conn)
        if obj is None:
            break

        # Print what you received
        print(f"[SERVER] Received: {type(obj).__name__}", end=' ')

        if isinstance(obj, dict):
            print(f"| keys: {list(obj.keys())}")
        else:
            print(f"| value: {obj}")

        # Send back a response
        response = {'status': 'ok', 'received': str(type(obj))}
        send_object(conn, response)

    print("[SERVER] Disconnected")

run_server()