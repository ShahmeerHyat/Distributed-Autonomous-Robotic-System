"""
shared_utils.py  —  Socket helpers following the paper's Connect.py protocol.

Protocol (identical to Edge/Connect.py):
  sender:   pickle.dumps(obj) → 6-byte big-endian length header → payload
  receiver: read 6-byte header → read exactly that many bytes → pickle.loads
"""

import pickle


def send_msg(sock, obj):
    data = pickle.dumps(obj)
    sock.send(len(data).to_bytes(6, byteorder='big'))
    sock.sendall(data)


def recv_msg(sock):
    raw = _recvall(sock, 6)
    if raw is None:
        return None
    length = int.from_bytes(raw, byteorder='big')
    data = _recvall(sock, length)
    if data is None:
        return None
    return pickle.loads(data)


def _recvall(sock, n):
    buf = b''
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf
