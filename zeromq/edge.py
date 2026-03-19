
import socket
import os
import sys
import struct
import time
from dotenv import load_dotenv

load_dotenv()


def send_file(server_ip, port, filepath):
    """Send any file to server"""
    
    filename = os.path.basename(filepath)
    filesize = os.path.getsize(filepath)
    
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    while True:
        try:
            sock.connect((server_ip, port))
            print("[CLIENT] Connected")
            break
        except ConnectionRefusedError:
            print("[CLIENT] Server not ready, retrying...")
            time.sleep(2)
    
    # Send filename length + filename + filesize
    name_bytes = filename.encode()
    header = struct.pack('!I', len(name_bytes)) + name_bytes
    header += struct.pack('!Q', filesize)
    sock.sendall(header)
    
    # Send file data in chunks
    sent = 0
    t0 = time.perf_counter()
    
    with open(filepath, 'rb') as f:
        while sent < filesize:
            chunk = f.read(65536)   # 64KB chunks
            if not chunk:
                break
            sock.sendall(chunk)
            sent += len(chunk)
            
            # Progress bar
            pct = sent / filesize * 100
            bar = '█' * int(pct / 2) + '░' * (50 - int(pct / 2))
            print(f"\r[CLIENT] [{bar}] {pct:.1f}%", end='', flush=True)
    
    elapsed = time.perf_counter() - t0
    speed_mb = (filesize / 1024 / 1024) / elapsed
    print(f"\n[CLIENT] Done in {elapsed:.2f}s ({speed_mb:.1f} MB/s)")
    sock.close()


def receive_file(port, save_folder='.'):
    """Receive any file from client"""
    
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('0.0.0.0', port))
    server.listen(1)
    print(f"[SERVER] Waiting for file on port {port}...")
    
    conn, addr = server.accept()
    print(f"[SERVER] Connection from {addr}")
    
    # Read filename length
    name_len = struct.unpack('!I', recv_exact(conn, 4))[0]
    filename  = recv_exact(conn, name_len).decode()
    filesize  = struct.unpack('!Q', recv_exact(conn, 8))[0]
    
    print(f"[SERVER] Receiving: {filename} ({filesize/1024:.1f} KB)")
    
    save_path = os.path.join(save_folder, filename)
    received = 0
    t0 = time.perf_counter()
    
    with open(save_path, 'wb') as f:
        while received < filesize:
            chunk = conn.recv(min(65536, filesize - received))
            if not chunk:
                break
            f.write(chunk)
            received += len(chunk)
            
            pct = received / filesize * 100
            bar = '█' * int(pct / 2) + '░' * (50 - int(pct / 2))
            print(f"\r[SERVER] [{bar}] {pct:.1f}%", end='', flush=True)
    
    elapsed = time.perf_counter() - t0
    speed_mb = (filesize / 1024 / 1024) / elapsed
    print(f"\n[SERVER] Saved to {save_path}")
    print(f"[SERVER] {elapsed:.2f}s ({speed_mb:.1f} MB/s)")
    conn.close()
    server.close()


def recv_exact(sock, n):
    """Receive exactly n bytes"""
    data = b''
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise ConnectionError("Connection closed")
        data += chunk
    return data



PORT = 9999
server_ip = os.getenv("SERVER_IP")
filepath  = sys.argv[2]
send_file(server_ip, PORT, filepath)