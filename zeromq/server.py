import socket
import pickle
import struct
import numpy as np
from ultralytics import YOLO
import time


# ---------------------------
# Load YOLO once before loop
# ---------------------------
print("[SERVER] Loading YOLO model...")
yolo_model = YOLO("yolov8n.pt")
print("[SERVER] YOLO model loaded.")


# ---------------------------
# Socket Helpers
# ---------------------------
def recv_exact(sock, n):
    data = b''
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            return None
        data += chunk
    return data


def send_object(sock, obj):
    payload = pickle.dumps(obj)
    header  = struct.pack('!Q', len(payload))
    sock.sendall(header + payload)


def recv_object(sock):
    header = recv_exact(sock, 8)
    if not header:
        return None
    size    = struct.unpack('!Q', header)[0]
    payload = recv_exact(sock, size)
    if not payload:
        return None
    return pickle.loads(payload)


# ---------------------------
# Print Benchmark Summary
# ---------------------------
def print_benchmark(preprocess_times, inference_times, send_times, total_times, session_duration):
    n = len(total_times)
    if n == 0:
        print("[SERVER] No frames processed — nothing to report.")
        return

    print("\n" + "=" * 60)
    print(f"  SERVER BENCHMARK — {session_duration:.1f}s | {n} frames | {n/session_duration:.1f} FPS")
    print("=" * 60)
    print(f"  {'Metric':<14} {'Min':>8} {'Max':>8} {'Avg':>8}")
    print("-" * 60)
    for label, data in [
        ("Preprocess",  preprocess_times),
        ("Inference",   inference_times),
        ("Send",        send_times),
        ("Total",       total_times),
    ]:
        print(f"  {label:<14} {min(data):>7.1f}ms {max(data):>7.1f}ms {sum(data)/n:>7.1f}ms")
    print("=" * 60 + "\n")


# ---------------------------
# Handle One Client Session
# ---------------------------
def handle_client(conn, addr):
    print(f"[SERVER] Edge device connected from {addr}")

    # --- Per-session metric accumulators ---
    recv_times        = []
    preprocess_times  = []
    inference_times   = []
    send_times        = []
    total_times       = []
    session_start     = time.perf_counter()

    while True:

        t_frame_start = time.perf_counter()

        # ---------- RECEIVE IMAGE FROM EDGE ----------
        try:
            data = recv_object(conn)
        except Exception as e:
            print(f"[SERVER] Receive error: {e}")
            break
        if data is None:
            print("[SERVER] Edge disconnected.")
            break

        # ---------- RECONSTRUCT IMAGE ----------
        t_pre_start = time.perf_counter()

        raw_image = data['image']
        width     = data['width']
        height    = data['height']

        frame = np.frombuffer(raw_image, dtype=np.uint8).reshape((height, width, 4))
        frame = frame[:, :, :3]    # Drop alpha, keep BGR
        frame = frame[:, :, ::-1]  # BGR → RGB

        t_pre_end = time.perf_counter()

        # ---------- RUN YOLO ----------
        t_infer_start = time.perf_counter()
        try:
            results         = yolo_model(frame, verbose=False)
            boxes           = results[0].boxes
            class_names     = results[0].names
            detected_labels = [class_names[int(cls)] for cls in boxes.cls] if len(boxes) > 0 else []
        except Exception as e:
            print(f"[SERVER] YOLO error: {e}")
            detected_labels = []
        t_infer_end = time.perf_counter()

        print(f"[SERVER] Detections: {detected_labels}")

        # ---------- SEND DETECTIONS BACK ----------
        t_send_start = time.perf_counter()
        try:
            send_object(conn, {'detections': detected_labels})
        except Exception as e:
            print(f"[SERVER] Send error: {e}")
            break
        t_send_end = time.perf_counter()

        t_frame_end = time.perf_counter()

        # ---------- ACCUMULATE METRICS ----------
        preprocess_ms = (t_pre_end    - t_pre_start)   * 1000
        inference_ms  = (t_infer_end  - t_infer_start) * 1000
        send_ms       = (t_send_end   - t_send_start)  * 1000
        total_ms      = (t_frame_end  - t_frame_start) * 1000

        preprocess_times.append(preprocess_ms)
        inference_times.append(inference_ms)
        send_times.append(send_ms)
        total_times.append(total_ms)

        # print(f"Pre: {preprocess_ms:.1f}ms | Infer: {inference_ms:.1f}ms | Send: {send_ms:.1f}ms | Total: {total_ms:.1f}ms")

    conn.close()
    print(f"[SERVER] Connection closed: {addr}")

    # ---------- PRINT SUMMARY ON DISCONNECT ----------
    session_duration = time.perf_counter() - session_start
    print_benchmark(preprocess_times, inference_times, send_times, total_times, session_duration)


# ---------------------------
# Start Server
# ---------------------------
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(('0.0.0.0', 9999))
server.listen(1)
print("[SERVER] Waiting for edge device on port 9999...")

while True:
    conn, addr = server.accept()
    handle_client(conn, addr)
    print("[SERVER] Waiting for next connection...")