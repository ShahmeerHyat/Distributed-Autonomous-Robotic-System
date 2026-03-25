from controller import Robot
import socket
import pickle
import struct
import time
import os

# ---------------------------
# Initialize Robot & Devices
# ---------------------------
robot = Robot()
timestep = int(robot.getBasicTimeStep())

# ----- Motors -----
left_motor  = robot.getDevice("left wheel motor")
right_motor = robot.getDevice("right wheel motor")
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))
MAX_SPEED = 6.28

# ----- Camera -----
camera = robot.getDevice("camera")
camera.enable(timestep)

# ----- Distance Sensors -----
proximity = []
for i in range(8):
    ps = robot.getDevice(f"ps{i}")
    ps.enable(timestep)
    proximity.append(ps)

print("[EDGE] Robot and devices initialized.")


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
# Connect to Server (with retry)
# ---------------------------
SERVER_IP   = os.getenv("SERVER_IP")
SERVER_PORT = 9999

def connect_to_server(ip, port):
    while True:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((ip, port))
            print(f"[EDGE] Connected to server {ip}:{port}")
            return s
        except Exception as e:
            print(f"[EDGE] Connection failed: {e} — retrying in 2s...")
            time.sleep(2)

sock = connect_to_server(SERVER_IP, SERVER_PORT)

print("[EDGE] Starting main loop...")

# ---------------------------
# Metrics Tracking
# ---------------------------
capture_times = []
send_times    = []
recv_times    = []
total_times   = []

BENCHMARK_DURATION = 10.0   # seconds to collect metrics
benchmark_start    = time.perf_counter()
benchmarking       = True

# ---------------------------
# Main Loop
# ---------------------------
while robot.step(timestep) != -1:

    # ---------- CAPTURE FRAME ----------
    t_capture_start = time.perf_counter()

    raw_image = camera.getImage()
    width     = camera.getWidth()
    height    = camera.getHeight()

    if not raw_image:
        continue

    t_capture_end = time.perf_counter()   # pure capture time isolated

    # ---------- SEND RAW IMAGE TO SERVER ----------
    payload = {
        'image' : raw_image,
        'width' : width,
        'height': height,
    }

    t0 = time.perf_counter()              # pure send starts here

    try:
        send_object(sock, payload)
    except Exception as e:
        print(f"[EDGE] Send failed: {e} — reconnecting...")
        sock = connect_to_server(SERVER_IP, SERVER_PORT)
        continue

    t1 = time.perf_counter()             # pure send ends here

    # ---------- RECEIVE DETECTIONS BACK ----------
    try:
        response = recv_object(sock)
    except Exception as e:
        print(f"[EDGE] Receive failed: {e} — reconnecting...")
        sock = connect_to_server(SERVER_IP, SERVER_PORT)
        continue

    t2 = time.perf_counter()             # recv ends here

    if response is None:
        print("[EDGE] Empty response — reconnecting...")
        sock = connect_to_server(SERVER_IP, SERVER_PORT)
        continue

    # ---------- USE DETECTIONS ----------
    detected_labels = response.get('detections', [])

    if detected_labels:
        print(f"[EDGE] Detected: {detected_labels}")
    else:
        print("[EDGE] Nothing detected")

    # Movement logic goes here later

    # ---------- COMPUTE & LOG TIMINGS ----------
    capture_ms = (t_capture_end - t_capture_start) * 1000
    send_ms    = (t1 - t0) * 1000
    recv_ms    = (t2 - t1) * 1000
    total_ms   = (t_capture_end - t_capture_start + t1 - t0 + t2 - t1) * 1000

    # print(f"Capture: {capture_ms:.1f}ms | Send: {send_ms:.1f}ms | Recv: {recv_ms:.1f}ms | Total: {total_ms:.1f}ms")

    # ---------- ACCUMULATE FOR BENCHMARK ----------
    if benchmarking:
        capture_times.append(capture_ms)
        send_times.append(send_ms)
        recv_times.append(recv_ms)
        total_times.append(total_ms)

        elapsed = time.perf_counter() - benchmark_start
        if elapsed >= BENCHMARK_DURATION:
            benchmarking = False

n = len(total_times)
print("\n" + "=" * 55)
print(f"  BENCHMARK RESULTS — {BENCHMARK_DURATION:.0f}s | {n} frames | {n/elapsed:.1f} FPS")
print("=" * 55)
print(f"  {'Metric':<10} {'Min':>8} {'Max':>8} {'Avg':>8}")
print("-" * 55)
for label, data in [
    ("Capture",  capture_times),
    ("Send",     send_times),
    ("Recv",     recv_times),
    ("Total",    total_times),
]:
    print(f"  {label:<10} {min(data):>7.1f}ms {max(data):>7.1f}ms {sum(data)/n:>7.1f}ms")
print("=" * 55 + "\n")