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
# Main Loop
# ---------------------------
while robot.step(timestep) != -1:

    # ---------- CAPTURE FRAME ----------
    raw_image = camera.getImage()
    width     = camera.getWidth()
    height    = camera.getHeight()

    if not raw_image:
        continue

    # ---------- SEND RAW IMAGE TO SERVER ----------
    payload = {
        'image' : raw_image,   # raw BGRA bytes from Webots camera
        'width' : width,
        'height': height,
    }

    try:
        send_object(sock, payload)
    except Exception as e:
        print(f"[EDGE] Send failed: {e} — reconnecting...")
        sock = connect_to_server(SERVER_IP, SERVER_PORT)
        continue

    # ---------- RECEIVE DETECTIONS BACK ----------
    try:
        response = recv_object(sock)
    except Exception as e:
        print(f"[EDGE] Receive failed: {e} — reconnecting...")
        sock = connect_to_server(SERVER_IP, SERVER_PORT)
        continue

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