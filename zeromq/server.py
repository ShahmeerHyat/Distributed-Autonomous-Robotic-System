import socket
import pickle
import struct
import numpy as np
from ultralytics import YOLO


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
# Handle One Client Session
# ---------------------------
def handle_client(conn, addr):
    print(f"[SERVER] Edge device connected from {addr}")

    while True:

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
        raw_image = data['image']
        width     = data['width']
        height    = data['height']

        # Webots sends BGRA — convert to RGB numpy array for YOLO
        frame = np.frombuffer(raw_image, dtype=np.uint8).reshape((height, width, 4))
        frame = frame[:, :, :3]   # Drop alpha, keep BGR
        frame = frame[:, :, ::-1] # BGR → RGB

        # ---------- RUN YOLO ----------
        try:
            results       = yolo_model(frame, verbose=False)
            boxes         = results[0].boxes
            class_names   = results[0].names

            if len(boxes) > 0:
                detected_labels = [class_names[int(cls)] for cls in boxes.cls]
            else:
                detected_labels = []

        except Exception as e:
            print(f"[SERVER] YOLO error: {e}")
            detected_labels = []

        print(f"[SERVER] Detections: {detected_labels}")

        # ---------- SEND DETECTIONS BACK ----------
        response = {'detections': detected_labels}

        try:
            send_object(conn, response)
        except Exception as e:
            print(f"[SERVER] Send error: {e}")
            break

    conn.close()
    print(f"[SERVER] Connection closed: {addr}")


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