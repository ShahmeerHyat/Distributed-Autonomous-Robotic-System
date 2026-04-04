from controller import Robot
from PIL import Image
import zmq
import msgpack
import io
import numpy as np
from ultralytics import YOLO
import cv2
print("Script Starte")
# ---------------------------
# Initialize Robot & Devices
# ---------------------------
robot = Robot()
timestep = int(robot.getBasicTimeStep())

# ----- Motors -----
left_motor = robot.getDevice("left wheel motor")
right_motor = robot.getDevice("right wheel motor")
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))
MAX_SPEED = 6.28

# ----- Camera -----
camera = robot.getDevice("camera")
camera.enable(timestep)

# ----- Distance Sensors (optional) -----
proximity = []
for i in range(8):
    ps = robot.getDevice(f"ps{i}")
    ps.enable(timestep)
    proximity.append(ps)

# ---------------------------
# Initialize ZMQ Publisher
# ---------------------------
ctx = zmq.Context()
sock = ctx.socket(zmq.PUB)
sock.bind("tcp://*:5555")

print("[INFO] Python controller initialized.")

# ---------------------------
# Initialize YOLO
# ---------------------------
yolo_model = YOLO("yolov8n.pt")  # make sure model is in controller folder

# ---------------------------
# Movement State Variables
# ---------------------------
state = "forward"
counter = 0

# ---------------------------
# Main Loop
# ---------------------------
while robot.step(timestep) != -1:

    # ---------- CAMERA ----------
    raw_image = camera.getImage()
    width = camera.getWidth()
    height = camera.getHeight()

    if raw_image:
        # Convert BGRA bytes to Pillow Image
        img = Image.frombytes("RGBA", (width, height), raw_image, "raw", "BGRA")

        # ---------- YOLO DETECTION ----------
        results = np.array(img.convert("RGB"))
        results = yolo_model(results)
        annotated = results[0].plot()  # annotated NumPy array

        # Convert annotated NumPy array to JPEG bytes
        annotated_img = Image.fromarray(annotated)
        buf = io.BytesIO()
        annotated_img.save(buf, format="JPEG")

        payload = {
            "time": robot.getTime(),
            "data": buf.getvalue()
        }

        # Send annotated frame to external server
        sock.send(msgpack.packb(payload, use_bin_type=True))

        # ---------- ROBOT REACTION TO DETECTION ----------
        detected_classes = results[0].names
        detected_labels = [detected_classes[int(cls)] for cls in results[0].boxes.cls] if len(results[0].boxes) > 0 else []
        # Example reaction: slow down if a "person" detected
        if "person" in detected_labels:
            left_motor.setVelocity(0.2 * MAX_SPEED)
            right_motor.setVelocity(0.2 * MAX_SPEED)
        else:
            # ---------- OBSTACLE AVOIDANCE ----------
            sensor_values = [ps.getValue() for ps in proximity]
            obstacle = max(sensor_values) > 80  # threshold

            if obstacle:
                left_motor.setVelocity(0.5 * MAX_SPEED)
                right_motor.setVelocity(-0.5 * MAX_SPEED)
            else:
                # ---------- FIXED PATH ----------
                if state == "forward":
                    left_motor.setVelocity(0.5 * MAX_SPEED)
                    right_motor.setVelocity(0.5 * MAX_SPEED)
                    counter += 1
                    if counter > 100:
                        state = "turn"
                        counter = 0
                elif state == "turn":
                    left_motor.setVelocity(0.5 * MAX_SPEED)
                    right_motor.setVelocity(-0.5 * MAX_SPEED)
                    counter += 1
                    if counter > 40:
                        state = "forward"
                        counter = 0