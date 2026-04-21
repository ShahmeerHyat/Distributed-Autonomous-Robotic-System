from controller import Robot    # type: ignore
import numpy as np
from ultralytics import YOLO
import time

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
# Load YOLO once before loop
# ---------------------------
print("[EDGE] Loading YOLO model...")
yolo_model = YOLO("yolov8s.pt")
print("[EDGE] YOLO model loaded.")

# ---------------------------
# CPU Optimization Settings
# ---------------------------
INFER_EVERY_N_FRAMES = 1
YOLO_RESOLUTION = 320

# ---------------------------
# Metrics Tracking
# ---------------------------
capture_times     = []
preprocess_times  = []
inference_times   = []
total_times       = []

session_start = time.perf_counter()

# ---------------------------
# Print Benchmark Summary
# ---------------------------
def print_benchmark():
    n_total = len(total_times)
    n_infer = len(inference_times)
    
    if n_total == 0:
        print("[EDGE] No frames processed — nothing to report.")
        return
        
    session_duration = time.perf_counter() - session_start

    print("\n" + "=" * 65)
    print(f"  BENCHMARK RESULTS — {session_duration:.1f}s | {n_total} frames | {n_total/session_duration:.1f} Effective FPS")
    print("=" * 65)
    print(f"  {'Metric':<14} {'Min':>8} {'Max':>8} {'Avg':>8}")
    print("-" * 65)

    avg_cap = sum(capture_times)/len(capture_times) if capture_times else 0
    avg_pre = sum(preprocess_times)/len(preprocess_times) if preprocess_times else 0
    avg_inf = sum(inference_times)/n_infer if n_infer > 0 else 0
    avg_tot = sum(total_times)/n_total if total_times else 0

    print(f"  {'Capture':<14} {min(capture_times, default=0):>7.1f}ms {max(capture_times, default=0):>7.1f}ms {avg_cap:>7.1f}ms")
    print(f"  {'Preprocess':<14} {min(preprocess_times, default=0):>7.1f}ms {max(preprocess_times, default=0):>7.1f}ms {avg_pre:>7.1f}ms")
    print(f"  {'Inference*':<14} {min(inference_times, default=0):>7.1f}ms {max(inference_times, default=0):>7.1f}ms {avg_inf:>7.1f}ms")
    print(f"  {'Total/Frame':<14} {min(total_times, default=0):>7.1f}ms {max(total_times, default=0):>7.1f}ms {avg_tot:>7.1f}ms")
    print("-" * 65)
    print(f"  *Inference ran on {n_infer} out of {n_total} frames (every {INFER_EVERY_N_FRAMES} frames)")
    print("=" * 65 + "\n")

# ---------------------------
# Tracking Config
# ---------------------------
TARGET_CLASS = "sports ball"
CENTER_TOL = 0.1
AREA_STOP_THRESHOLD = 0.25

# ---------------------------
# ByteTrack State
# ---------------------------
TARGET_TRACK_ID = None        # Lock onto first detected ball's track ID
last_target_box = None        # Fallback cache if tracker loses target briefly

# ---------------------------
# Main Loop
# ---------------------------
print("[EDGE] Starting ByteTrack tracking loop...")
frame_count = 0

def get_frame(camera):
    raw = camera.getImage()
    if not raw:
        return None
    w, h = camera.getWidth(), camera.getHeight()
    return np.ascontiguousarray(
        np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 4))[:, :, 2::-1]
    )


while robot.step(timestep) != -1:

    frame_count += 1
    target_box = None

    # ---------- ONE LINE CAPTURE + PREPROCESS ----------
    frame = get_frame(camera)
    if frame is None:
        continue

    # ---------- BYTETRACK ----------
    t_infer_start = time.perf_counter()

    try:
        results = yolo_model.track(
            frame,                        # <-- direct frame in, no extra steps
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
            imgsz=YOLO_RESOLUTION
        )[0]

        boxes = results.boxes
        names = results.names
        best_area = 0

        if boxes is not None and boxes.id is not None:
            for i, box in enumerate(boxes):
                cls_id   = int(box.cls[0])
                cls_name = names[cls_id]
                track_id = int(boxes.id[i])

                if cls_name != TARGET_CLASS:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                area = (x2 - x1) * (y2 - y1)

                if TARGET_TRACK_ID is None:
                    TARGET_TRACK_ID = track_id
                    print(f"[BYTETRACK] Locked onto Track ID: {TARGET_TRACK_ID}")

                if track_id == TARGET_TRACK_ID:
                    if area > best_area:
                        best_area  = area
                        target_box = (x1, y1, x2, y2)

        if target_box is None and TARGET_TRACK_ID is not None:
            print(f"[BYTETRACK] Lost Track ID {TARGET_TRACK_ID} — reacquiring...")
            TARGET_TRACK_ID = None

        last_target_box = target_box

    except Exception as e:
        print(f"[EDGE] Tracker error: {e}")
        last_target_box = None

    inference_times.append((time.perf_counter() - t_infer_start) * 1000)

    if target_box is None:
        target_box = last_target_box

    # ---------- CONTROL ----------
    if target_box is not None:
        x1, y1, x2, y2 = target_box
        w = camera.getWidth()

        obj_center_x   = (x1 + x2) / 2
        frame_center_x = w / 2
        error = (obj_center_x - frame_center_x) / frame_center_x
        area  = ((x2 - x1) * (y2 - y1)) / (w * camera.getHeight())

        if area > AREA_STOP_THRESHOLD:
            left_speed = right_speed = 0
            print(f"[EDGE] Target reached (Track ID {TARGET_TRACK_ID}) — stopping")

        elif abs(error) < CENTER_TOL:
            left_speed = right_speed = MAX_SPEED

        elif error > 0:
            left_speed  = MAX_SPEED
            right_speed = MAX_SPEED * (1 - abs(error))

        else:
            left_speed  = MAX_SPEED * (1 - abs(error))
            right_speed = MAX_SPEED

    else:
        left_speed = right_speed = 0

    left_motor.setVelocity(left_speed)
    right_motor.setVelocity(right_speed)