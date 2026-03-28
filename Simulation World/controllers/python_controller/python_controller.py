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
yolo_model = YOLO("yolov8n.pt")  # Nano model is perfect for CPU
print("[EDGE] YOLO model loaded.")

# ---------------------------
# CPU Optimization Settings
# ---------------------------
INFER_EVERY_N_FRAMES = 3   # YOLO only runs 1 out of every 3 frames
YOLO_RESOLUTION = 320      # Cut resolution in half (Default is 640)

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
    
    # Safely calculate averages
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
# Main Loop
# ---------------------------
print("[EDGE] Starting main loop...")
frame_count = 0
last_detected_labels = []

while robot.step(timestep) != -1:

    if frame_count >= 500:
        break
        
    t_frame_start = time.perf_counter()
    frame_count += 1

    # ---------- CAPTURE FRAME ----------
    t_capture_start = time.perf_counter()
    raw_image = camera.getImage()
    width     = camera.getWidth()
    height    = camera.getHeight()

    if not raw_image:
        continue

    t_capture_end = time.perf_counter()

    # ---------- PREPROCESS ----------
    t_pre_start = time.perf_counter()

    frame = np.frombuffer(raw_image, dtype=np.uint8).reshape((height, width, 4))
    # CPU Optimization: Single-pass slice to drop alpha (4th channel) and reverse BGR to RGB
    frame = np.ascontiguousarray(frame[:, :, 2::-1]) 

    t_pre_end = time.perf_counter()

    # ---------- RUN YOLO (CPU OPTIMIZED) ----------
    inference_ms = 0  # Default to 0 for skipped frames
    
    if frame_count % INFER_EVERY_N_FRAMES == 0:
        t_infer_start = time.perf_counter()

        try:
            # CPU Optimization: Lower imgsz dramatically reduces CPU load
            results         = yolo_model(frame, verbose=False, imgsz=YOLO_RESOLUTION)
            boxes           = results[0].boxes
            class_names     = results[0].names
            last_detected_labels = [class_names[int(cls)] for cls in boxes.cls] if len(boxes) > 0 else []
        except Exception as e:
            print(f"[EDGE] YOLO error: {e}")
            last_detected_labels = []

        t_infer_end = time.perf_counter()
        inference_ms = (t_infer_end - t_infer_start) * 1000
        inference_times.append(inference_ms)

    # Use the cached labels on skipped frames
    detected_labels = last_detected_labels

    # ---------- USE DETECTIONS ----------
    # Uncomment to see detections spam:
    # if detected_labels:
    #     print(f"[EDGE] Detected: {detected_labels}")

    # # ---------- COMPUTE & LOG TIMINGS ----------
    capture_ms    = (t_capture_end - t_capture_start) * 1000
    preprocess_ms = (t_pre_end     - t_pre_start)     * 1000
    total_ms      = (time.perf_counter() - t_frame_start) * 1000

    capture_times.append(capture_ms)
    preprocess_times.append(preprocess_ms)
    total_times.append(total_ms)

# Loop exited — simulation stopped
print_benchmark()