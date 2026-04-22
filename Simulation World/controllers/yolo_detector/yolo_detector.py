from controller import Robot    # type: ignore
import numpy as np
from ultralytics import YOLO
import time

# ---------------------------
# Initialize Robot & Devices
# ---------------------------
robot    = Robot()
timestep = int(robot.getBasicTimeStep())

left_motor  = robot.getDevice("left wheel motor")
right_motor = robot.getDevice("right wheel motor")
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))
MAX_SPEED = 6.28

camera = robot.getDevice("camera")
camera.enable(timestep)

print("[INIT] Robot and devices initialized.")

# ---------------------------
# Load YOLO once
# ---------------------------
print("[INIT] Loading YOLO model...")
yolo_model = YOLO("yolov8s.pt")
print("[INIT] YOLO model loaded.")

# ---------------------------
# Config
# ---------------------------
TARGET_CLASS     = "sports ball"

# Steering
CENTER_TOL       = 0.10
BASE_SPEED       = MAX_SPEED * 0.65
TURN_GAIN        = MAX_SPEED * 0.55
SEARCH_SPEED     = MAX_SPEED * 0.30

# Frame-skip
INFER_EVERY_N    = 1
YOLO_RESOLUTION  = 480

# Track lock
MAX_MISS_FRAMES  = 10

# ---------------------------
# Jitter Fix 1 — Warm-up
# ---------------------------
# Robot holds still for this many inference frames at startup.
# ByteTrack's Kalman filter needs ~5–8 frames to converge on a stable box.
# During this window we run YOLO but do NOT move or update last_error.
WARMUP_FRAMES    = 8

# ---------------------------
# Jitter Fix 2 — Adaptive smoothing
# ---------------------------
# At startup we use a much heavier filter (low alpha) so noisy early boxes
# don't kick the robot sideways. Once the tracker has been stable for
# STABLE_FRAMES consecutive detections, we switch to the normal alpha.
ERROR_ALPHA_INIT   = 0.10   # heavy smoothing during warmup / early tracking
ERROR_ALPHA_STABLE = 0.40   # normal smoothing once tracker is confident
STABLE_FRAMES      = 12     # detections needed before switching to normal alpha

# ---------------------------
# Jitter Fix 3 — Minimum detection confidence
# ---------------------------
# Ignore any YOLO box below this confidence to filter out weak early detections.
MIN_CONF         = 0.40

# ---------------------------
# Jitter Fix 4 — Error dead-band gate
# ---------------------------
# Only update last_error when the new raw error differs from the smoothed
# error by more than this amount. Tiny box wobbles (< JITTER_GATE) are ignored.
JITTER_GATE      = 0.03

# ---------------------------
# Metrics
# ---------------------------
inference_times = []
total_times     = []
session_start   = time.perf_counter()


def print_benchmark():
    n_total = len(total_times)
    n_infer = len(inference_times)
    if n_total == 0:
        print("[BENCH] No frames processed.")
        return
    duration = time.perf_counter() - session_start
    print("\n" + "=" * 60)
    print(f"  BENCHMARK — {duration:.1f}s | {n_total} frames | "
          f"{n_total/duration:.1f} effective FPS")
    print("=" * 60)
    avg_inf = sum(inference_times) / n_infer if n_infer else 0
    avg_tot = sum(total_times)     / n_total if n_total else 0
    print(f"  Inference  min={min(inference_times, default=0):.1f}ms  "
          f"max={max(inference_times, default=0):.1f}ms  avg={avg_inf:.1f}ms")
    print(f"  Total/frm  min={min(total_times, default=0):.1f}ms  "
          f"max={max(total_times, default=0):.1f}ms  avg={avg_tot:.1f}ms")
    print(f"  Inference ran on {n_infer}/{n_total} frames "
          f"(every {INFER_EVERY_N} frames)")
    print("=" * 60 + "\n")


# ---------------------------
# Frame capture
# ---------------------------
def get_frame(cam):
    raw = cam.getImage()
    if not raw:
        return None
    w, h = cam.getWidth(), cam.getHeight()
    return np.ascontiguousarray(
        np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 4))[:, :, 2::-1]
    )


# ---------------------------
# Steering
# ---------------------------
def compute_speeds(error):
    forward = BASE_SPEED * (1.0 - 0.35 * abs(error))
    delta   = TURN_GAIN  * error
    left_speed  = np.clip(forward + delta, -MAX_SPEED, MAX_SPEED)
    right_speed = np.clip(forward - delta, -MAX_SPEED, MAX_SPEED)
    direction   = "CENTRE" if abs(error) < CENTER_TOL else ("RIGHT" if error > 0 else "LEFT")
    return left_speed, right_speed, direction


# ---------------------------
# State
# ---------------------------
TARGET_TRACK_ID  = None
miss_streak      = 0
stable_streak    = 0       # consecutive confident detections after lock
last_error       = 0.0
infer_count      = 0       # counts only inference frames (for warmup gate)
frame_count      = 0

print(f"[RUN] Tracking: '{TARGET_CLASS}'  (warming up for {WARMUP_FRAMES} frames...)")

# ---------------------------
# Main Loop
# ---------------------------
while robot.step(timestep) != -1:

    t0 = time.perf_counter()
    frame_count += 1
    run_infer    = (frame_count % INFER_EVERY_N == 0)

    frame = get_frame(camera)
    if frame is None:
        continue

    target_box = None
    box_conf   = 0.0

    # ------------------------------------------------------------------
    # YOLO + ByteTrack
    # ------------------------------------------------------------------
    if run_infer:
        infer_count += 1
        t_infer = time.perf_counter()

        try:
            results = yolo_model.track(
                frame,
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
                    cls_name = names[int(box.cls[0])]
                    track_id = int(boxes.id[i])
                    conf     = float(box.conf[0])

                    if cls_name != TARGET_CLASS:
                        continue

                    # Jitter Fix 3: skip weak detections
                    if conf < MIN_CONF:
                        continue

                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    area = (x2 - x1) * (y2 - y1)

                    if TARGET_TRACK_ID is None:
                        TARGET_TRACK_ID = track_id
                        miss_streak     = 0
                        stable_streak   = 0
                        print(f"[TRACK] Locked onto Track ID: {TARGET_TRACK_ID}")

                    if track_id == TARGET_TRACK_ID and area > best_area:
                        best_area  = area
                        target_box = (x1, y1, x2, y2)
                        box_conf   = conf

            if target_box is None and TARGET_TRACK_ID is not None:
                miss_streak   += 1
                stable_streak  = 0   # reset stability on any miss
                if miss_streak >= MAX_MISS_FRAMES:
                    print(f"[TRACK] Lost ID {TARGET_TRACK_ID} "
                          f"after {miss_streak} misses — reacquiring...")
                    TARGET_TRACK_ID = None
                    miss_streak     = 0
            else:
                miss_streak = 0
                if target_box is not None:
                    stable_streak += 1

        except Exception as e:
            print(f"[TRACK] Error: {e}")

        inference_times.append((time.perf_counter() - t_infer) * 1000)

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------
    W = camera.getWidth()
    H = camera.getHeight()

    # Jitter Fix 1: hold still during warmup window
    in_warmup = (infer_count <= WARMUP_FRAMES)

    if target_box is not None and not in_warmup:
        x1, y1, x2, y2 = target_box
        raw_error = ((x1 + x2) / 2 - W / 2) / (W / 2)

        # Jitter Fix 4: ignore tiny wobbles below gate threshold
        if abs(raw_error - last_error) > JITTER_GATE:
            # Jitter Fix 2: adaptive alpha — heavy filter until tracker is stable
            alpha = (ERROR_ALPHA_STABLE
                     if stable_streak >= STABLE_FRAMES
                     else ERROR_ALPHA_INIT)
            last_error = alpha * raw_error + (1.0 - alpha) * last_error

        left_speed, right_speed, direction = compute_speeds(last_error)

        if run_infer:
            area_frac = ((x2 - x1) * (y2 - y1)) / (W * H)
            phase = "STABLE" if stable_streak >= STABLE_FRAMES else f"INIT({stable_streak})"
            print(f"[FOUND/{phase}]  id={TARGET_TRACK_ID}  conf={box_conf:.2f}  "
                  f"error={last_error:+.3f}  area={area_frac:.3f}  → {direction}")

    elif in_warmup:
        # Warmup: stay still, let ByteTrack converge
        left_speed = right_speed = 0.0
        if run_infer and target_box is not None:
            print(f"[WARMUP {infer_count}/{WARMUP_FRAMES}]  "
                  f"id={TARGET_TRACK_ID}  conf={box_conf:.2f}  — holding...")

    else:
        # No detection: spin to search
        left_speed  =  SEARCH_SPEED
        right_speed = -SEARCH_SPEED
        last_error *= 0.90
        if run_infer:
            print(f"[SEARCH]  miss={miss_streak}  id={TARGET_TRACK_ID}")

    left_motor.setVelocity(left_speed)
    right_motor.setVelocity(right_speed)

    total_times.append((time.perf_counter() - t0) * 1000)