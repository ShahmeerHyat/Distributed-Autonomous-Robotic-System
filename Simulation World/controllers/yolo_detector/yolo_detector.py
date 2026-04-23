from controller import Robot    # type: ignore
import numpy as np
from ultralytics import YOLO
import time
from collections import defaultdict

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
TARGET_CLASS       = "sports ball"
CENTER_TOL         = 0.10
BASE_SPEED         = MAX_SPEED * 0.65
TURN_GAIN          = MAX_SPEED * 0.55
SEARCH_SPEED       = MAX_SPEED * 0.30
INFER_EVERY_N      = 1
YOLO_RESOLUTION    = 680
MAX_MISS_FRAMES    = 10
WARMUP_FRAMES      = 8
ERROR_ALPHA_INIT   = 0.10
ERROR_ALPHA_STABLE = 0.40
STABLE_FRAMES      = 12
MIN_CONF           = 0.40
JITTER_GATE        = 0.03

# ---------------------------
# ── METRICS STORE ──────────────────────────────────────────────────────
# Every metric is appended here during the loop.
# print_benchmark() reads from this dict — nothing is computed mid-loop.
# ---------------------------
M = {
    # Latency
    "inference_ms"        : [],   # ms per YOLO call
    "total_ms"            : [],   # ms per full robot step

    # Detection quality
    "conf_target"         : [],   # confidence of the locked target each frame
    "conf_rejected"       : [],   # confidences of target-class boxes rejected (< MIN_CONF)
    "first_detect_area"   : None, # area fraction at the very first detection above MIN_CONF
    "false_positive_count": 0,    # target-class boxes with a different track ID than locked

    # Class confusion — every class YOLO outputs (not just target)
    "class_counts"        : defaultdict(int),  # {class_name: n_detections}

    # Tracking stability
    "track_lifetimes"     : [],   # frames each track ID survived before being dropped
    "miss_streaks"        : [],   # length of every miss burst that ended (re-detect or lock-drop)
    "id_switches"         : 0,    # how many times TARGET_TRACK_ID changed to a new ID
    "reacq_frames"        : [],   # frames elapsed between losing a track and relocking

    # Spatial / control quality
    "error_history"       : [],   # smoothed last_error each frame (for convergence analysis)
    "area_history"        : [],   # area fraction each detected frame (proxy for distance)
    "fresh_box_frames"    : 0,    # frames where motors got a brand-new box
    "cached_box_frames"   : 0,    # frames where motors acted on a cached (stale) box
}

# Internal tracking helpers (not printed, just used to feed M)
_track_start_frame   = {}   # {track_id: frame_count when first seen}
_lost_frame          = None # frame_count when track was last dropped
_current_miss_len    = 0    # running miss burst length
session_start        = time.perf_counter()


# -----------------------------------------------------------------------
def print_benchmark():
    duration = time.perf_counter() - session_start
    n_total  = len(M["total_ms"])
    n_infer  = len(M["inference_ms"])

    if n_total == 0:
        print("[BENCH] No frames processed.")
        return

    sep  = "=" * 68
    sep2 = "-" * 68

    def ms_row(label, data):
        if not data:
            return f"  {label:<26} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8}"
        return (f"  {label:<26}"
                f" {min(data):>7.1f}ms"
                f" {max(data):>7.1f}ms"
                f" {sum(data)/len(data):>7.1f}ms"
                f" {np.percentile(data, 95):>7.1f}ms")

    print("\n" + sep)
    print(f"  BENCHMARK REPORT   —   model: yolov8s   |   {duration:.1f}s session")
    print(sep)

    # ── 1. SESSION OVERVIEW ────────────────────────────────────────────
    eff_fps    = n_total  / duration
    infer_fps  = n_infer  / duration
    fresh_pct  = (M["fresh_box_frames"] /
                  max(1, M["fresh_box_frames"] + M["cached_box_frames"])) * 100

    print(f"\n  {'SESSION OVERVIEW'}")
    print(sep2)
    print(f"  Total steps          : {n_total}")
    print(f"  Inference steps      : {n_infer}  (every {INFER_EVERY_N} frame(s))")
    print(f"  Effective FPS        : {eff_fps:.1f}")
    print(f"  Inference FPS        : {infer_fps:.1f}")
    print(f"  Fresh box frames     : {M['fresh_box_frames']}  ({fresh_pct:.1f}%)")
    print(f"  Cached box frames    : {M['cached_box_frames']}  ({100-fresh_pct:.1f}%)")

    # ── 2. LATENCY ────────────────────────────────────────────────────
    print(f"\n  {'LATENCY':<26} {'Min':>8} {'Max':>8} {'Avg':>8} {'p95':>8}")
    print(sep2)
    print(ms_row("Inference (YOLO call)",  M["inference_ms"]))
    print(ms_row("Total per step",         M["total_ms"]))

    # ── 3. DETECTION QUALITY ──────────────────────────────────────────
    print(f"\n  DETECTION QUALITY")
    print(sep2)

    if M["conf_target"]:
        avg_c = sum(M["conf_target"]) / len(M["conf_target"])
        print(f"  Target conf (accepted) : "
              f"min={min(M['conf_target']):.3f}  "
              f"max={max(M['conf_target']):.3f}  "
              f"avg={avg_c:.3f}")
    else:
        print(f"  Target conf (accepted) : no detections above MIN_CONF={MIN_CONF}")

    if M["conf_rejected"]:
        avg_r = sum(M["conf_rejected"]) / len(M["conf_rejected"])
        print(f"  Target conf (rejected) : "
              f"min={min(M['conf_rejected']):.3f}  "
              f"max={max(M['conf_rejected']):.3f}  "
              f"avg={avg_r:.3f}  "
              f"count={len(M['conf_rejected'])}")
    else:
        print(f"  Target conf (rejected) : none")

    if M["first_detect_area"] is not None:
        print(f"  First detection area   : {M['first_detect_area']:.4f} "
              f"  (≈ {M['first_detect_area']*100:.1f}% of frame — "
              f"{'close' if M['first_detect_area'] > 0.10 else 'far/small'})")
    else:
        print(f"  First detection area   : never detected")

    print(f"  False positives        : {M['false_positive_count']} "
          f"(target-class boxes with wrong track ID)")

    # ── 4. CLASS CONFUSION ────────────────────────────────────────────
    print(f"\n  CLASS CONFUSION  (all classes YOLO output this session)")
    print(sep2)
    if M["class_counts"]:
        sorted_classes = sorted(M["class_counts"].items(),
                                key=lambda x: x[1], reverse=True)
        for cls, cnt in sorted_classes:
            marker = " ← TARGET" if cls == TARGET_CLASS else ""
            print(f"  {cls:<30} {cnt:>6} detections{marker}")
    else:
        print("  No detections at all.")

    # ── 5. TRACKING STABILITY ─────────────────────────────────────────
    print(f"\n  TRACKING STABILITY")
    print(sep2)
    print(f"  Track ID switches      : {M['id_switches']}")

    if M["track_lifetimes"]:
        avg_life = sum(M["track_lifetimes"]) / len(M["track_lifetimes"])
        print(f"  Track lifetimes (frms) : "
              f"min={min(M['track_lifetimes'])}  "
              f"max={max(M['track_lifetimes'])}  "
              f"avg={avg_life:.1f}  "
              f"n={len(M['track_lifetimes'])}")
    else:
        print(f"  Track lifetimes        : no completed tracks")

    if M["miss_streaks"]:
        avg_miss = sum(M["miss_streaks"]) / len(M["miss_streaks"])
        streak_dist = defaultdict(int)
        for s in M["miss_streaks"]:
            bucket = f"{((s-1)//5)*5+1}-{((s-1)//5)*5+5}"
            streak_dist[bucket] += 1
        dist_str = "  ".join(f"{k}:{v}" for k, v in sorted(streak_dist.items()))
        print(f"  Miss streak lengths    : "
              f"min={min(M['miss_streaks'])}  "
              f"max={max(M['miss_streaks'])}  "
              f"avg={avg_miss:.1f}")
        print(f"  Miss streak histogram  : {dist_str}")
    else:
        print(f"  Miss streaks           : none")

    if M["reacq_frames"]:
        avg_reacq = sum(M["reacq_frames"]) / len(M["reacq_frames"])
        print(f"  Reacquisition (frames) : "
              f"min={min(M['reacq_frames'])}  "
              f"max={max(M['reacq_frames'])}  "
              f"avg={avg_reacq:.1f}")
    else:
        print(f"  Reacquisition          : no re-locks needed")

    # ── 6. SPATIAL / CONTROL QUALITY ─────────────────────────────────
    print(f"\n  SPATIAL / CONTROL QUALITY")
    print(sep2)

    if M["error_history"]:
        abs_err   = [abs(e) for e in M["error_history"]]
        centred   = sum(1 for e in abs_err if e < CENTER_TOL)
        centred_p = centred / len(abs_err) * 100
        # Rolling convergence: compare first-quarter avg error vs last-quarter
        q = max(1, len(abs_err) // 4)
        early_err = sum(abs_err[:q])  / q
        late_err  = sum(abs_err[-q:]) / q
        print(f"  Lateral error (abs)    : "
              f"min={min(abs_err):.3f}  "
              f"max={max(abs_err):.3f}  "
              f"avg={sum(abs_err)/len(abs_err):.3f}")
        print(f"  Frames centred         : {centred}/{len(abs_err)} ({centred_p:.1f}%)")
        print(f"  Error convergence      : "
              f"early avg={early_err:.3f}  →  late avg={late_err:.3f}  "
              f"({'improving ✓' if late_err < early_err else 'not converging ✗'})")
    else:
        print(f"  Lateral error          : no tracking data")

    if M["area_history"]:
        print(f"  Apparent area (proxy)  : "
              f"min={min(M['area_history']):.4f}  "
              f"max={max(M['area_history']):.4f}  "
              f"avg={sum(M['area_history'])/len(M['area_history']):.4f}")
        # Area growth rate (d_area/d_frame) — positive = approaching
        if len(M["area_history"]) > 1:
            diffs     = [M["area_history"][i+1] - M["area_history"][i]
                         for i in range(len(M["area_history"])-1)]
            avg_rate  = sum(diffs) / len(diffs)
            print(f"  Avg area growth/frame  : {avg_rate:+.6f} "
                  f"({'approaching' if avg_rate > 0 else 'retreating/static'})")
    else:
        print(f"  Apparent area          : no tracking data")

    print("\n" + sep + "\n")


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
    forward     = BASE_SPEED * (1.0 - 0.35 * abs(error))
    delta       = TURN_GAIN  * error
    left_speed  = np.clip(forward + delta, -MAX_SPEED, MAX_SPEED)
    right_speed = np.clip(forward - delta, -MAX_SPEED, MAX_SPEED)
    direction   = "CENTRE" if abs(error) < CENTER_TOL else ("RIGHT" if error > 0 else "LEFT")
    return left_speed, right_speed, direction


# ---------------------------
# State
# ---------------------------
TARGET_TRACK_ID  = None
miss_streak      = 0
stable_streak    = 0
last_error       = 0.0
infer_count      = 0
frame_count      = 0

print(f"[RUN] Tracking: '{TARGET_CLASS}'  (warming up for {WARMUP_FRAMES} frames...)")

# ---------------------------
# Main Loop
# ---------------------------
while robot.step(timestep) != -1:
    if frame_count == 500:
        break
    t0 = time.perf_counter()
    frame_count += 1
    run_infer    = (frame_count % INFER_EVERY_N == 0)

    frame = get_frame(camera)
    if frame is None:
        continue

    W = camera.getWidth()
    H = camera.getHeight()

    target_box    = None
    box_conf      = 0.0
    got_fresh_box = False

    # ------------------------------------------------------------------
    # YOLO + ByteTrack
    # ------------------------------------------------------------------
    if run_infer:
        infer_count += 1
        t_infer      = time.perf_counter()

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

            if boxes is not None:
                id_tensor = boxes.id  # may be None if tracker hasn't assigned IDs yet

                for i, box in enumerate(boxes):
                    cls_name = names[int(box.cls[0])]
                    conf     = float(box.conf[0])

                    # Track every class YOLO sees (class confusion table)
                    M["class_counts"][cls_name] += 1

                    if cls_name != TARGET_CLASS:
                        continue

                    # Collect rejected confidences before the gate
                    if conf < MIN_CONF:
                        M["conf_rejected"].append(conf)
                        continue

                    if id_tensor is None:
                        continue

                    track_id = int(id_tensor[i])
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    area     = (x2 - x1) * (y2 - y1)
                    area_frac = area / (W * H)

                    # First-ever detection area
                    if M["first_detect_area"] is None:
                        M["first_detect_area"] = area_frac
                        print(f"[METRIC] First detection at area={area_frac:.4f} "
                              f"({area_frac*100:.1f}% of frame)")

                    # Track lifetime bookkeeping
                    if track_id not in _track_start_frame:
                        _track_start_frame[track_id] = frame_count

                    # Lock logic
                    if TARGET_TRACK_ID is None:
                        TARGET_TRACK_ID = track_id
                        miss_streak     = 0
                        stable_streak   = 0
                        M["id_switches"] += 1
                        if _lost_frame is not None:
                            M["reacq_frames"].append(frame_count - _lost_frame)
                        print(f"[TRACK] Locked onto Track ID: {TARGET_TRACK_ID}")

                    # False positive: target-class box that isn't our target
                    if track_id != TARGET_TRACK_ID:
                        M["false_positive_count"] += 1

                    if track_id == TARGET_TRACK_ID and area > best_area:
                        best_area  = area
                        target_box = (x1, y1, x2, y2)
                        box_conf   = conf

            # Miss / reacquire bookkeeping
            if target_box is None and TARGET_TRACK_ID is not None:
                miss_streak   += 1
                stable_streak  = 0
                _current_miss_len = miss_streak

                if miss_streak >= MAX_MISS_FRAMES:
                    # Record lifetime of the dropped track
                    if TARGET_TRACK_ID in _track_start_frame:
                        lifetime = frame_count - _track_start_frame[TARGET_TRACK_ID]
                        M["track_lifetimes"].append(lifetime)

                    M["miss_streaks"].append(miss_streak)
                    _lost_frame = frame_count

                    print(f"[TRACK] Lost ID {TARGET_TRACK_ID} "
                          f"after {miss_streak} misses — reacquiring...")
                    TARGET_TRACK_ID = None
                    miss_streak     = 0
            else:
                if miss_streak > 0:
                    M["miss_streaks"].append(miss_streak)
                miss_streak = 0
                if target_box is not None:
                    stable_streak += 1
                    M["conf_target"].append(box_conf)
                    got_fresh_box = True

        except Exception as e:
            print(f"[TRACK] Error: {e}")

        M["inference_ms"].append((time.perf_counter() - t_infer) * 1000)

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------
    in_warmup = (infer_count <= WARMUP_FRAMES)

    if target_box is not None and not in_warmup:
        x1, y1, x2, y2 = target_box
        raw_error = ((x1 + x2) / 2 - W / 2) / (W / 2)

        if abs(raw_error - last_error) > JITTER_GATE:
            alpha      = (ERROR_ALPHA_STABLE
                          if stable_streak >= STABLE_FRAMES
                          else ERROR_ALPHA_INIT)
            last_error = alpha * raw_error + (1.0 - alpha) * last_error

        left_speed, right_speed, direction = compute_speeds(last_error)

        area_frac = ((x2 - x1) * (y2 - y1)) / (W * H)

        # Record spatial metrics
        M["error_history"].append(last_error)
        M["area_history"].append(area_frac)

        if got_fresh_box:
            M["fresh_box_frames"] += 1
        else:
            M["cached_box_frames"] += 1

        if run_infer:
            phase = "STABLE" if stable_streak >= STABLE_FRAMES else f"INIT({stable_streak})"
            print(f"[FOUND/{phase}]  id={TARGET_TRACK_ID}  conf={box_conf:.2f}  "
                  f"error={last_error:+.3f}  area={area_frac:.3f}  → {direction}")

    elif in_warmup:
        left_speed = right_speed = 0.0
        if run_infer and target_box is not None:
            print(f"[WARMUP {infer_count}/{WARMUP_FRAMES}]  "
                  f"id={TARGET_TRACK_ID}  conf={box_conf:.2f}  — holding...")

    else:
        left_speed  =  SEARCH_SPEED
        right_speed = -SEARCH_SPEED
        last_error *= 0.90
        M["cached_box_frames"] += 1
        if run_infer:
            print(f"[SEARCH]  miss={miss_streak}  id={TARGET_TRACK_ID}")

    left_motor.setVelocity(left_speed)
    right_motor.setVelocity(right_speed)

    M["total_ms"].append((time.perf_counter() - t0) * 1000)

print_benchmark()