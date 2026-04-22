from controller import Robot
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# ---------------------------
# Initialize Robot
# ---------------------------
robot = Robot()
timestep = int(robot.getBasicTimeStep())

print("[INIT] Booting robot...")

for _ in range(5):
    robot.step(timestep)

# ---------------------------
# Devices
# ---------------------------
left_motor  = robot.getDevice("left wheel motor")
right_motor = robot.getDevice("right wheel motor")

left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))

MAX_SPEED = 6.28

camera = robot.getDevice("camera")
camera.enable(timestep)

print("[INIT] Devices ready.")

# ---------------------------
# Load CLIP Model
# ---------------------------
print("[INIT] Loading CLIP model...")

model_path = r"../../../Clip Model"
model      = CLIPModel.from_pretrained(model_path, local_files_only=True)
processor  = CLIPProcessor.from_pretrained(model_path, local_files_only=True)
model.eval()

print("[INIT] CLIP loaded successfully.")

# ---------------------------
# Settings
# ---------------------------
SEARCH_PROMPT        = "a round ball with black and white patches"
CENTER_TOL           = 0.05   # ±15% of frame width → considered centred
CONFIDENCE_THRESHOLD = 20.0
EMA_ALPHA            = 0.4

# Drive constants
BASE_SPEED  = MAX_SPEED * 0.40   # forward creep while centering
TURN_GAIN   = MAX_SPEED * 0.50   # steering authority
SEARCH_SPEED = MAX_SPEED * 0.30  # spin speed when object not found

# Frame-skip: run CLIP every N steps
INFERENCE_EVERY_N = 3

# ---------------------------
# Pre-compute patch grid once
# ---------------------------
GRID_SIZE = 7
_xs = torch.arange(GRID_SIZE).unsqueeze(0).repeat(GRID_SIZE, 1).flatten().float()


def get_frame_as_pil(cam):
    raw = cam.getImage()
    if not raw:
        return None
    w   = cam.getWidth()
    h   = cam.getHeight()
    img = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 4))
    return Image.fromarray(img[:, :, [2, 1, 0]])   # BGRA → RGB


def compute_speeds(error):
    """
    Given a normalised lateral error (−1 … +1), return motor speeds
    that keep the detected object centred in the frame.

    States
    ------
    CENTRED  : error within tolerance → drive straight
    LEFT     : object is left of centre → turn left
    RIGHT    : object is right of centre → turn right
    """
    if abs(error) < CENTER_TOL:
        # Already centred — drive straight
        return BASE_SPEED, BASE_SPEED, "CENTRED"

    # Reduce forward speed proportionally to how far off-centre the object is,
    # and add a turn delta to steer toward it.
    forward_speed = BASE_SPEED * (1.0 - 0.4 * abs(error))
    turn_delta    = TURN_GAIN  * error

    left_speed  = max(-MAX_SPEED, min(MAX_SPEED, forward_speed + turn_delta))
    right_speed = max(-MAX_SPEED, min(MAX_SPEED, forward_speed - turn_delta))

    direction = "RIGHT" if error > 0 else "LEFT"
    return left_speed, right_speed, direction


# ---------------------------
# Persistent state
# ---------------------------
last_logits = 0.0
last_error  = 0.0
step_count  = 0

print(f"[RUN] Searching for: '{SEARCH_PROMPT}'")

# ---------------------------
# Main Loop
# ---------------------------
while robot.step(timestep) != -1:

    step_count    += 1
    run_inference  = (step_count % INFERENCE_EVERY_N == 0)

    # ---------------------------
    # CLIP inference (frame-skipped)
    # ---------------------------
    if run_inference:
        image = get_frame_as_pil(camera)

        if image is not None:
            inputs = processor(
                text=[SEARCH_PROMPT],
                images=image,
                return_tensors="pt",
                padding=True
            )

            with torch.no_grad():
                outputs = model(**inputs)

            last_logits = outputs.logits_per_image.item()

            # ── Spatial heatmap → lateral error ──────────────────────
            patches   = outputs.vision_model_output.last_hidden_state[:, 1:, :]
            projected = model.visual_projection(patches)
            projected = projected / projected.norm(dim=-1, keepdim=True)

            text_emb  = outputs.text_embeds
            text_emb  = text_emb / text_emb.norm(dim=-1, keepdim=True)

            heatmap  = torch.matmul(projected[0], text_emb.t()).squeeze()
            weights  = torch.softmax(heatmap, dim=0)

            center_x  = (weights * _xs).sum().item()
            center    = (GRID_SIZE - 1) / 2
            raw_error = (center_x - center) / center        # −1 (left) … +1 (right)

            # EMA smoothing to reduce frame-to-frame jitter
            last_error = EMA_ALPHA * raw_error + (1.0 - EMA_ALPHA) * last_error

    # ---------------------------
    # Motor control
    # ---------------------------
    if last_logits > CONFIDENCE_THRESHOLD:
        left_speed, right_speed, state_label = compute_speeds(last_error)

        if run_inference:
            print(f"[FOUND]     score={last_logits:.2f}  "
                  f"error={last_error:+.2f}  → {state_label}")
    else:
        # Object not detected → spin in place to search
        last_error  = 0.0          # clear stale error before re-acquisition
        left_speed  =  SEARCH_SPEED
        right_speed = -SEARCH_SPEED

        if run_inference:
            print(f"[SEARCHING] score={last_logits:.2f}")

    left_motor.setVelocity(left_speed)
    right_motor.setVelocity(right_speed)