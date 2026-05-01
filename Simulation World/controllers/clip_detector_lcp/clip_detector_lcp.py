"""
clip_detector_lcp.py  —  LCP CLIP-powered object tracking on e-puck (Webots)

Loosely Coupled Protocol version:
  - The MasterOrchestrator starts in edge-only mode immediately.
    No workers need to be connected before the simulation begins.
  - Workers can join (or rejoin) at any point during the simulation by
    running worker.py --master-host <this machine's IP>.
  - The ARIMA load balancer picks up each worker after its preflight RTT
    measurement and begins distributing heads/neurons on the next inference call.
"""

from controller import Supervisor
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from distributedSystem.master import MasterOrchestrator
from concurrent.futures import ThreadPoolExecutor
import os
# ─────────────────────────────────────────────────────────────────────────────
# Robot init
# ─────────────────────────────────────────────────────────────────────────────

robot    = Supervisor()
timestep = int(robot.getBasicTimeStep())

print("[INIT] Booting robot...")
for _ in range(5):
    robot.step(timestep)

# ─────────────────────────────────────────────────────────────────────────────
# Devices
# ─────────────────────────────────────────────────────────────────────────────

left_motor  = robot.getDevice("left wheel motor")
right_motor = robot.getDevice("right wheel motor")
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))

MAX_SPEED = 6.28

robot_node = robot.getSelf()
ball_node  = robot.getFromDef("TENNIS_BALL")

_robot_init_t = list(robot_node.getField("translation").getSFVec3f())
_robot_init_r = list(robot_node.getField("rotation").getSFRotation())
_ball_init_t  = list(ball_node.getField("translation").getSFVec3f())
_ball_init_r  = list(ball_node.getField("rotation").getSFRotation())

camera = robot.getDevice("camera")
camera.enable(timestep)

RESET_EVERY_N_FRAMES = 800  


def reset_world():
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)

    # Signal ball controller to pause
    open("reset_flag.txt", "w").close()

    robot_node.getField("translation").setSFVec3f(_robot_init_t)
    robot_node.getField("rotation").setSFRotation(_robot_init_r)
    robot_node.resetPhysics()

    ball_node.getField("translation").setSFVec3f(_ball_init_t)
    ball_node.getField("rotation").setSFRotation(_ball_init_r)
    ball_node.resetPhysics()

    for _ in range(5):
        robot.step(timestep)

    # Clear the flag — ball controller resumes
    if os.path.exists("reset_flag.txt"):
        os.remove("reset_flag.txt")

    print("[RESET] World reset complete.")

print("[INIT] Devices ready.")

# ─────────────────────────────────────────────────────────────────────────────
# Load CLIP
# ─────────────────────────────────────────────────────────────────────────────

print("[INIT] Loading CLIP model...")

model_path = r"../../../Clip Model"
model      = CLIPModel.from_pretrained(model_path, local_files_only=True)
processor  = CLIPProcessor.from_pretrained(model_path, local_files_only=True)
model.eval()

print("[INIT] CLIP loaded.")

# ─────────────────────────────────────────────────────────────────────────────
# Settings
# ─────────────────────────────────────────────────────────────────────────────

SEARCH_PROMPT        = "a round ball with black and white patches"
CENTER_TOL           = 0.01
ROTATE_ONLY_TOL      = 0.40
CONFIDENCE_THRESHOLD = 20.0
EMA_ALPHA            = 0.4

BASE_SPEED   = MAX_SPEED
TURN_GAIN    = MAX_SPEED * 0.80
SEARCH_SPEED = MAX_SPEED * 0.60

INFERENCE_EVERY_N = 5

# ─────────────────────────────────────────────────────────────────────────────
# Pre-compute patch grid  (7×7 = 49 spatial patches for ViT-B/32)
# ─────────────────────────────────────────────────────────────────────────────

GRID_SIZE = 7
_xs = torch.arange(GRID_SIZE).unsqueeze(0).repeat(GRID_SIZE, 1).flatten().float()

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_frame_as_pil(cam):
    raw = cam.getImage()
    if not raw:
        return None
    w   = cam.getWidth()
    h   = cam.getHeight()
    img = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 4))
    return Image.fromarray(img[:, :, [2, 1, 0]])   # BGRA → RGB


def get_clip_metadata(clip_model) -> dict:
    cfg = clip_model.vision_model.config
    return {
        "num_heads":      cfg.num_attention_heads,
        "embed_dim":      cfg.hidden_size,
        "head_dim":       cfg.hidden_size // cfg.num_attention_heads,
        "mlp_hidden_dim": cfg.intermediate_size,
        "seq_length":     50,   # 49 patches + 1 CLS for ViT-B/32
    }


def compute_speeds(error: float):
    """
    Three-zone steering controller.

    STRICT_CENTER   |error| < CENTER_TOL        → drive straight
    ARC             CENTER_TOL ≤ |e| ≤ TOL      → proportional arc
    STATIONARY_TURN |error| > ROTATE_ONLY_TOL   → spin in place
    """
    abs_err = abs(error)

    if abs_err < CENTER_TOL:
        return BASE_SPEED, BASE_SPEED, "STRICT_CENTER"

    if abs_err > ROTATE_ONLY_TOL:
        turn_speed = TURN_GAIN * (error / abs_err)
        return turn_speed, -turn_speed, "STATIONARY_TURN"

    forward_speed = BASE_SPEED * (1.0 - abs_err / ROTATE_ONLY_TOL)
    turn_delta    = TURN_GAIN * error
    left_speed    = max(-MAX_SPEED, min(MAX_SPEED, forward_speed - turn_delta))
    right_speed   = max(-MAX_SPEED, min(MAX_SPEED, forward_speed + turn_delta))
    direction     = "ARC_RIGHT" if error < 0 else "ARC_LEFT"
    return left_speed, right_speed, direction


# ─────────────────────────────────────────────────────────────────────────────
# LCP MasterOrchestrator
#   Starts in edge-only mode. Workers connect at any time via TCP port 29500.
# ─────────────────────────────────────────────────────────────────────────────

orch = MasterOrchestrator(
    host  = "0.0.0.0",
    port  = 29500,
    model = model.vision_model,
    meta  = get_clip_metadata(model),
)

# ─────────────────────────────────────────────────────────────────────────────
# Pre-compute text embedding (once, before the loop)
# ─────────────────────────────────────────────────────────────────────────────

with torch.no_grad():
    text_inputs   = processor(text=[SEARCH_PROMPT], return_tensors="pt", padding=True)
    text_outputs  = model.text_model(**text_inputs)
    pooled_output = text_outputs.pooler_output
    text_emb      = model.text_projection(pooled_output)
    text_emb      = text_emb / text_emb.norm(dim=-1, keepdim=True)

# ─────────────────────────────────────────────────────────────────────────────
# Persistent state
# ─────────────────────────────────────────────────────────────────────────────

last_logits    = 0.0
last_error     = 0.0
step_count     = 0
_pending_future = None
_infer_pool    = ThreadPoolExecutor(max_workers=1)

print(f"[RUN] Searching for: '{SEARCH_PROMPT}'")
print("[RUN] Connect workers at any time: "
      "python worker.py --name <id> --master-host <this IP>")

# ─────────────────────────────────────────────────────────────────────────────
# Background inference  — runs in a thread so robot.step() never blocks
# ─────────────────────────────────────────────────────────────────────────────

def _clip_inference(image):
    """Returns (logits, raw_error). Called from the inference thread pool."""
    with torch.no_grad():
        img_inputs   = processor(images=image, return_tensors="pt")
        pixel_values = img_inputs["pixel_values"]

        hidden      = model.vision_model.embeddings(pixel_values)
        hidden      = model.vision_model.pre_layrnorm(hidden)
        last_hidden = orch.run_inference(hidden)

        cls_token    = last_hidden[:, 0, :]
        pooled       = model.vision_model.post_layernorm(cls_token)
        image_embeds = model.visual_projection(pooled)
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

        patches           = last_hidden[:, 1:, :]
        patches           = model.vision_model.post_layernorm(patches)
        projected_patches = model.visual_projection(patches)
        projected_patches = projected_patches / projected_patches.norm(dim=-1, keepdim=True)

        heatmap  = torch.matmul(projected_patches[0], text_emb.t()).squeeze()
        scale    = model.logit_scale.exp()
        logits   = (image_embeds @ text_emb.t()).item() * scale.item()
        weights  = torch.softmax(heatmap / 0.1, dim=0)
        center_x = (weights * _xs).sum().item()
        raw_error = (center_x - 3) / 3
        return logits, raw_error

# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

while robot.step(timestep) != -1:
    step_count += 1

    # ── Reset trigger ────────────────────────────────────────────────────
    if step_count % RESET_EVERY_N_FRAMES == 0:
        print(f"[RESET] Frame {step_count} — resetting world")
        reset_world()
        _pending_future = None   # stale result after reset is irrelevant
        last_logits = 0.0
        last_error  = 0.0
        continue

    # ── Collect completed inference result ───────────────────────────────
    if _pending_future is not None and _pending_future.done():
        try:
            logits, raw_error = _pending_future.result()
            last_logits = logits
            last_error  = EMA_ALPHA * raw_error + (1.0 - EMA_ALPHA) * last_error
        except Exception as e:
            print(f"[WARN] Inference error: {e}")
        _pending_future = None

    # ── Submit new inference if none is running ───────────────────────────
    if step_count % INFERENCE_EVERY_N == 0 and _pending_future is None:
        image = get_frame_as_pil(camera)
        if image is not None:
            _pending_future = _infer_pool.submit(_clip_inference, image)

    # ── Motor control ─────────────────────────────────────────────────────
    if last_logits > CONFIDENCE_THRESHOLD:
        left_speed, right_speed, state_label = compute_speeds(last_error)
    else:
        last_error  = 0.0
        left_speed  = SEARCH_SPEED
        right_speed = -SEARCH_SPEED

    left_motor.setVelocity(left_speed)
    right_motor.setVelocity(right_speed)
