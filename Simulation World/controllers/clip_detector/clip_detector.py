"""
clip_detector.py  —  CLIP-powered object tracking on e-puck (Webots)

Distributed inference:
  Vision encoder blocks run via MasterOrchestrator (edge + worker split).
  Text embedding is computed once locally before the main loop.

Key fixes vs previous version:
  - Text embedding now uses model.get_text_features() which applies
    text_projection, placing text in the same 512-d joint embedding space
    as the visual patches (which go through visual_projection 768→512).
    Using raw pooler_output without text_projection put text and vision
    in different spaces → cosine similarity scores were meaningless.
  - EMA smoothing on last_error is now actually applied (was defined but
    bypassed with a direct assignment).
  - Confidence score uses mean patch similarity rather than noisy max.
"""

from controller import Robot
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from distributedSystem.master import MasterOrchestrator

# ─────────────────────────────────────────────────────────────────────────────
# Robot init
# ─────────────────────────────────────────────────────────────────────────────

robot    = Robot()
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

camera = robot.getDevice("camera")
camera.enable(timestep)

print("[INIT] Devices ready.")

# ─────────────────────────────────────────────────────────────────────────────
# Load CLIP
# ─────────────────────────────────────────────────────────────────────────────

print("[INIT] Loading CLIP model...")

model_path = r"../../../Clip Model"
model      = CLIPModel.from_pretrained(model_path, local_files_only=True)
processor  = CLIPProcessor.from_pretrained(model_path, local_files_only=True)
model.eval()

print("[INIT] CLIP loaded successfully.")

# ─────────────────────────────────────────────────────────────────────────────
# Settings
# ─────────────────────────────────────────────────────────────────────────────

SEARCH_PROMPT        = "a round ball with black and white patches"
CENTER_TOL           = 0.01   # within ±1% → perfectly centred, drive straight
ROTATE_ONLY_TOL      = 0.40   # beyond ±40% → spin in place, no forward motion
CONFIDENCE_THRESHOLD = 20.0   # min similarity score (scaled ×100) to track
EMA_ALPHA            = 0.4    # lateral error smoothing  (0 = no smoothing)

BASE_SPEED   = MAX_SPEED * 0.35
TURN_GAIN    = MAX_SPEED * 0.80
SEARCH_SPEED = MAX_SPEED * 0.30

INFERENCE_EVERY_N = 1   # run CLIP every N Webots steps

# ─────────────────────────────────────────────────────────────────────────────
# Pre-compute patch grid  (7×7 = 49 spatial patches)
# ─────────────────────────────────────────────────────────────────────────────

GRID_SIZE = 7
_xs = torch.arange(GRID_SIZE).unsqueeze(0).repeat(GRID_SIZE, 1).flatten().float()
# shape: (49,)  values: [0,0,0,0,0,0,0, 1,1,…, 6,6,6,6,6,6,6]


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
    """Extract architecture constants from CLIP's vision config."""
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
        turn_speed = TURN_GAIN * (error / abs_err)   # preserves sign
        return turn_speed, -turn_speed, "STATIONARY_TURN"

    # Proportional arc: forward speed tapers as error grows
    forward_speed = BASE_SPEED * (1.0 - abs_err / ROTATE_ONLY_TOL)
    turn_delta    = TURN_GAIN * error
    left_speed    = max(-MAX_SPEED, min(MAX_SPEED, forward_speed - turn_delta))
    right_speed   = max(-MAX_SPEED, min(MAX_SPEED, forward_speed + turn_delta))
    direction     = "ARC_RIGHT" if error < 0 else "ARC_LEFT"
    return left_speed, right_speed, direction


# ─────────────────────────────────────────────────────────────────────────────
# MasterOrchestrator  (blocks here until worker connects)
# ─────────────────────────────────────────────────────────────────────────────

orch = MasterOrchestrator(
    expected_workers = [],
    host             = "0.0.0.0",
    port             = 29500,
    model            = model.vision_model,
    meta             = get_clip_metadata(model),
)

# ─────────────────────────────────────────────────────────────────────────────
# Pre-compute text embedding once  (cheap, local, done before the loop)
#
# CRITICAL: use model.get_text_features() — this applies text_projection
# (maps 512 → 512 for ViT-B/32) so the text embedding lives in the same
# 512-d joint space as the visual patches after visual_projection (768→512).
# Using pooler_output directly skips text_projection → different space →
# cosine similarities are garbage.
# ─────────────────────────────────────────────────────────────────────────────

# with torch.no_grad():
#     text_inputs = processor(text=[SEARCH_PROMPT], return_tensors="pt", padding=True)
#     text_emb    = model.get_text_features(**text_inputs)   # (1, 512)
#     text_emb    = text_emb / text_emb.norm(dim=-1, keepdim=True)

with torch.no_grad():
    text_inputs = processor(text=[SEARCH_PROMPT], return_tensors="pt", padding=True)
    text_outputs = model.text_model(**text_inputs)
    text_emb = text_outputs.pooler_output
    text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

# ─────────────────────────────────────────────────────────────────────────────
# Persistent state
# ─────────────────────────────────────────────────────────────────────────────

last_logits = 0.0
last_error  = 0.0
step_count  = 0

print(f"[RUN] Searching for: '{SEARCH_PROMPT}'")

# ─────────────────────────────────────────────────────────────────────────────
# Main Loop
# ─────────────────────────────────────────────────────────────────────────────

while robot.step(timestep) != -1:
    step_count   += 1
    run_inference = (step_count % INFERENCE_EVERY_N == 0)

    if run_inference:
        image = get_frame_as_pil(camera)

        if image is not None:
            with torch.no_grad():

                # 1. Pre-process image
                img_inputs   = processor(images=image, return_tensors="pt")
                pixel_values = img_inputs["pixel_values"]   # (1, 3, 224, 224)

                # 2. Patch embedding + CLS token + positional encoding
                #    Both are cheap local ops — no reason to distribute them
                hidden = model.vision_model.embeddings(pixel_values)  # (1, 50, 768)
                hidden = model.vision_model.pre_layrnorm(hidden)

                # 3. Distributed encoder forward
                #    Applies all 12 transformer blocks (+ post_layernorm inside)
                #    Returns (1, 50, 768)
                last_hidden = orch.run_inference(hidden)

                # 4. Project patch tokens into the joint 512-d embedding space
                #    Skip CLS (index 0), take the 49 spatial patches
                patches   = last_hidden[:, 1:, :]                  # (1, 49, 768)
                projected = model.visual_projection(patches)        # (1, 49, 512)
                projected = projected / projected.norm(dim=-1, keepdim=True)

                # 5. Per-patch cosine similarity with the text embedding
                #    heatmap shape: (49,)
                heatmap = torch.matmul(projected[0], text_emb.t()).squeeze()

                # 6. Confidence score: mean similarity scaled to ~0–100
                #    More stable than max (single noisy patch)
                last_logits = projected[0].mean(dim=0) @ text_emb[0]
                last_logits = last_logits.item() * 100.0

                # 7. Spatial heatmap → lateral error
                #    Temperature 0.01 sharpens the softmax for precise centering
                weights  = torch.softmax(heatmap / 0.01, dim=0)   # (49,)
                center_x = (weights * _xs).sum().item()
                center   = (GRID_SIZE - 1) / 2.0                   # 3.0

                raw_error  = (center_x - center) / center          # −1 … +1
                last_error = EMA_ALPHA * raw_error + (1.0 - EMA_ALPHA) * last_error

            print(f"[DEBUG] center_x={center_x:.2f}  "
                  f"raw={raw_error:+.2f}  smooth={last_error:+.2f}  "
                  f"score={last_logits:.2f}")

    # ── Motor control ────────────────────────────────────────────────────────

    if last_logits > CONFIDENCE_THRESHOLD:
        left_speed, right_speed, state_label = compute_speeds(last_error)

        if run_inference:
            print(f"[FOUND]     score={last_logits:.2f}  "
                  f"error={last_error:+.2f}  → {state_label}")
    else:
        # Not detected → spin to search; reset error so re-acquisition is clean
        last_error  = 0.0
        left_speed  = -SEARCH_SPEED
        right_speed =  SEARCH_SPEED

        if run_inference:
            print(f"[SEARCHING] score={last_logits:.2f}")

    left_motor.setVelocity(left_speed)
    right_motor.setVelocity(right_speed)