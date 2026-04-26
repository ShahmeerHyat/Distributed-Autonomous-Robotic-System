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
SEARCH_SPEED = MAX_SPEED * 0.60

INFERENCE_EVERY_N = 1   # run CLIP every N Webots steps

# ─────────────────────────────────────────────────────────────────────────────
# Pre-compute patch grid  (7×7 = 49 spatial patches)
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

with torch.no_grad():
    text_inputs = processor(text=[SEARCH_PROMPT], return_tensors="pt", padding=True)
    # Extract the hidden state of the EOT token
    text_outputs = model.text_model(**text_inputs)
    pooled_output = text_outputs.pooler_output 
    
    # CRITICAL: Map the 768-d text vector into the 512-d joint space
    text_emb = model.text_projection(pooled_output) 
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
                img_inputs   = processor(images=image, return_tensors="pt")
                pixel_values = img_inputs["pixel_values"]

                hidden = model.vision_model.embeddings(pixel_values)
                hidden = model.vision_model.pre_layrnorm(hidden)

                # 3. Distributed encoder forward (Returns raw last_hidden_state)
                last_hidden = orch.run_inference(hidden)

                # 4. Official Pooling & Post-Norm (For the Global Score/Confidence)
                cls_token = last_hidden[:, 0, :]
                pooled_output = model.vision_model.post_layernorm(cls_token) # Norm ONLY the CLS
                image_embeds = model.visual_projection(pooled_output)
                image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

                # 5. Patch Processing (For the Heatmap/Centering)
                # Official code doesn't explicitly norm patches for projection, but we do for heatmaps
                patches = last_hidden[:, 1:, :] 
                # Note: Usually, post_layernorm is applied to patches too if you want patch features
                patches = model.vision_model.post_layernorm(patches) 
                projected_patches = model.visual_projection(patches)
                projected_patches = projected_patches / projected_patches.norm(dim=-1, keepdim=True)

                # 6. Similarity & Logit Scale
                heatmap = torch.matmul(projected_patches[0], text_emb.t()).squeeze()

                # Use the official logit scale instead of hardcoded 100
                with torch.no_grad():
                    scale = model.logit_scale.exp()
                    
                # Use the pooled global embedding for the confidence score (matches official CLIP output)
                last_logits = (image_embeds @ text_emb.t()).item() * scale.item()
                
                weights  = torch.softmax(heatmap, dim=0)
                center_x = (weights * _xs).sum().item()
                center   = (GRID_SIZE - 1) / 2.0                   

                raw_error  = (center_x - center) / center
                last_error = EMA_ALPHA * raw_error + (1.0 - EMA_ALPHA) * last_error
                
                top_val, top_idx = torch.topk(heatmap, 1)

    # ── Motor control ────────────────────────────────────────────────────────

    if last_logits > CONFIDENCE_THRESHOLD:
        left_speed, right_speed, state_label = compute_speeds(last_error)

        # if run_inference:
        #     print(f"[FOUND]     score={last_logits:.2f}  "
        #           f"error={last_error:+.2f}  → {state_label}")
    else:
        # Not detected → spin to search; reset error so re-acquisition is clean
        last_error  = 0.0
        left_speed  = SEARCH_SPEED
        right_speed = -SEARCH_SPEED

        # if run_inference:
        #     print(f"[SEARCHING] score={last_logits:.2f}")

    left_motor.setVelocity(left_speed)
    right_motor.setVelocity(right_speed)