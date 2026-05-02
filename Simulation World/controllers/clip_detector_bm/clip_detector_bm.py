"""
clip_detector_bm.py  —  CLIP-powered object tracking on e-puck (Webots)
                         with full benchmark instrumentation.

Benchmark call order
────────────────────
1. Load CLIP model
2. Create MasterOrchestrator (blocks until worker connects + runs preflight)
3. bench.measure_edge_baseline()   — edge-only latency, NO worker dispatch
4. bench.measure_comm_overhead()   — serialised bytes per frame
5. Pre-compute text embedding
6. Main loop: tick_frame_start / record_inference / tick_frame_end
7. After BENCHMARK_AFTER_N_FRAMES: bench.report()
"""

from controller import Robot
import numpy as np
import time
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from distributedSystem_.master import MasterOrchestrator
from distributedSystem_.benchmarks import BenchmarkCollector

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
CENTER_TOL           = 0.01
ROTATE_ONLY_TOL      = 0.40
CONFIDENCE_THRESHOLD = 20
EMA_ALPHA            = 0.4

BASE_SPEED   = MAX_SPEED
TURN_GAIN    = MAX_SPEED * 0.80
SEARCH_SPEED = MAX_SPEED * 0.60

INFERENCE_EVERY_N        = 5
BENCHMARK_AFTER_N_FRAMES = 500

# ─────────────────────────────────────────────────────────────────────────────
# Patch grid
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
        "seq_length":     50,
    }

def compute_speeds(error: float):
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
# MasterOrchestrator  (blocks until worker connects + runs preflight)
# ─────────────────────────────────────────────────────────────────────────────

orch = MasterOrchestrator(
    expected_workers = ["pc_gpu"],
    host             = "0.0.0.0",
    port             = 29500,
    model            = model.vision_model,
    meta             = get_clip_metadata(model),
)

# ─────────────────────────────────────────────────────────────────────────────
# BenchmarkCollector — created right after orch so it can access arima state
# ─────────────────────────────────────────────────────────────────────────────

bench = BenchmarkCollector(
    orch                 = orch,
    vision_model         = model.vision_model,
    clip_model           = model,
    confidence_threshold = CONFIDENCE_THRESHOLD,
)

# M2: edge-only baseline — MUST run before any inference that dispatches
#     to the worker, so timing reflects pure CPU compute.
bench.measure_edge_baseline(n_runs=6)

# M4: communication overhead — measures serialised tensor sizes per frame.
bench.measure_comm_overhead()

# ─────────────────────────────────────────────────────────────────────────────
# Text embedding  (computed once locally before the loop)
# ─────────────────────────────────────────────────────────────────────────────

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

    bench.tick_frame_start()                           # M8: FPS timing

    run_inference = (step_count % INFERENCE_EVERY_N == 0)

    if run_inference:
        image = get_frame_as_pil(camera)

        if image is not None:
            with torch.no_grad():

                # Patch embedding (local, cheap)
                img_inputs   = processor(images=image, return_tensors="pt")
                pixel_values = img_inputs["pixel_values"]
                hidden = model.vision_model.embeddings(pixel_values)   # (1, 50, 768)
                hidden = model.vision_model.pre_layrnorm(hidden)

                # Distributed encoder forward — timed for M3
                t0          = time.perf_counter()
                last_hidden = orch.run_inference(hidden)
                inf_lat     = time.perf_counter() - t0

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

                weights  = torch.softmax(heatmap / 0.1, dim=0)
                center_x = (weights * _xs).sum().item()
                center   = (GRID_SIZE - 1) / 2.0

                raw_error  = (center_x - 3) / 3
                last_error = EMA_ALPHA * raw_error + (1.0 - EMA_ALPHA) * last_error

                top_val, top_idx = torch.topk(heatmap, 1)

            # Push to benchmark collector
            # ball_visible: use current confidence as proxy (no ground truth)
            bench.record_inference(
                latency      = inf_lat,
                last_logits  = last_logits,
                ball_visible = (last_logits > CONFIDENCE_THRESHOLD),
            )

            # print(f"[DEBUG] cx={center_x:.2f}  raw={raw_error:+.2f}  "
            #       f"smooth={last_error:+.2f}  score={last_logits:.2f}")

    # ── Motor control ─────────────────────────────────────────────────────────

    if last_logits > CONFIDENCE_THRESHOLD:
        left_speed, right_speed, state_label = compute_speeds(last_error)
        if run_inference:
            print(f"[FOUND]     score={last_logits:.2f}  "
                  f"error={last_error:+.2f}  → {state_label}")
    else:
        last_error  = 0.0
        left_speed  =  SEARCH_SPEED
        right_speed = -SEARCH_SPEED
        if run_inference:
            print(f"[SEARCHING] score={last_logits:.2f}")

    left_motor.setVelocity(left_speed)
    right_motor.setVelocity(right_speed)

    bench.tick_frame_end()                             # M8: FPS timing

    # ── Benchmark report trigger ───────────────────────────────────────────

    if step_count == BENCHMARK_AFTER_N_FRAMES:
        # M11: run circuit breaker test just before printing
        dummy = torch.randn(1, get_clip_metadata(model)["seq_length"],
                            get_clip_metadata(model)["embed_dim"])
        bench.simulate_worker_failure("pc_gpu", dummy)

        bench.report()
        break