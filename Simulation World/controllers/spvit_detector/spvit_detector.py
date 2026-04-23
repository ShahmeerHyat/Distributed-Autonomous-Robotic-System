from controller import Robot
import numpy as np
import torch
import time
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Import your custom distributed components
from distributedSystem.master import MasterOrchestrator

# ---------------------------
# Initialize Robot
# ---------------------------
robot = Robot()
timestep = int(robot.getBasicTimeStep())

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

# ---------------------------
# Load CLIP
# ---------------------------
model_path = r"../../../Clip Model"
clip       = CLIPModel.from_pretrained(model_path, local_files_only=True)
processor  = CLIPProcessor.from_pretrained(model_path, local_files_only=True)
clip.eval()

# ---------------------------
# CLIP-specific metadata
# ---------------------------
CLIP_META = {
    "embed_dim"     : 768,
    "num_heads"     : 12,
    "head_dim"      : 64,
    "mlp_hidden_dim": 3072,
    "seq_length"    : 50,   # 7x7 patches + 1 CLS token
}

# ---------------------------
# MasterOrchestrator
# ---------------------------
orch = MasterOrchestrator(
    expected_workers=["pc_gpu"],
    host="0.0.0.0",
    port=29500,
    model=clip.vision_model,   
    meta=CLIP_META,            
)

# ---------------------------
# Settings (Synchronized with clip_detector.py)
# ---------------------------
SEARCH_PROMPT        = "a round ball with black and white patches"
CENTER_TOL           = 0.03 
ROTATE_ONLY_TOL      = 0.40 
CONFIDENCE_THRESHOLD = 20.0
EMP_CENTER           = 3.02 
INFERENCE_EVERY_N    = 3

BASE_SPEED   = MAX_SPEED * 0.35
TURN_GAIN    = MAX_SPEED * 0.80
SEARCH_SPEED = MAX_SPEED * 0.30

GRID_SIZE = 7
_xs = torch.arange(GRID_SIZE).unsqueeze(0).repeat(GRID_SIZE, 1).flatten().float()

# ---------------------------
# Helper Functions
# ---------------------------
def get_frame_as_pil(cam):
    raw = cam.getImage()
    if not raw: return None
    w, h = cam.getWidth(), cam.getHeight()
    img = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 4))
    return Image.fromarray(img[:, :, [2, 1, 0]])

def compute_speeds(error):
    abs_err = abs(error)
    if abs_err < CENTER_TOL:
        return BASE_SPEED, BASE_SPEED, "STRICT_CENTER"
    if abs_err > ROTATE_ONLY_TOL:
        turn_speed = TURN_GAIN * (error / abs_err)
        return turn_speed, -turn_speed, "STATIONARY_TURN"
    
    forward_speed = BASE_SPEED * (1.0 - (abs_err / ROTATE_ONLY_TOL))
    turn_delta    = TURN_GAIN * error
    left_speed    = max(-MAX_SPEED, min(MAX_SPEED, forward_speed - turn_delta))
    right_speed   = max(-MAX_SPEED, min(MAX_SPEED, forward_speed + turn_delta))
    return left_speed, right_speed, "ARC"

# ---------------------------
# Distributed Logic
# ---------------------------
def _distributed_encoder_forward(current_state: torch.Tensor) -> torch.Tensor:
    """Manual block loop for CLIP Vision Transformer layers."""
    vision = clip.vision_model
    layers = vision.encoder.layers

    for i, block in enumerate(layers):
        raw_latency = {dev: 0.0 for dev in orch.all_devices}
        share_used  = {dev: 0.0 for dev in orch.all_devices}

        orch.arima.update_shares()

        # --- ATTENTION ---
        identity = current_state
        ln_x = block.layer_norm1(current_state)

        # Full QKV locally
        q = block.self_attn.q_proj(ln_x)
        k = block.self_attn.k_proj(ln_x)
        v = block.self_attn.v_proj(ln_x)

        def split_heads(x):
            B, S, D = x.shape
            H = CLIP_META["num_heads"]
            return x.view(B, S, H, -1).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        attn_parts = []

        for dev in orch.all_devices:
            h_range = orch.arima.get_indices(dev, CLIP_META["num_heads"])
            if len(h_range) == 0: continue

            q_s, k_s, v_s = q[:, h_range, :, :], k[:, h_range, :, :], v[:, h_range, :, :]

            if dev == "edge":
                t0 = time.time()
                scale = CLIP_META["head_dim"] ** -0.5
                probs = torch.nn.functional.softmax((q_s @ k_s.transpose(-2, -1)) * scale, dim=-1)
                attn_parts.append(probs @ v_s)
                raw_latency["edge"] += time.time() - t0
            else:
                res, lat = orch._dispatch_task(dev, "ATTN", i, (q_s, k_s, v_s), None, None)
                attn_parts.append(res)
                raw_latency[dev] += lat

        ctx = torch.cat(attn_parts, dim=1).transpose(1, 2).reshape(1, 50, 768)
        current_state = identity + block.self_attn.out_proj(ctx)

        # --- MLP ---
        identity = current_state
        ln_x_mlp = block.layer_norm2(current_state)
        
        mlp_parts = []
        for dev in orch.all_devices:
            n_range = orch.arima.get_indices(dev, CLIP_META["mlp_hidden_dim"])
            if len(n_range) == 0: continue

            if dev == "edge":
                t0 = time.time()
                w1 = block.mlp.fc1.weight[n_range.start:n_range.stop, :]
                b1 = block.mlp.fc1.bias[n_range.start:n_range.stop]
                w2 = block.mlp.fc2.weight[:, n_range.start:n_range.stop]
                # Note: CLIP uses GELU
                edge_mlp = torch.nn.functional.gelu(ln_x_mlp @ w1.t() + b1) @ w2.t()
                mlp_parts.append(edge_mlp)
                raw_latency["edge"] += time.time() - t0
            else:
                res, lat = orch._dispatch_task(dev, "MLP", i, ln_x_mlp, n_range.start, n_range.stop)
                mlp_parts.append(res)
                raw_latency[dev] += lat
        
        # Sum partial MLP results and add final bias (CLIP specific: fc2.bias)
        current_state = identity + torch.sum(torch.stack(mlp_parts), dim=0) + block.mlp.fc2.bias

        # ARIMA record keeping
        for dev in orch.all_devices:
            orch.arima.record_block_latency(dev, raw_latency[dev], orch.arima.current_shares[dev])

    return current_state

def clip_distributed_forward(pixel_values):
    vision = clip.vision_model
    hidden = vision.embeddings(pixel_values)
    hidden = vision.pre_layrnorm(hidden)
    hidden = _distributed_encoder_forward(hidden)
    hidden = vision.post_layernorm(hidden)
    return hidden

# ---------------------------
# Pre-compute text embedding
# ---------------------------
text_inputs = processor(text=[SEARCH_PROMPT], return_tensors="pt", padding=True)
with torch.no_grad():
    text_outputs = clip.text_model(**text_inputs)
    # text_embeds in CLIP is the projected pooler_output
    text_emb = clip.text_projection(text_outputs.pooler_output)
    text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

# ---------------------------
# Main Loop
# ---------------------------
last_logits, last_error, step_count = 0.0, 0.0, 0
logit_scale = clip.logit_scale.exp() # CLIP's internal temperature

print(f"[RUN] Distributed search for: '{SEARCH_PROMPT}'")

while robot.step(timestep) != -1:
    step_count += 1
    run_inference = (step_count % INFERENCE_EVERY_N == 0)

    if run_inference:
        image = get_frame_as_pil(camera)
        if image:
            img_inputs = processor(images=image, return_tensors="pt")
            pixel_values = img_inputs["pixel_values"]

            with torch.no_grad():
                # 1. Distributed ViT Back-bone
                last_hidden = clip_distributed_forward(pixel_values) 

                # 2. Projection & Normalization (Crucial for CLIP)
                # Global Embedding (CLS token at index 0)
                cls_token = last_hidden[:, 0, :]
                img_emb_global = clip.visual_projection(cls_token)
                img_emb_global = img_emb_global / img_emb_global.norm(dim=-1, keepdim=True)

                # 3. Calculate Logits (Confidence)
                last_logits = (img_emb_global @ text_emb.t()).item() * logit_scale.item()

                # 4. Spatial Heatmap (Patches 1 to end)
                patches = last_hidden[:, 1:, :]
                patch_projected = clip.visual_projection(patches)
                patch_projected = patch_projected / patch_projected.norm(dim=-1, keepdim=True)

                heatmap = torch.matmul(patch_projected[0], text_emb.t()).squeeze()
                weights = torch.softmax(heatmap / 0.1, dim=0)
                center_x = (weights * _xs).sum().item()
                last_error = (center_x - EMP_CENTER) / EMP_CENTER

            print(f"[FOUND] score={last_logits:.2f} error={last_error:+.2f}")

    # Control Logic
    if last_logits > CONFIDENCE_THRESHOLD:
        left_speed, right_speed, _ = compute_speeds(last_error)
    else:
        last_error, left_speed, right_speed = 0.0, -SEARCH_SPEED, SEARCH_SPEED
        if run_inference: print(f"[SEARCHING] score={last_logits:.2f}")

    left_motor.setVelocity(left_speed)
    right_motor.setVelocity(right_speed)