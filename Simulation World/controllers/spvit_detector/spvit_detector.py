from controller import Robot
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

from distributedSystem.master import MasterOrchestrator
from distributedSystem.shared_utils import get_head_weights, merge_n_projections, ready_for_math

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
# (overrides vit_b_16 defaults in MasterOrchestrator)
# ---------------------------
CLIP_META = {
    "embed_dim"     : 768,
    "num_heads"     : 12,
    "head_dim"      : 64,
    "mlp_hidden_dim": 3072,
    "seq_length"    : 50,   # 7×7 patches + 1 CLS token
}

# ---------------------------
# MasterOrchestrator
# (pass clip.vision_model so it uses CLIP's encoder blocks)
# ---------------------------
orch = MasterOrchestrator(
    expected_workers=["pc_gpu"],
    host="0.0.0.0",
    port=29500,
    model=clip.vision_model,   # <-- CLIP vision encoder, not vit_b_16
    meta=CLIP_META,            # <-- CLIP-specific metadata for ARIMA splitting
)

# ---------------------------
# Settings
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


def get_frame_as_pil(cam):
    raw = cam.getImage()
    if not raw:
        return None
    w   = cam.getWidth()
    h   = cam.getHeight()
    img = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 4))
    return Image.fromarray(img[:, :, [2, 1, 0]])


def clip_distributed_forward(pixel_values: torch.Tensor) -> torch.Tensor:
    """
    Replaces CLIPModel's black-box vision forward pass with a manual
    block loop that offloads head/neuron slices via MasterOrchestrator.
    Returns last_hidden_state: (1, 50, 768)
    """
    vision = clip.vision_model

    # --- Patch embedding + CLS + positional encoding (always local, cheap) ---
    hidden = vision.embeddings(pixel_values)   # (1, 50, 768)
    hidden = vision.pre_layrnorm(hidden)

    # --- Block loop (this is what gets distributed) ---
    # We reuse orch.run_inference but we need to swap block attribute access.
    # Instead of calling run_inference directly (which uses torchvision attr names),
    # we run an adapted loop here using CLIP's attr names.
    hidden = _distributed_encoder_forward(hidden)

    # --- Post layernorm (local, cheap) ---
    hidden = vision.post_layernorm(hidden)
    return hidden


def _distributed_encoder_forward(current_state: torch.Tensor) -> torch.Tensor:
    """
    Block-by-block forward using CLIP attribute names,
    with ARIMA-driven head/neuron splitting identical to master.py.
    """
    vision  = clip.vision_model
    blocks  = vision.encoder.layers

    for i, block in enumerate(blocks):

        raw_latency = {dev: 0.0 for dev in orch.all_devices}
        share_used  = {dev: 0.0 for dev in orch.all_devices}

        probing_workers = {
            w for w in orch.expected_workers
            if w in orch.arima.breakers and orch.arima.breakers[w].is_half_open
        }

        orch.arima.update_shares()

        # ── ATTENTION ────────────────────────────────────────────────────
        identity = current_state
        ln_x     = block.layer_norm1(current_state)   # CLIP name

        # Dispatch workers
        attn_futures = {}
        for w_name in orch.expected_workers:
            h_range = orch.arima.get_indices(w_name, CLIP_META["num_heads"])
            if len(h_range) > 0:
                attn_futures[w_name] = orch.executor.submit(
                    orch._dispatch_task, w_name, "ATTN", i,
                    ln_x, h_range.start, h_range.stop,
                )
                share_used[w_name] += orch.arima.current_shares[w_name]

        # Local edge slice
        edge_h = orch.arima.get_indices("edge", CLIP_META["num_heads"])
        share_used["edge"] += orch.arima.current_shares["edge"]

        t0 = time.time()
        if len(edge_h) > 0:
            # CLIP attr name for in_proj_weight
            edge_attn_w = get_head_weights(
                block.self_attn.in_proj_weight, edge_h,
                CLIP_META["embed_dim"], CLIP_META["head_dim"],
            )
            edge_qkv = ln_x @ edge_attn_w.t()
        else:
            edge_qkv = torch.tensor([])
        raw_latency["edge"] += time.time() - t0

        # Collect & merge
        qkv_parts = []
        for dev in orch.all_devices:
            if dev == "edge":
                qkv_parts.append(edge_qkv)
            elif dev in attn_futures:
                res, lat = attn_futures[dev].result()
                if res is None:
                    res = torch.zeros_like(edge_qkv) if edge_qkv.numel() > 0 else torch.tensor([])
                qkv_parts.append(res)
                raw_latency[dev] += lat

        merged_qkv  = merge_n_projections(qkv_parts)
        merged_qkv += block.self_attn.in_proj_bias   # CLIP attr name
        q, k, v     = torch.chunk(merged_qkv, 3, dim=-1)
        q, k, v     = (ready_for_math(t, CLIP_META) for t in (q, k, v))

        scale      = CLIP_META["head_dim"] ** -0.5
        attn_probs = torch.nn.functional.softmax(
            (q @ k.transpose(-2, -1)) * scale, dim=-1
        )
        ctx        = (attn_probs @ v).transpose(1, 2).reshape(
            1, CLIP_META["seq_length"], CLIP_META["embed_dim"]
        )
        attn_out      = block.self_attn.out_proj(ctx)   # CLIP attr name
        current_state = identity + attn_out

        # ── MLP ───────────────────────────────────────────────────────────
        identity  = current_state
        ln_x_mlp  = block.layer_norm2(current_state)   # CLIP attr name

        mlp_futures = {}
        for w_name in orch.expected_workers:
            n_range = orch.arima.get_indices(w_name, CLIP_META["mlp_hidden_dim"])
            if len(n_range) > 0:
                mlp_futures[w_name] = orch.executor.submit(
                    orch._dispatch_task, w_name, "MLP", i,
                    ln_x_mlp, n_range.start, n_range.stop,
                )
                share_used[w_name] = (
                    share_used[w_name] + orch.arima.current_shares[w_name]
                ) / 2.0

        edge_n = orch.arima.get_indices("edge", CLIP_META["mlp_hidden_dim"])
        t0     = time.time()
        if len(edge_n) > 0:
            # CLIP uses fc1/fc2 instead of mlp[0]/mlp[3]
            w1       = block.mlp.fc1.weight[edge_n.start:edge_n.stop, :]
            b1       = block.mlp.fc1.bias[edge_n.start:edge_n.stop]
            w2       = block.mlp.fc2.weight[:, edge_n.start:edge_n.stop]
            edge_mlp = torch.nn.functional.gelu(ln_x_mlp @ w1.t() + b1) @ w2.t()
        else:
            edge_mlp = torch.tensor([])
        raw_latency["edge"] += time.time() - t0

        mlp_parts = [edge_mlp] if edge_mlp.numel() > 0 else []
        for w_name, fut in mlp_futures.items():
            res, lat = fut.result()
            if res is not None and res.numel() > 0:
                mlp_parts.append(res)
            raw_latency[w_name] += lat

        # CLIP uses fc2 bias (master.py used mlp[3].bias)
        mlp_final     = torch.sum(torch.stack(mlp_parts), dim=0) + block.mlp.fc2.bias
        current_state = identity + mlp_final

        # ── ARIMA bookkeeping (identical to master.py) ────────────────────
        for dev in orch.all_devices:
            s = share_used[dev]
            orch.arima.record_block_latency(dev, raw_latency[dev], s if s > 0 else 0.0)

        for w_name in probing_workers:
            s = share_used.get(w_name, 0.0)
            if s > 0 and raw_latency[w_name] < 999.0:
                orch.arima.notify_probe_result(w_name, raw_latency[w_name] / s)
            else:
                orch.arima.notify_probe_result(w_name, 999.0)

    return current_state


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
    return left_speed, right_speed, "ARC_RIGHT" if error < 0 else "ARC_LEFT"


# ---------------------------
# Pre-compute text embedding once (cheap, local)
# ---------------------------
text_inputs = processor(text=[SEARCH_PROMPT], return_tensors="pt", padding=True)
with torch.no_grad():
    text_emb = clip.get_text_features(**text_inputs)
    text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

# ---------------------------
# Persistent state
# ---------------------------
import time
last_logits = 0.0
last_error  = 0.0
step_count  = 0

print(f"[RUN] Searching for: '{SEARCH_PROMPT}'")

# ---------------------------
# Main Loop
# ---------------------------
while robot.step(timestep) != -1:
    step_count   += 1
    run_inference = (step_count % INFERENCE_EVERY_N == 0)

    if run_inference:
        image = get_frame_as_pil(camera)

        if image is not None:
            # Process image into pixel_values tensor
            img_inputs   = processor(images=image, return_tensors="pt")
            pixel_values = img_inputs["pixel_values"]

            with torch.no_grad():
                # Distributed CLIP vision forward
                last_hidden = clip_distributed_forward(pixel_values)  # (1, 50, 768)

                # Heatmap from patch tokens (skip CLS at index 0)
                patches   = last_hidden[:, 1:, :]                     # (1, 49, 768)
                projected = clip.visual_projection(patches)
                projected = projected / projected.norm(dim=-1, keepdim=True)

                # Similarity score for confidence
                img_emb     = projected.mean(dim=1)
                last_logits = (img_emb @ text_emb.t()).squeeze().item() * 100.0

                # Spatial heatmap → lateral error
                heatmap  = torch.matmul(projected[0], text_emb.t()).squeeze()
                weights  = torch.softmax(heatmap / 0.1, dim=0)
                center_x = (weights * _xs).sum().item()
                last_error = (center_x - EMP_CENTER) / EMP_CENTER

            print(f"[FOUND] score={last_logits:.2f}  error={last_error:+.2f}")

    if last_logits > CONFIDENCE_THRESHOLD:
        left_speed, right_speed, label = compute_speeds(last_error)
    else:
        last_error  = 0.0
        left_speed  = -SEARCH_SPEED
        right_speed = SEARCH_SPEED
        if run_inference:
            print(f"[SEARCHING] score={last_logits:.2f}")

    left_motor.setVelocity(left_speed)
    right_motor.setVelocity(right_speed)