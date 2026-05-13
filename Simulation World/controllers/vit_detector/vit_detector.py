"""
vit_detector.py  —  SPViT distributed inference on e-puck (Webots)

Runs the ViT architecture from the SPViT paper with true head-split attention
and neuron-split MLP via ViTMasterOrchestrator.

Setup:
  1. Drop the vit_detector/ folder into your Webots controllers directory.
  2. On your laptop: python distributedSystem/worker_vit.py --name laptop --master_ip <robot_ip>
  3. Set this file as the robot controller in Webots.

ViT config below matches the SPViT paper settings for CIFAR-10.
Change VIT_CONFIG to match whatever checkpoint you load.
"""

from controller import Robot
import numpy as np
import torch
from PIL import Image
from distributedSystem.master_vit import ViTMasterOrchestrator

# ─────────────────────────────────────────────────────────────────────────────
# ViT config  —  SPViT paper settings for CIFAR-10
# ─────────────────────────────────────────────────────────────────────────────

VIT_CONFIG = dict(
    image_size  = 32,
    patch_size  = 4,
    num_classes = 10,
    dim         = 512,
    depth       = 6,
    heads       = 8,
    mlp_dim     = 512,
    dim_head    = 64,
    dropout     = 0.0,
    emb_dropout = 0.0,
    pool        = 'cls',
    channels    = 3,
)

# Path to a trained checkpoint, or None to use random weights (for shape/math verification)
CHECKPOINT = None

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# CIFAR-10 normalisation constants
_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
_STD  = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)

# ─────────────────────────────────────────────────────────────────────────────
# Robot init
# ─────────────────────────────────────────────────────────────────────────────

robot    = Robot()
timestep = int(robot.getBasicTimeStep())

print("[INIT] Booting robot …")
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
# ViTMasterOrchestrator  (blocks until worker connects)
# ─────────────────────────────────────────────────────────────────────────────

orch = ViTMasterOrchestrator(
    expected_workers = ["laptop"],
    host             = "0.0.0.0",
    port             = 6688,            # paper's default port (Connect.py)
    vit_config       = VIT_CONFIG,
    checkpoint       = CHECKPOINT,
)

# ─────────────────────────────────────────────────────────────────────────────
# Image preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def get_frame_tensor(cam):
    raw = cam.getImage()
    if not raw:
        return None
    w   = cam.getWidth()
    h   = cam.getHeight()
    img = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 4))
    pil = Image.fromarray(img[:, :, [2, 1, 0]])                  # BGRA → RGB
    pil = pil.resize((VIT_CONFIG["image_size"], VIT_CONFIG["image_size"]))
    arr = np.array(pil, dtype=np.float32) / 255.0
    arr = (arr - _MEAN) / _STD
    return torch.tensor(arr).permute(2, 0, 1).unsqueeze(0)       # (1, 3, H, W)

# ─────────────────────────────────────────────────────────────────────────────
# Settings
# ─────────────────────────────────────────────────────────────────────────────

INFERENCE_EVERY_N    = 5      # run ViT every N Webots steps
CONFIDENCE_THRESHOLD = 0.4    # min softmax probability to act on result

# ─────────────────────────────────────────────────────────────────────────────
# Persistent state
# ─────────────────────────────────────────────────────────────────────────────

last_class = -1
last_conf  = 0.0
step_count = 0

print("[RUN] Starting distributed ViT inference loop …")

# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

while robot.step(timestep) != -1:
    step_count   += 1
    run_inference = (step_count % INFERENCE_EVERY_N == 0)

    if run_inference:
        img_tensor = get_frame_tensor(camera)

        if img_tensor is not None:
            with torch.no_grad():
                logits = orch.run_inference(img_tensor)          # (1, num_classes)
                probs  = torch.softmax(logits, dim=-1)[0]
                last_conf, pred = probs.max(dim=0)
                last_class = pred.item()
                last_conf  = last_conf.item()

                top3 = probs.topk(3)
                top3_str = "  ".join(
                    f"{CIFAR10_CLASSES[idx.item()]}={val.item():.3f}"
                    for val, idx in zip(top3.values, top3.indices)
                )
                print(f"[ViT] step={step_count:5d}  "
                      f"→ {CIFAR10_CLASSES[last_class]} ({last_conf:.3f})  |  {top3_str}")

    # ── Motor control ────────────────────────────────────────────────────────
    # Simple demo: drive forward when confident, spin to search otherwise.
    if last_conf > CONFIDENCE_THRESHOLD:
        left_motor.setVelocity(MAX_SPEED * 0.5)
        right_motor.setVelocity(MAX_SPEED * 0.5)
    else:
        left_motor.setVelocity(MAX_SPEED * 0.4)
        right_motor.setVelocity(-MAX_SPEED * 0.4)
