"""
Webots Robot Controller — Person Following with SPViT
======================================================
This is the Python file you paste into Webots as the robot controller.

What it does every timestep:
    1. Gets camera frame from Webots
    2. Runs SPViT distributed inference (person / no-person)
    3. Steers robot toward detected person

Setup in Webots:
    - Robot: e-puck (has camera + 2 wheels)
    - Set controller to this file
    - Make sure spvit/ folder is in the same directory

Folder structure Webots needs:
    controllers/
    └── robot_controller/
        ├── robot_controller.py   ← this file
        └── spvit/
            ├── vit.py
            ├── arima_v.py
            └── coordinator.py
"""

import sys
import os
import numpy as np
import time

# Add spvit folder to path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'spvit'))

import torch
import torchvision.transforms as T
from spvit.vit import VisionTransformer
from spvit.coordinator import SPViTCoordinator

# ─────────────────────────────────────────────
# WEBOTS IMPORTS
# ─────────────────────────────────────────────
try:
    from controller import Robot, Camera, Motor
    WEBOTS_AVAILABLE = True
except ImportError:
    # Running outside Webots for testing
    WEBOTS_AVAILABLE = False
    print("[WARNING] Webots controller module not found — running in test mode")


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
TIMESTEP = 64           # ms per simulation step
MAX_SPEED = 6.28        # rad/s (e-puck max)
IMAGE_SIZE = 32         # resize camera frame to this
NUM_DEVICES = 3         # simulated devices (coordinator + 2 workers)
USE_DISTRIBUTED = False # set True if workers are running (simulate_devices.py)

# Person detection thresholds
DETECTION_CONFIDENCE = 0.6   # min confidence to act
CENTER_THRESHOLD = 0.15      # how centered person must be before going straight


# ─────────────────────────────────────────────
# IMAGE PREPROCESSING
# ─────────────────────────────────────────────
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


def webots_image_to_tensor(camera):
    """
    Convert Webots camera output to PyTorch tensor.
    Webots gives raw BGRA bytes — we convert to RGB numpy then to tensor.
    """
    width = camera.getWidth()
    height = camera.getHeight()

    # Get raw image bytes from Webots
    raw = camera.getImage()

    # Convert to numpy array (BGRA format from Webots)
    img_array = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 4))

    # Drop alpha channel, convert BGR → RGB
    img_rgb = img_array[:, :, :3][:, :, ::-1].copy()

    # Apply transforms → (1, 3, IMAGE_SIZE, IMAGE_SIZE)
    tensor = transform(img_rgb).unsqueeze(0)
    return tensor, img_rgb


def estimate_person_position(img_rgb, class_id, confidence):
    """
    Estimate where in the frame the person is (left/center/right).

    Simple heuristic: in a real system you'd use bounding box detection.
    Here we use a basic brightness/motion heuristic on the detected frame
    to estimate person position for steering.

    Returns: 'left', 'center', 'right', or 'none'
    """
    if class_id != 1 or confidence < DETECTION_CONFIDENCE:
        return 'none'

    width = img_rgb.shape[1]
    left_half = img_rgb[:, :width//2, :]
    right_half = img_rgb[:, width//2:, :]

    # Use pixel brightness difference as rough person position estimate
    left_brightness = float(np.mean(left_half))
    right_brightness = float(np.mean(right_half))

    diff = (left_brightness - right_brightness) / 255.0

    if abs(diff) < CENTER_THRESHOLD:
        return 'center'
    elif diff > 0:
        return 'left'
    else:
        return 'right'


# ─────────────────────────────────────────────
# ROBOT CONTROLLER STATES
# ─────────────────────────────────────────────
class RobotState:
    SEARCHING   = 'searching'    # no person — rotating to find
    FOLLOWING   = 'following'    # person detected — moving toward
    CENTERING   = 'centering'    # person detected but off-center — turning
    STOPPED     = 'stopped'      # person lost — waiting


# ─────────────────────────────────────────────
# MAIN CONTROLLER
# ─────────────────────────────────────────────

def run_controller():
    """Main robot control loop."""

    # ── Initialize Webots ─────────────────────────────────────────────────────
    if WEBOTS_AVAILABLE:
        robot = Robot()

        # Camera
        camera = robot.getDevice('camera')
        camera.enable(TIMESTEP)
        print(f"[Robot] Camera: {camera.getWidth()}x{camera.getHeight()}")

        # Motors (e-puck has left/right wheel)
        left_motor  = robot.getDevice('left wheel motor')
        right_motor = robot.getDevice('right wheel motor')
        left_motor.setPosition(float('inf'))   # velocity control mode
        right_motor.setPosition(float('inf'))
        left_motor.setVelocity(0.0)
        right_motor.setVelocity(0.0)

    else:
        # Test mode — simulate 50 steps with random "camera" input
        print("[Test Mode] Simulating 50 inference steps without Webots")

    # ── Load ViT model ────────────────────────────────────────────────────────
    print("[Robot] Loading ViT model...")
    model = VisionTransformer(
        image_size=IMAGE_SIZE,
        patch_size=4,
        in_channels=3,
        num_classes=2,      # 0=no_person, 1=person
        embed_dim=256,
        depth=6,
        num_heads=8,
        dropout=0.1,
    )

    # Load pretrained weights if available
    weights_path = os.path.join(os.path.dirname(__file__), 'model_weights.pth')
    if os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"[Robot] Loaded weights from {weights_path}")
    else:
        print("[Robot] No weights found — using random init (for demo only)")

    model.eval()
    print(f"[Robot] Model parameters: {model.get_num_parameters():,}")

    # ── Initialize SPViT Coordinator ──────────────────────────────────────────
    coordinator = SPViTCoordinator(
        model=model,
        num_devices=NUM_DEVICES,
        total_heads=8,
        embed_dim=256,
        num_classes=2,
    )

    if USE_DISTRIBUTED:
        # Connect to worker processes (run simulate_devices.py first)
        coordinator.connect_to_workers()
    else:
        print("[Robot] Running LOCAL inference (no distributed workers)")
        print("[Robot] Set USE_DISTRIBUTED=True and run simulate_devices.py for full SPViT")

    # ── State machine ─────────────────────────────────────────────────────────
    state = RobotState.SEARCHING
    no_person_count = 0
    step = 0

    print("[Robot] Starting person-following loop...")
    print("-" * 50)

    # ── Main loop ─────────────────────────────────────────────────────────────
    while True:
        if WEBOTS_AVAILABLE:
            if robot.step(TIMESTEP) == -1:
                break  # Webots simulation ended
        else:
            if step >= 50:
                break

        step += 1

        # ── Get camera frame ──────────────────────────────────────────────────
        if WEBOTS_AVAILABLE:
            tensor, img_rgb = webots_image_to_tensor(camera)
        else:
            # Test mode: random tensor
            tensor = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
            img_rgb = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

        # ── SPViT Inference ───────────────────────────────────────────────────
        t_start = time.time()
        class_id, confidence, stats = coordinator.infer(tensor)
        t_infer = (time.time() - t_start) * 1000

        class_names = ['no_person', 'person']
        print(f"[Step {step:04d}] "
              f"Detection: {class_names[class_id]} ({confidence:.2f}) | "
              f"Inference: {t_infer:.1f}ms | "
              f"Heads: {stats['head_partition']}")

        # ── Estimate person position ──────────────────────────────────────────
        position = estimate_person_position(img_rgb, class_id, confidence)

        # ── State machine update ──────────────────────────────────────────────
        if position == 'none':
            no_person_count += 1
            if no_person_count > 10:
                state = RobotState.SEARCHING
            else:
                state = RobotState.STOPPED
        else:
            no_person_count = 0
            if position == 'center':
                state = RobotState.FOLLOWING
            else:
                state = RobotState.CENTERING

        # ── Motor commands ────────────────────────────────────────────────────
        if WEBOTS_AVAILABLE:
            if state == RobotState.FOLLOWING:
                # Person is centered — go straight
                left_motor.setVelocity(MAX_SPEED * 0.7)
                right_motor.setVelocity(MAX_SPEED * 0.7)

            elif state == RobotState.CENTERING:
                if position == 'left':
                    # Person on left — turn left
                    left_motor.setVelocity(-MAX_SPEED * 0.3)
                    right_motor.setVelocity(MAX_SPEED * 0.3)
                else:
                    # Person on right — turn right
                    left_motor.setVelocity(MAX_SPEED * 0.3)
                    right_motor.setVelocity(-MAX_SPEED * 0.3)

            elif state == RobotState.SEARCHING:
                # Rotate slowly to find person
                left_motor.setVelocity(MAX_SPEED * 0.3)
                right_motor.setVelocity(-MAX_SPEED * 0.3)

            elif state == RobotState.STOPPED:
                left_motor.setVelocity(0.0)
                right_motor.setVelocity(0.0)

        print(f"         State: {state} | Position: {position}")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    if USE_DISTRIBUTED:
        coordinator.stop()

    avg_time = np.mean(coordinator.inference_times) * 1000
    print(f"\n[Robot] Done. Average inference time: {avg_time:.1f}ms over {step} steps")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == '__main__':
    run_controller()
