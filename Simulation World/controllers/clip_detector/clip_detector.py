from controller import Robot
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from distributedSystem.master import MasterOrchestrator

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
CENTER_TOL           = 0.01 # ±15% of frame width → considered centred
ROTATE_ONLY_TOL      = 0.40 # If error is > 40%, stop moving forward and just spin
CONFIDENCE_THRESHOLD = 20.0
EMA_ALPHA            = 0.4
EMP_CENTER           = 2.95 # empirically determined center_x value when object is perfectly centered (for error normalization)  

# Drive constants
BASE_SPEED  = MAX_SPEED * 0.35   # forward creep while centering
TURN_GAIN   = MAX_SPEED * 0.80   # steering authority
SEARCH_SPEED = MAX_SPEED * 0.30  # spin speed when object not found

# Frame-skip: run CLIP every N steps
INFERENCE_EVERY_N = 5

# ---------------------------
# Pre-compute patch grid once
# ---------------------------
GRID_SIZE = 7
_xs = torch.arange(GRID_SIZE).unsqueeze(0).repeat(GRID_SIZE, 1).flatten().float()
# print(f"[INIT] _XS grid pre-computed: {_xs.shape}  values={_xs.numpy()} ...")

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
    Improved control logic: 
    1. If very far off, spin in place.
    2. If somewhat off, arc toward the object.
    3. If centered, drive straight.
    """
    abs_err = abs(error)
    
    # CASE 1: Centered (Drive straight)
    if abs_err < CENTER_TOL:
        return BASE_SPEED, BASE_SPEED, "STRICT_CENTER"

    # CASE 2: Far off-center (Rotate in place)
    # This prevents the robot from 'orbiting' the object
    if abs_err > ROTATE_ONLY_TOL:
        # High gain rotation, zero forward velocity
        turn_speed = TURN_GAIN * (error / abs_err) # Keeps direction sign
        return turn_speed, -turn_speed, "STATIONARY_TURN"

    # CASE 3: Moderate error (Arcing/Proportional steering)
    # We use a non-linear scaling so it slows down forward speed as it turns
    forward_speed = BASE_SPEED * (1.0 - (abs_err / ROTATE_ONLY_TOL))
    turn_delta    = TURN_GAIN * error
    
    left_speed  = max(-MAX_SPEED, min(MAX_SPEED, forward_speed - turn_delta))
    right_speed = max(-MAX_SPEED, min(MAX_SPEED, forward_speed + turn_delta))

    direction = "ARC_RIGHT" if error < 0 else "ARC_LEFT"
    return left_speed, right_speed, direction

def get_clip_metadata(model):
    # Extracts constants from the CLIP Vision Config
    config = model.vision_model.config
    return {
        "num_heads": config.num_attention_heads,
        "embed_dim": config.hidden_size,
        "head_dim": config.hidden_size // config.num_attention_heads,
        "mlp_hidden_dim": config.intermediate_size,
        "seq_length": 50, # 49 patches + 1 class token
    }

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

orch = MasterOrchestrator(
    expected_workers=["pc_gpu"],
    host="0.0.0.0",            # Listen on all interfaces
    port=29500,                # Default port
    model=model.visual_model,  # The Vision Tower
    meta=get_clip_metadata(model)             # The metadata dictionary
)

with torch.no_grad():
    text_inputs = processor(text=[SEARCH_PROMPT], return_tensors="pt", padding=True)
    # Use the model's get_text_features to get the normalized text embedding
    text_embeds = model.get_text_features(**text_inputs)
    text_emb_norm = text_embeds / text_embeds.norm(dim=-1, keepdim=True)


while robot.step(timestep) != -1:
    step_count    += 1
    run_inference  = (step_count % INFERENCE_EVERY_N == 0)

    # ---------------------------
    # CLIP inference (frame-skipped)
    # ---------------------------
    if run_inference:
        image = get_frame_as_pil(camera)

        if image is not None:
            # inputs = processor(
            #     text=[SEARCH_PROMPT],
            #     images=image,
            #     return_tensors="pt",
            #     padding=True
            # )

            # with torch.no_grad():
            #     outputs = model(**inputs)
            
            # 1. Pre-process the image
            inputs = processor(images=image, return_tensors="pt")
            
            # 2. RUN DISTRIBUTED INFERENCE
            # returns raw tensor: [batch, 50_tokens, 768_dim]
            with torch.no_grad():
                outputs = orch.run_inference(inputs['pixel_values'])

            # 3. FIX: ACCESS PATCHES DIRECTLY
            # The orchestrator returns the hidden state. 
            # [:, 1:, :] skips the CLS token and takes the 49 spatial patches.
            patches = outputs[:, 1:, :] 
            
            # 4. Project and Normalize Patches
            projected = model.visual_projection(patches)
            projected = projected / projected.norm(dim=-1, keepdim=True)

            # 6. Heatmap Math
            # Result: [49] similarity scores
            heatmap = torch.matmul(projected[0], text_emb_norm.t()).squeeze()
            
            # FIX: Confidence proxy (since .logits_per_image is gone)
            # We scale the max similarity to act as a confidence score
            last_logits = heatmap.max().item() * 100 

            # FIX: Softmax Temperature (0.01 makes centering much more precise)
            weights = torch.softmax(heatmap / 0.01, dim=0)

            # 7. Spatial Calculations
            center_x  = (weights * _xs).sum().item()
            center    = (GRID_SIZE - 1) / 2
            raw_error = (center_x - center) / center

            print(f"DEBUG: center_x={center_x:.2f} raw_error={raw_error:+.2f}")
            last_error = raw_error

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
        left_speed  = -SEARCH_SPEED
        right_speed = SEARCH_SPEED

        if run_inference:
            
            print(f"[SEARCHING] score={last_logits:.2f}")

    left_motor.setVelocity(left_speed)
    right_motor.setVelocity(right_speed)