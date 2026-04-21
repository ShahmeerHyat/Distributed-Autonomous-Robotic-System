from controller import Robot
import numpy as np
import torch
import time
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# ---------------------------
# Initialize Robot & Devices
# ---------------------------
robot = Robot()
timestep = int(robot.getBasicTimeStep())

# ----- Motors -----
left_motor  = robot.getDevice("left wheel motor")
right_motor = robot.getDevice("right wheel motor")
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))
MAX_SPEED = 6.28

# ----- Camera -----
camera = robot.getDevice("camera")
camera.enable(timestep)

print("[ViT-EDGE] Robot initialized.")

# ---------------------------
# Load ViT (CLIP) Model
# ---------------------------
print("[ViT-EDGE] Loading CLIP-ViT model...")
model_id = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)
model.eval() # Set to evaluation mode
print("[ViT-EDGE] ViT loaded.")

# ---------------------------
# Semantic Config
# ---------------------------
SEARCH_PROMPT = "a round ball with black and white patches"
CENTER_TOL = 0.15
# Lowered threshold slightly; adjust based on your lighting/simulation
CONFIDENCE_THRESHOLD = 18.0 

def get_frame_as_pil(camera):
    raw = camera.getImage()
    if not raw: return None
    w, h = camera.getWidth(), camera.getHeight()
    img_np = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 4))[:, :, 2::-1]
    return Image.fromarray(img_np.astype('uint8'), 'RGB')

# ---------------------------
# Main Loop
# ---------------------------
print(f"[ViT-EDGE] Searching for: '{SEARCH_PROMPT}'")

while robot.step(timestep) != -1:
    pil_img = get_frame_as_pil(camera)
    if pil_img is None: continue

    # 1. Prepare Inputs
    inputs = processor(text=[SEARCH_PROMPT], images=pil_img, return_tensors="pt", padding=True)

    # 2. Inference
    with torch.no_grad():
        outputs = model(**inputs)
        
    # Get global similarity scores
    logits_per_image = outputs.logits_per_image 

    # 3. Spatial Localization (Heatmap Logic)
    # last_hidden_state: [1, 50, 768] (1 CLS + 49 patches)
    # text_embeds: [1, 512]
    
    last_hidden_state = outputs.vision_model_output.last_hidden_state
    text_embeds = outputs.text_embeds

    # Extract spatial patches (indices 1 to 49)
    patches = last_hidden_state[:, 1:, :] # [1, 49, 768]
    
    # Project patches from 768 -> 512
    projected_patches = model.visual_projection(patches) # [1, 49, 512]
    
    # Normalize for cosine similarity calculation
    projected_patches = projected_patches / projected_patches.norm(dim=-1, keepdim=True)
    norm_text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

    # Calculate similarity: [49, 512] @ [512, 1] -> [49, 1]
    heatmap = torch.matmul(projected_patches[0], norm_text_embeds.t()).squeeze()
    
    # Find the patch with the highest match
    best_patch_idx = torch.argmax(heatmap).item()

    # Convert patch index to X-coordinate (7x7 grid for patch-32)
    grid_size = 7 
    patch_x = best_patch_idx % grid_size
    
    # Normalize X to [-1.0, 1.0] for steering
    # patch_x: 0, 1, 2 (left) | 3 (center) | 4, 5, 6 (right)
    error = (patch_x - 3) / 3.0
    
    # 4. Control Logic (Proportional Steering)
    if logits_per_image.item() > CONFIDENCE_THRESHOLD:
        print(f"[FOUND] Confidence: {logits_per_image.item():.2f} | Error: {error:.2f}")
        
        if abs(error) < CENTER_TOL:
            # Move Forward
            left_speed = right_speed = MAX_SPEED * 0.7
        elif error > 0: 
            # Target is to the right
            left_speed  = MAX_SPEED * 0.6
            right_speed = MAX_SPEED * 0.1
        else: 
            # Target is to the left
            left_speed  = MAX_SPEED * 0.1
            right_speed = MAX_SPEED * 0.6
    else:
        print(f"[SEARCHING] Confidence: {logits_per_image.item():.2f}")
        # Search mode: spin in place
        left_speed  = MAX_SPEED * 0.3
        right_speed = -MAX_SPEED * 0.3

    left_motor.setVelocity(left_speed)
    right_motor.setVelocity(right_speed)