# SPViT Robot — Person Following with Distributed ViT Inference

Person-following robot simulation using Webots + SPViT distributed inference.

## Project Structure

```
robot_spvit/
├── spvit/
│   ├── vit.py           # Vision Transformer model
│   ├── arima_v.py       # ARIMA-V adaptive scheduler
│   └── coordinator.py   # Multi-device inference manager
├── controllers/
│   └── robot_controller.py   # Webots robot brain
├── simulate_devices.py       # Launch fake edge devices
└── README.md
```

## How to Run

### Option A — Local inference (simplest, no distributed)
1. Open Webots → File → Open Sample World → `e-puck.wbt`
2. Set robot controller to `robot_controller.py`
3. Make sure `USE_DISTRIBUTED = False` in `robot_controller.py`
4. Press Play ▶ in Webots

### Option B — Full SPViT distributed (3 simulated devices)
**Terminal 1:**
```bash
python simulate_devices.py
```

**Terminal 2 (or Webots):**
```bash
# Set USE_DISTRIBUTED = True in robot_controller.py first
python controllers/robot_controller.py
```

### Test without Webots
```bash
cd controllers
python robot_controller.py
# Runs 50 steps with random camera input
```

## Install dependencies
```bash
pip install torch torchvision numpy
```

## How SPViT Works Here

```
Camera Frame (32x32)
      ↓
Patch Embedding (coordinator)
      ↓
ARIMA-V decides: Device 0 → heads [0,1,2,3]
                 Device 1 → heads [4,5]
                 Device 2 → heads [6,7]
      ↓
All 3 devices run attention in PARALLEL via sockets
      ↓
Coordinator merges partial outputs
      ↓
MLP + Classification → person / no_person
      ↓
Motor commands → robot follows person
```

## PDC Component (for your report)

The parallel & distributed computing happens in `coordinator.py`:
- `SPViTCoordinator.infer()` — splits work across devices
- `ARIMAVScheduler.get_partition()` — adaptive head allocation
- Workers run on localhost sockets simulating WiFi devices
- Threading enables true parallel execution

## References
- SPViT paper: IEEE TMC 2025 — Zhao et al.
- ViT paper: ICLR 2021 — Dosovitskiy et al.
