# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Distributed Autonomous Robotic System — a split inference framework that distributes CLIP (ViT-B/32) transformer computation across an edge device and remote workers to minimize inference latency. The robot runs in Webots simulator, tracks a moving ball, and uses ARIMA-V adaptive load balancing to distribute attention heads and MLP neurons across devices.

## How to Run

**Model Download (one-time):**
```bash
python download_model.py
```
Downloads OpenAI CLIP ViT-B/32 from HuggingFace into `Clip Model/`.

**Simulation:**
1. Open Webots, load `Simulation World/worlds/Bot Training.wbt`
2. The e-puck robot uses `clip_detector` as its controller
3. Workers connect to master at `localhost:29500` before or after simulation starts

**Benchmarking (no Webots needed):**
```bash
cd "Simulation World/controllers/clip_detector_bm"
python benchmark.py --runs 20 --bandwidth 160
```

**Starting a Worker:**
```bash
cd "Simulation World/controllers/clip_detector/distributedSystem"
python worker.py --master-host <ip> --master-port 29500
```

## Architecture

### Distributed Inference Split

The core technique splits each transformer layer between the edge device and remote workers:

- **Attention heads** → split by **concatenation**: each device handles a slice of the 12 heads. Results are `torch.cat`-ed because heads capture independent features.
- **MLP neurons** → split by **summation**: each device handles a range of the 3072 hidden neurons. Results are `torch.sum`-ed because neurons are independent contributors.

Tensor flow:
```
clip_detector.py  →  MasterOrchestrator  →  [slice & send]  →  CLIPWorker(s)
                                         ←  [partial results] ←
                       [reconstruct & continue inference]
```

### Key Files

| File | Role |
|------|------|
| `clip_detector/clip_detector.py` | Webots robot controller; loads CLIP, captures camera frames, calls master, computes steering |
| `distributedSystem/master.py` | `MasterOrchestrator` — accepts worker connections, slices attention/MLP tensors, sends work, collects and merges results |
| `distributedSystem/worker.py` | `CLIPWorker` — connects to master, computes assigned head/neuron ranges, returns partial outputs |
| `distributedSystem/shared_utils.py` | Network I/O (`send_msg`/`recv_msg` via `torch.save`), `MultiDeviceARIMAManager`, `CircuitBreaker`, `to_head_space`, `sum_mlp_parts` |
| `distributedSystem/benchmark.py` | Runs instrumented inference across split ratios, measures edge/worker/comm latency |
| `paper-implementation/spvit/main.ipynb` | Theoretical background and SPViT reference |

### ARIMA-V Load Balancer (`shared_utils.py`)

`MultiDeviceARIMAManager` continuously adjusts each worker's compute share:
- Tracks normalized latency history per device
- Predicts next latency using a differenced moving average (ARIMA approximation)
- Sets shares **inversely proportional** to predicted latency
- Hysteresis constants: `DROP_MULT=3.0` (trip threshold), `ADMIT_MULT=1.8` (recovery threshold)
- Enforces minimum share floors to prevent "death spiral" starvation

### Circuit Breaker (`shared_utils.py`)

`CircuitBreaker` has three states (CLOSED → OPEN → HALF_OPEN → CLOSED):
- Trips when consecutive failures exceed threshold or latency exceeds `DROP_MULT × baseline`
- In OPEN state, master routes all work locally
- Probes recovery with a single request in HALF_OPEN state

### Network Protocol

`send_msg` / `recv_msg` in `shared_utils.py` frame messages with a 4-byte big-endian length prefix, serializing tensors with `torch.save` / `torch.load`. Master listens on TCP port `29500`.

## Dependencies

No `requirements.txt` is present. Required packages:
- `torch`, `transformers` (CLIP model inference)
- `Pillow` (image preprocessing)
- `numpy`
- Webots Python API (`from controller import Robot`) — provided by Webots installation

The `yolo_venv/` directory is a venv for the YOLO variant controllers (not the primary approach).
