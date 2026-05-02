# CLIP-SPViT — Distributed Autnomous Robot System

Open-vocabulary object tracking on a simulated e-puck robot using Webots, CLIP ViT-B/32, and a distributed inference system based on the SPViT framework (Zhao et al., IEEE TMC 2025).

The robot tracks any object described by a **natural language prompt** — no retraining required. Inference is split across an edge CPU and a remote GPU worker using Head-Width and Neuron-Width splitting, governed by an AIMD-based adaptive load balancer with circuit-breaker fault tolerance.

---

## Project Structure

```
Distributed-Autonomous-Robotic-System/
├── distributedSystem/
│   ├── master.py          # MasterOrchestrator — block-by-block distributed forward pass
│   ├── worker.py          # CLIPWorker — runs on remote GPU/CPU machine
│   ├── splitInfer.py      # MultiDeviceEvaluator — AIMD scheduler + EMA compute tracking
│   ├── comms.py           # Socket protocol — send_msg / recv_msg / probe_rtt / CircuitBreaker
│   ├── helper.py          # to_head_space / merge_heads / allocate_heads / get_model_metadata
│   └── benchmarks.py      # BenchmarkCollector — measures all paper comparison metrics
│
├── controllers/
│   ├── clip_detector_lcp.py   # Main robot controller (LCP — worker-optional mode)
│   ├── clip_detector_bm.py    # Benchmarking variant with full metric collection
│   └── ball_motion.py         # Ball controller (sine-wave / straight path)
│
├── Clip Model/                # Local CLIP ViT-B/32 checkpoint (not committed)
└── README.md
```

---

## How It Works

```
Camera frame (480×480, BGRA→RGB)
        ↓
CLIP patch embedding + pre-layernorm        [edge, local]
        ↓
┌─────────────────────────────────────────────────────┐
│  MasterOrchestrator — 12 encoder blocks             │
│                                                     │
│  For each block:                                    │
│    AIMD evaluator → allocate_heads()                │
│    ├── Edge CPU:   heads [0 .. H_edge)   [local]    │
│    └── GPU Worker: heads [H_edge .. 12)  [socket]   │
│                                                     │
│    Results concatenated → out_proj                  │
│    MLP similarly split by neuron range              │
└─────────────────────────────────────────────────────┘
        ↓
post_layernorm  [edge, local]
        ↓
CLS token → visual_projection → cosine sim with text_emb → confidence score
Patch tokens → per-patch similarity → softmax heatmap → lateral error
        ↓
Three-zone steering controller → motor commands
        ↓
Robot tracks object described by SEARCH_PROMPT
```

---

## Setup

### 1. Dependencies

```bash
pip install torch transformers Pillow numpy
```

Webots R2025a must be installed separately: https://cyberbotics.com

### 2. CLIP model

Download `openai/clip-vit-base-patch32` from HuggingFace and place it at `../../../Clip Model/` relative to the controller directory, or change `model_path` in the controller to point to your local checkpoint.

```python
# In clip_detector_lcp.py
model_path = r"../../../Clip Model"
```

### 3. World file

Open `simulation_world.wbt` in Webots. Ensure the e-puck node has `supervisor TRUE` set — this is required for world reset functionality.

The ball node must have `DEF TENNIS_BALL` set so the supervisor can find and reset it.

---

## Running the System

### Option A — Edge-only (no worker, simplest)

No worker needed. The `MasterOrchestrator` runs all 12 encoder blocks locally.

1. Open Webots and load the world file
2. The controller (`clip_detector_lcp.py`) starts automatically
3. The robot begins searching for the object described by `SEARCH_PROMPT`

### Option B — Distributed (edge + GPU worker)

**On the worker machine:**
```bash
cd Distributed-Autonomous-Robotic-System
python distributedSystem/worker.py \
    --name pc_gpu \
    --master_ip <edge-machine-IP> \
    --port 29500 \
    --gpu
```

**On the edge machine (Webots):**

The `MasterOrchestrator` listens on port 29500 and blocks until the expected worker connects. Once connected it runs the preflight RTT measurement and begins distributing inference.

Both machines must have the CLIP checkpoint locally — the worker loads its own copy.

### Option C — With benchmarking

Use `clip_detector_bm.py` as the controller. It collects all metrics and prints a full report after `BENCHMARK_AFTER_N_FRAMES` frames:

```
═══════════════════════════════════════════════════════════════════════════
  DISTRIBUTED CLIP BENCHMARK REPORT
═══════════════════════════════════════════════════════════════════════════
  M1.  Model parameters
  M2.  Edge-only inference latency
  M3.  Distributed inference latency + speedup
  M4.  Communication overhead (MB/frame)
  M5/6. ATTN / MLP share of edge compute
  M7.  ARIMA scheduler overhead (µs/block)
  M8.  Frames per second
  M9.  Ball loss rate
  M10. Share stability (EMA smoothing validation)
  M11. Circuit breaker recovery (simulated worker failure)
```

---

## Changing the Tracked Object

Edit `SEARCH_PROMPT` in the controller — no retraining required:

```python
SEARCH_PROMPT = "a round ball with black and white patches"   # default
SEARCH_PROMPT = "a red fire extinguisher"                     # any object
SEARCH_PROMPT = "a yellow traffic cone"
```

---

## Key Design Decisions

### Why CLIP instead of a classification ViT?

The original SPViT targets ImageNet classification with a fixed label set. CLIP's joint text-image embedding space enables zero-shot detection from free-form text, making the robot's perception module general-purpose without task-specific training.

### Why AIMD instead of ARIMA-V?

The original ARIMA-V scheduler stores raw latency in its prediction history. Raw latency is confounded with workload share: a device doing 80% of the work and taking 200 ms appears slower than one doing 20% taking 50 ms, even if both run at identical throughput. This feedback loop collapses all work to the edge device.

Our `MultiDeviceEvaluator` (in `splitInfer.py`) separates two signals:
- **Compute EMA** — smoothed per-block compute time, passive signal
- **AIMD Latency Score** — aggressive multiplicative decrease on spikes, slow additive recovery

Shares are derived from a weighted combination of both, preventing the collapse.

### Why is distributed currently slower than edge-only?

Python's `torch.save()` serialisation adds pickle framing that inflates tensor payloads to ~6.5 MB/frame. On an 80 Mb/s link this costs ~650 ms/frame in transmission alone, exceeding the compute savings from offloading. Replacing with raw FP16 struct-packed transmission would reduce overhead to ~0.4 MB/frame, at which point the split becomes economically viable. This is the single most impactful remaining engineering change.

---

## Distributed System Components

| File | Responsibility |
|---|---|
| `master.py` | Block-by-block forward pass, dispatches ATTN/MLP tasks to workers |
| `worker.py` | Receives tasks, computes attention or MLP slice, returns result |
| `splitInfer.py` | AIMD evaluator, compute EMA, score normalisation |
| `comms.py` | Socket send/recv, RTT probing, CircuitBreaker state machine |
| `helper.py` | `to_head_space`, `allocate_heads`, `sum_mlp_parts`, model metadata |
| `benchmarks.py` | Collects M1–M11 metrics, prints paper comparison table |

### Circuit Breaker States

```
CLOSED ──(latency spike / dispatch fail)──► OPEN
  ▲                                           │
  │                                    exponential backoff
  │                                    T_k = min(4·2^(k-1), 32) blocks
  │                                           │
  └──(probe success)──── HALF_OPEN ◄──────────┘
                              │
                         (probe fail)
                              │
                           back to OPEN
```

---

## References

1. S. Zhao, T. Liu, H. Jin, D. Yao — *SPViT: Accelerate Vision Transformer Inference on Mobile Devices via Adaptive Splitting and Offloading* — IEEE Transactions on Mobile Computing, vol. 24, no. 10, 2025
2. A. Radford et al. — *Learning Transferable Visual Models From Natural Language Supervision* (CLIP) — ICML 2021
3. A. Dosovitskiy et al. — *An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale* — ICLR 2021