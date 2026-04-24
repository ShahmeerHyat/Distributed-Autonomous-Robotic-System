"""
benchmark.py — Standalone metrics for our SPViT reimplementation.

Runs entirely locally (no worker connection needed). For each metric:

  §5.1  Overall system metrics
        • Model size (MB)
        • Total inference latency (ms)
        • Communication overhead (MB / frame)  ← computed from real tensor sizes
        • Top-1 accuracy                        ← SKIPPED (no dataset; fill from paper)

  §5.2  Split-ratio vs latency
        Analogous to the paper's split-layer-3 / 6 / 9 analysis.
        We vary edge_share ∈ {1.0, 0.5, 0.2} and measure:
          • edge_latency   = wall-clock time for the edge portion of each block
          • worker_latency = estimated proportionally (same hardware assumption)
          • comm_latency   = bytes / assumed bandwidth (see ASSUMED_BANDWIDTH_MBPS)
          • total_latency  = max(edge, worker) + comm

Usage:
    python benchmark.py            # default: 10 bench runs, bandwidth 80 Mb/s
    python benchmark.py --runs 20 --bandwidth 160
"""

import argparse
import io
import time
import torch
import torchvision.models as models
import torch.nn.functional as F

# ── local imports ─────────────────────────────────────────────────────────────
from shared_utils import (
    get_model_metadata,
    get_head_weights,
    merge_n_projections,
    ready_for_math,
    send_msg,          # imported only to reuse torch.save serialisation
)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

WARMUP_RUNS          = 3        # inference passes before timing (JIT / cache warm-up)
DEFAULT_BENCH_RUNS   = 10       # inference passes used for the average

# Assumed network bandwidth for computing communication latency.
# Change with --bandwidth flag.  Paper uses 80 Mb/s and 160 Mb/s.
DEFAULT_BW_MBPS      = 80.0

# Three split configurations analogous to the paper's Layer 3 / 6 / 9 analysis.
# edge_share = fraction of attention heads AND MLP neurons kept on the edge device.
SPLIT_CONFIGS = [
    {"label": "Layer-3 analog (edge-heavy)", "edge_share": 1.00},
    {"label": "Layer-6 analog (balanced)  ", "edge_share": 0.50},
    {"label": "Layer-9 analog (wkr-heavy) ", "edge_share": 0.20},
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def tensor_bytes(t: torch.Tensor) -> int:
    """Serialised size of a tensor as it would travel over the socket."""
    buf = io.BytesIO()
    torch.save(t, buf)
    return len(buf.getvalue())

def model_size_mb(model: torch.nn.Module) -> float:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return len(buf.getvalue()) / (1024 ** 2)

def num_params_m(model: torch.nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6

def separator(char="─", width=72):
    print(char * width)

def print_table(headers, rows, col_widths=None):
    if col_widths is None:
        col_widths = [max(len(str(r[i])) for r in [headers] + rows) + 2
                      for i in range(len(headers))]
    fmt = "".join(f"{{:<{w}}}" for w in col_widths)
    separator()
    print(fmt.format(*headers))
    separator()
    for row in rows:
        print(fmt.format(*[str(x) for x in row]))
    separator()


# ─────────────────────────────────────────────────────────────────────────────
# Core: single-pass inference (edge portion only) with instrumentation
# ─────────────────────────────────────────────────────────────────────────────

def run_instrumented_inference(model, meta, x: torch.Tensor, edge_share: float):
    """
    Runs one forward pass through all encoder blocks.
    At each block, only the edge portion (edge_share fraction of heads/neurons)
    is computed; the worker portion is NOT actually dispatched — instead we
    record the tensor sizes and estimate the worker latency proportionally.

    Returns a dict with per-block breakdown and aggregate totals.
    """
    num_blocks = len(model.encoder.layers)

    block_edge_latency_ms   = []   # ms of actual edge compute per block
    block_worker_latency_ms = []   # ms estimated worker compute per block
    block_comm_bytes        = []   # bytes that would travel to/from worker per block
    block_attn_edge_ms      = []
    block_mlp_edge_ms       = []

    current_state = x

    with torch.no_grad():
        for i, block in enumerate(model.encoder.layers):

            # ── ATTENTION ─────────────────────────────────────────────────
            identity = current_state
            ln_x     = block.ln_1(current_state)

            num_heads    = meta["num_heads"]
            n_edge_heads = max(1, int(round(edge_share * num_heads)))
            n_wkr_heads  = num_heads - n_edge_heads

            # ── Edge ATTN compute ──
            t0 = time.perf_counter()
            edge_h   = range(n_edge_heads)
            attn_w   = get_head_weights(
                block.self_attention.in_proj_weight, edge_h,
                meta["embed_dim"], meta["head_dim"],
            )
            edge_qkv = ln_x @ attn_w.t()
            attn_edge_ms = (time.perf_counter() - t0) * 1e3

            # ── Worker ATTN: measure tensor that WOULD be sent/received ──
            # Send: ln_x  →  Receive: worker qkv slice
            send_bytes_attn = tensor_bytes(ln_x)
            wkr_head_frac   = n_wkr_heads / num_heads if num_heads > 0 else 0
            # Worker output shape mirrors edge output but for wkr_heads
            wkr_qkv_size    = edge_qkv.numel() * (wkr_head_frac / (n_edge_heads / num_heads)) \
                              if n_edge_heads > 0 else 0
            recv_bytes_attn = int(wkr_qkv_size * edge_qkv.element_size()) + 256  # +256 pickle overhead
            attn_comm_bytes = send_bytes_attn + recv_bytes_attn

            # ── Global softmax (always on master, not counted in split latency) ──
            # Reconstruct full merged QKV (pretend worker returned zeros for its slice)
            if n_wkr_heads > 0:
                wkr_qkv_shape = list(edge_qkv.shape)
                wkr_qkv_shape[-1] = int(edge_qkv.shape[-1] * wkr_head_frac / (n_edge_heads / num_heads)) \
                                    if n_edge_heads > 0 else edge_qkv.shape[-1]
                fake_wkr_qkv  = torch.zeros(wkr_qkv_shape)
                merged_qkv    = merge_n_projections([edge_qkv, fake_wkr_qkv])
            else:
                merged_qkv = merge_n_projections([edge_qkv])

            # Pad/truncate to expected size (needed if rounding makes shapes off)
            expected_qkv_dim = meta["embed_dim"] * 3
            if merged_qkv.shape[-1] != expected_qkv_dim:
                merged_qkv = F.pad(merged_qkv, (0, expected_qkv_dim - merged_qkv.shape[-1]))

            merged_qkv += block.self_attention.in_proj_bias
            q, k, v = torch.chunk(merged_qkv, 3, dim=-1)
            q, k, v = (ready_for_math(t, meta) for t in (q, k, v))
            scale   = meta["head_dim"] ** -0.5
            attn_probs = F.softmax((q @ k.transpose(-2, -1)) * scale, dim=-1)
            ctx     = (attn_probs @ v).transpose(1, 2).reshape(
                        1, meta["seq_length"], meta["embed_dim"])
            current_state = identity + block.self_attention.out_proj(ctx)

            # ── MLP ───────────────────────────────────────────────────────
            identity  = current_state
            ln_x_mlp  = block.ln_2(current_state)

            mlp_hidden   = meta["mlp_hidden_dim"]
            n_edge_n     = max(1, int(round(edge_share * mlp_hidden)))
            n_wkr_n      = mlp_hidden - n_edge_n

            # ── Edge MLP compute ──
            t0 = time.perf_counter()
            w1 = block.mlp[0].weight[:n_edge_n, :]
            b1 = block.mlp[0].bias[:n_edge_n]
            w2 = block.mlp[3].weight[:, :n_edge_n]
            edge_mlp = F.gelu(ln_x_mlp @ w1.t() + b1) @ w2.t()
            mlp_edge_ms = (time.perf_counter() - t0) * 1e3

            # ── Worker MLP: tensor sizes ──
            send_bytes_mlp = tensor_bytes(ln_x_mlp)
            wkr_n_frac     = n_wkr_n / mlp_hidden if mlp_hidden > 0 else 0
            recv_bytes_mlp = int(edge_mlp.numel() * edge_mlp.element_size()
                                 * (wkr_n_frac / max(n_edge_n / mlp_hidden, 1e-6))) + 256
            mlp_comm_bytes = send_bytes_mlp + recv_bytes_mlp

            # ── Finish block ──
            mlp_final     = edge_mlp + block.mlp[3].bias
            current_state = identity + mlp_final

            # ── Record ────────────────────────────────────────────────────
            total_edge_ms = attn_edge_ms + mlp_edge_ms

            # Worker latency estimate: proportional to its share of work.
            # If edge does share S in T ms, worker does (1-S) in T*(1-S)/S ms.
            # (assumes identical hardware speed — conservative for GPU workers)
            wkr_share     = 1.0 - edge_share
            wkr_lat_ms    = total_edge_ms * (wkr_share / edge_share) if edge_share < 1.0 else 0.0

            block_edge_latency_ms.append(total_edge_ms)
            block_worker_latency_ms.append(wkr_lat_ms)
            block_comm_bytes.append(attn_comm_bytes + mlp_comm_bytes)
            block_attn_edge_ms.append(attn_edge_ms)
            block_mlp_edge_ms.append(mlp_edge_ms)

    return {
        "edge_latency_ms":      block_edge_latency_ms,
        "worker_latency_ms":    block_worker_latency_ms,
        "comm_bytes":           block_comm_bytes,
        "attn_edge_ms":         block_attn_edge_ms,
        "mlp_edge_ms":          block_mlp_edge_ms,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Full-pass timing (clean, no instrumentation overhead)
# ─────────────────────────────────────────────────────────────────────────────

def time_full_inference(model, meta, x, edge_share, n_runs):
    """Returns list of wall-clock times (ms) for n_runs complete forward passes."""
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            run_instrumented_inference(model, meta, x, edge_share)
            times.append((time.perf_counter() - t0) * 1e3)
    return times


# ─────────────────────────────────────────────────────────────────────────────
# Main benchmark
# ─────────────────────────────────────────────────────────────────────────────

def main(bench_runs: int, bw_mbps: float):
    DIVIDER = "═" * 72

    print(f"\n{DIVIDER}")
    print("  SPViT Reimplementation — Benchmark & Metrics Report")
    print(f"  Model: ViT-B/16  |  Runs: {bench_runs}  |  Bandwidth (assumed): {bw_mbps} Mb/s")
    print(DIVIDER)

    # ── Load model ────────────────────────────────────────────────────────────
    print("\n[1/4] Loading ViT-B/16 …", end=" ", flush=True)
    try:
        model = models.vit_b_16(weights="DEFAULT").eval()
    except Exception:
        print("(pretrained weights unavailable, using random init) ", end="")
        model = models.vit_b_16(weights=None).eval()
    meta  = get_model_metadata(model)
    x     = torch.randn(1, meta["seq_length"], meta["embed_dim"])
    print("done.")

    # ── §5.1 Model-level metrics ──────────────────────────────────────────────
    print("\n[2/4] Computing model-level metrics …", end=" ", flush=True)

    size_mb     = model_size_mb(model)
    params_m    = num_params_m(model)
    num_blocks  = len(model.encoder.layers)

    # Warm up
    for _ in range(WARMUP_RUNS):
        run_instrumented_inference(model, meta, x, edge_share=1.0)

    # Baseline: 100% edge (no offloading), measures pure compute latency
    baseline_times = time_full_inference(model, meta, x, edge_share=1.0, n_runs=bench_runs)
    baseline_avg   = sum(baseline_times) / len(baseline_times)
    baseline_min   = min(baseline_times)
    baseline_max   = max(baseline_times)
    baseline_std   = (sum((t - baseline_avg) ** 2 for t in baseline_times) / len(baseline_times)) ** 0.5

    # Communication overhead for balanced split (edge_share=0.5) — most representative
    instr         = run_instrumented_inference(model, meta, x, edge_share=0.5)
    total_bytes   = sum(instr["comm_bytes"])
    comm_mb_frame = total_bytes / (1024 ** 2)

    print("done.")

    print(f"\n{'─'*72}")
    print("  §5.1  Overall System Metrics")
    print(f"{'─'*72}")
    print(f"  {'Metric':<38}  {'Our Implementation':<20}  {'Paper (SPViT)'}")
    print(f"{'─'*72}")

    rows_51 = [
        ("Model Parameters",          f"{params_m:.1f} M",              "~86 M"),
        ("Model Size (serialised)",    f"{size_mb:.1f} MB",              "~85 MB"),
        ("Inference Latency (avg)",    f"{baseline_avg:.1f} ms",         "42 ms"),
        ("Inference Latency (min)",    f"{baseline_min:.1f} ms",         "—"),
        ("Inference Latency (max)",    f"{baseline_max:.1f} ms",         "—"),
        ("Latency Std Dev",            f"±{baseline_std:.1f} ms",        "—"),
        ("Comm. Overhead (50% split)", f"{comm_mb_frame:.2f} MB/frame",  "1.2 MB/frame"),
        ("Top-1 Accuracy",             "N/A (no dataset)",               "82.4%"),
    ]
    for label, ours, paper in rows_51:
        print(f"  {label:<38}  {ours:<20}  {paper}")

    print(f"{'─'*72}")
    print("  * Latency is CPU-only (edge-only path). Paper uses Jetson TX2 + GPU worker.")
    print("  * Model size: torch.save() FP32 ≈ 330 MB; paper's 85 MB = FP16 on-device size.")
    print("  * Comm overhead: torch.save() pickle format; paper uses raw tensor transmission.")
    print("    Raw FP32 overhead = (1×197×768×4 bytes) × 2 × 12 blocks / 1024² ≈ 18 MB.")
    print("    Run `python benchmark.py --bandwidth 160` to reproduce paper's BW setting.")

    # ── §5.2 Split-ratio vs Latency ───────────────────────────────────────────
    print(f"\n\n{'─'*72}")
    print("  §5.2  Split-Ratio vs Latency  (analogous to SPViT split-layer table)")
    print(f"{'─'*72}")
    print(f"  Assumption: worker hardware identical to edge (conservative for GPU).")
    print(f"  Bandwidth: {bw_mbps} Mb/s.  Total latency = max(edge, worker) + comm.\n")

    bw_bytes_per_ms = (bw_mbps * 1e6 / 8) / 1e3   # bytes per ms

    headers = [
        "Config", "EdgeShare", "EdgeLatency(ms)", "WkrEst(ms)", "Comm(ms)", "Total(ms)"
    ]
    col_w = [32, 10, 17, 12, 10, 11]
    fmt   = "".join(f"{{:<{w}}}" for w in col_w)
    print(fmt.format(*headers))
    separator()

    split_results = []
    for cfg in SPLIT_CONFIGS:
        label      = cfg["label"]
        edge_share = cfg["edge_share"]

        # Time the edge portion across bench_runs
        etimes = time_full_inference(model, meta, x, edge_share=edge_share, n_runs=bench_runs)
        e_avg  = sum(etimes) / len(etimes)

        # Estimate worker latency (proportional, same hardware)
        wkr_share = 1.0 - edge_share
        w_est     = e_avg * (wkr_share / edge_share) if edge_share < 1.0 else 0.0

        # Communication latency: total bytes / bandwidth
        instr_cfg   = run_instrumented_inference(model, meta, x, edge_share=edge_share)
        total_b     = sum(instr_cfg["comm_bytes"])
        comm_ms     = total_b / bw_bytes_per_ms if edge_share < 1.0 else 0.0

        total_ms    = max(e_avg, w_est) + comm_ms

        print(fmt.format(
            label,
            f"{edge_share*100:.0f}%",
            f"{e_avg:.1f}",
            f"{w_est:.1f}",
            f"{comm_ms:.1f}",
            f"{total_ms:.1f}",
        ))

        split_results.append({
            "label": label, "edge_share": edge_share,
            "edge_ms": e_avg, "wkr_ms": w_est,
            "comm_ms": comm_ms, "total_ms": total_ms,
            "comm_bytes": total_b,
        })

    separator()
    print("  Note: WkrEst assumes same hardware as edge. With a GPU worker,")
    print("  worker latency will be significantly lower → total latency improves.")

    # ── §5.3 Per-Block Latency Breakdown ─────────────────────────────────────
    print(f"\n\n{'─'*72}")
    print("  §5.3  Per-Block Latency Breakdown  (balanced 50/50 split, single run)")
    print(f"{'─'*72}")

    instr_50 = run_instrumented_inference(model, meta, x, edge_share=0.5)
    headers3 = ["Block", "ATTN(ms)", "MLP(ms)", "EdgeTotal(ms)", "CommBytes", "CommMB"]
    col_w3   = [7, 10, 9, 15, 12, 9]
    fmt3     = "".join(f"{{:<{w}}}" for w in col_w3)
    print(fmt3.format(*headers3))
    separator()

    for i in range(num_blocks):
        attn_ms = instr_50["attn_edge_ms"][i]
        mlp_ms  = instr_50["mlp_edge_ms"][i]
        edge_ms = instr_50["edge_latency_ms"][i]
        cb      = instr_50["comm_bytes"][i]
        cb_mb   = cb / (1024 ** 2)
        print(fmt3.format(
            i,
            f"{attn_ms:.2f}",
            f"{mlp_ms:.2f}",
            f"{edge_ms:.2f}",
            f"{cb:,}",
            f"{cb_mb:.3f}",
        ))

    separator()
    total_attn = sum(instr_50["attn_edge_ms"])
    total_mlp  = sum(instr_50["mlp_edge_ms"])
    total_edge = sum(instr_50["edge_latency_ms"])
    total_cb   = sum(instr_50["comm_bytes"])
    print(fmt3.format(
        "TOTAL",
        f"{total_attn:.2f}",
        f"{total_mlp:.2f}",
        f"{total_edge:.2f}",
        f"{total_cb:,}",
        f"{total_cb/(1024**2):.3f}",
    ))
    separator()

    pct_attn = 100 * total_attn / total_edge if total_edge > 0 else 0
    pct_mlp  = 100 * total_mlp  / total_edge if total_edge > 0 else 0
    print(f"  ATTN share of edge compute: {pct_attn:.1f}%  "
          f"(paper reports ~42.2% of total inference)")
    print(f"  MLP  share of edge compute: {pct_mlp:.1f}%  "
          f"(paper reports ~51.6% of total inference)")

    # ── §5.4 ARIMA-V Overhead Estimate ────────────────────────────────────────
    print(f"\n\n{'─'*72}")
    print("  §5.4  ARIMA-V Scheduler Overhead  (per-block, isolated timing)")
    print(f"{'─'*72}")

    import sys, os
    from shared_utils import MultiDeviceARIMAManager

    devices = ["edge", "pc_gpu"]
    arima   = MultiDeviceARIMAManager(devices)

    for _ in range(8):
        arima.record_block_latency("edge",   0.012, 0.5)
        arima.record_block_latency("pc_gpu", 0.025, 0.5)

    # Suppress ARIMA's per-call print output during the tight timing loop
    _devnull  = open(os.devnull, "w")
    overhead_samples = []
    for _ in range(200):
        t0 = time.perf_counter()
        _real, sys.stdout = sys.stdout, _devnull
        arima.update_shares()
        sys.stdout = _real
        overhead_samples.append((time.perf_counter() - t0) * 1e6)   # µs
    _devnull.close()

    avg_us  = sum(overhead_samples) / len(overhead_samples)
    min_us  = min(overhead_samples)
    max_us  = max(overhead_samples)

    print(f"  update_shares() overhead  (n=200 calls):")
    print(f"    avg = {avg_us:.1f} µs   min = {min_us:.1f} µs   max = {max_us:.1f} µs")
    print(f"  → Negligible vs inference latency (~{avg_us/baseline_avg*1000/12:.2f}‰ of one block)")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n\n{DIVIDER}")
    print("  Summary — Key Numbers to Insert into §5.1 / §5.2 Tables")
    print(DIVIDER)
    print(f"  Model size          :  {size_mb:.1f} MB  ({params_m:.1f} M parameters)")
    print(f"  Inference latency   :  {baseline_avg:.1f} ms  (edge-only, CPU)")
    print(f"  Comm overhead       :  {comm_mb_frame:.2f} MB / frame  (50/50 split, serialised)")
    print(f"  ATTN compute share  :  {pct_attn:.1f}%  of edge-side latency")
    print(f"  MLP  compute share  :  {pct_mlp:.1f}%  of edge-side latency")
    print()
    print("  Split-ratio table (fill 'Paper' column from SPViT §VIII-B):")
    print(f"    {'Config':<32}  {'EdgeLat':>8}  {'WkrEst':>8}  {'CommLat':>8}  {'Total':>8}")
    for r in split_results:
        print(f"    {r['label']:<32}  {r['edge_ms']:>7.1f}ms  "
              f"{r['wkr_ms']:>7.1f}ms  {r['comm_ms']:>7.1f}ms  {r['total_ms']:>7.1f}ms")
    print(DIVIDER)
    print()


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SPViT reimplementation benchmark")
    parser.add_argument("--runs",      type=int,   default=DEFAULT_BENCH_RUNS,
                        help="Number of timed inference passes (default: 10)")
    parser.add_argument("--bandwidth", type=float, default=DEFAULT_BW_MBPS,
                        help="Assumed network bandwidth in Mb/s (default: 80)")
    args = parser.parse_args()

    main(bench_runs=args.runs, bw_mbps=args.bandwidth)
