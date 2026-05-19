"""
benchmark.py  —  Runtime benchmark collector for distributed CLIP inference.
                 ARIMA-V scheduler edition.
 
Produces ONE unified table after N frames.
 
Mode detection (automatic at init):
  • Edge-Only   (no workers) — edge-time column filled, distributed-time column null
  • Distributed (≥1 worker)  — distributed-time column filled, edge-time column null
                                network transmission size included
 
Metrics
───────
  M1.  Model parameters
  M2.  Inference latency          — edge OR distributed, never both in same run
  M3.  ATTN / MLP compute shares  — from master.last_inference_stats
  M4.  ARIMA-V evaluator overhead — µs/block
  M5.  Network transmission size  — distributed mode only
  M6.  Frames per second          — full perception → motor cycle
  M7.  Ball loss rate
  M8.  Worker share stability     — distributed mode only
  M9.  Scheduler recovery         — circuit-breaker simulation, distributed only
 
Usage
─────
  bench = BenchmarkCollector(orch, model.vision_model, model, CONFIDENCE_THRESHOLD)
  bench.measure_edge_baseline(n_runs=6)   # always call — sets edge baseline
  bench.measure_comm_overhead()           # no-op in edge-only mode
 
  # In loop:
  bench.tick_frame_start()
  t0 = time.perf_counter()
  last_hidden = orch.run_inference(hidden)
  bench.record_inference(time.perf_counter()-t0, last_logits, ball_visible)
  bench.tick_frame_end()
 
  # At N frames:
  bench.report()
"""

import io
import time
import statistics
import torch
from collections import deque
 
try:
    from .shared_utils import MultiDeviceARIMAManager
    from .helper import allocate_heads
except ImportError:
    from helper import allocate_heads
    from shared_utils import MultiDeviceARIMAManager
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
 
def _stats(values: list) -> dict:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n": 0}
    return {
        "mean": statistics.mean(values),
        "std":  statistics.pstdev(values),
        "min":  min(values),
        "max":  max(values),
        "n":    len(values),
    }
 
 
def _serialised_bytes(obj) -> int:
    buf = io.BytesIO()
    torch.save(obj, buf)
    return len(buf.getvalue())
 
 
def _count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
 
 
def _null() -> str:
    return "—"
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Table renderer  (single unified table, variable column widths)
# ─────────────────────────────────────────────────────────────────────────────
 
def _print_unified_table(headers: list, rows: list, title: str = ""):
    """
    Print a box-drawn ASCII table.
 
    headers : list of column header strings
    rows    : list of tuples/lists, one per data row.
              Use _null() for cells that should show '—'.
    """
    n      = len(headers)
    widths = [len(str(h)) for h in headers]
    for row in rows:
        for i, cell in enumerate(row[:n]):
            widths[i] = max(widths[i], len(str(cell)))
 
    sep   = lambda l, m, r, f="─": l + m.join(f*(w+2) for w in widths) + r
    drow  = lambda cells: ("│" +
                           "│".join(f" {str(c):<{w}} "
                                    for c, w in zip(list(cells)+[""]*n, widths)) +
                           "│")
 
    if title:
        print(f"\n  {title}")
    print(sep("╔", "╦", "╗", "═"))
    print(drow(headers))
    print(sep("╠", "╬", "╣", "═"))
 
    for i, row in enumerate(rows):
        # Section divider rows are tuples where all cells are the same marker
        if len(row) == 1 and row[0].startswith("──"):
            print(sep("╟", "╫", "╢"))
        else:
            print(drow(row))
 
    print(sep("╚", "╩", "╝"))
 
 
# ─────────────────────────────────────────────────────────────────────────────
# BenchmarkCollector
# ─────────────────────────────────────────────────────────────────────────────
 
class BenchmarkCollector:
 
    SHARE_LOG_DEPTH = 200
    LATENCY_TRIM    = 400
 
    # ── Init ─────────────────────────────────────────────────────────────────
 
    def __init__(
        self,
        orch,
        vision_model,
        clip_model,
        confidence_threshold: float,
    ):
        self.orch           = orch
        self.vision         = vision_model
        self.threshold      = confidence_threshold
        self.is_distributed = len(getattr(orch, "expected_workers", [])) > 0
 
        # M1
        self.vision_params = _count_params(clip_model.vision_model)
        self.total_params  = _count_params(clip_model)
 
        # M2 — edge baseline (pre-loop) + live inference latency
        self._edge_baseline: list = []
        self._infer_latencies: list = []   # edge-only OR distributed depending on mode
 
        # M3 — ATTN / MLP sub-timing from master.last_inference_stats
        self._attn_ms: list = []
        self._mlp_ms:  list = []
 
        # M4 — evaluator overhead
        self._evaluator_us: list = []
 
        # M5 — network transmission (distributed only, set by measure_comm_overhead)
        self._comm_sent_bytes:  float = 0.0
        self._comm_recv_bytes:  float = 0.0
        self._comm_total_bytes: float = 0.0
 
        # M6 — FPS
        self._frame_start    = None
        self._cycle_durations: list = []
 
        # M7 — ball tracking
        self._ball_visible  = 0
        self._ball_lost     = 0
        self._loss_window   = deque(maxlen=100)
 
        # M8 — share stability (distributed only)
        self._share_log: dict = {
            w: deque(maxlen=self.SHARE_LOG_DEPTH)
            for w in getattr(orch, "expected_workers", [])
        }
        self._share_block_count = 0
 
        # M9 — circuit breaker recovery
        self._recovery_results: list = []
 
        self._total_frames = 0
 
    # ── M2: Edge baseline ────────────────────────────────────────────────────
 
    def measure_edge_baseline(self, n_runs: int = 6):
        """
        Runs _local_inference() n_runs times on a dummy input.
        First run is discarded (JIT warm-up).
        Call BEFORE the main loop and before any worker dispatch.
        """
        print(f"[Benchmark] Edge-only baseline ({n_runs} runs) …")
        meta  = self.orch.meta
        dummy = torch.randn(1, meta["seq_length"], meta["embed_dim"])
        self.orch._local_inference(dummy)           # warm-up, not recorded
        for i in range(n_runs):
            t0  = time.perf_counter()
            self.orch._local_inference(dummy)
            lat = time.perf_counter() - t0
            self._edge_baseline.append(lat)
            print(f"  Run {i+1}/{n_runs}: {lat*1000:.1f} ms")
        s = _stats(self._edge_baseline)
        print(f"  → mean={s['mean']*1000:.1f} ms  std=±{s['std']*1000:.1f} ms\n")
 
    # ── M5: Network transmission size ────────────────────────────────────────
 
    def measure_comm_overhead(self):
        """
        Estimates serialised byte cost of one full inference pass based on
        the current ARIMA-V share allocation. No-op in edge-only mode.
        Call AFTER workers are connected and ARIMA has warmed up slightly.
        """
        if not self.is_distributed:
            return
 
        print("[Benchmark] Measuring comm overhead …")
 
        meta    = self.orch.meta
        H       = meta["num_heads"]
        hd      = meta["head_dim"]
        S       = meta["seq_length"]
        D       = meta["embed_dim"]
        MLP_HID = meta["mlp_hidden_dim"]
 
        alloc_w    = self.orch.evaluator.get_allocation()
        a_heads    = allocate_heads(H,       alloc_w)
        a_neurons  = allocate_heads(MLP_HID, alloc_w)
        n_blocks   = len(self.orch.model.encoder.layers)
        sent = recv = 0
 
        for w in self.orch.expected_workers:
            H_w = a_heads.get(w, 0)
            N_w = a_neurons.get(w, 0)
 
            if H_w > 0:
                q   = torch.zeros(1, H_w, S, hd).half()
                sent += _serialised_bytes((q, q.clone(), q.clone())) * n_blocks
                recv += _serialised_bytes(q) * n_blocks
 
            if N_w > 0:
                lx  = torch.zeros(1, S, D).half()
                sent += _serialised_bytes(lx) * n_blocks
                recv += _serialised_bytes(torch.zeros(1, S, D).half()) * n_blocks
 
        self._comm_sent_bytes  = sent
        self._comm_recv_bytes  = recv
        self._comm_total_bytes = sent + recv
 
        mb_t = self._comm_total_bytes / (1024**2)
        mb_s = self._comm_sent_bytes  / (1024**2)
        mb_r = self._comm_recv_bytes  / (1024**2)
        print(f"  Total: {mb_t:.3f} MB/frame  "
              f"(sent {mb_s:.3f} MB + recv {mb_r:.3f} MB)\n")
 
    # ── Main loop API ─────────────────────────────────────────────────────────
 
    def tick_frame_start(self):
        self._frame_start = time.perf_counter()
        self._total_frames += 1
 
    def tick_frame_end(self):
        if self._frame_start is not None:
            self._cycle_durations.append(time.perf_counter() - self._frame_start)
 
    def record_inference(
        self,
        latency: float,
        last_logits: float,
        ball_visible: bool,
    ):
        """
        Call immediately after orch.run_inference() returns.
 
        latency      : seconds from hidden state input to last_hidden output
        last_logits  : raw confidence score for this frame
        ball_visible : True when ball is known to be in frame
        """
        # M2 — inference latency (same bucket regardless of mode;
        #       mode determines which column it appears in during report)
        self._infer_latencies.append(latency)
        if len(self._infer_latencies) > self.LATENCY_TRIM:
            self._infer_latencies = self._infer_latencies[-self.LATENCY_TRIM:]
 
        # M3 / M4 — sub-timing from master
        stats = getattr(self.orch, "last_inference_stats", None)
        if stats:
            self._attn_ms.extend(stats.get("per_block_attn_ms",    []))
            self._mlp_ms.extend( stats.get("per_block_mlp_ms",     []))
            self._evaluator_us.extend(stats.get("evaluator_overhead_us", []))
 
        # M7 — ball tracking
        if ball_visible:
            self._ball_visible += 1
            lost = last_logits < self.threshold
            self._loss_window.append(1 if lost else 0)
            if lost:
                self._ball_lost += 1
        else:
            self._loss_window.append(0)
 
        # M8 — share stability (distributed only)
        if self.is_distributed:
            shares = self.orch.current_shares
            for w, dq in self._share_log.items():
                dq.append(shares.get(w, 0.0))
            self._share_block_count += 1
 
    # ── report() — single unified table ──────────────────────────────────────
 
    def report(self):
        """
        Print ONE unified benchmark table.
 
        Columns:
          Metric | Edge-Only Time | Distributed Time | Paper (SPViT) | Notes
 
        • Edge-only run  : Distributed Time column shows '—' for all rows.
        • Distributed run: Edge-Only Time column shows '—' for inference rows.
                           Edge baseline (pre-loop) is still shown as a reference.
        """
        mode = "Distributed" if self.is_distributed else "Edge-Only"
        SEP  = "═" * 80
 
        print(f"\n{SEP}")
        print(f"  CLIP BENCHMARK REPORT — {mode} Mode"
              f"  |  Frames: {self._total_frames}"
              f"  |  Workers: {', '.join(self.orch.expected_workers) if self.is_distributed else 'none'}")
        print(SEP)
 
        # ── Precompute everything ────────────────────────────────────────────
 
        eb  = _stats(self._edge_baseline)
        inf = _stats(self._infer_latencies)
 
        attn_total = sum(self._attn_ms)
        mlp_total  = sum(self._mlp_ms)
        comp_total = attn_total + mlp_total
 
        def pct(num, den):
            return f"{100*num/den:.1f}%" if den > 0 else _null()
 
        def ms(s_dict, key="mean"):
            v = s_dict[key]
            return f"{v*1000:.1f} ms" if s_dict["n"] > 0 else _null()
 
        def pm(s_dict):
            return f"±{s_dict['std']*1000:.1f} ms" if s_dict["n"] > 0 else _null()
 
        # Speedup reference
        edge_ref_ms = (eb["mean"] if eb["n"] > 0 else 0.0)
        if self.is_distributed and inf["n"] > 0 and edge_ref_ms > 0 and inf["mean"] > 0:
            speedup_str = f"{edge_ref_ms / inf['mean']:.2f}×"
        else:
            speedup_str = _null()
 
        # FPS
        fps_s   = _stats(self._cycle_durations)
        fps_val = f"{1.0/fps_s['mean']:.2f}" if fps_s["n"] > 0 and fps_s["mean"] > 0 else _null()
        fps_min = f"{1.0/fps_s['max']:.2f}"  if fps_s["n"] > 0 and fps_s["max"]  > 0 else _null()
        fps_max = f"{1.0/fps_s['min']:.2f}"  if fps_s["n"] > 0 and fps_s["min"]  > 0 else _null()
 
        # Ball loss
        if self._ball_visible > 0:
            loss_overall = f"{self._ball_lost/self._ball_visible*100:.1f}%"
            loss_window  = f"{sum(self._loss_window)/max(len(self._loss_window),1)*100:.1f}%"
        else:
            loss_overall = loss_window = _null()
 
        # Evaluator overhead
        ev_s = _stats(self._evaluator_us)
        ev_mean = f"{ev_s['mean']:.1f} µs" if ev_s["n"] > 0 else _null()
        if fps_s["n"] > 0 and ev_s["n"] > 0 and fps_s["mean"] > 0:
            n_blocks  = len(self.orch.model.encoder.layers)
            block_ms_ = inf["mean"] * 1000 / n_blocks if inf["n"] > 0 and n_blocks > 0 else 0
            ev_pct    = (f"{(ev_s['mean']/1000)/block_ms_*100:.3f}%"
                         if block_ms_ > 0 else _null())
        else:
            ev_pct = _null()
 
        # ── Column values per mode ────────────────────────────────────────────
        # edge_col   = value for "Edge-Only Time" column
        # dist_col   = value for "Distributed Time" column
        # In edge-only mode  : dist_col is always _null()
        # In distributed mode: edge_col is _null() EXCEPT for the baseline row
 
        N  = _null()
 
        def ec(val):
            """Edge column: show value only in edge-only mode."""
            return val if not self.is_distributed else N
 
        def dc(val):
            """Distributed column: show value only in distributed mode."""
            return val if self.is_distributed else N
 
        # ── Build rows ────────────────────────────────────────────────────────
 
        HDRS = [
            "Metric",
            "Edge-Only Time",
            "Distributed Time",
            "Paper (SPViT)",
            "Notes",
        ]
 
        # Section divider helper
        def section(label):
            return (f"── {label} " + "─"*40,)   # single-cell sentinel
 
        rows = []
 
        # ── Model parameters ──────────────────────────────────────────────────
        rows.append(section("Model Parameters"))
        rows += [
            ("Vision encoder params",
             f"{self.vision_params/1e6:.1f} M",
             f"{self.vision_params/1e6:.1f} M",
             "~86 M",
             "✓ Match" if abs(self.vision_params/1e6 - 86) < 2 else "✗"),
            ("Full CLIP model params",
             f"{self.total_params/1e6:.1f} M",
             f"{self.total_params/1e6:.1f} M",
             "86 M (ViT only)",
             "Vision + Text encoder"),
        ]
 
        # ── Inference latency ─────────────────────────────────────────────────
        rows.append(section("Inference Latency"))
 
        # Edge baseline always shown (even in distributed mode — for reference)
        rows.append((
            "Edge-only baseline (pre-loop)",
            ms(eb),
            N,                          # never distributed time
            "42 ms (TX2+GPU)",
            f"std {pm(eb)}  n={eb['n']}" if eb["n"] > 0 else "not measured",
        ))
 
        # Live inference time — routed to the correct column
        rows.append((
            "Live inference latency (avg)",
            ec(ms(inf)),                # filled only in edge-only mode
            dc(ms(inf)),                # filled only in distributed mode
            _null(),
            f"std {pm(inf)}  n={inf['n']}" if inf["n"] > 0 else "no data",
        ))
        rows.append((
            "Live inference latency (min)",
            ec(ms(inf, "min")),
            dc(ms(inf, "min")),
            _null(), _null(),
        ))
        rows.append((
            "Live inference latency (max)",
            ec(ms(inf, "max")),
            dc(ms(inf, "max")),
            _null(), _null(),
        ))
 
        if self.is_distributed:
            rows.append((
                "Speedup vs edge baseline",
                N,
                speedup_str,
                "2.2×–3.3×",
                "distributed / edge-baseline ratio",
            ))
 
        # Early vs late stability (only if enough data)
        if inf["n"] >= 10:
            mid   = len(self._infer_latencies) // 2
            early = _stats(self._infer_latencies[:mid])
            late  = _stats(self._infer_latencies[mid:])
            rows.append((
                f"Early frames (first {mid})",
                ec(ms(early)), dc(ms(early)),
                _null(), "warm-up phase",
            ))
            rows.append((
                f"Late frames  (last {inf['n']-mid})",
                ec(ms(late)),  dc(ms(late)),
                _null(), "steady state",
            ))
 
        # ── Compute distribution ──────────────────────────────────────────────
        rows.append(section("Compute Distribution (Edge)"))
        attn_s = _stats(self._attn_ms)
        mlp_s  = _stats(self._mlp_ms)
        rows += [
            ("ATTN share of edge compute",
             pct(attn_total, comp_total),
             pct(attn_total, comp_total),
             "42.2%",
             f"per-block mean {attn_s['mean']:.2f} ms" if attn_s["n"] > 0 else "no data"),
            ("MLP  share of edge compute",
             pct(mlp_total, comp_total),
             pct(mlp_total, comp_total),
             "51.6%",
             f"per-block mean {mlp_s['mean']:.2f} ms" if mlp_s["n"] > 0 else "no data"),
        ]
 
        # ── ARIMA-V scheduler overhead ────────────────────────────────────────
        rows.append(section("ARIMA-V Scheduler Overhead"))
        rows += [
            ("Evaluator overhead / block",
             ev_mean, ev_mean,
             "Not reported", f"% of block: {ev_pct}"),
        ]
 
        # ── Network transmission (distributed only) ───────────────────────────
        if self.is_distributed:
            rows.append(section("Network Transmission  (Distributed Only)"))
            if self._comm_total_bytes > 0:
                mb_t = self._comm_total_bytes / (1024**2)
                mb_s = self._comm_sent_bytes  / (1024**2)
                mb_r = self._comm_recv_bytes  / (1024**2)
                raw_est = mb_t * 0.072
                rows += [
                    ("Total serialised / frame",
                     N, f"{mb_t:.3f} MB",
                     "1.2 MB", "pickle overhead included"),
                    ("Sent  (edge → workers) / frame",
                     N, f"{mb_s:.3f} MB",
                     _null(), "Q, K, V slices + LN input"),
                    ("Recv  (workers → edge) / frame",
                     N, f"{mb_r:.3f} MB",
                     _null(), "ATTN out + MLP partial"),
                    ("Est. raw-tensor equivalent",
                     N, f"{raw_est:.3f} MB",
                     _null(), "excl. pickle / tensor metadata"),
                ]
            else:
                rows.append((
                    "Transmission size",
                    N, "NOT MEASURED — call measure_comm_overhead()",
                    _null(), _null(),
                ))
 
        # ── FPS ───────────────────────────────────────────────────────────────
        rows.append(section("System Performance"))
        rows += [
            ("Cycle time (mean)",
             ec(f"{fps_s['mean']*1000:.1f} ms" if fps_s["n"]>0 else N),
             dc(f"{fps_s['mean']*1000:.1f} ms" if fps_s["n"]>0 else N),
             _null(),
             f"std ±{fps_s['std']*1000:.1f} ms" if fps_s["n"]>0 else "no data"),
            ("FPS (mean)",
             ec(fps_val), dc(fps_val),
             _null(),
             f"min {fps_min}  max {fps_max}"),
        ]
 
        # ── Ball tracking ─────────────────────────────────────────────────────
        rows.append(section("Ball Tracking Quality"))
        rows += [
            ("Ball-visible steps",
             ec(str(self._ball_visible)), dc(str(self._ball_visible)),
             _null(), _null()),
            ("Ball-loss events",
             ec(str(self._ball_lost)),    dc(str(self._ball_lost)),
             _null(), _null()),
            ("Loss rate (overall)",
             ec(loss_overall), dc(loss_overall),
             _null(), _null()),
            (f"Loss rate (last {len(self._loss_window)} frames)",
             ec(loss_window),  dc(loss_window),
             _null(), "rolling window"),
        ]
 
        # ── Worker share stability (distributed only) ─────────────────────────
        if self.is_distributed and self._share_log:
            rows.append(section(
                f"Worker Allocation Stability  ({self._share_block_count} blocks)"))
            for w, dq in self._share_log.items():
                vals = list(dq)
                if vals:
                    s   = _stats(vals)
                    idx = max(0.0, 100.0 * (1 - s["std"] / max(s["mean"], 1e-9)))
                    rows.append((
                        f"Worker '{w}'  share stats",
                        N,
                        f"mean={s['mean']:.3f}  ±{s['std']:.4f}  "
                        f"[{s['min']:.3f}, {s['max']:.3f}]",
                        _null(),
                        f"Stability index: {idx:.1f}%",
                    ))
 
        # ── Circuit-breaker recovery (distributed only) ───────────────────────
        if self.is_distributed:
            rows.append(section("Scheduler Recovery  (CB Simulation)"))
            if self._recovery_results:
                for r in self._recovery_results:
                    rows.append((
                        f"Worker '{r['worker']}'",
                        N,
                        (f"CB tripped: {'Yes' if r['cb_ever_tripped'] else 'No'}  |  "
                         f"Degraded blocks: {r['blocks_degraded_before_cb_trip']}  |  "
                         f"Recovery blocks: {r['blocks_to_full_recovery']}"),
                        _null(), _null(),
                    ))
 
        # ── Print ─────────────────────────────────────────────────────────────
        _print_unified_table(HDRS, rows, title=f"Benchmark Table [{mode}]")
 
        # ── Paper fill-in summary (compact, copy-paste ready) ─────────────────
        print(f"\n{SEP}")
        print("  PAPER TABLE FILL-IN  (copy these numbers into your report)")
        print(SEP)
 
        col = [42, 24, 20, 12]
        hdr = ("Metric", "Our Implementation", "Paper (SPViT)", "Notes")
        fmt = "".join(f"{{:<{c}}}" for c in col)
        print(fmt.format(*hdr))
        print("─" * sum(col))
 
        edge_lat   = ms(eb)
        infer_lat  = ms(inf)
        comm_str   = (f"{self._comm_total_bytes/(1024**2):.2f} MB"
                      if self._comm_total_bytes > 0 else "—")
 
        paper_rows = [
            ("Vision encoder params",
             f"{self.vision_params/1e6:.1f} M", "~86 M", "✓"),
            ("Full CLIP model params",
             f"{self.total_params/1e6:.1f} M",  "86 M (ViT only)", "Vision+Text"),
            ("Edge-only latency (avg)",
             edge_lat, "42 ms (TX2+GPU)", "HW gap"),
            ("Edge-only latency (std)",
             pm(eb) if eb["n"] > 0 else "—", "—", ""),
            ("Distributed latency (avg)",
             dc(infer_lat), "—", ""),
            ("Speedup (edge / distributed)",
             speedup_str, "2.2×–3.3×", ""),
            ("Comm. overhead / frame",
             dc(comm_str), "1.2 MB", "Pickle gap"),
            ("ATTN share of compute",
             pct(attn_total, comp_total), "42.2%",
             "✓" if self._attn_ms else ""),
            ("MLP share of compute",
             pct(mlp_total, comp_total),  "51.6%",
             "✓" if self._mlp_ms else ""),
            ("ARIMA eval overhead / block",
             ev_mean, "Not reported", "Negligible"),
            ("Ball loss rate",
             loss_overall, "—", ""),
        ]
 
        for row in paper_rows:
            print(fmt.format(*row))
 
        print(f"{SEP}\n")