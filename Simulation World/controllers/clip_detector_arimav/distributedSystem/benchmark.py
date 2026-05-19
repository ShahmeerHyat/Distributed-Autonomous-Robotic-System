"""
benchmark.py  —  Runtime benchmark collector for distributed CLIP inference.
                 Adapted for the ARIMA-V scheduler.

Automatically detects whether workers are connected and adjusts what it measures:
  • Edge-Only mode  (no workers) — reports edge compute time only.
  • Distributed mode (≥1 worker) — reports edge baseline, distributed compute
                                    time, and network transmission size.

Metrics collected
─────────────────
  M1.  Model parameters
  M2.  Edge compute time          — baseline (pre-loop) + live (main loop)
  M3.  Distributed compute time   — distributed mode only
  M4.  Network transmission size  — serialised bytes sent / received per frame
                                    (distributed mode only)
  M5.  ATTN share of edge compute
  M6.  MLP  share of edge compute
  M7.  ARIMA evaluator overhead   — update_shares() per block (µs)
  M8.  Frames Per Second
  M9.  Ball loss rate
  M10. Worker share stability     — distributed mode only
  M11. Scheduler recovery         — circuit breaker simulation (distributed only)

Usage
─────
  from distributedSystem.benchmark import BenchmarkCollector

  bench = BenchmarkCollector(
      orch                  = orch,
      vision_model          = model.vision_model,
      clip_model            = model,
      confidence_threshold  = CONFIDENCE_THRESHOLD,
  )

  bench.measure_edge_baseline(n_runs=6)
  bench.measure_comm_overhead()   # no-op in edge-only mode

  # Inside the main loop:
  bench.tick_frame_start()
  ...
  t0 = time.perf_counter()
  last_hidden = orch.run_inference(hidden)
  bench.record_inference(
      latency      = time.perf_counter() - t0,
      last_logits  = last_logits,
      ball_visible = (last_logits > CONFIDENCE_THRESHOLD),
  )
  ...
  bench.tick_frame_end()

  if step_count == BENCHMARK_AFTER_N_FRAMES:
      bench.simulate_worker_failure(worker_name, dummy_tensor)
      bench.report()
"""

import io
import time
import statistics
import torch
from collections import deque

try:
    from .shared_utils import allocate_heads
except ImportError:
    from shared_utils import allocate_heads


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
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

def _serialised_bytes(tensor) -> int:
    buf = io.BytesIO()
    torch.save(tensor, buf)
    return len(buf.getvalue())

def _count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())

def _fmt_table(title: str, headers: list, rows: list):
    """Print a box-drawn ASCII table to stdout."""
    if not rows:
        print(f"\n┌─ {title}")
        print("│  (no data)")
        print("└" + "─" * 40)
        return

    n = len(headers)
    widths = [len(str(h)) for h in headers]
    for row in rows:
        for i in range(min(n, len(row))):
            widths[i] = max(widths[i], len(str(row[i])))

    def hline(left, mid, right, fill="─"):
        return left + mid.join(fill * (w + 2) for w in widths) + right

    def drow(cells):
        padded = list(cells) + [""] * max(0, n - len(cells))
        return "│" + "│".join(f" {str(c):<{w}} " for c, w in zip(padded, widths)) + "│"

    print(f"\n┌─ {title}")
    print(hline("├", "┬", "┤"))
    print(drow(headers))
    print(hline("├", "┼", "┤"))
    for row in rows:
        print(drow(row))
    print(hline("└", "┴", "┘"))


# ─────────────────────────────────────────────────────────────────────────────
# BenchmarkCollector
# ─────────────────────────────────────────────────────────────────────────────

class BenchmarkCollector:

    SHARE_LOG_DEPTH = 200
    LATENCY_TRIM    = 400

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
        self.is_distributed = len(orch.expected_workers) > 0

        # M1: Model parameters
        self.vision_params = _count_params(clip_model.vision_model)
        self.total_params  = _count_params(clip_model)

        # M2: Edge compute time
        self._edge_baseline_latencies: list = []   # from measure_edge_baseline()
        self._edge_live_latencies: list     = []   # from record_inference() edge-only

        # M3: Distributed compute time (distributed mode only)
        self._dist_latencies: list = []

        # M4: Network transmission size (distributed mode only)
        self._comm_bytes_sent:  float = 0.0   # edge → workers, per frame
        self._comm_bytes_recv:  float = 0.0   # workers → edge, per frame
        self._comm_bytes_total: float = 0.0   # sent + recv, per frame

        # M5 / M6: ATTN & MLP sub-timing
        self._attn_ms_all: list = []
        self._mlp_ms_all:  list = []

        # M7: ARIMA evaluator overhead
        self._evaluator_overhead_us: list = []

        # M8: FPS
        self._frame_start = None
        self._cycle_durations: list = []

        # M9: Ball loss rate
        self._ball_visible_steps = 0
        self._ball_loss_events   = 0
        self._loss_window        = deque(maxlen=100)

        # M10: Worker share stability (distributed mode only)
        self._share_log = {
            w: deque(maxlen=self.SHARE_LOG_DEPTH)
            for w in orch.expected_workers
        }
        self._share_block_count = 0

        # M11: Scheduler recovery
        self._recovery_results: list = []

        self._total_frames = 0

    # ─────────────────────────────────────────────────────────────────────────
    # M2: Edge-only baseline  (call ONCE before the main loop)
    # ─────────────────────────────────────────────────────────────────────────

    def measure_edge_baseline(self, n_runs: int = 6):
        """
        Run _local_inference() n_runs times on a random input and record
        wall-clock latency. First run discarded for JIT/cache warm-up.
        Call before any worker dispatch so timing is pure edge compute.
        """
        print(f"[Benchmark] Measuring edge-only baseline ({n_runs} runs) …")
        meta  = self.orch.meta
        dummy = torch.randn(1, meta["seq_length"], meta["embed_dim"])

        self.orch._local_inference(dummy)   # warm-up — not recorded

        for i in range(n_runs):
            t0  = time.perf_counter()
            self.orch._local_inference(dummy)
            lat = time.perf_counter() - t0
            self._edge_baseline_latencies.append(lat)
            print(f"  Edge run {i+1}/{n_runs}: {lat*1000:.1f} ms")

        s = _stats(self._edge_baseline_latencies)
        print(f"  Edge baseline → mean={s['mean']*1000:.1f}ms  "
              f"std=±{s['std']*1000:.1f}ms\n")

    # ─────────────────────────────────────────────────────────────────────────
    # M4: Network transmission size  (call ONCE before the main loop)
    # ─────────────────────────────────────────────────────────────────────────

    def measure_comm_overhead(self):
        """
        Compute serialised byte cost of one full distributed inference pass,
        split into bytes sent (edge→workers) and bytes received (workers→edge).
        No-op in edge-only mode.
        """
        if not self.is_distributed:
            return

        print("[Benchmark] Measuring communication overhead …")

        meta    = self.orch.meta
        H       = meta["num_heads"]
        hd      = meta["head_dim"]
        S       = meta["seq_length"]
        D       = meta["embed_dim"]
        MLP_HID = meta["mlp_hidden_dim"]

        alloc_weights = self.orch.arima.current_shares   # ARIMA-V snapshot
        alloc_heads   = allocate_heads(H, alloc_weights)
        alloc_mlp     = allocate_heads(MLP_HID, alloc_weights)

        total_sent = 0
        total_recv = 0
        num_blocks = len(self.orch.model.encoder.layers)

        for w_name in self.orch.expected_workers:
            H_w = alloc_heads.get(w_name, 0)
            N_w = alloc_mlp.get(w_name, 0)

            if H_w > 0:
                q_s       = torch.zeros(1, H_w, S, hd).half()
                sent_attn = _serialised_bytes((q_s, q_s.clone(), q_s.clone()))
                recv_attn = _serialised_bytes(q_s.half())
                total_sent += sent_attn * num_blocks
                total_recv += recv_attn * num_blocks

            if N_w > 0:
                ln_x     = torch.zeros(1, S, D).half()
                sent_mlp = _serialised_bytes(ln_x)
                recv_mlp = _serialised_bytes(torch.zeros(1, S, D).half())
                total_sent += sent_mlp * num_blocks
                total_recv += recv_mlp * num_blocks

        self._comm_bytes_sent  = total_sent
        self._comm_bytes_recv  = total_recv
        self._comm_bytes_total = total_sent + total_recv

        mb_t = self._comm_bytes_total / (1024 ** 2)
        mb_s = self._comm_bytes_sent  / (1024 ** 2)
        mb_r = self._comm_bytes_recv  / (1024 ** 2)
        print(f"  Comm overhead: {self._comm_bytes_total:,} B = {mb_t:.3f} MB / frame")
        print(f"    Sent (edge→workers): {mb_s:.3f} MB   "
              f"Recv (workers→edge): {mb_r:.3f} MB\n")

    # ─────────────────────────────────────────────────────────────────────────
    # Main loop push API
    # ─────────────────────────────────────────────────────────────────────────

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

        Parameters
        ----------
        latency      : seconds from hidden-state input to last_hidden output
        last_logits  : confidence score for this frame
        ball_visible : True when the ball is known to be in frame
        """
        # M2 / M3: route latency to the right bucket
        if self.is_distributed:
            self._dist_latencies.append(latency)
            if len(self._dist_latencies) > self.LATENCY_TRIM:
                self._dist_latencies = self._dist_latencies[-self.LATENCY_TRIM:]
        else:
            self._edge_live_latencies.append(latency)
            if len(self._edge_live_latencies) > self.LATENCY_TRIM:
                self._edge_live_latencies = self._edge_live_latencies[-self.LATENCY_TRIM:]

        # M5 / M6 / M7: per-block breakdown from master stats
        stats = getattr(self.orch, "last_inference_stats", None)
        if stats:
            self._attn_ms_all.extend(stats.get("per_block_attn_ms", []))
            self._mlp_ms_all.extend(stats.get("per_block_mlp_ms", []))
            self._evaluator_overhead_us.extend(stats.get("evaluator_overhead_us", []))

        # M9: ball loss rate
        if ball_visible:
            self._ball_visible_steps += 1
            lost = last_logits < self.threshold
            self._loss_window.append(1 if lost else 0)
            if lost:
                self._ball_loss_events += 1
        else:
            self._loss_window.append(0)

        # M10: share stability
        shares = self.orch.current_shares
        for w, dq in self._share_log.items():
            dq.append(shares.get(w, 0.0))
        self._share_block_count += 1

    # ─────────────────────────────────────────────────────────────────────────
    # M11: Scheduler recovery test  (run explicitly, outside main loop)
    # ─────────────────────────────────────────────────────────────────────────

    def simulate_worker_failure(
        self,
        worker_name: str,
        dummy_input: torch.Tensor,
        recovery_delay: float = 10.0,
    ) -> dict:
        """
        Simulate a worker crash by injecting high latency into ARIMA history
        and observing circuit breaker behaviour, then measure recovery time.
        No-op in edge-only mode.
        """
        if not self.is_distributed:
            return {"error": "No workers in edge-only mode — skipped."}

        FAIL_LAT = 999.0
        arima    = self.orch.arima
        breaker  = arima.breakers.get(worker_name)

        if breaker is None:
            return {"error": f"No circuit breaker for worker '{worker_name}'"}

        result = {
            "worker":                         worker_name,
            "cb_ever_tripped":                False,
            "blocks_degraded_before_cb_trip": 0,
            "blocks_to_full_recovery":        0,
        }

        print(f"[Benchmark] Simulating '{worker_name}' failure …")

        # Phase 1: inject failure latency until CB trips
        degraded = 0
        for _ in range(50):
            arima.record_block_latency(worker_name, latency=FAIL_LAT, share_used=0.5)
            arima.update_shares()
            if breaker.is_open:
                result["cb_ever_tripped"] = True
                break
            if arima.current_shares.get(worker_name, 0.0) > 0:
                degraded += 1

        result["blocks_degraded_before_cb_trip"] = degraded
        print(f"  CB tripped after {degraded} degraded block(s).")

        # Phase 2: fast-forward cooldown to HALF_OPEN
        while breaker.is_open:
            breaker.tick()
        print("  CB is now HALF_OPEN. Probing recovery …")

        # Phase 3: inject healthy latency until probe clears
        GOOD_LAT        = 0.01
        recovery_blocks = 0
        for _ in range(30):
            arima.notify_probe_result(worker_name, GOOD_LAT)
            recovery_blocks += 1
            if breaker.is_closed:
                break

        result["blocks_to_full_recovery"] = recovery_blocks
        share_now = arima.current_shares.get(worker_name, 0.0)
        print(f"  Recovery in {recovery_blocks} block(s). "
              f"Share restored to {share_now:.2f}\n")

        self._recovery_results.append(result)
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # report()  —  print all benchmarks as ASCII tables
    # ─────────────────────────────────────────────────────────────────────────

    def report(self):
        mode = "Distributed" if self.is_distributed else "Edge-Only"
        SEP  = "═" * 72

        print(f"\n{SEP}")
        print(f"  CLIP BENCHMARK REPORT  (ARIMA-V scheduler) — {mode} Mode")
        print(f"  Total frames observed: {self._total_frames}")
        print(f"{SEP}")

        # ── Table 1: Model Parameters ─────────────────────────────────────
        match_v = "✓ Match" if abs(self.vision_params/1e6 - 86) < 2 else "✗ (paper: ~86 M)"
        _fmt_table(
            "Table 1: Model Parameters",
            ["Component", "Parameters", "Notes"],
            [
                ("Vision encoder",  f"{self.vision_params/1e6:.1f} M", match_v),
                ("Full CLIP model", f"{self.total_params/1e6:.1f} M",  "Vision + Text"),
            ],
        )

        # ── Table 2: Edge Compute Time ────────────────────────────────────
        eb = _stats(self._edge_baseline_latencies)
        el = _stats(self._edge_live_latencies)

        attn_total = sum(self._attn_ms_all)
        mlp_total  = sum(self._mlp_ms_all)
        comp_total = attn_total + mlp_total

        def pct(num, den):
            return f"{100*num/den:.1f}%" if den > 0 else "N/A"

        edge_rows = []
        if eb["n"] > 0:
            edge_rows.append((
                "Baseline latency (pre-loop)",
                f"{eb['mean']*1000:.1f} ms",
                f"±{eb['std']*1000:.1f} ms",
                f"{eb['min']*1000:.1f} ms",
                f"{eb['max']*1000:.1f} ms",
                str(eb["n"]),
            ))
        if el["n"] > 0:
            edge_rows.append((
                "Live latency (main loop)",
                f"{el['mean']*1000:.1f} ms",
                f"±{el['std']*1000:.1f} ms",
                f"{el['min']*1000:.1f} ms",
                f"{el['max']*1000:.1f} ms",
                str(el["n"]),
            ))
        if self._attn_ms_all:
            attn_s = _stats(self._attn_ms_all)
            mlp_s  = _stats(self._mlp_ms_all)
            edge_rows += [
                ("ATTN share of compute",
                 pct(attn_total, comp_total), "paper: 42.2%", "—", "—", "—"),
                ("MLP  share of compute",
                 pct(mlp_total, comp_total),  "paper: 51.6%", "—", "—", "—"),
                ("Per-block ATTN time",
                 f"{attn_s['mean']:.2f} ms", f"±{attn_s['std']:.2f} ms",
                 "—", "—", str(attn_s["n"])),
                ("Per-block MLP  time",
                 f"{mlp_s['mean']:.2f} ms",  f"±{mlp_s['std']:.2f} ms",
                 "—", "—", str(mlp_s["n"])),
            ]

        _fmt_table(
            "Table 2: Edge Compute Time",
            ["Metric", "Mean / Value", "Std / Notes", "Min", "Max", "N"],
            edge_rows if edge_rows else [("(no data — call measure_edge_baseline())", "", "", "", "", "")],
        )

        # ── Table 3: Distributed Compute Time (distributed mode only) ─────
        if self.is_distributed:
            d = _stats(self._dist_latencies)
            if d["n"] > 0:
                mid   = len(self._dist_latencies) // 2
                early = _stats(self._dist_latencies[:mid])
                late  = _stats(self._dist_latencies[mid:])

                e_mean = eb["mean"] if eb["n"] > 0 else (el["mean"] if el["n"] > 0 else 0.0)
                speedup = (f"{e_mean/d['mean']:.2f}×"
                           if d["mean"] > 0 and e_mean > 0 else "N/A")

                dist_rows = [
                    ("Inference latency (overall)",
                     f"{d['mean']*1000:.1f} ms",
                     f"±{d['std']*1000:.1f} ms",
                     f"{d['min']*1000:.1f} ms",
                     f"{d['max']*1000:.1f} ms",
                     str(d["n"])),
                    (f"Early frames (first {mid})",
                     f"{early['mean']*1000:.1f} ms", "—", "—", "—", str(early["n"])),
                    (f"Late  frames (last {d['n']-mid})",
                     f"{late['mean']*1000:.1f} ms",  "—", "—", "—", str(late["n"])),
                    ("Speedup vs edge baseline",
                     speedup, "paper: 2.2×–3.3×", "—", "—", "—"),
                ]
                _fmt_table(
                    "Table 3: Distributed Compute Time",
                    ["Metric", "Mean / Value", "Std / Notes", "Min", "Max", "N"],
                    dist_rows,
                )
            else:
                print("\n┌─ Table 3: Distributed Compute Time")
                print("│  No distributed latency data collected.")
                print("└" + "─" * 40)

        # ── Table 4: Network Transmission Size (distributed mode only) ────
        if self.is_distributed:
            if self._comm_bytes_total > 0:
                fps_mean = 0.0
                if self._cycle_durations:
                    cd_s     = _stats(self._cycle_durations)
                    fps_mean = 1.0 / cd_s["mean"] if cd_s["mean"] > 0 else 0.0

                bw_note = (f"@ {fps_mean:.1f} FPS" if fps_mean > 0 else "FPS not available")
                bw_mbps = self._comm_bytes_total / (1024 ** 2) * fps_mean

                trans_rows = [
                    ("Total serialised / frame",
                     f"{self._comm_bytes_total/(1024**2):.3f} MB",
                     "pickle overhead included"),
                    ("Sent  (edge → workers) / frame",
                     f"{self._comm_bytes_sent/(1024**2):.3f} MB",
                     "Q, K, V slices + LN input per block"),
                    ("Recv  (workers → edge) / frame",
                     f"{self._comm_bytes_recv/(1024**2):.3f} MB",
                     "ATTN output + MLP partial output per block"),
                    ("Est. raw-tensor equiv. / frame",
                     f"{self._comm_bytes_total*0.072/(1024**2):.3f} MB",
                     "excl. pickle / tensor metadata"),
                    ("Est. bandwidth @ live FPS",
                     f"{bw_mbps:.2f} MB/s",
                     bw_note),
                ]
                _fmt_table(
                    "Table 4: Network Transmission Size  (based on current ARIMA-V allocation)",
                    ["Metric", "Value", "Notes"],
                    trans_rows,
                )
            else:
                print("\n┌─ Table 4: Network Transmission Size")
                print("│  NOT MEASURED. Call bench.measure_comm_overhead() before the main loop.")
                print("└" + "─" * 40)

        # ── Table 5: Scheduler Overhead ───────────────────────────────────
        if self._evaluator_overhead_us:
            s = _stats(self._evaluator_overhead_us)

            ref_lats = self._dist_latencies if self.is_distributed else self._edge_live_latencies
            block_ms = 0.0
            if ref_lats:
                n_blocks = len(self.orch.model.encoder.layers)
                block_ms = _stats(ref_lats)["mean"] * 1000 / n_blocks if n_blocks > 0 else 0.0
            overhead_pct = (f"{(s['mean']/1000)/block_ms*100:.3f}%"
                            if block_ms > 0 else "N/A")

            _fmt_table(
                "Table 5: Scheduler Overhead  (ARIMA-V update_shares per block)",
                ["Metric", "Mean / Value", "Std / Notes", "Min", "Max", "N"],
                [
                    ("Evaluator overhead / block",
                     f"{s['mean']:.1f} µs", f"±{s['std']:.1f} µs",
                     f"{s['min']:.1f} µs", f"{s['max']:.1f} µs", str(s["n"])),
                    ("% of per-block latency",
                     overhead_pct, "negligible target: < 1%", "—", "—", "—"),
                ],
            )
        else:
            print("\n┌─ Table 5: Scheduler Overhead")
            print("│  No evaluator overhead data.")
            print("└" + "─" * 40)

        # ── Table 6: System Performance (FPS) ────────────────────────────
        if self._cycle_durations:
            s       = _stats(self._cycle_durations)
            fps     = 1.0 / s["mean"] if s["mean"] > 0 else 0.0
            fps_min = 1.0 / s["max"]  if s["max"]  > 0 else 0.0
            fps_max = 1.0 / s["min"]  if s["min"]  > 0 else 0.0
            _fmt_table(
                "Table 6: System Performance  (full perception → motor cycle)",
                ["Metric", "Mean / Value", "Std / Notes", "Min", "Max", "N"],
                [
                    ("Cycle time",
                     f"{s['mean']*1000:.1f} ms", f"±{s['std']*1000:.1f} ms",
                     f"{s['min']*1000:.1f} ms", f"{s['max']*1000:.1f} ms",
                     str(s["n"])),
                    ("FPS",
                     f"{fps:.2f}", "—",
                     f"{fps_min:.2f}", f"{fps_max:.2f}", "—"),
                ],
            )
        else:
            print("\n┌─ Table 6: System Performance")
            print("│  No cycle data.")
            print("└" + "─" * 40)

        # ── Table 7: Ball Tracking Quality ───────────────────────────────
        if self._ball_visible_steps > 0:
            overall = self._ball_loss_events / self._ball_visible_steps * 100
            window  = sum(self._loss_window) / max(len(self._loss_window), 1) * 100
            _fmt_table(
                "Table 7: Ball Tracking Quality",
                ["Metric", "Value", "Notes"],
                [
                    ("Ball-visible steps",        str(self._ball_visible_steps), ""),
                    ("Loss events (total)",        str(self._ball_loss_events),   ""),
                    ("Loss rate (overall)",        f"{overall:.1f}%",             ""),
                    (f"Loss rate (last {len(self._loss_window)} frames)",
                     f"{window:.1f}%", "rolling window"),
                ],
            )
        else:
            print("\n┌─ Table 7: Ball Tracking Quality")
            print("│  No ball-visible frames.")
            print("└" + "─" * 40)

        # ── Table 8: Worker Allocation Stability (distributed mode only) ─
        if self.is_distributed and self._share_log:
            alloc_rows = []
            for w, dq in self._share_log.items():
                vals = list(dq)
                if vals:
                    s   = _stats(vals)
                    idx = max(0.0, 100.0 * (1.0 - s["std"] / max(s["mean"], 1e-9)))
                    alloc_rows.append((
                        w,
                        f"{s['mean']:.3f}",
                        f"±{s['std']:.4f}",
                        f"{s['min']:.3f}",
                        f"{s['max']:.3f}",
                        f"{idx:.1f}%",
                    ))
            _fmt_table(
                f"Table 8: Worker Allocation Stability  (ARIMA-V EMA, {self._share_block_count} blocks)",
                ["Worker", "Mean Share", "Std", "Min", "Max", "Stability Index"],
                alloc_rows,
            )

        # ── Table 9: Scheduler Recovery (distributed mode only) ──────────
        if self.is_distributed:
            if self._recovery_results:
                _fmt_table(
                    "Table 9: Scheduler Recovery  (circuit breaker simulation)",
                    ["Worker", "CB Tripped", "Degraded Blocks", "Recovery Blocks"],
                    [
                        (
                            r["worker"],
                            "Yes" if r["cb_ever_tripped"] else "No",
                            str(r["blocks_degraded_before_cb_trip"]),
                            str(r["blocks_to_full_recovery"]),
                        )
                        for r in self._recovery_results
                    ],
                )
            else:
                print("\n┌─ Table 9: Scheduler Recovery")
                print("│  Not run. Call bench.simulate_worker_failure() outside the main loop.")
                print("└" + "─" * 40)

        print(f"\n{SEP}\n")
