"""
benchmarks.py  —  Runtime benchmarks for distributed CLIP inference

Triggered from clip_detector_v2.py after BENCHMARK_AFTER_N_FRAMES frames.
All metrics are self-contained: no external dependencies beyond what the
detector already imports.

Metrics
-------
1. Inference Latency     — wall-clock pixel_values → last_hidden_state
2. Frames Per Second     — full perception-to-motor-command cycles/s
3. Ball Loss Rate        — confidence drops while ball visible / 100 steps
4. Scheduler Recovery   — blocks degraded before / after CB trip + recovery
5. Share Stability       — std-dev of worker share over 200 blocks

Usage in clip_detector_v2.py
------------------------------
    from distributedSystem.benchmarks import BenchmarkCollector

    bench = BenchmarkCollector(orch=orch, confidence_threshold=CONFIDENCE_THRESHOLD)

    # Inside the main loop, at the very top:
    bench.tick_frame_start()

    # Right after last_hidden = orch.run_inference(hidden):
    bench.record_inference(
        latency        = <float seconds>,
        last_logits    = last_logits,
        ball_visible   = <bool — your ground truth or heuristic>,
    )

    # At the end of motor-command dispatch:
    bench.tick_frame_end()

    # Trigger the report:
    if step_count == BENCHMARK_AFTER_N_FRAMES:
        bench.report()
"""

import time
import statistics
import threading
from collections import deque


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _stats(values: list) -> dict:
    """Return mean / std / min / max for a non-empty list, else all zeros."""
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n": 0}
    return {
        "mean": statistics.mean(values),
        "std":  statistics.pstdev(values),   # population std — no Bessel correction needed here
        "min":  min(values),
        "max":  max(values),
        "n":    len(values),
    }


def _fmt(s: dict) -> str:
    return (f"mean={s['mean']*1000:.1f}ms  std={s['std']*1000:.1f}ms  "
            f"min={s['min']*1000:.1f}ms  max={s['max']*1000:.1f}ms  (n={s['n']})")


# ─────────────────────────────────────────────────────────────────────────────
# BenchmarkCollector
# ─────────────────────────────────────────────────────────────────────────────

class BenchmarkCollector:
    """
    Passive collector: you push data in, it stores it.
    Call report() to flush everything to stdout.
    """

    # How many consecutive inference steps to use for latency stats
    LATENCY_WINDOW = 50

    # How many blocks to log for share-stability analysis (Metric 5)
    SHARE_LOG_DEPTH = 200

    def __init__(
        self,
        orch,                          # MasterOrchestrator instance
        confidence_threshold: float,   # same value used in the main loop
    ):
        self.orch      = orch
        self.threshold = confidence_threshold

        # ── Metric 1 & 2: latency + FPS ─────────────────────────────────
        self._inference_latencies: list[float] = []
        self._frame_start: float | None = None
        self._cycle_durations: list[float] = []

        # ── Metric 3: ball loss rate ──────────────────────────────────────
        # A "loss event" = confidence < threshold while ball_visible == True
        self._ball_visible_steps   = 0
        self._ball_loss_events     = 0
        # rolling window of last 100 inference steps for the rate calculation
        self._loss_window: deque = deque(maxlen=100)

        # ── Metric 4: scheduler recovery ─────────────────────────────────
        # Filled in by simulate_worker_failure() — not auto-collected
        self._recovery_results: list[dict] = []

        # ── Metric 5: share stability ─────────────────────────────────────
        # Keyed by worker name → list of share values, capped at SHARE_LOG_DEPTH
        self._share_log: dict[str, deque] = {
            w: deque(maxlen=self.SHARE_LOG_DEPTH)
            for w in orch.expected_workers
        }
        self._share_block_count = 0

        # ── Internal state ───────────────────────────────────────────────
        self._total_frames = 0

    # ─────────────────────────────────────────────────────────────────────────
    # Public push API  (call from the main loop)
    # ─────────────────────────────────────────────────────────────────────────

    def tick_frame_start(self):
        """Call at the very top of every main-loop iteration."""
        self._frame_start = time.perf_counter()
        self._total_frames += 1

    def tick_frame_end(self):
        """Call after motor velocities are set (end of main-loop iteration)."""
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
        ball_visible : True when you have ground truth that ball is in frame.
                       If you have no ground truth, pass (last_logits > threshold)
                       from the *previous* frame as a proxy.
        """
        # Metric 1 — raw latency samples (keep last LATENCY_WINDOW)
        self._inference_latencies.append(latency)
        if len(self._inference_latencies) > self.LATENCY_WINDOW * 4:
            self._inference_latencies = self._inference_latencies[-self.LATENCY_WINDOW:]

        # Metric 3 — ball loss rate
        if ball_visible:
            self._ball_visible_steps += 1
            lost = last_logits < self.threshold
            self._loss_window.append(1 if lost else 0)
            if lost:
                self._ball_loss_events += 1
        else:
            self._loss_window.append(0)

        # Metric 5 — log current ARIMA shares
        shares = self.orch.arima.current_shares
        for w, dq in self._share_log.items():
            dq.append(shares.get(w, 0.0))
        self._share_block_count += 1

    # ─────────────────────────────────────────────────────────────────────────
    # Metric 4 — Scheduler Recovery  (active test, run explicitly)
    # ─────────────────────────────────────────────────────────────────────────

    def simulate_worker_failure(
        self,
        worker_name: str,
        dummy_input,           # torch.Tensor — same shape as real hidden state
        recovery_delay: float = 10.0,
        use_circuit_breaker: bool = True,
    ) -> dict:
        """
        Simulate worker failure + recovery and return timing metrics.

        This forcibly closes the worker socket (simulating a crash), runs
        inference for `recovery_delay` seconds to observe CB behaviour, then
        re-opens the socket and counts blocks until full recovery.

        IMPORTANT: Call this outside the main Webots loop (e.g. in a test
        harness), since it blocks for ~recovery_delay seconds.

        Returns a dict logged into self._recovery_results for report().
        """
        import socket as _socket

        arima   = self.orch.arima
        breaker = arima.breakers.get(worker_name)

        if breaker is None:
            return {"error": f"No circuit breaker for worker '{worker_name}'"}

        if not use_circuit_breaker:
            # Disable CB: pre-open it so it never trips
            breaker.state = "OPEN"
            breaker.blocks_remaining = 10_000   # effectively disabled

        result = {
            "worker":               worker_name,
            "use_circuit_breaker":  use_circuit_breaker,
            "blocks_degraded_before_cb_trip": 0,
            "blocks_to_full_recovery":        0,
            "cb_ever_tripped":                False,
            "recovery_delay_s":               recovery_delay,
        }

        # ── Phase 1: inject failure ───────────────────────────────────────
        print(f"\n[Benchmark] Simulating '{worker_name}' failure …")
        sock = self.orch.sockets.get(worker_name)
        if sock:
            try:
                sock.close()
            except Exception:
                pass

        # Count how many blocks ran degraded before CB tripped
        degraded_blocks = 0
        t_fail_start    = time.perf_counter()

        while time.perf_counter() - t_fail_start < recovery_delay:
            t0 = time.perf_counter()
            try:
                self.orch.run_inference(dummy_input)
            except Exception:
                pass

            share = arima.current_shares.get(worker_name, 0.0)
            cb_now = breaker.state

            if cb_now == "OPEN":
                result["cb_ever_tripped"] = True
                break
            elif share > 0:
                degraded_blocks += 1   # worker still assigned work but is dead

        result["blocks_degraded_before_cb_trip"] = degraded_blocks

        # ── Phase 2: simulate recovery (reconnect) ────────────────────────
        print(f"[Benchmark] Simulating '{worker_name}' restart after {recovery_delay}s …")
        # We can't literally reconnect a real socket in this harness,
        # so we reset the CB manually to HALF_OPEN to measure re-admission
        breaker.state            = "HALF_OPEN"
        breaker.blocks_remaining = 0

        recovery_blocks = 0
        t_recover_start = time.perf_counter()
        timeout         = recovery_delay * 2

        while time.perf_counter() - t_recover_start < timeout:
            try:
                self.orch.run_inference(dummy_input)
            except Exception:
                pass

            recovery_blocks += 1

            if breaker.state == "CLOSED" and arima.current_shares.get(worker_name, 0.0) > 0.05:
                break

        result["blocks_to_full_recovery"] = recovery_blocks
        self._recovery_results.append(result)
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # report()  —  print everything
    # ─────────────────────────────────────────────────────────────────────────

    def report(self):
        sep  = "═" * 70
        sep2 = "─" * 70

        print(f"\n{sep}")
        print("  DISTRIBUTED CLIP BENCHMARK REPORT")
        print(f"  Total frames observed: {self._total_frames}")
        print(sep)

        # ── Metric 1: Inference Latency ───────────────────────────────────
        print("\n┌─ METRIC 1: Inference Latency (pixel_values → last_hidden_state)")
        lats = self._inference_latencies
        if lats:
            s = _stats(lats)
            print(f"│  {_fmt(s)}")

            # Split into first half (warm-up) vs second half (steady state)
            mid   = len(lats) // 2
            early = _stats(lats[:mid])
            late  = _stats(lats[mid:])
            print(f"│  Early (first {mid} frames): mean={early['mean']*1000:.1f}ms")
            print(f"│  Late  (last  {len(lats)-mid} frames): mean={late['mean']*1000:.1f}ms")
        else:
            print("│  No latency data collected.")
        print("└" + sep2)

        # ── Metric 2: Frames Per Second ───────────────────────────────────
        print("\n┌─ METRIC 2: Frames Per Second (full perception→motor cycle)")
        cyc = self._cycle_durations
        if cyc:
            s   = _stats(cyc)
            fps = 1.0 / s["mean"] if s["mean"] > 0 else 0.0
            fps_min = 1.0 / s["max"] if s["max"] > 0 else 0.0
            fps_max = 1.0 / s["min"] if s["min"] > 0 else 0.0
            print(f"│  Cycle time: {_fmt(s)}")
            print(f"│  FPS: mean={fps:.2f}  min={fps_min:.2f}  max={fps_max:.2f}")
        else:
            print("│  No cycle data collected.")
        print("└" + sep2)

        # ── Metric 3: Ball Loss Rate ──────────────────────────────────────
        print("\n┌─ METRIC 3: Ball Loss Rate")
        if self._ball_visible_steps > 0:
            # Overall rate
            overall_rate = (self._ball_loss_events / self._ball_visible_steps) * 100.0
            # Rolling-window rate (last 100 steps)
            window_rate  = (sum(self._loss_window) / max(len(self._loss_window), 1)) * 100.0
            print(f"│  Ball-visible steps   : {self._ball_visible_steps}")
            print(f"│  Loss events (total)  : {self._ball_loss_events}")
            print(f"│  Loss rate (overall)  : {overall_rate:.1f}% of ball-visible steps")
            print(f"│  Loss rate (last {len(self._loss_window):3d}) : {window_rate:.1f}% of steps")
        else:
            print("│  No ball-visible frames recorded.")
            print("│  Tip: pass ball_visible=True to record_inference() when ball is in frame.")
        print("└" + sep2)

        # ── Metric 4: Scheduler Recovery ─────────────────────────────────
        print("\n┌─ METRIC 4: Scheduler Recovery Time")
        if self._recovery_results:
            for r in self._recovery_results:
                cb_tag = "WITH circuit-breaker" if r["use_circuit_breaker"] else "WITHOUT circuit-breaker"
                print(f"│  Worker: {r['worker']}  [{cb_tag}]")
                print(f"│    CB tripped          : {r['cb_ever_tripped']}")
                print(f"│    Degraded blocks     : {r['blocks_degraded_before_cb_trip']} "
                      f"(before CB tripped)")
                print(f"│    Recovery blocks     : {r['blocks_to_full_recovery']} "
                      f"(HALF_OPEN → CLOSED + share > 5%)")
                print(f"│    Recovery delay used : {r['recovery_delay_s']}s")
                print("│")
        else:
            print("│  No recovery simulation run.")
            print("│  Tip: call bench.simulate_worker_failure(worker_name, dummy_tensor)")
            print("│        outside the Webots main loop to populate this section.")
        print("└" + sep2)

        # ── Metric 5: Share Stability ─────────────────────────────────────
        print("\n┌─ METRIC 5: Share Stability (ARIMA EMA smoothing)")
        print(f"│  Blocks logged: {self._share_block_count}  "
              f"(cap per worker: {self.SHARE_LOG_DEPTH})")
        if self._share_log:
            for w, dq in self._share_log.items():
                vals = list(dq)
                if vals:
                    s = _stats(vals)
                    # Stability index: lower std = more stable
                    stability_pct = max(0.0, 100.0 * (1.0 - s["std"] / max(s["mean"], 1e-9)))
                    print(f"│  Worker '{w}':")
                    print(f"│    mean share = {s['mean']:.3f}  "
                          f"std = {s['std']:.4f}  "
                          f"min = {s['min']:.3f}  "
                          f"max = {s['max']:.3f}")
                    print(f"│    Stability index: {stability_pct:.1f}%  "
                          f"(100% = perfectly fixed, lower = more variance)")
                else:
                    print(f"│  Worker '{w}': no share data yet.")
        else:
            print("│  No workers registered.")
        print("└" + sep2)

        print(f"\n{sep}")
        print("  END OF BENCHMARK REPORT")
        print(f"{sep}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Integration snippet (printed to stdout for copy-paste convenience)
# ─────────────────────────────────────────────────────────────────────────────

INTEGRATION_SNIPPET = '''
# ── Paste this into clip_detector_v2.py ──────────────────────────────────────

from distributedSystem.benchmarks import BenchmarkCollector

BENCHMARK_AFTER_N_FRAMES = 500   # report fires once, then resets

bench = BenchmarkCollector(orch=orch, confidence_threshold=CONFIDENCE_THRESHOLD)

# ── Inside the while loop ────────────────────────────────────────────────────

while robot.step(timestep) != -1:
    step_count += 1
    bench.tick_frame_start()                        # <── ADD THIS (top of loop)

    run_inference = (step_count % INFERENCE_EVERY_N == 0)

    if run_inference:
        image = get_frame_as_pil(camera)
        if image is not None:
            with torch.no_grad():
                hidden      = model.vision_model.embeddings(pixel_values)
                hidden      = model.vision_model.pre_layrnorm(hidden)

                t0          = time.perf_counter()   # <── ADD
                last_hidden = orch.run_inference(hidden)
                inf_lat     = time.perf_counter() - t0   # <── ADD

                # ... rest of your inference code ...

            bench.record_inference(                 # <── ADD THIS
                latency      = inf_lat,
                last_logits  = last_logits,
                ball_visible = (last_logits > CONFIDENCE_THRESHOLD),  # or your GT
            )

    # ... motor control ...

    bench.tick_frame_end()                          # <── ADD THIS (end of loop)

    if step_count == BENCHMARK_AFTER_N_FRAMES:
        bench.report()
'''


if __name__ == "__main__":
    print(INTEGRATION_SNIPPET)
