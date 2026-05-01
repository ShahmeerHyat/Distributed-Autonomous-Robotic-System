"""
master.py  —  LCP (Loosely Coupled Protocol) MasterOrchestrator

Key differences from the original clip_detector version:
  - No expected_workers list. The server socket stays open permanently in a
    background daemon thread, so any worker can join at any time.
  - _handle_new_connection() registers the socket, runs preflight, then adds
    the worker to the ARIMA manager under a lock — all without touching the
    ongoing inference loop.
  - run_inference() snapshots (block_devices, block_shares) at the start of
    each transformer block so that workers joining or leaving mid-call cannot
    corrupt tensor shapes or head assignments.
  - If a worker disconnects mid-dispatch, _cleanup_worker() removes it in a
    background thread so the current inference block finishes correctly with
    zeros filling the missing head slice.
  - Reconnecting workers (same name) have their old socket replaced and ARIMA
    state reset before new preflight data is seeded.
  - send_msg / recv_msg now use fp16 on the wire (handled transparently in
    shared_utils) — ~50% bandwidth reduction with negligible accuracy loss.
"""

import threading
import time
import socket
import torch
from concurrent.futures import ThreadPoolExecutor

from distributedSystem.shared_utils import (
    MultiDeviceARIMAManager,
    get_model_metadata,
    to_head_space,
    sum_mlp_parts,
    indices_from_shares,
    send_msg,
    recv_msg,
    tune_socket,
)

# ─────────────────────────────────────────────────────────────────────────────
PREFLIGHT_PINGS    = 8
PROBE_FAIL_LATENCY = 999.0
DEBUG_PRINT_EVERY  = 10   # print allocation table every N inference calls


def _print_split_table(inference_num, block_devices, block_shares, num_heads,
                       mlp_dim, total_latency):
    """Print a compact allocation + latency table to the Webots console."""
    n_blocks = 12   # ViT-B/32 has 12 transformer blocks
    sep = "─" * 72
    print(f"\n{sep}")
    print(f"  [Master] Inference #{inference_num}  —  {len(block_devices)} device(s) active")
    print(f"  {'Device':<12}  {'Share':>6}  {'Heads':>13}  {'MLP neurons':>22}  {'Latency':>9}")
    print(f"  {'──────':<12}  {'──────':>6}  {'─────────────':>13}  {'──────────────────────':>22}  {'─────────':>9}")
    for dev in block_devices:
        share   = block_shares.get(dev, 0.0)
        h       = indices_from_shares(block_devices, block_shares, dev, num_heads)
        n       = indices_from_shares(block_devices, block_shares, dev, mlp_dim)
        h_str   = f"{h.start}-{h.stop-1} ({len(h)}/{num_heads})" if h else "—"
        n_str   = f"{n.start}-{n.stop-1} ({len(n)}/{mlp_dim})" if n else "—"
        lat_ms  = total_latency.get(dev, 0.0) * 1000 / n_blocks
        lat_str = f"{lat_ms:.1f}ms/blk"
        print(f"  {dev:<12}  {share:>5.1%}  {h_str:>13}  {n_str:>22}  {lat_str:>9}")
    print(sep + "\n")


class MasterOrchestrator:

    def __init__(
        self,
        host:  str  = "0.0.0.0",
        port:  int  = 29500,
        model        = None,
        meta:  dict  = {},
    ):
        self.model = model
        self.meta  = meta

        # Protects: sockets, all_devices, arima.*
        self._lock           = threading.RLock()
        self._shutdown_event = threading.Event()

        # Start edge-only; workers are added dynamically.
        self.sockets:     dict = {}
        self.all_devices: list = ["edge"]
        self.arima        = MultiDeviceARIMAManager(["edge"])
        self.last_inference_stats: dict = {}
        self._inference_count = 0

        # Large pool — one future per worker per block, so headroom matters.
        self.executor = ThreadPoolExecutor(max_workers=64)

        # ── Persistent listener ───────────────────────────────────────────
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind((host, port))
        self._srv.listen(32)
        self._srv.settimeout(1.0)   # Non-blocking accept loop

        self._listener = threading.Thread(
            target=self._listener_loop, daemon=True, name="MasterListener"
        )
        self._listener.start()
        print(f"[Master] Edge-only mode. Listening for workers on {host}:{port}")

    # ── Listener loop ────────────────────────────────────────────────────────

    def _listener_loop(self):
        while not self._shutdown_event.is_set():
            try:
                conn, addr = self._srv.accept()
            except socket.timeout:
                continue
            except Exception as e:
                if not self._shutdown_event.is_set():
                    print(f"[Master] Listener error: {e}")
                continue
            # Handle each connection in its own thread so accept() keeps running.
            threading.Thread(
                target=self._handle_new_connection,
                args=(conn, addr),
                daemon=True,
            ).start()
        try:
            self._srv.close()
        except Exception:
            pass

    # ── Connection handler ───────────────────────────────────────────────────

    def _handle_new_connection(self, conn, addr):
        try:
            tune_socket(conn)
            msg = recv_msg(conn)
            if not msg or msg[0] != "REGISTER":
                conn.close()
                return

            name = msg[1]

            # Register socket immediately (needed for preflight pings).
            with self._lock:
                is_reconnect = name in self.sockets
                if is_reconnect:
                    try:
                        self.sockets[name].close()
                    except Exception:
                        pass
                    print(f"[Master] Worker '{name}' reconnected from {addr[0]}. Re-running preflight...")
                else:
                    print(f"[Master] New worker '{name}' from {addr[0]}. Running preflight...")
                self.sockets[name] = conn

            # ── Preflight (outside lock — takes ~8 RTTs) ──────────────────
            samples = self._measure_rtt(name)
            if all(s >= PROBE_FAIL_LATENCY for s in samples):
                print(f"[Master] Preflight failed for '{name}'. Dropping connection.")
                with self._lock:
                    if self.sockets.get(name) is conn:
                        self.sockets.pop(name, None)
                conn.close()
                return

            med_rtt = sorted(samples)[len(samples) // 2]
            print(f"  {name}: median RTT = {med_rtt*1000:.1f}ms  "
                  f"[{', '.join(f'{r*1000:.1f}' for r in samples)}]ms")

            # Measure actual ATTN dispatch latency (includes WiFi data transfer).
            # This is much more accurate than RTT alone for seeding ARIMA because
            # real dispatches send ~300 KB of tensor data per block.
            task_lat = self._measure_task_latency(name)
            if task_lat < PROBE_FAIL_LATENCY:
                print(f"  {name}: ATTN task latency (50% heads) = {task_lat*1000:.1f}ms  "
                      f"(vs RTT {med_rtt*1000:.1f}ms — "
                      f"data overhead = {(task_lat-med_rtt)*1000:.1f}ms)")
            else:
                print(f"  {name}: task latency measurement failed — using RTT only")

            # ── Add to ARIMA (under lock) ─────────────────────────────────
            with self._lock:
                if is_reconnect:
                    self.arima.reset_device(name)
                else:
                    self.arima.add_device(name)
                    self.all_devices.append(name)

                # Seed with RTT first, then overwrite with actual task latency
                # so ARIMA starts with a realistic picture of dispatch cost.
                self.arima.prime(name, samples, nominal_share=0.5)
                if task_lat < PROBE_FAIL_LATENCY:
                    self.arima.prime_task(name, task_lat, nominal_share=0.5)

                # Pre-trip only if truly catastrophic (>DROP_MULT × assumed edge).
                # With DROP_MULT=15 this threshold is intentionally very high.
                edge_guess = 0.05
                if med_rtt / 0.5 > edge_guess * MultiDeviceARIMAManager.DROP_MULT:
                    print(f"  [Preflight] '{name}' is extremely slow → pre-tripping CB")
                    self.arima.breakers[name].trip(reason="pre-flight RTT too high")

            print(f"[Master] Worker '{name}' is active.")

        except Exception as e:
            print(f"[Master] Error handling connection from {addr}: {e}")
            try:
                conn.close()
            except Exception:
                pass

    # ── Preflight measurements ───────────────────────────────────────────────

    def _measure_rtt(self, name: str) -> list:
        with self._lock:
            sock = self.sockets.get(name)
        if sock is None:
            return [PROBE_FAIL_LATENCY] * PREFLIGHT_PINGS

        samples = []
        dummy   = torch.zeros(1, 1)
        for _ in range(PREFLIGHT_PINGS):
            try:
                t0 = time.time()
                send_msg(sock, ("PING", 0, dummy, 0, 0))
                resp = recv_msg(sock)
                rtt  = time.time() - t0
                samples.append(rtt if resp is not None else PROBE_FAIL_LATENCY)
            except Exception:
                samples.append(PROBE_FAIL_LATENCY)
        return samples

    def _measure_task_latency(self, name: str, n_samples: int = 4) -> float:
        """
        Dispatch real-sized ATTN tensors (fp16, 50% head share) to get the true
        round-trip latency including WiFi data-transfer overhead.  A PING only
        measures propagation delay; actual dispatches are dominated by the
        transfer of ~300 KB of tensor data per block per worker.

        Returns median latency in seconds, or PROBE_FAIL_LATENCY on failure.
        """
        with self._lock:
            sock = self.sockets.get(name)
        if sock is None or not self.meta:
            return PROBE_FAIL_LATENCY

        H  = self.meta["num_heads"]
        hd = self.meta["head_dim"]
        S  = self.meta["seq_length"]
        half_H = max(1, H // 2)

        dummy_q = torch.zeros(1, half_H, S, hd)
        dummy_k = torch.zeros(1, half_H, S, hd)
        dummy_v = torch.zeros(1, half_H, S, hd)

        samples = []
        for _ in range(n_samples):
            try:
                t0 = time.time()
                send_msg(sock, ("ATTN", 0, (dummy_q, dummy_k, dummy_v)))
                resp = recv_msg(sock)
                lat  = time.time() - t0
                samples.append(lat if resp is not None else PROBE_FAIL_LATENCY)
            except Exception:
                samples.append(PROBE_FAIL_LATENCY)

        valid = [s for s in samples if s < PROBE_FAIL_LATENCY]
        return sorted(valid)[len(valid) // 2] if valid else PROBE_FAIL_LATENCY

    # ── Worker cleanup ───────────────────────────────────────────────────────

    def _cleanup_worker(self, name: str):
        """Remove a disconnected worker. Safe to call from any thread."""
        with self._lock:
            if name not in self.sockets:
                return
            try:
                self.sockets[name].close()
            except Exception:
                pass
            self.sockets.pop(name, None)
            # Only remove from ARIMA if add_device() was already called.
            if name in self.arima.devices:
                self.arima.remove_device(name)
            if name in self.all_devices:
                self.all_devices.remove(name)
        print(f"[Master] Worker '{name}' removed after disconnect.")

    # ── Dispatch ─────────────────────────────────────────────────────────────

    def _dispatch_task(self, worker_name, task_type, block_idx, payload,
                       start_idx=None, end_idx=None):
        try:
            with self._lock:
                sock = self.sockets.get(worker_name)
            if sock is None:
                return None, PROBE_FAIL_LATENCY

            t0 = time.time()

            if task_type == "ATTN":
                send_msg(sock, ("ATTN", block_idx, payload))
            elif task_type == "MLP":
                send_msg(sock, ("MLP", block_idx, payload, start_idx, end_idx))
            elif task_type == "PING":
                send_msg(sock, ("PING", 0, payload, 0, 0))
            else:
                raise ValueError(f"Unknown task type: {task_type!r}")

            res = recv_msg(sock)
            lat = time.time() - t0

            if res is None:
                raise ConnectionError(f"Worker '{worker_name}' disconnected mid-task.")
            return res, lat

        except Exception as exc:
            print(f"\n  [ERROR] Dispatch to '{worker_name}' failed: {exc}")
            # Trip the circuit breaker on actual connection failure.
            # This is the ONLY place the CB fires — not from slow latency.
            with self._lock:
                if worker_name in self.arima.breakers:
                    self.arima.breakers[worker_name].trip(
                        reason=f"dispatch exception: {type(exc).__name__}"
                    )
            # Cleanup in a background thread — don't block the inference future.
            threading.Thread(
                target=self._cleanup_worker, args=(worker_name,), daemon=True
            ).start()
            return None, PROBE_FAIL_LATENCY

    # ── Inference ────────────────────────────────────────────────────────────

    def run_inference(self, x: torch.Tensor) -> torch.Tensor:
        """
        Distributed forward through clip.vision_model.encoder.layers.

        x : (1, seq_length, embed_dim)
            Output of vision.embeddings + pre_layernorm (applied by caller).

        Returns last_hidden_state (1, seq_length, embed_dim).
        Caller applies post_layernorm and visual_projection.

        Thread safety: snapshots (block_devices, block_shares) at the start of
        each block. Workers that join mid-call are visible from the next call.
        Workers that disconnect mid-call cause zeros to fill their head slice.
        """
        H  = self.meta["num_heads"]
        hd = self.meta["head_dim"]
        S  = self.meta["seq_length"]
        D  = self.meta["embed_dim"]

        self._inference_count += 1
        should_print   = (self._inference_count % DEBUG_PRINT_EVERY == 0)
        total_latency  = {}   # accumulated across all 12 blocks for the summary
        snap0_devices  = None # block-0 snapshot for the summary table
        snap0_shares   = None

        current_state = x

        for i, block in enumerate(self.model.encoder.layers):

            # ── Block snapshot ────────────────────────────────────────────
            # Hold the lock only for the snapshot; release before any compute.
            with self._lock:
                self.arima.update_shares()
                block_devices = list(self.arima.devices)          # ["edge", ...]
                block_shares  = dict(self.arima.current_shares)   # {dev: float}
                probing_workers = {
                    w for w in block_devices if w != "edge"
                    and w in self.arima.breakers
                    and self.arima.breakers[w].is_half_open
                }

            active_workers = [d for d in block_devices if d != "edge"]
            raw_latency    = {d: 0.0 for d in block_devices}
            share_used     = {d: 0.0 for d in block_devices}

            if i == 0:
                snap0_devices = block_devices
                snap0_shares  = block_shares

            ln_1 = block.layer_norm1
            ln_2 = block.layer_norm2
            attn = block.self_attn
            fc1  = block.mlp.fc1
            fc2  = block.mlp.fc2

            # ── ATTENTION ─────────────────────────────────────────────────
            identity = current_state
            ln_x     = ln_1(current_state)

            q_full = attn.q_proj(ln_x)
            k_full = attn.k_proj(ln_x)
            v_full = attn.v_proj(ln_x)

            # Dispatch head slices to workers in parallel.
            attn_futures = {}
            for w in active_workers:
                h_range = indices_from_shares(block_devices, block_shares, w, H)
                if len(h_range) == 0:
                    continue
                q_s = to_head_space(q_full, h_range, H, hd)
                k_s = to_head_space(k_full, h_range, H, hd)
                v_s = to_head_space(v_full, h_range, H, hd)
                attn_futures[w] = self.executor.submit(
                    self._dispatch_task, w, "ATTN", i, (q_s, k_s, v_s),
                )
                share_used[w] += block_shares.get(w, 0.0)

            # Edge computes its head slice locally.
            edge_h = indices_from_shares(block_devices, block_shares, "edge", H)
            share_used["edge"] += block_shares.get("edge", 1.0)

            t0 = time.time()
            if len(edge_h) > 0:
                q_e = to_head_space(q_full, edge_h, H, hd)
                k_e = to_head_space(k_full, edge_h, H, hd)
                v_e = to_head_space(v_full, edge_h, H, hd)
                scale     = hd ** -0.5
                attn_probs = torch.softmax((q_e @ k_e.transpose(-2, -1)) * scale, dim=-1)
                edge_attn  = attn_probs @ v_e
            else:
                edge_attn = None
            raw_latency["edge"] += time.time() - t0

            # Collect and merge head outputs in device order (preserves head indices).
            head_parts = []
            for dev in block_devices:
                if dev == "edge":
                    if edge_attn is not None:
                        head_parts.append(edge_attn)
                elif dev in attn_futures:
                    res, lat = attn_futures[dev].result()
                    raw_latency[dev] += lat
                    if res is not None and res.numel() > 0:
                        head_parts.append(res.to(ln_x.device))
                    else:
                        # Worker failed mid-block: fill zeros to preserve shape.
                        h_range = indices_from_shares(block_devices, block_shares, dev, H)
                        if len(h_range) > 0:
                            head_parts.append(
                                torch.zeros(1, len(h_range), S, hd, device=ln_x.device)
                            )

            if not head_parts:
                # Full edge fallback — should only happen during a transient race.
                q_e = to_head_space(q_full, range(H), H, hd)
                k_e = to_head_space(k_full, range(H), H, hd)
                v_e = to_head_space(v_full, range(H), H, hd)
                scale = hd ** -0.5
                attn_probs = torch.softmax((q_e @ k_e.transpose(-2, -1)) * scale, dim=-1)
                head_parts = [attn_probs @ v_e]

            ctx      = torch.cat(head_parts, dim=1).transpose(1, 2).reshape(1, S, D)
            attn_out = attn.out_proj(ctx)
            current_state = identity + attn_out

            # ── MLP ───────────────────────────────────────────────────────
            identity  = current_state
            ln_x_mlp  = ln_2(current_state)
            act_fn    = block.mlp.activation_fn   # quick_gelu for CLIP ViT-B/32

            mlp_futures = {}
            for w in active_workers:
                n_range = indices_from_shares(block_devices, block_shares, w, self.meta["mlp_hidden_dim"])
                if len(n_range) == 0:
                    continue
                mlp_futures[w] = self.executor.submit(
                    self._dispatch_task, w, "MLP", i,
                    ln_x_mlp, n_range.start, n_range.stop,
                )
                share_used[w] = (share_used[w] + block_shares.get(w, 0.0)) / 2.0

            edge_n = indices_from_shares(block_devices, block_shares, "edge", self.meta["mlp_hidden_dim"])

            t0 = time.time()
            if len(edge_n) > 0:
                w1 = fc1.weight[edge_n.start:edge_n.stop, :]
                b1 = fc1.bias[edge_n.start:edge_n.stop]
                w2 = fc2.weight[:, edge_n.start:edge_n.stop]
                edge_mlp = act_fn(ln_x_mlp @ w1.t() + b1) @ w2.t()
            else:
                edge_mlp = None
            raw_latency["edge"] += time.time() - t0

            mlp_parts = [edge_mlp] if edge_mlp is not None else []
            for w, fut in mlp_futures.items():
                res, lat = fut.result()
                raw_latency[w] += lat
                if res is not None and res.numel() > 0:
                    mlp_parts.append(res.to(ln_x_mlp.device))

            if not mlp_parts:
                # Full edge MLP fallback.
                edge_mlp_full = act_fn(ln_x_mlp @ fc1.weight.t() + fc1.bias) @ fc2.weight.t()
                mlp_parts = [edge_mlp_full]

            mlp_final     = sum_mlp_parts(mlp_parts) + fc2.bias
            current_state = identity + mlp_final

            # ── ARIMA bookkeeping (under lock) ────────────────────────────
            with self._lock:
                for dev in block_devices:
                    s = share_used.get(dev, 0.0)
                    self.arima.record_block_latency(dev, raw_latency.get(dev, 0.0), s)

                for w in probing_workers:
                    s = share_used.get(w, 0.0)
                    lat = raw_latency.get(w, PROBE_FAIL_LATENCY)
                    if s > 0 and lat < PROBE_FAIL_LATENCY:
                        self.arima.notify_probe_result(w, lat / s)
                    else:
                        self.arima.notify_probe_result(w, PROBE_FAIL_LATENCY)

            # ── Accumulate latency for the debug summary ──────────────────
            if should_print:
                for dev in block_devices:
                    total_latency[dev] = total_latency.get(dev, 0.0) + raw_latency.get(dev, 0.0)

        # ── Per-N-inference allocation + latency summary ──────────────────────
        if should_print and snap0_devices and len(snap0_devices) > 1:
            _print_split_table(
                self._inference_count,
                snap0_devices, snap0_shares,
                H, self.meta["mlp_hidden_dim"],
                total_latency,
            )

        return current_state

    # ── Shutdown ─────────────────────────────────────────────────────────────

    def shutdown(self):
        self._shutdown_event.set()
        with self._lock:
            for w_name, sock in list(self.sockets.items()):
                try:
                    send_msg(sock, ("QUIT", 0, None, 0, 0))
                    sock.close()
                except Exception:
                    pass
            self.sockets.clear()
        self.executor.shutdown(wait=False)
        print("[Master] Shut down.")


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test  (no Webots required)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=29500)
    args = p.parse_args()

    from transformers import CLIPModel

    MODEL_PATH = r"../../../../Clip Model"
    clip = CLIPModel.from_pretrained(MODEL_PATH, local_files_only=True)
    clip.eval()

    meta = get_model_metadata(clip.vision_model)
    orch = MasterOrchestrator(host="0.0.0.0", port=args.port,
                              model=clip.vision_model, meta=meta)

    dummy = torch.randn(1, meta["seq_length"], meta["embed_dim"])

    for run in range(3):
        t0     = time.time()
        out    = orch.run_inference(dummy)
        elapsed = time.time() - t0
        print(f"Run {run+1}: {elapsed:.4f}s  shape={tuple(out.shape)}")
        time.sleep(0.5)

    orch.shutdown()
