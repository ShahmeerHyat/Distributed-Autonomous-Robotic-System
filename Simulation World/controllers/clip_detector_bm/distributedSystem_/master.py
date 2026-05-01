"""
master.py  —  CLIP-aware MasterOrchestrator (AIMD-based scheduler)

ATTN protocol (CLIP mode):
  Master projects Q, K, V from ln_x via attn.q_proj / k_proj / v_proj.
  Each device (edge + workers) receives a head-sliced portion as
  (B, H_slice, S, head_dim) tensors.
  Workers return their attention output in the same shape.
  Master concatenates all parts on the head dimension, reshapes to
  (B, S, embed_dim), and applies attn.out_proj.

MLP protocol:
  Each device computes act_fn(x @ w1_slice.t() + b1_slice) @ w2_slice.t()
  and returns (B, S, embed_dim). Master sums all parts and adds fc2.bias once.
"""

import time
import torch
import socket
from concurrent.futures import ThreadPoolExecutor
from transformers import CLIPModel

# Relative imports when used as a package; bare imports when run standalone.
try:
    from .splitInfer import MultiDeviceEvaluator
    from .comms import send_msg, recv_msg, probe_rtt, CircuitBreaker
    from .helper import get_model_metadata, to_head_space, sum_mlp_parts, allocate_heads
except ImportError:
    from splitInfer import MultiDeviceEvaluator
    from comms import send_msg, recv_msg, probe_rtt, CircuitBreaker
    from helper import get_model_metadata, to_head_space, sum_mlp_parts, allocate_heads

# ─────────────────────────────────────────────────────────────────────────────
PREFLIGHT_PINGS    = 8
PROBE_FAIL_LATENCY = 999.0

MODEL_PATH = r"../../../../Clip Model"


class MasterOrchestrator:

    def __init__(
        self,
        expected_workers: list,
        host: str  = "0.0.0.0",
        port: int  = 29500,
        model      = None,
        meta: dict = None,
    ):
        self.expected_workers = expected_workers
        self.all_devices      = ["edge"] + expected_workers
        self.evaluator        = MultiDeviceEvaluator(self.all_devices)
        self.breakers         = {w: CircuitBreaker() for w in expected_workers}
        self.last_inference_stats = {}

        if model is not None:
            self.model = model
        else:
            clip       = CLIPModel.from_pretrained(MODEL_PATH, local_files_only=True).eval()
            self.model = clip.vision_model

        self.meta = meta or get_model_metadata(self.model)

        self.sockets: dict        = {}                                                                                                                                                          
        self.worker_addrs: dict   = {}                                                                                                                                                          
        self._preflight_rtt: dict = {}   # median RTT per worker, set in _preflight   
        self._wait_for_workers(host, port)
        self.executor = ThreadPoolExecutor(max_workers=max(1, len(expected_workers)))
        self._preflight()

    @property
    def current_shares(self) -> dict:
        """Normalized allocation weights for each device (sum to 1.0)."""
        return self.evaluator.get_allocation()

    # ── Connection ──────────────────────────────────────────────────────────

    def _wait_for_workers(self, host: str, port: int):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((host, port))
        srv.listen(len(self.expected_workers))
        print(f"[Master] Waiting for {len(self.expected_workers)} worker(s) on {host}:{port} …")

        while len(self.sockets) < len(self.expected_workers):
            conn, addr = srv.accept()
            msg = recv_msg(conn)
            if msg and msg[0] == "REGISTER":
                name = msg[1]
                self.sockets[name]      = conn
                self.worker_addrs[name] = (addr[0], port)
                print(f"  Worker '{name}' connected from {addr[0]}")

        srv.close()
        print("[Master] All workers connected.\n")

    def _preflight(self):
        print("=" * 60)
        print("[Master] Pre-flight RTT probing …")

        for w_name in self.expected_workers:
            host, port = self.worker_addrs[w_name]
            rtts = []

            for _ in range(PREFLIGHT_PINGS):
                rtt = probe_rtt(host, port)
                rtts.append(rtt if rtt is not None else PROBE_FAIL_LATENCY)

            median_rtt = sorted(rtts)[len(rtts) // 2]
            print(f"  {w_name}: RTTs={[f'{r:.3f}s' for r in rtts]} → median {median_rtt:.3f}s")
            self._preflight_rtt[w_name] = median_rtt
            for rtt in rtts:
                if rtt < PROBE_FAIL_LATENCY:
                    self.evaluator.record_step(w_name, latency=rtt, compute_time=0.0)

            if median_rtt >= PROBE_FAIL_LATENCY:
                self.breakers[w_name].trip(reason="pre-flight unreachable")
            elif self.evaluator.needs_probe(w_name):
                self.breakers[w_name].trip(reason="pre-flight RTT too high")

        # Seed edge with zeros — local device, no network latency.
        self.evaluator.record_step("edge", latency=0.0, compute_time=0.0)

        print("[Master] Pre-flight complete.\n" + "=" * 60 + "\n")

    # ── Local inference (edge-only, used by BenchmarkCollector) ─────────────

    def _local_inference(self, x: torch.Tensor) -> torch.Tensor:
        """
        Runs the full encoder forward pass locally on the edge device only.
        No tasks are dispatched to workers. Used exclusively during preflight
        to measure pure edge compute latency without network interference.
        """
        current_state = x
        H  = self.meta["num_heads"]
        hd = self.meta["head_dim"]
        S  = self.meta["seq_length"]
        D  = self.meta["embed_dim"]

        for block in self.model.encoder.layers:
            ln_1 = block.layer_norm1
            ln_2 = block.layer_norm2
            attn = block.self_attn
            fc1  = block.mlp.fc1
            fc2  = block.mlp.fc2

            identity = current_state
            ln_x     = ln_1(current_state)
            q = attn.q_proj(ln_x)
            k = attn.k_proj(ln_x)
            v = attn.v_proj(ln_x)

            def reshape(t):
                return t.view(1, S, H, hd).transpose(1, 2)

            scale      = hd ** -0.5
            attn_probs = torch.softmax(
                (reshape(q) @ reshape(k).transpose(-2, -1)) * scale, dim=-1
            )
            ctx           = (attn_probs @ reshape(v)).transpose(1, 2).reshape(1, S, D)
            current_state = identity + attn.out_proj(ctx)

            identity      = current_state
            ln_x_mlp      = ln_2(current_state)
            act_fn        = block.mlp.activation_fn
            mlp_out       = act_fn(ln_x_mlp @ fc1.weight.t() + fc1.bias) @ fc2.weight.t() + fc2.bias
            current_state = identity + mlp_out

        return current_state

    # ── Dispatch ────────────────────────────────────────────────────────────

    def _dispatch_task(
        self,
        worker_name,
        task_type,
        block_idx,
        payload,
        start_idx=None,
        end_idx=None,
    ) -> tuple:
        """
        RTT is estimated from the preflight median rather than measured inline.                                                                                                                
        Paying an extra round-trip per dispatch on every block of a 12-block                                                                                                                   
        sequential encoder adds ~24 pings/frame (~120 ms on a 5 ms LAN) for no                                                                                                                 
        runtime benefit — the evaluator gets a clean signal from preflight RTT                                                                                                                 
        plus the per-block update after results arrive. 
        """
        try:
            sock = self.sockets[worker_name]
            t0   = time.perf_counter()
            
            if task_type == "ATTN":
                assert isinstance(payload, tuple) and len(payload) == 3
                send_msg(sock, ("ATTN", block_idx, payload))
            elif task_type == "MLP":
                send_msg(sock, ("MLP", block_idx, payload, start_idx, end_idx))
            elif task_type == "PING":
                send_msg(sock, ("PING", 0, payload, 0, 0))
            else:
                raise ValueError(f"Unknown task type: {task_type!r}")

            res          = recv_msg(sock)
            total_time   = time.perf_counter() - t0

            if res is None:
                raise ConnectionError(f"Worker '{worker_name}' disconnected.")
            net_t  = self._preflight_rtt.get(worker_name, total_time * 0.3)                                                                                                                    
            comp_t = max(0.0, total_time - net_t)                                                                                                                                              
            return res, net_t, comp_t   

        except Exception as exc:
            print(f"\n  [ERROR] Dispatch to '{worker_name}' failed: {exc}")
            return None, PROBE_FAIL_LATENCY, 0.0

    # ── Inference ────────────────────────────────────────────────────────────

    def run_inference(self, x: torch.Tensor) -> torch.Tensor:
        """
        Block-by-block distributed forward through clip.vision_model.encoder.layers.

        x : (1, seq_length, embed_dim)  — output of vision.embeddings + pre_layrnorm
        Returns last hidden state (1, seq_length, embed_dim).
        """
        current_state = x
        H  = self.meta["num_heads"]
        hd = self.meta["head_dim"]
        S  = self.meta["seq_length"]
        D  = self.meta["embed_dim"]

        _blk_attn_ms      = []
        _blk_mlp_ms       = []
        _blk_evaluator_us = []

        for i, block in enumerate(self.model.encoder.layers):

            network_time = {dev: 0.0 for dev in self.all_devices}
            compute_time = {dev: 0.0 for dev in self.all_devices}

            # ── Tick breakers + build allocation ──────────────────────────────
            for w_name in self.expected_workers:
                self.breakers[w_name].tick()

            _t_eval    = time.perf_counter()
            raw_scores = self.evaluator.get_allocation()
            # Force score to 0 for fully-open breakers so they get no allocation.
            scores = {
                dev: (
                    score
                    if dev == "edge"
                    or self.breakers[dev].is_closed
                    or self.breakers[dev].is_half_open
                    else 0.0
                )
                for dev, score in raw_scores.items()
            }
            allocs = allocate_heads(H, scores)
            _blk_evaluator_us.append((time.perf_counter() - _t_eval) * 1e6)

            # Build head ranges in device order so merge is always sorted.
            start    = 0
            h_ranges = {}
            for dev in self.all_devices:
                h_ranges[dev] = range(start, start + allocs[dev])
                start += allocs[dev]

            # ── Block layers ──────────────────────────────────────────────────
            ln_1 = block.layer_norm1
            ln_2 = block.layer_norm2
            attn = block.self_attn
            fc1  = block.mlp.fc1
            fc2  = block.mlp.fc2

            # ── ATTENTION ─────────────────────────────────────────────────────
            _t_attn  = time.perf_counter()
            identity = current_state
            ln_x     = ln_1(current_state)

            q_full = attn.q_proj(ln_x)
            k_full = attn.k_proj(ln_x)
            v_full = attn.v_proj(ln_x)

            # Dispatch workers in parallel.
            attn_futures = {}
            for w_name in self.expected_workers:
                h_range = h_ranges[w_name]
                if len(h_range) == 0:
                    continue
                q_s = to_head_space(q_full, h_range, H, hd)
                k_s = to_head_space(k_full, h_range, H, hd)
                v_s = to_head_space(v_full, h_range, H, hd)
                attn_futures[w_name] = self.executor.submit(
                    self._dispatch_task,
                    w_name, "ATTN", i, (q_s, k_s, v_s),
                )

            # Edge computes locally while workers are running.
            t0        = time.perf_counter()
            edge_attn = None
            if len(h_ranges["edge"]) > 0:
                q_e = to_head_space(q_full, h_ranges["edge"], H, hd)
                k_e = to_head_space(k_full, h_ranges["edge"], H, hd)
                v_e = to_head_space(v_full, h_ranges["edge"], H, hd)
                scale      = hd ** -0.5
                attn_probs = torch.softmax((q_e @ k_e.transpose(-2, -1)) * scale, dim=-1)
                edge_attn  = attn_probs @ v_e
            compute_time["edge"] += time.perf_counter() - t0

            # Collect + merge attention heads in device order.
            head_parts = []
            for dev in self.all_devices:
                if dev == "edge":
                    if edge_attn is not None:
                        head_parts.append(edge_attn)
                elif dev in attn_futures:
                    res, net_t, comp_t = attn_futures[dev].result()
                    network_time[dev] += net_t
                    compute_time[dev] += comp_t

                    if res is not None and res.numel() > 0:
                        head_parts.append(res.to(ln_x.device))
                    else:
                        n = len(h_ranges[dev])
                        if n > 0:
                            head_parts.append(
                                torch.zeros(1, n, S, hd, device=ln_x.device)
                            )
                        self.breakers[dev].trip(reason="attn dispatch returned None")

            # (1, H_total, S, hd) → (1, S, D)
            ctx      = torch.cat(head_parts, dim=1).transpose(1, 2).reshape(1, S, D)
            attn_out = attn.out_proj(ctx)
            current_state = identity + attn_out
            _blk_attn_ms.append((time.perf_counter() - _t_attn) * 1e3)

            # ── MLP ───────────────────────────────────────────────────────────
            _t_mlp   = time.perf_counter()
            identity = current_state
            ln_x_mlp = ln_2(current_state)

            # Reuse same score ratios for neuron split.
            mlp_allocs = allocate_heads(self.meta["mlp_hidden_dim"], scores)
            mlp_start  = 0
            mlp_ranges = {}
            for dev in self.all_devices:
                mlp_ranges[dev] = range(mlp_start, mlp_start + mlp_allocs[dev])
                mlp_start += mlp_allocs[dev]

            mlp_futures = {}
            for w_name in self.expected_workers:
                n_range = mlp_ranges[w_name]
                if len(n_range) == 0:
                    continue
                mlp_futures[w_name] = self.executor.submit(
                    self._dispatch_task,
                    w_name, "MLP", i,
                    ln_x_mlp, n_range.start, n_range.stop,
                )

            t0       = time.perf_counter()
            edge_mlp = None
            edge_n   = mlp_ranges["edge"]
            if len(edge_n) > 0:
                w1     = fc1.weight[edge_n.start:edge_n.stop, :]
                b1     = fc1.bias[edge_n.start:edge_n.stop]
                w2     = fc2.weight[:, edge_n.start:edge_n.stop]
                act_fn = block.mlp.activation_fn
                edge_mlp = act_fn(ln_x_mlp @ w1.t() + b1) @ w2.t()
            compute_time["edge"] += time.perf_counter() - t0

            mlp_parts = [edge_mlp] if edge_mlp is not None else []
            for w_name, fut in mlp_futures.items():
                res, net_t, comp_t = fut.result()
                network_time[w_name] += net_t
                compute_time[w_name] += comp_t
                if res is not None and res.numel() > 0:
                    mlp_parts.append(res.to(ln_x_mlp.device))
                else:
                    self.breakers[w_name].trip(reason="mlp dispatch returned None")

            current_state = identity + sum_mlp_parts(mlp_parts) + fc2.bias
            _blk_mlp_ms.append((time.perf_counter() - _t_mlp) * 1e3)

            # ── Evaluator update ──────────────────────────────────────────────
            for dev in self.all_devices:
                self.evaluator.record_step(
                    dev,
                    latency=network_time[dev],
                    compute_time=compute_time[dev],
                )

            # ── Circuit breaker feedback ───────────────────────────────────────
            for w_name in self.expected_workers:
                cb     = self.breakers[w_name]
                net_t  = network_time[w_name]
                failed = net_t >= PROBE_FAIL_LATENCY

                if failed and cb.is_closed:
                    cb.trip(reason="dispatch failed")
                elif self.evaluator.needs_probe(w_name) and cb.is_closed:
                    cb.trip(reason="link score degraded")
                elif cb.is_half_open:
                    if not failed:
                        cb.on_probe_success()
                    else:
                        cb.on_probe_failure(reason="still unreachable")

            # Uncomment to enable per-block diagnostics:
            # for dev in self.all_devices:
            #     print(self.evaluator.get_diagnostics(dev))

        self.last_inference_stats = {
            "per_block_attn_ms":    _blk_attn_ms,
            "per_block_mlp_ms":     _blk_mlp_ms,
            "evaluator_overhead_us": _blk_evaluator_us,
        }

        return current_state

    # ── Shutdown ─────────────────────────────────────────────────────────────

    def shutdown(self):
        for w_name, sock in self.sockets.items():
            try:
                send_msg(sock, ("QUIT", 0, None, 0, 0))
                sock.close()
            except Exception:
                pass
        self.executor.shutdown(wait=False)
        print("[Master] Shut down.")


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="CLIP MasterOrchestrator standalone test")
    p.add_argument("--port", type=int, default=29500)
    args = p.parse_args()

    orch = MasterOrchestrator(
        expected_workers=["pc_gpu"],
        host="0.0.0.0",
        port=args.port,
    )

    dummy_input = torch.randn(1, 50, 768)

    t0      = time.time()
    output  = orch.run_inference(dummy_input)
    elapsed = time.time() - t0

    print(
        f"\n[Master] Inference complete in {elapsed:.4f}s  "
        f"|  output shape: {tuple(output.shape)}"
    )

    orch.shutdown()
