"""
simulate_devices.py — Run Fake Edge Devices on One Computer
============================================================
Launches multiple WorkerDevice processes on localhost,
each simulating a different edge device (Pi, Manifold, etc.)

Run this BEFORE starting the robot controller:
    python simulate_devices.py

Then in another terminal (or Webots), set USE_DISTRIBUTED=True
and run the robot controller.
"""

import sys
import os
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'spvit'))

from spvit.coordinator import WorkerDevice, BASE_PORT

NUM_DEVICES = 3
TOTAL_HEADS = 8


def start_worker(device_id, head_indices):
    """Start one worker device in its own thread."""
    worker = WorkerDevice(
        device_id=device_id,
        port=BASE_PORT + device_id,
        head_indices=head_indices,
        embed_dim=256,
    )
    worker.start()


def main():
    # Split 8 heads across 3 devices
    # Device 0 (fastest): heads 0-3 (4 heads)
    # Device 1 (medium):  heads 4-5 (2 heads)
    # Device 2 (slowest): heads 6-7 (2 heads)
    # In real SPViT, ARIMA-V adjusts this dynamically
    head_splits = [
        [0, 1, 2, 3],   # Device 0 — simulated Manifold (fast)
        [4, 5],          # Device 1 — simulated Pi (medium)
        [6, 7],          # Device 2 — simulated Pi (slower)
    ]

    print("=" * 50)
    print("SPViT Device Simulator")
    print("=" * 50)
    print(f"Starting {NUM_DEVICES} simulated devices...")
    print(f"Head allocation: {head_splits}")
    print()

    threads = []
    for d in range(NUM_DEVICES):
        t = threading.Thread(
            target=start_worker,
            args=(d, head_splits[d]),
            daemon=True
        )
        t.start()
        threads.append(t)
        time.sleep(0.1)

    print(f"\nAll {NUM_DEVICES} workers listening. Waiting for coordinator...")
    print("Press Ctrl+C to stop.\n")

    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        print("\nShutting down simulated devices...")


if __name__ == '__main__':
    main()
