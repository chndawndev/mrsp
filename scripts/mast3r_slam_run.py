#!/usr/bin/env python
"""Run MASt3R-SLAM headless on one sequence's extracted frames, uncalibrated
mode, measuring wall time and peak GPU memory (docs/pipelines/mast3r_slam.md,
Q1). GPU selection follows scripts/gpu_status.py's pattern: pick the freest
single index, export CUDA_VISIBLE_DEVICES explicitly, log nvidia-smi before
the run.

This does NOT compute any evaluation metric and does not import anything
from src/eval/ or src/gt/ -- it only invokes MASt3R-SLAM's own main.py as a
subprocess and records timing/memory around it.

Usage (from the mast3r-slam conda env):
    python scripts/mast3r_slam_run.py --sequence c1_cecum_t1_v1 --gpu 7
"""
from __future__ import annotations

import argparse
import os
import subprocess
import threading
import time
from pathlib import Path

REPO = Path("/data1_ycao/chua/projects/mrsp")
MAST3R_REPO = REPO / "scratch/pipelines/MASt3R-SLAM"
SCRATCH_ROOT = REPO / "scratch/mast3r_slam"
LOG_DIR = REPO / "logs"


def log(msg: str) -> None:
    prefix = f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}]"
    print(f"{prefix} {msg}", flush=True)


def poll_gpu_mem(gpu_index: int, stop_event: threading.Event, samples: list[int]) -> None:
    while not stop_event.is_set():
        out = subprocess.run(
            ["nvidia-smi", f"--id={gpu_index}", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True,
        ).stdout.strip()
        try:
            samples.append(int(out))
        except ValueError:
            pass
        stop_event.wait(2.0)


def run(sequence: str, gpu_index: int) -> dict:
    dataset_path = SCRATCH_ROOT / sequence
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"{dataset_path} missing -- run scripts/mast3r_slam_prepare_input.py first"
        )

    nvidia_smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
    (LOG_DIR / "mast3r_slam_run.log").open("a").write(
        f"\n=== nvidia-smi before run, {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} ===\n"
        + nvidia_smi
    )

    samples: list[int] = []
    stop_event = threading.Event()
    poller = threading.Thread(target=poll_gpu_mem, args=(gpu_index, stop_event, samples))
    poller.start()

    cmd = [
        "python", "main.py",
        "--dataset", str(dataset_path),
        "--config", "config/base.yaml",
        "--save-as", sequence,
        "--no-viz",
    ]
    log(f"launching: {' '.join(cmd)} (CUDA_VISIBLE_DEVICES={gpu_index})")
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    t0 = time.time()
    proc = subprocess.run(
        cmd, cwd=MAST3R_REPO, env=env,
        capture_output=True, text=True,
    )
    elapsed = time.time() - t0
    stop_event.set()
    poller.join()

    (LOG_DIR / f"mast3r_slam_run_{sequence}.stdout.log").write_text(proc.stdout)
    (LOG_DIR / f"mast3r_slam_run_{sequence}.stderr.log").write_text(proc.stderr)

    result = {
        "sequence": sequence,
        "gpu_index": gpu_index,
        "returncode": proc.returncode,
        "elapsed_seconds": elapsed,
        "peak_gpu_mem_mib": max(samples) if samples else None,
        "n_gpu_mem_samples": len(samples),
    }
    log(f"result: {result}")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence", required=True)
    ap.add_argument("--gpu", type=int, required=True)
    args = ap.parse_args()
    run(args.sequence, args.gpu)
