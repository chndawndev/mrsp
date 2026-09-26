#!/usr/bin/env python
"""Launcher for the D1 Stage B full-corpus evaluation: queries free GPUs,
launches one scripts/eval_d1_sequence.py shard per qualifying GPU
(scratch/.venv's python -- this is geometry/warp ray-casting work, not
EndoDAC's torch env). Meant to run detached (tmux); this process just
launches shards and waits/logs.

Usage: python scripts/eval_d1_run_corpus.py
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/data1_ycao/chua/projects/mrsp")
sys.path.insert(0, str(REPO / "scripts"))

from gpu_status import get_gpu_stats  # noqa: E402

EVAL_PYTHON = str(REPO / "scratch/.venv/bin/python")
MIN_FREE_GB_PER_GPU = 8.0
LOG_DIR = REPO / "logs"


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}", flush=True)


def select_gpus() -> list[int]:
    log("=== GPU availability check (scripts/gpu_status.py) ===")
    table = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
    log(table)
    gpus = get_gpu_stats()
    qualifying = [g["index"] for g in gpus if g["memory_free_mib"] / 1000 >= MIN_FREE_GB_PER_GPU]
    for g in gpus:
        flag = "SELECTED" if g["index"] in qualifying else "skip"
        log(f"  GPU {g['index']} ({g['name']}): {g['memory_free_mib']/1000:.1f}GB free, "
            f"{g['utilization_pct']}% util -- {flag}")
    if not qualifying:
        raise RuntimeError(f"no GPU with >= {MIN_FREE_GB_PER_GPU}GB free -- cannot start the full run")
    return qualifying


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=str, default=None,
                         help="comma-separated GPU indices to use, overriding auto-selection "
                              "(e.g. to pin to a specific idle GPU and skip one that's shared "
                              "with another active job even if it clears the free-memory threshold)")
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if args.gpus:
        gpus = [int(x) for x in args.gpus.split(",")]
        log(f"using explicit GPU override: {gpus} (auto-selection skipped)")
    else:
        gpus = select_gpus()
    log(f"launching {len(gpus)} shard(s) on GPUs {gpus}")

    procs = []
    for shard_index, gpu_index in enumerate(gpus):
        shard_log = LOG_DIR / f"eval_d1_run_corpus_shard{shard_index}.log"
        cmd = (
            f"CUDA_VISIBLE_DEVICES={gpu_index} {EVAL_PYTHON} "
            f"{REPO / 'scripts/eval_d1_sequence.py'} "
            f"--shard-index {shard_index} --shard-total {len(gpus)} "
            f">> {shard_log} 2>&1"
        )
        log(f"shard {shard_index}: GPU {gpu_index}, log {shard_log}")
        p = subprocess.Popen(cmd, shell=True, cwd=str(REPO))
        procs.append((shard_index, gpu_index, p))

    exit_codes = {}
    for shard_index, gpu_index, p in procs:
        rc = p.wait()
        exit_codes[shard_index] = rc
        log(f"shard {shard_index} (GPU {gpu_index}) exited with code {rc}")

    failed = {i: rc for i, rc in exit_codes.items() if rc != 0}
    if failed:
        log(f"*** {len(failed)} shard(s) exited non-zero: {failed} -- check per-shard logs ***")
        sys.exit(1)
    log("all shards completed")


if __name__ == "__main__":
    main()
