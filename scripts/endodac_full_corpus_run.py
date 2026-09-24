#!/usr/bin/env python
"""Launcher for the D1 Stage A full-corpus EndoDAC run: queries free GPUs,
launches one scripts/endodac_inference.py shard per qualifying GPU, each
in the `endodac` conda env. Meant to run detached (tmux) -- this process
itself just launches shards and waits/logs, it does not do any inference.

Usage: python scripts/endodac_full_corpus_run.py
(run from the project root; typically under tmux, e.g.
  tmux new-session -d -s endodac_full_run \
    "python scripts/endodac_full_corpus_run.py 2>&1 | tee logs/endodac_full_run.log"
)
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/data1_ycao/chua/projects/mrsp")
sys.path.insert(0, str(REPO / "scripts"))

from gpu_status import get_gpu_stats  # noqa: E402

ENDODAC_PYTHON = "/data1_ycao/chua/miniforge3/envs/endodac/bin/python"
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
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    gpus = select_gpus()
    log(f"launching {len(gpus)} shard(s) on GPUs {gpus}")

    procs = []
    for shard_index, gpu_index in enumerate(gpus):
        shard_log = LOG_DIR / f"endodac_full_run_shard{shard_index}.log"
        cmd = (
            f"CUDA_VISIBLE_DEVICES={gpu_index} {ENDODAC_PYTHON} "
            f"{REPO / 'scripts/endodac_inference.py'} "
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
