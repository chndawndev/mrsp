#!/usr/bin/env python
"""GPU availability guard: print nvidia-smi's table, recommend the freest GPU
index, and exit non-zero if no GPU has at least --min-free-gb free memory.

Use at the start of any GPU job, e.g.:
    python scripts/gpu_status.py --min-free-gb 8 || exit 1
    export CUDA_VISIBLE_DEVICES=$(python scripts/gpu_status.py --min-free-gb 8 --print-index)
"""
from __future__ import annotations

import argparse
import subprocess
import sys

QUERY_FIELDS = ["index", "name", "memory.used", "memory.total", "memory.free", "utilization.gpu"]


def get_gpu_table() -> str:
    return subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True).stdout


def get_gpu_stats() -> list[dict]:
    result = subprocess.run(
        ["nvidia-smi", f"--query-gpu={','.join(QUERY_FIELDS)}", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    )
    gpus = []
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        gpus.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "memory_used_mib": int(parts[2]),
                "memory_total_mib": int(parts[3]),
                "memory_free_mib": int(parts[4]),
                "utilization_pct": int(parts[5]),
            }
        )
    return gpus


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--min-free-gb",
        type=float,
        default=8.0,
        help="minimum free memory (GB, 1 GB = 1000 MiB) required on at least one GPU (default: 8)",
    )
    ap.add_argument(
        "--print-index",
        action="store_true",
        help="also print just the recommended GPU index on its own line (for use in $(...) capture)",
    )
    args = ap.parse_args()
    min_free_mib = args.min_free_gb * 1000

    try:
        table = get_gpu_table()
    except FileNotFoundError:
        print("gpu_status: nvidia-smi not found -- no NVIDIA GPU / driver available", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as exc:
        print(f"gpu_status: nvidia-smi failed: {exc.stderr}", file=sys.stderr)
        return 1

    print(table)

    gpus = get_gpu_stats()
    if not gpus:
        print("gpu_status: nvidia-smi reported zero GPUs", file=sys.stderr)
        return 1

    freest = max(gpus, key=lambda g: g["memory_free_mib"])
    free_gb = freest["memory_free_mib"] / 1000
    print(
        f"Recommended GPU: index {freest['index']} ({freest['name']}) -- "
        f"{free_gb:.1f} GB free, {freest['utilization_pct']}% utilized"
    )

    if args.print_index:
        print(freest["index"])

    if freest["memory_free_mib"] < min_free_mib:
        print(
            f"gpu_status: no GPU has >= {args.min_free_gb:.1f} GB free "
            f"(best is index {freest['index']} with {free_gb:.1f} GB)",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
