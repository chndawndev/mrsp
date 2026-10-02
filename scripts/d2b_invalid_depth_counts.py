#!/usr/bin/env python
"""D2b pre-flight 3 (docs/d2b_eval.md): count invalid predicted depth
pixels (non-finite, or not strictly positive; docs/eval_protocol.md
"2026-10-03: Invalid predicted depth") in a pipeline's saved primary
outputs, every frame of every registered sequence, on the pipeline's own
depth grid. Counting only; no evaluation metric.

Usage: scratch/.venv/bin/python scripts/d2b_invalid_depth_counts.py --pipeline endodac --workers 16
"""
from __future__ import annotations

import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/data1_ycao/chua/projects/mrsp")
sys.path.insert(0, str(REPO / "scripts"))
from cut3r_run_corpus import discover_sequences  # noqa: E402

ROOTS = {"endodac": REPO / "results/pipelines/endodac_full_run",
         "mast3r_slam": REPO / "results/pipelines/mast3r_slam_full_run",
         "cut3r": REPO / "results/pipelines/cut3r_full_run"}
OUT_DIR = REPO / "results/d2b_eval/preflight"


def count(args) -> dict:
    pipeline, seq = args
    d = ROOTS[pipeline] / seq / "depth"
    files = sorted(d.glob("*_pred_depth.npy" if pipeline == "endodac" else "*.npz"))
    if not files:
        raise RuntimeError(f"{pipeline}/{seq}: no depth files")
    n_px = n_nonfinite = n_nonpositive = n_frames_bad = 0
    for f in files:
        if pipeline == "endodac":
            z = np.load(f)
        else:
            with np.load(f) as npz:
                z = npz["z"]
        finite = np.isfinite(z)
        nf = int((~finite).sum())
        npos = int((finite & (z <= 0)).sum())
        n_px += z.size
        n_nonfinite += nf
        n_nonpositive += npos
        n_frames_bad += int(nf + npos > 0)
    return {"pipeline": pipeline, "sequence": seq, "n_depth_frames": len(files), "n_pixels": n_px,
            "n_nonfinite": n_nonfinite, "n_finite_nonpositive": n_nonpositive,
            "n_invalid": n_nonfinite + n_nonpositive, "n_frames_with_invalid": n_frames_bad}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", required=True, choices=sorted(ROOTS))
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seqs = discover_sequences()
    t0 = time.time()
    rows = []
    with Pool(args.workers) as pool:
        for k, r in enumerate(pool.imap_unordered(count, [(args.pipeline, s) for s in seqs])):
            rows.append(r)
            if (k + 1) % 10 == 0 or k + 1 == len(seqs):
                el = time.time() - t0
                print(f"[{time.strftime('%H:%M:%SZ', time.gmtime())}] {k + 1}/{len(seqs)} elapsed {el / 60:.1f} min "
                      f"ETA {el / (k + 1) * (len(seqs) - k - 1) / 60:.1f} min", flush=True)
    df = pd.DataFrame(rows).sort_values("sequence")
    df.to_csv(OUT_DIR / f"invalid_depth_counts_{args.pipeline}.csv", index=False)
    print(f"{args.pipeline}: {len(df)} sequences, {int(df.n_depth_frames.sum())} frames, {int(df.n_pixels.sum())} pixels, "
          f"non-finite {int(df.n_nonfinite.sum())}, finite non-positive {int(df.n_finite_nonpositive.sum())}, "
          f"sequences with any invalid pixel: {df.loc[df.n_invalid > 0, 'sequence'].tolist()}")


if __name__ == "__main__":
    main()
