#!/usr/bin/env python
"""Dump the pixel population of the Stage 2 MASt3R ray-direction check
(MASt3R's own confident pixels on one frame) as integer grid indices, with
MASt3R's implied unit ray at each, so that the CUT3R Stage 1 ray-direction
check (scripts/cut3r_stage1_checks.py) compares EndoDAC, MASt3R-SLAM and
CUT3R on literally the same pixels.

Reuses scripts/mast3r_slam_resolution_mapping.py
::mast3r_confident_pixels_original_coords (one deterministic MASt3R
forward pass). The original-frame coordinates it returns come from the
pixel-center mapping (corrected 2026-10-01); the integer grid indices are
recovered by the inverse mapping and checked to be exact.

No evaluation metric. Usage (mast3r-slam conda env, one GPU):
    python scripts/cut3r_dump_mast3r_confident_pixels.py --sequence c1_cecum_t1_v1 --gpu <i>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/data1_ycao/chua/projects/mrsp")
sys.path.insert(0, str(REPO / "scripts"))

import mast3r_slam_resolution_mapping as rm  # noqa: E402

OUT_ROOT = REPO / "results/pipelines/cut3r_stage1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence", required=True)
    ap.add_argument("--frame-index", type=int, default=0)
    ap.add_argument("--gpu", type=int, required=True)
    args = ap.parse_args()

    m = json.loads((rm.PERFRAME_ROOT / "c1_cecum_t1_v1" / "resolution_mapping.json").read_text())["resolution_mapping"]
    ox, oy, dirs = rm.mast3r_confident_pixels_original_coords(args.sequence, args.frame_index, args.gpu, m)
    px, py = rm.original_to_model(ox, oy, m)
    gx, gy = np.round(px).astype(np.int64), np.round(py).astype(np.int64)
    err = max(float(np.abs(px - gx).max()), float(np.abs(py - gy).max()))
    if err > 1e-6:
        raise RuntimeError(f"confident pixels are not integer grid pixels (max deviation {err})")
    out_dir = OUT_ROOT / args.sequence
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"mast3r_confident_pixels_frame{args.frame_index}.npz"
    np.savez_compressed(out, gx=gx, gy=gy, ox=ox, oy=oy, mast3r_dirs=dirs.astype(np.float64))
    print(f"{len(gx)} confident pixels -> {out}")


if __name__ == "__main__":
    main()
