#!/usr/bin/env python
"""CUT3R Stage 2, step 1: per-frame comparison of two inference paths on
one sequence (docs/pipelines/cut3r.md). Reads two output directories of
scripts/cut3r_run.py. No evaluation metric.

Per frame:
  - translation difference, model units: ||t_a - t_b||, also relative to
    the trajectory extent (max distance of any frame from frame 0 in
    path a);
  - rotation difference, degrees: angle of R_a^T R_b;
  - Z-depth: max abs difference, and max and median relative difference
    |z_a - z_b| / |z_a|, over all grid pixels (every pixel has finite,
    positive z in both paths; checked).

DECISION RULE AND THRESHOLDS -- fixed and committed before any comparison
was run (2026-10-02):
  EQUIVALENT iff, for every frame of every compared sequence,
      relative translation difference <= 1e-4 of the trajectory extent,
      rotation difference            <= 1e-4 rad (0.00573 deg),
      max relative Z difference      <= 1e-4.
  Otherwise NOT EQUIVALENT.
Why 1e-4: float32 carries about 7 significant digits (eps 1.2e-7). The two
paths run the same weights on the same inputs and differ only in how
frames are batched through the encoder, which changes the order of
float32 operations. Such differences start at ~1e-7 relative per
operation and can accumulate through 36 transformer blocks, the DPT head
and up to several hundred recurrent state updates; 1e-4 (about 1,000 eps)
is taken as the upper end of what accumulation of round-off can produce.
A difference above it is three or more orders above eps and is treated as
a real difference between the paths, whatever its cause.

Usage: scratch/.venv/bin/python scripts/cut3r_compare_paths.py --a <tag_a> --b <tag_b>
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

REPO = Path("/data1_ycao/chua/projects/mrsp")
OUT_ROOT = REPO / "results/pipelines/cut3r_stage1"
REL_TRANSLATION_MAX = 1e-4
ROTATION_MAX_RAD = 1e-4
REL_DEPTH_MAX = 1e-4


def dist(x: np.ndarray) -> dict:
    q = np.percentile(x, [50, 95, 99])
    return {"min": float(x.min()), "median": float(q[0]), "p95": float(q[1]), "p99": float(q[2]),
            "max": float(x.max()), "argmax_frame": int(x.argmax())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="tag under results/pipelines/cut3r_stage1 (reference path)")
    ap.add_argument("--b", required=True)
    args = ap.parse_args()
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] thresholds (fixed before the comparison): "
          f"relative translation <= {REL_TRANSLATION_MAX}, rotation <= {ROTATION_MAX_RAD} rad "
          f"({np.degrees(ROTATION_MAX_RAD):.5f} deg), max relative Z <= {REL_DEPTH_MAX}", flush=True)
    da, db = OUT_ROOT / args.a, OUT_ROOT / args.b
    ma, mb = (json.loads((d / "MANIFEST.json").read_text()) for d in (da, db))
    if ma["status"] != "ok" or mb["status"] != "ok":
        raise SystemExit("a run did not finish ok")
    Pa, Pb = np.load(da / "poses_c2w.npy"), np.load(db / "poses_c2w.npy")
    if Pa.shape != Pb.shape:
        raise SystemExit(f"pose arrays differ in shape: {Pa.shape} vs {Pb.shape}")
    n = len(Pa)
    extent = float(np.linalg.norm(Pa[:, :3, 3] - Pa[0, :3, 3], axis=1).max())
    dt = np.linalg.norm(Pa[:, :3, 3] - Pb[:, :3, 3], axis=1)
    cosang = (np.einsum("nij,nij->n", Pa[:, :3, :3], Pb[:, :3, :3]) - 1) / 2
    rot_rad = np.arccos(np.clip(cosang, -1, 1))
    # arccos loses precision near 0; use the chord for small angles
    chord = np.linalg.norm(Pa[:, :3, :3] - Pb[:, :3, :3], axis=(1, 2)) / np.sqrt(2)
    rot_rad = np.where(rot_rad < 1e-3, chord, rot_rad)

    z_abs, z_rel_max, z_rel_med, n_bad = np.zeros(n), np.zeros(n), np.zeros(n), 0
    for i in range(n):
        with np.load(da / "depth" / f"{i:04d}.npz") as fa, np.load(db / "depth" / f"{i:04d}.npz") as fb:
            za, zb = fa["z"].astype(np.float64), fb["z"].astype(np.float64)
        if za.shape != zb.shape:
            raise SystemExit(f"frame {i}: depth shapes differ")
        ok = np.isfinite(za) & np.isfinite(zb) & (za > 0) & (zb > 0)
        n_bad += int((~ok).sum())
        d = np.abs(za - zb)
        z_abs[i] = d.max()
        rel = d / np.abs(za)
        z_rel_max[i], z_rel_med[i] = rel.max(), np.median(rel)
    if n_bad:
        raise SystemExit(f"{n_bad} pixels are non-finite or non-positive in one of the paths")

    res = {
        "a": args.a, "b": args.b, "path_a": ma.get("path", "parallel"), "path_b": mb.get("path", "parallel"),
        "n_frames": n, "grid": [ma["grid_h"], ma["grid_w"]],
        "trajectory_extent_model_units": extent,
        "translation_diff_model_units": dist(dt),
        "translation_diff_relative_to_extent": dist(dt / extent),
        "rotation_diff_deg": dist(np.degrees(rot_rad)),
        "z_max_abs_diff_model_units": dist(z_abs),
        "z_max_relative_diff": dist(z_rel_max),
        "z_median_relative_diff": dist(z_rel_med),
        "bit_identical": bool((dt == 0).all() and (rot_rad == 0).all() and (z_abs == 0).all()),
        "thresholds": {"relative_translation": REL_TRANSLATION_MAX, "rotation_rad": ROTATION_MAX_RAD,
                       "relative_depth": REL_DEPTH_MAX},
        "n_frames_over_threshold": {
            "translation": int((dt / extent > REL_TRANSLATION_MAX).sum()),
            "rotation": int((rot_rad > ROTATION_MAX_RAD).sum()),
            "depth": int((z_rel_max > REL_DEPTH_MAX).sum()),
        },
        "peak_gpu_allocated_mib": {"a": ma["peak_gpu_allocated_mib"], "b": mb["peak_gpu_allocated_mib"]},
        "peak_gpu_reserved_mib": {"a": ma["peak_gpu_reserved_mib"], "b": mb["peak_gpu_reserved_mib"]},
        "inference_seconds": {"a": ma["inference_seconds"], "b": mb["inference_seconds"]},
        "per_frame": {"translation_rel": (dt / extent).tolist(), "rotation_deg": np.degrees(rot_rad).tolist(),
                      "z_max_rel": z_rel_max.tolist(), "z_median_rel": z_rel_med.tolist()},
    }
    res["equivalent_on_this_sequence"] = not any(res["n_frames_over_threshold"].values())
    out = OUT_ROOT / f"compare__{args.a}__vs__{args.b}.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "per_frame"}, indent=2))


if __name__ == "__main__":
    main()
