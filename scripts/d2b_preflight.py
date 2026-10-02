#!/usr/bin/env python
"""D2b evaluation pre-flight 2 (docs/d2b_eval.md): CUT3R adapter check,
no evaluation metric. On one sequence: frames with pose and depth; the
adapter's grid mapping against the markers located on CUT3R's own grid in
Stage 1 (results/pipelines/cut3r_stage1/grid_alignment/); fraction of
valid (non-vignette) pixels with no depth; invalid predicted depth pixels
(non-finite or <= 0) and the full-resolution pixels they make
unavailable; the adapter's full-resolution depth against an independent
scalar bilinear lookup.

Usage: scratch/.venv/bin/python scripts/d2b_preflight.py [--sequence c1_cecum_t1_v1 ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/data1_ycao/chua/projects/mrsp")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from eval.evaluable import load_vignette_mask  # noqa: E402  (mask loader only; no metric)
from pipeline2_preflight import gt_frame_count  # noqa: E402
from pipeline_adapters import (  # noqa: E402
    CUT3R_MAPPING_JSON, ORIGINAL_H, ORIGINAL_W, BilinearGridMap, Cut3rAdapter, bilinear_reference,
    load_cut3r_mapping, load_mast3r_mapping, original_to_model,
)

OUT_DIR = REPO / "results/d2b_eval/preflight"
SEED = 20261003


def check(seq: str) -> dict:
    m = load_cut3r_mapping()
    gm = BilinearGridMap(m)
    n_gt = gt_frame_count(seq)
    ad = Cut3rAdapter(seq, n_gt, gm)
    valid = ~load_vignette_mask(ORIGINAL_W, ORIGINAL_H)
    out = {"sequence": seq, "mapping_parameters": {k: m[k] for k in ("model_grid_w", "model_grid_h", "scale_w", "scale_h", "half_crop_w", "half_crop_h")}}

    # mapping: adapter formula vs markers located on CUT3R's own grid (Stage 1, external test)
    marker = json.loads(CUT3R_MAPPING_JSON.read_text())
    pts = np.array([[r["ox"], r["oy"]] for r in marker["markers"]])
    found = np.array([[r["found_px"], r["found_py"]] for r in marker["markers"]])
    pred = np.stack(original_to_model(pts[:, 0], pts[:, 1], m), axis=1)
    out["mapping_vs_stage1_markers"] = {
        "n_markers": int(len(pts)), "max_abs_residual_grid_px": float(np.abs(found - pred).max()),
        "mean_residual_x": float((found - pred)[:, 0].mean()), "mean_residual_y": float((found - pred)[:, 1].mean()),
        "corner_aligned_mean_residual_x": float((found[:, 0] - (pts[:, 0] / m["scale_w"] - m["half_crop_w"])).mean()),
    }
    mm = load_mast3r_mapping()
    out["same_parameters_as_mast3r_mapping"] = all(m[k] == mm[k] for k in ("model_grid_w", "model_grid_h", "scale_w", "scale_h", "half_crop_w", "half_crop_h"))

    both = sorted(set(ad.frames_with_depth) & set(ad.frames_with_pose))
    out["counts"] = {"n_gt_frames": n_gt, "n_frames_with_depth": len(ad.frames_with_depth),
                     "n_frames_with_pose": len(ad.frames_with_pose), "n_frames_with_pose_and_depth": len(both),
                     "n_frames_missing_either": n_gt - len(both)}

    n_no_depth, n_unavail_from_invalid = 0, 0
    for i in ad.frames_with_depth:
        _, av = ad.depth_native(i)
        n_no_depth += int((valid & ~av).sum())
        n_unavail_from_invalid += int((valid & gm.available & ~av).sum())
    crop = int((valid & ~gm.available).sum())
    out["pixels"] = {
        "n_valid_non_vignette_per_frame": int(valid.sum()),
        "n_valid_with_no_depth_from_crop_per_frame": crop,
        "frac_valid_with_no_depth_from_crop": crop / int(valid.sum()),
        "frac_valid_with_no_depth_all_frames": n_no_depth / (int(valid.sum()) * len(ad.frames_with_depth)),
        "n_invalid_predicted_depth_grid_pixels": ad.n_invalid_depth_pixels,
        "frames_with_invalid_predicted_depth": sorted(ad.frames_with_invalid_depth),
        "n_valid_fullres_pixels_unavailable_because_of_invalid_depth": n_unavail_from_invalid,
    }

    rng = np.random.default_rng(SEED)
    picks = rng.choice(np.flatnonzero(valid & gm.available), size=10, replace=False)
    frames = [0, int(rng.choice(ad.frames_with_depth))] + sorted(ad.frames_with_invalid_depth)[:1]
    worst = 0.0
    for f in frames:
        z = ad.depth_grid(f)
        full, av = ad.depth_native(f)
        for p in picks:
            if not av[p]:
                continue
            y, x = divmod(int(p), ORIGINAL_W)
            px, py = original_to_model(np.array([float(x)]), np.array([float(y)]), m)
            worst = max(worst, abs(float(full[p]) - bilinear_reference(z, float(px[0]), float(py[0]))))
    full0, av0 = ad.depth_native(0)
    out["round_trip"] = {"seed": SEED, "frames": frames, "n_pixels": 10, "max_abs_diff_vs_scalar_bilinear": worst,
                         "unavailable_pixels_all_nan": bool(np.isnan(full0[~av0]).all()),
                         "available_pixels_all_finite_positive": bool((np.isfinite(full0[av0]) & (full0[av0] > 0)).all())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence", action="append")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    res = {s: check(s) for s in (args.sequence or ["c1_cecum_t1_v1"])}
    (OUT_DIR / "preflight2_cut3r_adapter.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
