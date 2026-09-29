#!/usr/bin/env python
"""MASt3R-SLAM Stage 3 pre-flight items 2 and 3 (docs/pipelines/
mast3r_slam.md): (2) reconcile the vignette-mask convention Stage 2 used
backwards, recompute the crop-loss fraction on the actually-valid
population; (3) a comparable EndoDAC ray-direction check.

Item 2: `src/eval/evaluable.py::load_vignette_mask` docstring (read, not
modified): "True = vignette (non-evaluable)". `evaluable_pixel_mask`
computes evaluability from `~vignette_mask` (mask False = valid/imaged).
Stage 2's `mast3r_slam_resolution_mapping.py::vignette_crop_fraction`
selected `np.nonzero(vignette_mask)` -- the True (invalid, ~102k-pixel
black-border) population -- and reported the crop-loss fraction of THAT,
backwards from what "valid pixels with no depth" means. This script
recomputes it on `~vignette_mask` (the ~1.356M-pixel valid population),
the correct population per src/eval/evaluable.py's own usage.

Item 3: docs/pipelines/endodac.md lines 914-915 already report EndoDAC's
predicted pinhole intrinsics (mean over 217 pairs, at its own 320x256
feed resolution, resize-only no crop): fx=265.1, fy=269.9, cx=158.7,
cy=130.4. Computes EndoDAC's implied ray direction for the same frame (0)
and same pixel set as Stage 2's MASt3R ray-direction check, compared
against this project's own Scaramuzza model (geometry.camera.unproject,
unlocked src/geometry/).

Does NOT import from src/eval/ or src/eval_ext/ (this task's hard
constraint) and does NOT modify src/eval/evaluable.py -- read only, to
confirm the convention.

Usage (mast3r-slam conda env, GPU needed for the MASt3R forward pass
this reuses via mast3r_slam_resolution_mapping.ray_direction_check):
    python scripts/mast3r_slam_check_vignette_and_endodac_ray.py --gpu 7
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/data1_ycao/chua/projects/mrsp")
PERFRAME_ROOT = REPO / "results/pipelines/mast3r_slam_perframe"
INTRINSICS_PATH = "/data1_ycao/chua/datasets/C3VDv2/camera_intrinsics.txt"
ORIGINAL_W, ORIGINAL_H = 1350, 1080

# EndoDAC's own predicted pinhole intrinsics, mean over 217 pairs, at its
# own 320x256 feed resolution -- docs/pipelines/endodac.md:914-915.
ENDODAC_FEED_W, ENDODAC_FEED_H = 320, 256
ENDODAC_FX, ENDODAC_FY, ENDODAC_CX, ENDODAC_CY = 265.1, 269.9, 158.7, 130.4

sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))


def confirm_vignette_convention() -> dict:
    """Reads (not modifies) src/eval/evaluable.py's docstring/usage to
    confirm the True=vignette, False=valid convention, quoted verbatim."""
    text = (REPO / "src/eval/evaluable.py").read_text()
    normalized = " ".join(text.split())
    assert "True = vignette (non-evaluable)" in normalized, (
        "src/eval/evaluable.py's load_vignette_mask docstring changed -- "
        "re-confirm the convention before trusting this script's fix"
    )
    assert "hit & ~vignette_mask & (d_hit <= MAX_HIT_DISTANCE_MM)" in normalized, (
        "src/eval/evaluable.py's evaluable_pixel_mask changed -- re-confirm the convention"
    )
    return {
        "file": "src/eval/evaluable.py",
        "load_vignette_mask_docstring": "True = vignette (non-evaluable)",
        "evaluable_pixel_mask_formula": "hit & ~vignette_mask & (d_hit <= MAX_HIT_DISTANCE_MM)",
        "conclusion": "True = vignette/invalid (black border); False = valid/imaged. "
                      "Evaluable candidates are the ~vignette_mask (False) population.",
    }


def recompute_crop_fraction_on_valid_pixels(m: dict) -> dict:
    from mast3r_slam_resolution_mapping import (
        load_vignette_mask, original_to_model, EXPECTED_VIGNETTE_PIXEL_COUNT,
    )

    vignette_mask = load_vignette_mask(ORIGINAL_W, ORIGINAL_H).reshape(ORIGINAL_H, ORIGINAL_W)
    valid_mask = ~vignette_mask
    valid_ys, valid_xs = np.nonzero(valid_mask)

    px, py = original_to_model(valid_xs.astype(np.float64), valid_ys.astype(np.float64), m)
    covered = (
        (px >= 0) & (px <= m["model_grid_w"] - 1) &
        (py >= 0) & (py <= m["model_grid_h"] - 1)
    )
    n_valid = len(valid_xs)
    n_uncovered = int((~covered).sum())
    return {
        "n_vignette_pixels_true_invalid": EXPECTED_VIGNETTE_PIXEL_COUNT,
        "n_valid_pixels_false_population": n_valid,
        "n_valid_pixels_total_check": ORIGINAL_W * ORIGINAL_H - EXPECTED_VIGNETTE_PIXEL_COUNT,
        "n_valid_pixels_uncovered_by_model_grid": n_uncovered,
        "fraction_valid_pixels_no_depth_from_cropping": n_uncovered / n_valid if n_valid else float("nan"),
    }


def endodac_ray_direction_check(sequence: str, frame_index: int, gpu: int, m: dict) -> dict:
    """Reruns MASt3R-SLAM's own confident-pixel selection on the SAME
    frame (index 0) via the exact same code path Stage 2's check used
    (mast3r_slam_resolution_mapping.ray_direction_check, imported, not
    duplicated), then evaluates EndoDAC's implied pinhole ray at those
    SAME original-frame pixel coordinates -- literally "the same valid
    pixels as the Stage 2 MASt3R check", not a redefined population.
    Deterministic (torch.inference_mode, no dropout at eval), so this
    reproduces Stage 2's confident-pixel set exactly; the Scaramuzza
    numbers recomputed here are also compared against Stage 2's saved
    numbers as a consistency check on that determinism assumption."""
    from geometry.camera import CameraIntrinsics, unproject
    from mast3r_slam_resolution_mapping import mast3r_confident_pixels_original_coords

    ox, oy, mast3r_dirs = mast3r_confident_pixels_original_coords(sequence, frame_index, gpu, m)

    intr = CameraIntrinsics.from_file(INTRINSICS_PATH)
    px_orig = np.stack([ox, oy], axis=-1)
    geom_dirs = unproject(px_orig, intr)

    mast3r_dot = np.clip(np.sum(mast3r_dirs * geom_dirs, axis=-1), -1.0, 1.0)
    mast3r_angle_deg = np.degrees(np.arccos(mast3r_dot))
    mast3r_recheck = {
        "n_pixels_compared": int(len(ox)),
        "median_angle_deg": float(np.median(mast3r_angle_deg)),
        "p95_angle_deg": float(np.percentile(mast3r_angle_deg, 95)),
        "mean_angle_deg": float(np.mean(mast3r_angle_deg)),
        "max_angle_deg": float(np.max(mast3r_angle_deg)),
    }

    ex = ox * (ENDODAC_FEED_W / ORIGINAL_W)
    ey = oy * (ENDODAC_FEED_H / ORIGINAL_H)
    ray_x = (ex - ENDODAC_CX) / ENDODAC_FX
    ray_y = (ey - ENDODAC_CY) / ENDODAC_FY
    ray_z = np.ones_like(ray_x)
    endodac_dirs = np.stack([ray_x, ray_y, ray_z], axis=-1)
    endodac_dirs = endodac_dirs / np.linalg.norm(endodac_dirs, axis=-1, keepdims=True)

    dot = np.clip(np.sum(endodac_dirs * geom_dirs, axis=-1), -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(dot))

    endodac_result = {
        "n_pixels_compared": int(len(ox)),
        "pixel_population": "MASt3R's own confident pixels (frame 0), same set both checks use",
        "median_angle_deg": float(np.median(angle_deg)),
        "p95_angle_deg": float(np.percentile(angle_deg, 95)),
        "mean_angle_deg": float(np.mean(angle_deg)),
        "max_angle_deg": float(np.max(angle_deg)),
    }

    check_path = PERFRAME_ROOT / sequence / "resolution_mapping.json"
    prior = json.loads(check_path.read_text())
    stage2_saved = prior["ray_direction_check"]
    determinism_check = {
        "stage2_saved_median_deg": stage2_saved["median_angle_deg"],
        "this_rerun_median_deg": mast3r_recheck["median_angle_deg"],
        "diff_deg": abs(stage2_saved["median_angle_deg"] - mast3r_recheck["median_angle_deg"]),
    }

    return {
        "mast3r_check_rerun_this_script": mast3r_recheck,
        "mast3r_check_stage2_saved": stage2_saved,
        "determinism_check": determinism_check,
        "endodac_check": endodac_result,
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence", default="c1_cecum_t1_v1")
    ap.add_argument("--frame-index", type=int, default=0)
    ap.add_argument("--gpu", type=int, required=True)
    args = ap.parse_args()

    from mast3r_slam_resolution_mapping import compute_resize_mapping, SCRATCH_ROOT

    convention = confirm_vignette_convention()
    print("vignette convention (confirmed by reading src/eval/evaluable.py, not modified):")
    print(json.dumps(convention, indent=2))

    sample_frame = SCRATCH_ROOT / args.sequence / f"{args.frame_index:04d}.png"
    m = compute_resize_mapping(sample_frame)

    corrected = recompute_crop_fraction_on_valid_pixels(m)
    print("\ncorrected crop fraction (valid pixels):")
    print(json.dumps(corrected, indent=2))

    endodac = endodac_ray_direction_check(args.sequence, args.frame_index, args.gpu, m)
    print("\nEndoDAC vs MASt3R ray-direction comparison:")
    print(json.dumps(endodac, indent=2))

    out = {
        "vignette_convention": convention,
        "corrected_valid_pixel_crop_fraction": corrected,
        "endodac_vs_mast3r_ray_comparison": endodac,
    }
    out_path = PERFRAME_ROOT / args.sequence / "vignette_and_endodac_ray_check.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
