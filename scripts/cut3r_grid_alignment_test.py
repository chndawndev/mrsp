#!/usr/bin/env python
"""External test of CUT3R's input grid mapping (docs/pipelines/cut3r.md,
question 4): resize alignment (pixel-center vs corner) and crop, measured
from markers, not from a formula applied forward and back.

A synthetic 1350x1080 PNG with Gaussian markers at known sub-pixel
positions is passed through the loader CUT3R's online demo executes:
vendored demo.py::prepare_input -> src/dust3r/utils/image.py::load_images.
Markers are found on the resulting internal grid (the normalized tensor
the encoder receives) by thresholding + connected components, with no
mapping formula involved, and matched to the known positions. Then:

  1. a free least-squares fit  grid = a * original + b  per axis gives the
     measured scale and offset (the offset contains alignment and crop);
  2. residuals against the two candidate mappings
        center:  p = (o + 0.5) / scale - 0.5 - half_crop
        corner:  p =  o / scale - half_crop
     with scale = original / resized size and half_crop taken from the
     grid size (resized long side 512; (resized - grid) / 2 per axis);
  3. the adapter mapping scripts/pipeline_adapters.py::original_to_model
     (pixel-center) is checked against the located markers;
  4. the fraction of VALID (non-vignette) pixels with no depth under the
     adapter's bilinear rule (scripts/pipeline_adapters.py
     ::BilinearGridMap), using the fixed vignette mask.

CPU only, no weights, no evaluation metric.

Usage (cut3r conda env):
    python scripts/cut3r_grid_alignment_test.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path("/data1_ycao/chua/projects/mrsp")
CUT3R_REPO = REPO / "scratch/pipelines/CUT3R"
SCRATCH = REPO / "scratch/cut3r_grid_alignment"
OUT_DIR = REPO / "results/pipelines/cut3r_stage1/grid_alignment"
SIZE = 512
TOLERANCE = 0.02  # grid pixels; the two candidate formulas differ by 0.31

sys.path.insert(0, str(REPO / "scripts"))
import mast3r_slam_grid_alignment_test as mt  # noqa: E402  marker_positions, render (same markers as the MASt3R-SLAM test)
from mast3r_slam_resolution_mapping import load_vignette_mask  # noqa: E402  (duplicate of the locked loader; reads the mask file only)
from pipeline_adapters import ORIGINAL_H, ORIGINAL_W, BilinearGridMap, original_to_model  # noqa: E402

# those modules put the MASt3R-SLAM checkout on sys.path; CUT3R must not see it
sys.path[:] = [p for p in sys.path if "MASt3R-SLAM" not in p]


def detect_markers(grid: np.ndarray) -> np.ndarray:
    """(n, 2) intensity-weighted centroids (x, y) of connected bright
    blobs. No mapping formula is used."""
    from scipy import ndimage

    labels, n = ndimage.label(grid > 0.02 * grid.max())
    idx = np.arange(1, n + 1)
    cy_cx = np.array(ndimage.center_of_mass(grid, labels, idx))
    return cy_cx[:, ::-1]


def main():
    import cv2
    import PIL

    SCRATCH.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pts = mt.marker_positions()
    img8 = mt.render(pts)
    png = SCRATCH / "0000.png"
    if not cv2.imwrite(str(png), img8[:, :, ::-1]):
        raise RuntimeError(f"could not write {png}")

    os.chdir(CUT3R_REPO)
    sys.path.insert(0, str(CUT3R_REPO))
    import demo  # vendored demo.py

    views = demo.prepare_input(img_paths=[str(png)], img_mask=[True], size=SIZE, revisit=1, update=True)
    if len(views) != 1:
        raise RuntimeError(f"expected 1 view, got {len(views)}")
    img = views[0]["img"]  # (1, 3, H, W), normalized to [-1, 1]
    gh, gw = int(img.shape[-2]), int(img.shape[-1])
    true_shape = views[0]["true_shape"].numpy().tolist()
    grid = (img[0, 0].numpy().astype(np.float64) + 1.0) / 2.0
    grid[grid < 1e-6] = 0.0

    found_all = detect_markers(grid)
    # match to known markers by nearest neighbour under the rough scale gw / W (markers are > 30 grid px apart)
    rough = pts * (gw / ORIGINAL_W)
    d = np.linalg.norm(found_all[None, :, :] - rough[:, None, :], axis=2)
    j = d.argmin(axis=1)
    if len(found_all) != len(pts) or len(set(j.tolist())) != len(pts) or d.min(axis=1).max() > 10:
        raise RuntimeError(f"marker matching failed: {len(found_all)} blobs for {len(pts)} markers, "
                           f"max match distance {d.min(axis=1).max():.2f}")
    found = found_all[j]

    # 1. free fit per axis
    fit = {}
    for k, name in enumerate("xy"):
        A = np.stack([pts[:, k], np.ones(len(pts))], axis=1)
        (a, b), *_ = np.linalg.lstsq(A, found[:, k], rcond=None)
        res = found[:, k] - (a * pts[:, k] + b)
        fit[name] = {"scale_grid_per_original": float(a), "offset": float(b),
                     "original_per_grid": float(1 / a), "fit_residual_max_abs": float(np.abs(res).max())}

    # 2. candidate mappings. Resized size as load_images computes it; crop from the grid size.
    S = max(ORIGINAL_W, ORIGINAL_H)
    rw, rh = int(round(ORIGINAL_W * SIZE / S)), int(round(ORIGINAL_H * SIZE / S))
    m = {"original_w": ORIGINAL_W, "original_h": ORIGINAL_H, "model_grid_w": gw, "model_grid_h": gh,
         "resized_w": rw, "resized_h": rh,
         "scale_w": ORIGINAL_W / rw, "scale_h": ORIGINAL_H / rh,
         "half_crop_w": (rw - gw) / 2, "half_crop_h": (rh - gh) / 2}
    center = np.stack(original_to_model(pts[:, 0], pts[:, 1], m), axis=1)
    corner = np.stack([pts[:, 0] / m["scale_w"] - m["half_crop_w"], pts[:, 1] / m["scale_h"] - m["half_crop_h"]], axis=1)
    expected_fit = {
        "x": {"scale": 1 / m["scale_w"], "offset_center": 0.5 / m["scale_w"] - 0.5 - m["half_crop_w"],
              "offset_corner": -m["half_crop_w"]},
        "y": {"scale": 1 / m["scale_h"], "offset_center": 0.5 / m["scale_h"] - 0.5 - m["half_crop_h"],
              "offset_corner": -m["half_crop_h"]},
    }

    def stats(pred):
        r = found - pred
        return {"mean_dx": float(r[:, 0].mean()), "mean_dy": float(r[:, 1].mean()),
                "std_dx": float(r[:, 0].std(ddof=1)), "std_dy": float(r[:, 1].std(ddof=1)),
                "max_abs_dx": float(np.abs(r[:, 0]).max()), "max_abs_dy": float(np.abs(r[:, 1]).max())}

    # 4. valid pixels with no depth under the adapter's bilinear rule
    gm = BilinearGridMap(m)
    valid = ~load_vignette_mask(ORIGINAL_W, ORIGINAL_H)
    no_depth = valid & ~gm.available
    X, Y = np.meshgrid(np.arange(ORIGINAL_W, dtype=np.float64), np.arange(ORIGINAL_H, dtype=np.float64))
    px, py = original_to_model(X.ravel(), Y.ravel(), m)

    out = {
        "seed": mt.SEED, "n_markers": int(len(pts)), "sigma_original_px": mt.SIGMA,
        "versions": {"PIL": PIL.__version__, "cv2": cv2.__version__},
        "loader": "scratch/pipelines/CUT3R/demo.py::prepare_input -> src/dust3r/utils/image.py::load_images",
        "true_shape_reported_by_loader": true_shape,
        "mapping_parameters": m,
        "free_fit_grid_equals_a_times_original_plus_b": fit,
        "expected_fit": expected_fit,
        "units": "grid pixels; residual = located marker - formula prediction",
        "residual_vs_center_aligned_adapter_formula": stats(center),
        "residual_vs_corner_aligned": stats(corner),
        "valid_pixels": {
            "n_valid_non_vignette": int(valid.sum()), "n_valid_with_no_depth": int(no_depth.sum()),
            "frac_valid_with_no_depth": float(no_depth.sum() / valid.sum()),
            "by_cause": {"above_grid": int((no_depth & (py < 0)).sum()), "below_grid": int((no_depth & (py > gh - 1)).sum()),
                         "left_of_grid": int((no_depth & (px < 0)).sum()), "right_of_grid": int((no_depth & (px > gw - 1)).sum())},
        },
        "markers": [{"ox": float(p[0]), "oy": float(p[1]), "found_px": float(f[0]), "found_py": float(f[1])}
                    for p, f in zip(pts, found)],
    }
    print(json.dumps({k: v for k, v in out.items() if k != "markers"}, indent=2))
    (OUT_DIR / "grid_alignment_test.json").write_text(json.dumps(out, indent=2))
    c = out["residual_vs_center_aligned_adapter_formula"]
    if max(c["max_abs_dx"], c["max_abs_dy"]) > TOLERANCE:
        raise SystemExit("center-aligned adapter formula does NOT match the markers located on CUT3R's grid")


if __name__ == "__main__":
    main()
