#!/usr/bin/env python
"""Direct test of MASt3R-SLAM's input resize alignment: is its internal
512x400 grid pixel-center aligned to the 1350x1080 input
    ox = (px + half_crop_w + 0.5) * scale_w - 0.5
or corner aligned (Stage 2's documented mapping)
    ox = (px + half_crop_w) * scale_w ?

A synthetic 1350x1080 image with Gaussian markers at known sub-pixel
positions is written as PNG and passed through the code path MASt3R-SLAM
executes for an RGB folder: mast3r_slam.dataloader.RGBFiles (cv2.imread,
BGR->RGB, /255) -> mast3r_slam.frame.create_frame (main.py:263), which
calls mast3r_slam.mast3r_utils.resize_img -> dust3r's _resize_pil_image
(PIL LANCZOS) -> integer center crop. Markers are located on the
resulting grid (frame.uimg, the image the model consumes before
normalization) by intensity-weighted centroid and compared against both
formulas. CPU only, no model weights, no evaluation metric.

Usage (mast3r-slam conda env):
    python scripts/mast3r_slam_grid_alignment_test.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path("/data1_ycao/chua/projects/mrsp")
MAST3R_REPO = REPO / "scratch/pipelines/MASt3R-SLAM"
SCRATCH = REPO / "scratch/mast3r_slam_grid_alignment"
OUT_DIR = REPO / "results/pipelines/mast3r_slam_grid_alignment"
ORIGINAL_W, ORIGINAL_H = 1350, 1080
SIGMA = 8.0   # marker std, original pixels (about 3 grid pixels)
WINDOW = 12   # centroid half-window, grid pixels (about 4 sigma)
SEED = 20261001
ADAPTER_TOLERANCE = 0.02  # grid pixels; the two candidate formulas differ by 0.31

sys.path.insert(0, str(MAST3R_REPO))


def marker_positions() -> np.ndarray:
    """(n, 2) marker centers (x, y) in original pixel-index coordinates
    (integer = pixel center). Asymmetric jittered lattice with sub-pixel
    offsets; every marker at least 60 px from each image edge."""
    rng = np.random.default_rng(SEED)
    xs = np.linspace(90, ORIGINAL_W - 90, 9)
    ys = np.linspace(90, ORIGINAL_H - 90, 7)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
    pts += rng.uniform(-20, 20, size=pts.shape)
    return pts


def render(pts: np.ndarray) -> np.ndarray:
    """uint8 (H, W, 3). Each marker is a Gaussian evaluated at pixel
    centers (pixel index i has its center at coordinate i)."""
    yy, xx = np.mgrid[0:ORIGINAL_H, 0:ORIGINAL_W].astype(np.float64)
    img = np.zeros((ORIGINAL_H, ORIGINAL_W))
    for x, y in pts:
        img += np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * SIGMA**2))
    img8 = np.clip(np.round(img * 250.0), 0, 255).astype(np.uint8)
    return np.repeat(img8[:, :, None], 3, axis=2)


def centroid(gray: np.ndarray, cx: float, cy: float) -> tuple[float, float]:
    x0, y0 = int(round(cx)), int(round(cy))
    xa, xb = x0 - WINDOW, x0 + WINDOW + 1
    ya, yb = y0 - WINDOW, y0 + WINDOW + 1
    if xa < 0 or ya < 0 or xb > gray.shape[1] or yb > gray.shape[0]:
        raise RuntimeError(f"centroid window for marker near ({cx:.1f}, {cy:.1f}) leaves the grid")
    win = gray[ya:yb, xa:xb].astype(np.float64)
    yy, xx = np.mgrid[ya:yb, xa:xb].astype(np.float64)
    s = win.sum()
    if s <= 0:
        raise RuntimeError(f"no marker signal near ({cx:.1f}, {cy:.1f})")
    return float((win * xx).sum() / s), float((win * yy).sum() / s)


def main():
    import cv2
    import PIL
    import torch
    from mast3r_slam.config import load_config
    from mast3r_slam.dataloader import RGBFiles
    from mast3r_slam.frame import create_frame
    from mast3r_slam.mast3r_utils import resize_img

    SCRATCH.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pts = marker_positions()
    img8 = render(pts)
    png = SCRATCH / "0000.png"
    if not cv2.imwrite(str(png), img8[:, :, ::-1]):
        raise RuntimeError(f"could not write {png}")

    # MASt3R-SLAM's own path: dataset -> create_frame (main.py:263).
    load_config(str(MAST3R_REPO / "config" / "base.yaml"))
    dataset = RGBFiles(str(SCRATCH))
    if len(dataset) != 1:
        raise RuntimeError(f"expected exactly 1 synthetic frame, found {len(dataset)}")
    _, img = dataset[0]
    if img.shape != (ORIGINAL_H, ORIGINAL_W, 3):
        raise RuntimeError(f"dataset image shape {img.shape}")
    frame = create_frame(0, img, torch.zeros(1), img_size=dataset.img_size, device="cpu")
    grid = frame.uimg.numpy()[:, :, 0].astype(np.float64)  # (400, 512), in [0, 1]
    model_in = frame.img[0, 0].numpy().astype(np.float64)   # normalized tensor the encoder receives
    if not np.allclose(model_in, (grid - 0.5) / 0.5, atol=1e-6):
        raise RuntimeError("frame.img is not the normalized frame.uimg")

    _, (sw, sh, hcw, hch) = resize_img(img, dataset.img_size, return_transformation=True)
    gh, gw = grid.shape

    corner = np.stack([pts[:, 0] / sw - hcw, pts[:, 1] / sh - hch], axis=1)
    center = np.stack([(pts[:, 0] + 0.5) / sw - 0.5 - hcw, (pts[:, 1] + 0.5) / sh - 0.5 - hch], axis=1)
    found = np.array([centroid(grid, cx, cy) for cx, cy in center])
    # Seed the centroid window from the other hypothesis too: the result
    # must not depend on which formula placed the window.
    found_alt = np.array([centroid(grid, cx, cy) for cx, cy in corner])

    def stats(pred):
        d = found - pred
        return {
            "mean_dx": float(d[:, 0].mean()), "mean_dy": float(d[:, 1].mean()),
            "std_dx": float(d[:, 0].std(ddof=1)), "std_dy": float(d[:, 1].std(ddof=1)),
            "max_abs_dx": float(np.abs(d[:, 0]).max()), "max_abs_dy": float(np.abs(d[:, 1]).max()),
            "rms": float(np.sqrt((d**2).sum(axis=1).mean())),
        }

    # The mapping the evaluation adapter actually applies.
    sys.path.insert(0, str(REPO / "scripts"))
    from pipeline_adapters import original_to_model as adapter_original_to_model
    m = {"scale_w": float(sw), "scale_h": float(sh), "half_crop_w": float(hcw), "half_crop_h": float(hch)}
    adapter_pred = np.stack(adapter_original_to_model(pts[:, 0], pts[:, 1], m), axis=1)

    out = {
        "seed": SEED, "n_markers": int(len(pts)), "sigma_original_px": SIGMA,
        "centroid_half_window_grid_px": WINDOW,
        "versions": {"PIL": PIL.__version__, "cv2": cv2.__version__},
        "grid_w": int(gw), "grid_h": int(gh),
        "scale_w": float(sw), "scale_h": float(sh), "half_crop_w": float(hcw), "half_crop_h": float(hch),
        "units": "grid pixels; residual = located marker - formula prediction",
        "predicted_separation_center_minus_corner": {
            "dx": float(0.5 / sw - 0.5), "dy": float(0.5 / sh - 0.5),
        },
        "residual_vs_corner_aligned": stats(corner),
        "residual_vs_center_aligned": stats(center),
        "residual_vs_adapter_original_to_model": stats(adapter_pred),
        "centroid_window_seed_dependence_max_abs": float(np.abs(found - found_alt).max()),
        "markers": [
            {"ox": float(p[0]), "oy": float(p[1]), "found_px": float(f[0]), "found_py": float(f[1]),
             "corner_px": float(c[0]), "corner_py": float(c[1]),
             "center_px": float(k[0]), "center_py": float(k[1])}
            for p, f, c, k in zip(pts, found, corner, center)
        ],
    }
    print(json.dumps({k: v for k, v in out.items() if k != "markers"}, indent=2))
    with open(OUT_DIR / "grid_alignment_test.json", "w") as f:
        json.dump(out, f, indent=2)
    worst = max(out["residual_vs_adapter_original_to_model"][k] for k in ("max_abs_dx", "max_abs_dy"))
    if worst > ADAPTER_TOLERANCE:
        raise SystemExit(f"adapter mapping is off by up to {worst:.4f} grid px (> {ADAPTER_TOLERANCE}) "
                         "from the markers located on MASt3R-SLAM's own grid")


if __name__ == "__main__":
    os.chdir(MAST3R_REPO)
    main()
