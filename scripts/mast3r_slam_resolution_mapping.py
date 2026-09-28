#!/usr/bin/env python
"""MASt3R-SLAM Stage 2 step 5 (docs/pipelines/mast3r_slam.md): the exact
mapping between MASt3R's internal resized+cropped grid and the original
1350x1080 input pixel grid, a round-trip test of that mapping, the
fraction of vignette-mask pixels with no depth because of cropping, and
a per-pixel ray-direction comparison between MASt3R's implied ray and
this project's own Scaramuzza camera model, for one frame.

Does NOT import from src/eval/ or src/gt/ (locked, and this task's hard
constraint bars them regardless): the fixed vignette mask asset
(docs/eval_protocol.md section 8) is loaded by src/eval/evaluable.py,
which lives under the locked path, so here it's read directly from the
same .npy file with a duplicated 3-line unpack -- same "duplicate the
small piece, don't import src/eval/" precedent Stage 1 used for Umeyama
(scratch/pipelines/endodac_scale_analysis.py).

`src/geometry/camera.py` is NOT locked (only src/eval/ and src/gt/ are,
per CLAUDE.md's Layout section) -- imported directly for `unproject`.

Computes no evaluation metric.

Usage (mast3r-slam conda env, GPU needed only for the ray-direction
check's single MASt3R forward pass):
    python scripts/mast3r_slam_resolution_mapping.py --sequence c1_cecum_t1_v1 --gpu 7
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path("/data1_ycao/chua/projects/mrsp")
MAST3R_REPO = REPO / "scratch/pipelines/MASt3R-SLAM"
PERFRAME_ROOT = REPO / "results/pipelines/mast3r_slam_perframe"
SCRATCH_ROOT = REPO / "scratch/mast3r_slam"
INTRINSICS_PATH = "/data1_ycao/chua/datasets/C3VDv2/camera_intrinsics.txt"
VIGNETTE_MASK_PATH = REPO / "results/pipelines/oracle_gap_vignette/c1_cecum_t1_v1_vignette_mask.npy"
EXPECTED_VIGNETTE_PIXEL_COUNT = 102049
ORIGINAL_W, ORIGINAL_H = 1350, 1080

sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(MAST3R_REPO))


def load_vignette_mask(width: int, height: int) -> np.ndarray:
    """Duplicated from src/eval/evaluable.py::load_vignette_mask (not
    imported -- that file is under the locked src/eval/ path). Same
    format: packed bits, True = vignette (non-evaluable)."""
    packed = np.load(VIGNETTE_MASK_PATH)
    mask = np.unpackbits(packed)[: width * height].astype(bool)
    if mask.size != width * height:
        raise RuntimeError(
            f"vignette mask unpacks to {mask.size} px, expected {width * height}"
        )
    n = int(mask.sum())
    if n != EXPECTED_VIGNETTE_PIXEL_COUNT:
        raise RuntimeError(
            f"vignette mask has {n} px set, expected {EXPECTED_VIGNETTE_PIXEL_COUNT}"
        )
    return mask


def compute_resize_mapping(sample_frame_path: Path) -> dict:
    from mast3r_slam.mast3r_utils import resize_img

    img = np.asarray(Image.open(sample_frame_path).convert("RGB")).astype(np.float32) / 255.0
    assert img.shape[1] == ORIGINAL_W and img.shape[0] == ORIGINAL_H, (
        f"expected {ORIGINAL_W}x{ORIGINAL_H}, got {img.shape[1]}x{img.shape[0]}"
    )
    res, (scale_w, scale_h, half_crop_w, half_crop_h) = resize_img(
        img, 512, return_transformation=True
    )
    cropped_h, cropped_w = res["true_shape"][0]
    return {
        "original_w": ORIGINAL_W, "original_h": ORIGINAL_H,
        "model_grid_w": int(cropped_w), "model_grid_h": int(cropped_h),
        "scale_w": float(scale_w), "scale_h": float(scale_h),
        "half_crop_w": float(half_crop_w), "half_crop_h": float(half_crop_h),
    }


def model_to_original(px_model: np.ndarray, py_model: np.ndarray, m: dict) -> tuple[np.ndarray, np.ndarray]:
    """Inverse mapping: model-grid pixel -> original 1350x1080 pixel."""
    ox = (px_model + m["half_crop_w"]) * m["scale_w"]
    oy = (py_model + m["half_crop_h"]) * m["scale_h"]
    return ox, oy


def original_to_model(ox: np.ndarray, oy: np.ndarray, m: dict) -> tuple[np.ndarray, np.ndarray]:
    """Forward mapping: original 1350x1080 pixel -> model-grid pixel."""
    px = ox / m["scale_w"] - m["half_crop_w"]
    py = oy / m["scale_h"] - m["half_crop_h"]
    return px, py


def round_trip_test(m: dict) -> dict:
    rng = np.random.default_rng(0)
    n = 2000
    ox = rng.uniform(0, ORIGINAL_W - 1, n)
    oy = rng.uniform(0, ORIGINAL_H - 1, n)
    px, py = original_to_model(ox, oy, m)
    ox2, oy2 = model_to_original(px, py, m)
    err = np.sqrt((ox - ox2) ** 2 + (oy - oy2) ** 2)
    corners = np.array([[0, 0], [ORIGINAL_W - 1, 0], [0, ORIGINAL_H - 1], [ORIGINAL_W - 1, ORIGINAL_H - 1],
                         [ORIGINAL_W / 2, ORIGINAL_H / 2]])
    cpx, cpy = original_to_model(corners[:, 0], corners[:, 1], m)
    cox2, coy2 = model_to_original(cpx, cpy, m)
    corner_err = np.sqrt((corners[:, 0] - cox2) ** 2 + (corners[:, 1] - coy2) ** 2)
    return {
        "n_samples": n, "max_error_px": float(err.max()), "mean_error_px": float(err.mean()),
        "corner_points": corners.tolist(), "corner_round_trip_error_px": corner_err.tolist(),
    }


def vignette_crop_fraction(m: dict) -> dict:
    vignette_mask = load_vignette_mask(ORIGINAL_W, ORIGINAL_H).reshape(ORIGINAL_H, ORIGINAL_W)
    vig_ys, vig_xs = np.nonzero(vignette_mask)

    px, py = original_to_model(vig_xs.astype(np.float64), vig_ys.astype(np.float64), m)
    covered = (
        (px >= 0) & (px <= m["model_grid_w"] - 1) &
        (py >= 0) & (py <= m["model_grid_h"] - 1)
    )
    n_vignette = len(vig_xs)
    n_uncovered = int((~covered).sum())
    return {
        "n_vignette_pixels": n_vignette,
        "n_vignette_pixels_uncovered_by_model_grid": n_uncovered,
        "fraction_vignette_pixels_no_depth_from_cropping": n_uncovered / n_vignette if n_vignette else float("nan"),
    }


def ray_direction_check(sequence: str, frame_index: int, gpu: int, m: dict) -> dict:
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.chdir(MAST3R_REPO)  # load_mast3r() uses the relative path "checkpoints/..."
    import torch
    from mast3r_slam.config import load_config, config
    from mast3r_slam.frame import create_frame
    from mast3r_slam.mast3r_utils import load_mast3r, mast3r_inference_mono
    import lietorch

    load_config(str(MAST3R_REPO / "config" / "base.yaml"))
    torch.set_grad_enabled(False)
    device = "cuda:0"

    frame_path = SCRATCH_ROOT / sequence / f"{frame_index:04d}.png"
    img = np.asarray(Image.open(frame_path).convert("RGB")).astype(np.float32) / 255.0

    model = load_mast3r(device=device)
    frame = create_frame(frame_index, img, lietorch.Sim3.Identity(1, device=device), img_size=512, device=device)
    X, C = mast3r_inference_mono(model, frame)
    h, w = frame.img_shape.reshape(-1).tolist()
    X = X.reshape(int(h), int(w), 3).cpu().numpy()
    C = C.reshape(int(h), int(w)).cpu().numpy()

    q_conf = config["tracking"]["Q_conf"]
    c_conf = config["tracking"]["C_conf"]
    valid = C > max(q_conf, c_conf)
    ys, xs = np.nonzero(valid)

    mast3r_dirs = X[ys, xs, :]
    mast3r_dirs = mast3r_dirs / np.linalg.norm(mast3r_dirs, axis=-1, keepdims=True)

    ox, oy = model_to_original(xs.astype(np.float64), ys.astype(np.float64), m)
    in_bounds = (ox >= 0) & (ox < ORIGINAL_W) & (oy >= 0) & (oy < ORIGINAL_H)

    from geometry.camera import CameraIntrinsics, unproject
    intr = CameraIntrinsics.from_file(INTRINSICS_PATH)
    px_orig = np.stack([ox[in_bounds], oy[in_bounds]], axis=-1)
    geom_dirs = unproject(px_orig, intr)

    dot = np.sum(mast3r_dirs[in_bounds] * geom_dirs, axis=-1)
    dot = np.clip(dot, -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(dot))

    return {
        "frame_index": frame_index,
        "n_confident_pixels": int(valid.sum()),
        "n_pixels_compared_in_bounds": int(in_bounds.sum()),
        "median_angle_deg": float(np.median(angle_deg)),
        "p95_angle_deg": float(np.percentile(angle_deg, 95)),
        "mean_angle_deg": float(np.mean(angle_deg)),
        "max_angle_deg": float(np.max(angle_deg)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence", required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--ray-check-frame", type=int, default=0)
    args = ap.parse_args()

    sample_frame = SCRATCH_ROOT / args.sequence / "0000.png"
    m = compute_resize_mapping(sample_frame)
    print("resolution mapping:", json.dumps(m, indent=2))

    rt = round_trip_test(m)
    print("round-trip test:", json.dumps(rt, indent=2))

    vig = vignette_crop_fraction(m)
    print("vignette crop fraction:", json.dumps(vig, indent=2))

    ray = ray_direction_check(args.sequence, args.ray_check_frame, args.gpu, m)
    print("ray direction check:", json.dumps(ray, indent=2))

    out = {
        "resolution_mapping": m, "round_trip_test": rt,
        "vignette_crop_fraction": vig, "ray_direction_check": ray,
    }
    out_dir = PERFRAME_ROOT / args.sequence
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "resolution_mapping.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
