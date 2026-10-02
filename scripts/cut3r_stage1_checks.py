#!/usr/bin/env python
"""CUT3R Stage 1 checks on one sequence (docs/pipelines/cut3r.md,
questions 5 to 8). Reads scripts/cut3r_run.py's saved outputs and the GT
pose / depth from the dataset archive.

Computes NO evaluation metric (no coverage, recall, false reassurance,
false alarm, region IoU, localization error) and imports nothing from
src/eval/ or src/eval_ext/. Small pieces are duplicated or imported from
the MASt3R-SLAM feasibility scripts instead (same precedent):
Umeyama from scripts/mast3r_slam_trajectory_quality_perframe.py, GT
loaders from scripts/mast3r_slam_sequence_descriptives.py. The depth-scale
definition restates src/eval/scale_recovery.py::compute_depth_scale's
docstring (per-frame median(GT) / median(pred) over GT-valid pixels;
sequence scale = median of the per-frame ratios) without calling it.

  5. Pose convention: per frame pair (stride 1 and stride 10), GT relative
     transform vs the predicted relative transform read as camera-to-world
     (A) and as its inverse (B): rotation error and translation-direction
     cosine. Translation direction is sign sensitive.
  6. Ray direction: angle between each model's implied per-pixel ray and
     this project's Scaramuzza ray (src/geometry/camera.py::unproject),
     frame 0, on MASt3R's confident pixels (the population of the Stage 2
     check, dumped by scripts/cut3r_dump_mast3r_confident_pixels.py),
     for EndoDAC, MASt3R-SLAM and CUT3R side by side. Original pixel
     coordinates from the pixel-center grid mapping.
  7. Trajectory quality (not an evaluation metric): ATE after Sim(3)
     (position-only Umeyama), endpoint drift fraction, per-frame depth
     scale median and relative IQR.
  8. Missing frames.

Usage: scratch/.venv/bin/python scripts/cut3r_stage1_checks.py --sequence c1_cecum_t1_v1
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

REPO = Path("/data1_ycao/chua/projects/mrsp")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from geometry.camera import CameraIntrinsics, unproject  # noqa: E402
from mast3r_slam_resolution_mapping import model_to_original  # noqa: E402
from mast3r_slam_sequence_descriptives import depth_members, load_gt_depth_mm, load_gt_poses  # noqa: E402
from mast3r_slam_trajectory_quality_perframe import path_length, umeyama  # noqa: E402
from mast3r_slam_check_vignette_and_endodac_ray import (  # noqa: E402
    ENDODAC_CX, ENDODAC_CY, ENDODAC_FEED_H, ENDODAC_FEED_W, ENDODAC_FX, ENDODAC_FY,
)
from pipeline_adapters import ORIGINAL_H, ORIGINAL_W, BilinearGridMap  # noqa: E402

OUT_ROOT = REPO / "results/pipelines/cut3r_stage1"
REGISTERED_DIR = Path("/data1_ycao/chua/datasets/C3VDv2/registered_videos")
INTRINSICS_PATH = "/data1_ycao/chua/datasets/C3VDv2/camera_intrinsics.txt"


def rot_angle_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def angle_stats(a_dirs: np.ndarray, b_dirs: np.ndarray) -> dict:
    ang = np.degrees(np.arccos(np.clip(np.sum(a_dirs * b_dirs, axis=-1), -1.0, 1.0)))
    return {"n_pixels": int(len(ang)), "median_angle_deg": float(np.median(ang)),
            "p95_angle_deg": float(np.percentile(ang, 95)), "mean_angle_deg": float(ang.mean()),
            "max_angle_deg": float(ang.max())}


def pose_convention(pred_c2w: np.ndarray, gt: list[np.ndarray], stride: int) -> dict:
    rA, rB, cA, cB = [], [], [], []
    n_skipped = 0
    for i in range(0, len(pred_c2w) - stride, stride):
        j = i + stride
        T = np.linalg.inv(pred_c2w[i]) @ pred_c2w[j]   # A: saved matrices are camera-to-world
        Ti = np.linalg.inv(T)                           # B: saved matrices are world-to-camera
        G = np.linalg.inv(gt[i]) @ gt[j]
        tg = G[:3, 3]
        if np.linalg.norm(tg) < 1e-6 or np.linalg.norm(T[:3, 3]) < 1e-12:
            n_skipped += 1
            continue
        rA.append(rot_angle_deg(G[:3, :3].T @ T[:3, :3]))
        rB.append(rot_angle_deg(G[:3, :3].T @ Ti[:3, :3]))
        tg = tg / np.linalg.norm(tg)
        cA.append(float(tg @ (T[:3, 3] / np.linalg.norm(T[:3, 3]))))
        cB.append(float(tg @ (Ti[:3, 3] / np.linalg.norm(Ti[:3, 3]))))
    cA, cB = np.array(cA), np.array(cB)

    def side(r, c):
        return {"mean_rot_err_deg": float(np.mean(r)), "median_rot_err_deg": float(np.median(r)),
                "mean_translation_cosine": float(c.mean()), "median_translation_cosine": float(np.median(c)),
                "frac_pairs_cosine_positive": float((c > 0).mean())}
    return {"stride": stride, "n_pairs": int(len(cA)), "n_pairs_skipped_zero_translation": n_skipped,
            "A_saved_is_camera_to_world": side(rA, cA), "B_saved_is_world_to_camera": side(rB, cB),
            "frac_pairs_A_cosine_greater_than_B": float((cA > cB).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence", required=True)
    args = ap.parse_args()
    seq = args.sequence
    out_dir = OUT_ROOT / seq
    manifest = json.loads((out_dir / "MANIFEST.json").read_text())
    if manifest["status"] != "ok":
        raise SystemExit("CUT3R run did not finish ok")
    gh, gw = manifest["grid_h"], manifest["grid_w"]
    m = json.loads((OUT_ROOT / "grid_alignment" / "grid_alignment_test.json").read_text())["mapping_parameters"]
    if (m["model_grid_h"], m["model_grid_w"]) != (gh, gw):
        raise RuntimeError("grid of the run differs from the grid of the marker test")

    gt = load_gt_poses(seq)
    n_gt = len(gt)
    pred = np.load(out_dir / "poses_c2w.npy")
    res = {"sequence": seq, "n_gt_frames": n_gt}

    # ---------------- 8. missing frames ----------------
    depth_files = sorted(int(p.stem) for p in (out_dir / "depth").glob("*.npz"))
    res["missing_frames"] = {
        "n_frames_with_pose": int(len(pred)), "n_frames_with_depth": len(depth_files),
        "frames_missing_pose": [i for i in range(n_gt) if i >= len(pred)],
        "frames_missing_depth": sorted(set(range(n_gt)) - set(depth_files)),
        "frames_nonfinite_pose": manifest["frames_nonfinite_pose"],
        "frames_nonfinite_depth": manifest["frames_nonfinite_depth"],
        "n_nonpositive_z_pixels": manifest["n_nonpositive_z_pixels"],
        "frames_with_any_nonpositive_z": manifest["frames_with_any_nonpositive_z"],
    }
    if len(pred) != n_gt or depth_files != list(range(n_gt)):
        raise RuntimeError("frame count mismatch -- pose / trajectory checks below assume one prediction per GT frame")

    # ---------------- 5. pose convention ----------------
    res["pose_convention"] = [pose_convention(pred, gt, s) for s in (1, 10)]

    # internal consistency (independent of GT): CUT3R also predicts each frame's points in the first
    # frame's coordinates (pts3d_in_other_view). If the saved pose is camera-to-world, pose @ self-view
    # points reproduces them; if it is world-to-camera, inv(pose) @ self-view points does.
    cons = []
    for p_self in sorted((out_dir / "pointmap_self").glob("*.npy")):
        p_other = out_dir / "pointmap_other" / p_self.name
        if not p_other.exists():
            continue
        i = int(p_self.stem)
        xs, xo = np.load(p_self).reshape(-1, 3).astype(np.float64), np.load(p_other).reshape(-1, 3).astype(np.float64)
        T = pred[i]
        Ti = np.linalg.inv(T)
        scale = float(np.median(np.linalg.norm(xo, axis=1)))
        eA = np.linalg.norm(xs @ T[:3, :3].T + T[:3, 3] - xo, axis=1)
        eB = np.linalg.norm(xs @ Ti[:3, :3].T + Ti[:3, 3] - xo, axis=1)
        cons.append({"frame": i, "median_err_A_c2w_rel": float(np.median(eA) / scale),
                     "median_err_B_w2c_rel": float(np.median(eB) / scale)})
    res["pose_convention_pointmap_consistency"] = cons

    # ---------------- 7. trajectory quality + depth scale ----------------
    gt_pos = np.array([g[:3, 3] for g in gt])
    traj = {}
    for name, pos in [("A_camera_to_world", pred[:, :3, 3]),
                      ("B_world_to_camera", np.array([np.linalg.inv(p)[:3, 3] for p in pred]))]:
        R, t, c, aligned = umeyama(pos, gt_pos)
        gl = path_length(gt_pos)
        traj[name] = {"sim3_scale_s_pose": float(c),
                      "ate_rmse_mm": float(np.sqrt(np.mean(np.sum((aligned - gt_pos) ** 2, axis=1)))),
                      "gt_path_length_mm": gl,
                      "endpoint_drift_mm": float(np.linalg.norm(aligned[-1] - gt_pos[-1])),
                      "endpoint_drift_frac_of_gt_path": float(np.linalg.norm(aligned[-1] - gt_pos[-1]) / gl),
                      "pred_path_length_native": path_length(pos)}
    res["trajectory_quality"] = traj

    gm = BilinearGridMap(m)
    ratios = []
    with zipfile.ZipFile(REGISTERED_DIR / f"{seq}.zip") as zf:
        members = depth_members(zf)
        for i in range(n_gt):
            gt_mm, gt_valid = load_gt_depth_mm(zf, members[i])
            with np.load(out_dir / "depth" / f"{i:04d}.npz") as npz:
                z = npz["z"]  # key "conf" is never read
            d = gm.sample(z)
            use = gt_valid.ravel() & gm.available
            if not use.any():
                raise RuntimeError(f"frame {i}: no GT-valid pixel with a prediction")
            pm = float(np.median(d[use]))
            if pm == 0.0:
                raise RuntimeError(f"frame {i}: predicted depth median is 0")
            ratios.append(float(np.median(gt_mm.ravel()[use])) / pm)
    ratios = np.array(ratios)
    q1, q3 = np.percentile(ratios, [25, 75])
    res["depth_scale"] = {
        "definition": "per frame median(GT mm) / median(pred) over GT-valid pixels that have a prediction "
                      "(bilinear, pixel-center mapping); sequence value = median over frames",
        "depth_scale_median": float(np.median(ratios)), "depth_scale_relative_iqr": float((q3 - q1) / np.median(ratios)),
        "min": float(ratios.min()), "max": float(ratios.max()), "n_frames": int(len(ratios)),
        "n_frames_ratio_nonpositive": int((ratios <= 0).sum()),
    }

    # ---------------- 6. ray direction, three pipelines, same pixels ----------------
    cp = np.load(out_dir / "mast3r_confident_pixels_frame0.npz")
    gx, gy = cp["gx"], cp["gy"]
    ox, oy = model_to_original(gx.astype(np.float64), gy.astype(np.float64), m)
    if not (np.allclose(ox, cp["ox"]) and np.allclose(oy, cp["oy"])):
        raise RuntimeError("MASt3R and CUT3R grids do not map to the same original coordinates")
    inb = (ox >= 0) & (ox <= ORIGINAL_W - 1) & (oy >= 0) & (oy <= ORIGINAL_H - 1)
    intr = CameraIntrinsics.from_file(INTRINSICS_PATH)
    geom = unproject(np.stack([ox, oy], axis=-1), intr)
    geom = geom / np.linalg.norm(geom, axis=-1, keepdims=True)

    pm0 = np.load(out_dir / "pointmap_self" / "0000.npy").astype(np.float64)
    cut = pm0[gy, gx]
    cut = cut / np.linalg.norm(cut, axis=-1, keepdims=True)
    mast = cp["mast3r_dirs"]
    ex, ey = ox * (ENDODAC_FEED_W / ORIGINAL_W), oy * (ENDODAC_FEED_H / ORIGINAL_H)
    endo = np.stack([(ex - ENDODAC_CX) / ENDODAC_FX, (ey - ENDODAC_CY) / ENDODAC_FY, np.ones_like(ex)], axis=-1)
    endo = endo / np.linalg.norm(endo, axis=-1, keepdims=True)
    saved = json.loads((REPO / "results/pipelines/mast3r_slam_perframe/c1_cecum_t1_v1/vignette_and_endodac_ray_check.json").read_text())
    res["ray_direction"] = {
        "frame_index": 0, "pixel_population": "MASt3R's confident pixels on frame 0 (Stage 2 check's population)",
        "n_pixels": int(len(gx)), "n_pixels_outside_original_image": int((~inb).sum()),
        "original_coordinates": "pixel-center grid mapping",
        "endodac": angle_stats(endo, geom), "mast3r_slam": angle_stats(mast, geom), "cut3r": angle_stats(cut, geom),
        "cut3r_vs_mast3r_slam": angle_stats(cut, mast),
        "previously_saved_with_corner_aligned_coordinates": {
            "endodac_median_deg": saved["endodac_vs_mast3r_ray_comparison"]["endodac_check"]["median_angle_deg"],
            "mast3r_slam_median_deg": saved["endodac_vs_mast3r_ray_comparison"]["mast3r_check_stage2_saved"]["median_angle_deg"],
        },
    }
    # CUT3R on all of its own grid pixels of frame 0 (no confidence selection), for reference
    yy, xx = np.mgrid[0:gh, 0:gw]
    aox, aoy = model_to_original(xx.ravel().astype(np.float64), yy.ravel().astype(np.float64), m)
    ageom = unproject(np.stack([aox, aoy], axis=-1), intr)
    ageom = ageom / np.linalg.norm(ageom, axis=-1, keepdims=True)
    acut = pm0.reshape(-1, 3)
    acut = acut / np.linalg.norm(acut, axis=-1, keepdims=True)
    res["ray_direction"]["cut3r_all_grid_pixels"] = angle_stats(acut, ageom)
    # equivalent pinhole focal of CUT3R's own pointmap (descriptive): median over pixels of radius / tan(angle)
    r = np.hypot(xx.ravel() - (gw - 1) / 2, yy.ravel() - (gh - 1) / 2)
    th = np.arccos(np.clip(acut[:, 2], -1, 1))
    ok = (r > 20) & (th > 1e-3)
    res["ray_direction"]["cut3r_implied_focal_grid_px_median"] = float(np.median(r[ok] / np.tan(th[ok])))
    res["ray_direction"]["cut3r_weiszfeld_focal_frame0"] = float(np.load(out_dir / "focal_weiszfeld.npy")[0])

    (out_dir / "stage1_checks.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
