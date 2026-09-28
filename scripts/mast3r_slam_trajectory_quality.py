#!/usr/bin/env python
"""Trajectory quality (NOT an evaluation metric) for MASt3R-SLAM's keyframe
poses on one sequence: Sim(3)-aligned ATE and endpoint drift as a fraction
of GT path length, compared with EndoDAC's already-recorded values for the
same sequence (docs/pipelines/mast3r_slam.md, Q5).

Self-contained closed-form Umeyama, duplicated (not imported) from
scratch/pipelines/endodac_scale_analysis.py:29-43, which also does not
import src/eval/scale_recovery.py -- same reasoning applies here: this
task's hard constraint bars running anything under src/eval/ on MASt3R-SLAM
outputs, even for a non-metric trajectory-quality number.

Uses hypothesis A from scripts/mast3r_slam_pose_check.py's result (T_WC as
saved = camera-to-world), i.e. no inversion.

Usage:
    python scripts/mast3r_slam_trajectory_quality.py --sequence c1_cecum_t1_v1
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np

REPO = Path("/data1_ycao/chua/projects/mrsp")
DATASET_ROOT = Path("/data1_ycao/chua/datasets/C3VDv2")
REGISTERED_DIR = DATASET_ROOT / "registered_videos"
MAST3R_REPO = REPO / "scratch/pipelines/MASt3R-SLAM"
FPS = 30.0


def load_gt_poses(sequence: str) -> list[np.ndarray]:
    zpath = REGISTERED_DIR / f"{sequence}.zip"
    with zipfile.ZipFile(zpath) as zf:
        lines = zf.read("pose.txt").decode().strip().split("\n")
    return [np.array([float(x) for x in line.split(",")]).reshape(4, 4).T for line in lines]


def load_pred_keyframe_positions(sequence: str) -> tuple[list[int], np.ndarray]:
    txt_path = MAST3R_REPO / "logs" / sequence / f"{sequence}.txt"
    frame_idxs, positions = [], []
    with open(txt_path) as f:
        for line in f:
            vals = [float(x) for x in line.strip().split()]
            t, tx, ty, tz = vals[:4]
            frame_idxs.append(round(t * FPS))
            positions.append([tx, ty, tz])
    return frame_idxs, np.array(positions)


def umeyama(src, dst):
    N = src.shape[0]
    mu_src, mu_dst = src.mean(axis=0), dst.mean(axis=0)
    src_c, dst_c = src - mu_src, dst - mu_dst
    var_src = (src_c ** 2).sum() / N
    Sigma = (dst_c.T @ src_c) / N
    U, D, Vt = np.linalg.svd(Sigma)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    c = np.trace(np.diag(D) @ S) / var_src
    t = mu_dst - c * R @ mu_src
    aligned = (c * (R @ src.T).T) + t
    return R, t, c, aligned


def path_length(positions: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())


def compute(sequence: str) -> dict:
    gt_poses = load_gt_poses(sequence)
    frame_idxs, pred_pos = load_pred_keyframe_positions(sequence)
    gt_pos = np.array([gt_poses[i][:3, 3] for i in frame_idxs])

    R, t, c, aligned = umeyama(pred_pos, gt_pos)
    ate_rmse = float(np.sqrt(np.mean(np.sum((aligned - gt_pos) ** 2, axis=1))))

    gt_path_len = path_length(gt_pos)
    endpoint_drift_mm = float(np.linalg.norm(aligned[-1] - gt_pos[-1]))
    endpoint_drift_frac = endpoint_drift_mm / gt_path_len if gt_path_len > 0 else float("nan")

    result = {
        "sequence": sequence,
        "n_keyframes": len(frame_idxs),
        "keyframe_frame_indices": frame_idxs,
        "sim3_scale": float(c),
        "ate_rmse_mm": ate_rmse,
        "gt_path_length_mm": gt_path_len,
        "endpoint_drift_mm": endpoint_drift_mm,
        "endpoint_drift_frac_of_gt_path": endpoint_drift_frac,
    }
    print(json.dumps(result, indent=1))
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence", required=True)
    args = ap.parse_args()
    compute(args.sequence)
