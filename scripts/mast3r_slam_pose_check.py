#!/usr/bin/env python
"""MASt3R-SLAM keyframe pose convention check (docs/pipelines/mast3r_slam.md,
Q3), same per-pair method as EndoDAC's decisive check
(scratch/pipelines/endodac_pose_validation_perstep.py, docs/pipelines/
endodac.md section 7): per consecutive KEYFRAME pair (no global alignment
freedom), compare rotation-angle error and translation-direction cosine
similarity between the GT relative transform and both the predicted
relative transform as saved and its inverse. Decisive because translation
direction is sign-flip-sensitive while near-identity rotations are not.

This computes NO evaluation metric (no coverage/recall/IoU/localization
error) and does not import anything from src/eval/ or src/gt/ -- it is a
convention check on raw predicted vs. raw GT poses only, mirroring the
already-approved EndoDAC precedent for exactly this kind of question.

MASt3R-SLAM's saved trajectory file (logs/<save-as>/<seq>.txt) is
keyframe-only, TUM format: "timestamp tx ty tz qx qy qz qw" (Sim3 saved
without the scale component -- see docs/pipelines/mast3r_slam.md Q2).
RGBFiles assigns timestamp = frame_index / 30.0, so frame_index =
round(timestamp * 30) recovers the original extracted-frame index for GT
lookup.

Usage:
    python scripts/mast3r_slam_pose_check.py --sequence c1_cecum_t1_v1
"""
from __future__ import annotations

import argparse
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


def load_pred_keyframes(sequence: str) -> list[tuple[float, int, np.ndarray]]:
    """Returns list of (timestamp, frame_index, T_WC) for each keyframe,
    T_WC built from the TUM tx ty tz qx qy qz qw columns (quaternion ->
    rotation matrix, standard Hamilton convention)."""
    txt_path = MAST3R_REPO / "logs" / sequence / f"{sequence}.txt"
    out = []
    with open(txt_path) as f:
        for line in f:
            vals = [float(x) for x in line.strip().split()]
            t, tx, ty, tz, qx, qy, qz, qw = vals[:8]
            frame_idx = round(t * FPS)
            R = quat_to_rot(qx, qy, qz, qw)
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = [tx, ty, tz]
            out.append((t, frame_idx, T))
    return out


def quat_to_rot(qx, qy, qz, qw) -> np.ndarray:
    n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)],
    ])


def rot_angle_deg(R: np.ndarray) -> float:
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    return float(np.degrees(np.arccos(c)))


def check(sequence: str) -> dict:
    gt_poses = load_gt_poses(sequence)
    kfs = load_pred_keyframes(sequence)

    rot_err_A, rot_err_B = [], []
    cos_sim_A, cos_sim_B = [], []
    pairs_used = []

    for (t_i, fi, T_i), (t_j, fj, T_j) in zip(kfs[:-1], kfs[1:]):
        if fi == fj:
            continue  # degenerate (shouldn't happen, guards against timestamp rounding collisions)
        T_pred = np.linalg.inv(T_i) @ T_j
        T_pred_inv = np.linalg.inv(T_pred)

        C2W_i, C2W_j = gt_poses[fi], gt_poses[fj]
        G = np.linalg.inv(C2W_i) @ C2W_j

        R_gt, t_gt = G[:3, :3], G[:3, 3]
        R_A, t_A = T_pred[:3, :3], T_pred[:3, 3]
        R_B, t_B = T_pred_inv[:3, :3], T_pred_inv[:3, 3]

        rot_err_A.append(rot_angle_deg(R_gt.T @ R_A))
        rot_err_B.append(rot_angle_deg(R_gt.T @ R_B))

        t_gt_n = t_gt / (np.linalg.norm(t_gt) + 1e-12)
        t_A_n = t_A / (np.linalg.norm(t_A) + 1e-12)
        t_B_n = t_B / (np.linalg.norm(t_B) + 1e-12)
        cos_sim_A.append(float(t_gt_n @ t_A_n))
        cos_sim_B.append(float(t_gt_n @ t_B_n))
        pairs_used.append((fi, fj))

        print(f"pair (kf {fi},{fj}): rot_err_A={rot_err_A[-1]:6.2f} deg, rot_err_B={rot_err_B[-1]:6.2f} deg, "
              f"cos_sim_A={cos_sim_A[-1]:+.4f}, cos_sim_B={cos_sim_B[-1]:+.4f}")

    result = {
        "sequence": sequence,
        "n_pairs": len(pairs_used),
        "pairs": pairs_used,
        "hypothesis_A_T_as_saved": {
            "mean_rot_err_deg": float(np.mean(rot_err_A)),
            "mean_translation_cosine": float(np.mean(cos_sim_A)),
        },
        "hypothesis_B_inv_T": {
            "mean_rot_err_deg": float(np.mean(rot_err_B)),
            "mean_translation_cosine": float(np.mean(cos_sim_B)),
        },
    }
    print("\n=== summary ===")
    print(f"n_pairs = {result['n_pairs']}")
    print(f"Hypothesis A (T_WC as saved): mean rot err = {result['hypothesis_A_T_as_saved']['mean_rot_err_deg']:.2f} deg, "
          f"mean translation-direction cosine sim = {result['hypothesis_A_T_as_saved']['mean_translation_cosine']:+.4f}")
    print(f"Hypothesis B (inv(T_WC)):     mean rot err = {result['hypothesis_B_inv_T']['mean_rot_err_deg']:.2f} deg, "
          f"mean translation-direction cosine sim = {result['hypothesis_B_inv_T']['mean_translation_cosine']:+.4f}")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence", required=True)
    args = ap.parse_args()
    check(args.sequence)
