#!/usr/bin/env python
"""Trajectory quality (NOT an evaluation metric) for MASt3R-SLAM's
per-frame reconstructed poses (scripts/mast3r_slam_reconstruct_poses.py's
poses_per_frame.csv) on one sequence, ALL frames -- unlike Stage 1's
mast3r_slam_trajectory_quality.py, which only covered the 20 keyframes
and a partial GT path (frames 0-168). This is Stage 2 step 7 (docs/
pipelines/mast3r_slam.md).

Same self-contained closed-form Umeyama as Stage 1's script (duplicated,
not imported, from scratch/pipelines/endodac_scale_analysis.py:29-43 --
same reasoning: this task's hard constraint bars src/eval/ even for
non-metric numbers). Kept as a SEPARATE file from Stage 1's script (not
overwritten) since Stage 1's 20-keyframe numbers are still cited in the
doc.

Usage:
    python scripts/mast3r_slam_trajectory_quality_perframe.py --sequence c1_cecum_t1_v1
"""
from __future__ import annotations

import argparse
import csv
import json
import zipfile
from pathlib import Path

import numpy as np

REPO = Path("/data1_ycao/chua/projects/mrsp")
DATASET_ROOT = Path("/data1_ycao/chua/datasets/C3VDv2")
REGISTERED_DIR = DATASET_ROOT / "registered_videos"
PERFRAME_ROOT = REPO / "results/pipelines/mast3r_slam_perframe"


def resolve_pose_txt_member(zf: zipfile.ZipFile) -> str:
    """Most C3VDv2 registered_videos archives store pose.txt at the
    archive root, but at least one (c1_cecum_t1_v3.zip, per
    src/geometry/coverage_mesh.py::_resolve_zip_member's docstring --
    read, not imported, same duplicate-small-helper precedent as
    elsewhere in this project) wraps everything in a <video_name>/
    subfolder instead. Match on basename rather than assuming root."""
    candidates = [n for n in zf.namelist() if n == "pose.txt" or n.endswith("/pose.txt")]
    if not candidates:
        raise RuntimeError("no pose.txt member found in archive")
    if len(candidates) > 1:
        raise RuntimeError(f"ambiguous pose.txt member: {candidates}")
    return candidates[0]


def load_gt_poses(sequence: str) -> list[np.ndarray]:
    zpath = REGISTERED_DIR / f"{sequence}.zip"
    with zipfile.ZipFile(zpath) as zf:
        member = resolve_pose_txt_member(zf)
        lines = zf.read(member).decode().strip().split("\n")
    return [np.array([float(x) for x in line.split(",")]).reshape(4, 4).T for line in lines]


def load_pred_per_frame_positions(sequence: str, out_root: Path = PERFRAME_ROOT) -> tuple[list[int], np.ndarray]:
    out_dir = out_root / sequence
    frame_idxs, positions = [], []
    with open(out_dir / "poses_per_frame.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["reconstructed"] != "True":
                continue
            frame_idxs.append(int(row["frame_id"]))
            positions.append([float(row["tx"]), float(row["ty"]), float(row["tz"])])
    order = np.argsort(frame_idxs)
    frame_idxs = [frame_idxs[i] for i in order]
    positions = np.array(positions)[order]
    return frame_idxs, positions


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


def compute(sequence: str, out_root: Path = PERFRAME_ROOT) -> dict:
    gt_poses = load_gt_poses(sequence)
    frame_idxs, pred_pos = load_pred_per_frame_positions(sequence, out_root)
    gt_pos = np.array([gt_poses[i][:3, 3] for i in frame_idxs])

    R, t, c, aligned = umeyama(pred_pos, gt_pos)
    ate_rmse = float(np.sqrt(np.mean(np.sum((aligned - gt_pos) ** 2, axis=1))))

    gt_path_len = path_length(gt_pos)
    endpoint_drift_mm = float(np.linalg.norm(aligned[-1] - gt_pos[-1]))
    endpoint_drift_frac = endpoint_drift_mm / gt_path_len if gt_path_len > 0 else float("nan")

    result = {
        "sequence": sequence,
        "n_frames_reconstructed": len(frame_idxs),
        "frame_index_range": [frame_idxs[0], frame_idxs[-1]],
        "sim3_scale": float(c),
        "ate_rmse_mm": ate_rmse,
        "gt_path_length_mm": gt_path_len,
        "endpoint_drift_mm": endpoint_drift_mm,
        "endpoint_drift_frac_of_gt_path": endpoint_drift_frac,
    }
    print(json.dumps(result, indent=1))
    out_dir = out_root / sequence
    with open(out_dir / "trajectory_quality_perframe.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence", required=True)
    ap.add_argument("--out-root", default=str(PERFRAME_ROOT))
    args = ap.parse_args()
    compute(args.sequence, Path(args.out_root))
