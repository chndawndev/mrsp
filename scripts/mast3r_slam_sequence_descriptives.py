#!/usr/bin/env python
"""MASt3R-SLAM Stage 3 step 3 (docs/pipelines/mast3r_slam.md): per-sequence
descriptives, no thresholds, no evaluation metric. Called by
scripts/mast3r_slam_run_corpus.py after a sequence's run + reconstruction
+ trajectory-quality steps finish.

Does NOT import from src/eval/ or src/eval_ext/ (this task's hard
constraint). GT depth loading duplicates the frozen raw-to-mm conversion
(depth_mm = raw/65535*100, mask raw==0/raw==65535) directly here rather
than importing src/gt/depth.py -- same conservative precedent Stage 1/2
used for src/eval/ (mixing GT-loading code with MASt3R output in one
call is exactly what that precedent exists to keep separate, so it's
duplicated here too even though this task's constraint only names
src/eval/src/eval_ext explicitly).

Depth-scale definition matches scripts/d1_stage_a_summary.py's (which
itself calls the locked src/eval/scale_recovery.py -- read here, not
imported, to confirm the formula): per-frame ratio =
median(GT depth mm, GT-valid pixels) / median(MASt3R Z, same pixels,
mapped through the resolution mapping), then
depth_scale_median = median(ratios), depth_scale_relative_iqr =
(q75-q25)/median.

Usage:
    python scripts/mast3r_slam_sequence_descriptives.py --sequence <name>
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import tifffile

REPO = Path("/data1_ycao/chua/projects/mrsp")
DATASET_ROOT = Path("/data1_ycao/chua/datasets/C3VDv2")
REGISTERED_DIR = DATASET_ROOT / "registered_videos"
FULL_RUN_ROOT = REPO / "results/pipelines/mast3r_slam_full_run"

sys.path.insert(0, str(REPO / "scripts"))
from mast3r_slam_resolution_mapping import original_to_model  # noqa: E402

# The resize/crop mapping is a fixed function of the (constant, dataset-wide)
# 1350x1080 input size and MASt3R's 512px long-edge target -- identical for
# every sequence (round-trip tested in Stage 2,
# results/pipelines/mast3r_slam_perframe/c1_cecum_t1_v1/resolution_mapping.json).
# Read from that already-validated file instead of recomputing via
# mast3r_slam.mast3r_utils.resize_img, which imports torch at module level
# and isn't otherwise needed here (this script reads already-saved
# depth/pose data only) -- lets this script run in scratch/.venv, not the
# heavier mast3r-slam conda env.
_RESOLUTION_MAPPING_REFERENCE = REPO / "results/pipelines/mast3r_slam_perframe/c1_cecum_t1_v1/resolution_mapping.json"


def load_resolution_mapping() -> dict:
    return json.loads(_RESOLUTION_MAPPING_REFERENCE.read_text())["resolution_mapping"]


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


def rot_angle_deg(R: np.ndarray) -> float:
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    return float(np.degrees(np.arccos(c)))


def gt_motion_over_range(gt_poses: list[np.ndarray], frame_indices: list[int]) -> dict:
    total_path = 0.0
    total_rot = 0.0
    for i, j in zip(frame_indices[:-1], frame_indices[1:]):
        Ci, Cj = gt_poses[i], gt_poses[j]
        total_path += float(np.linalg.norm(Cj[:3, 3] - Ci[:3, 3]))
        G = np.linalg.inv(Ci) @ Cj
        total_rot += rot_angle_deg(G[:3, :3])
    return {"total_path_length_mm": total_path, "total_rotation_deg": total_rot}


def depth_members(zf: zipfile.ZipFile) -> dict[int, str]:
    """frame_index -> member name, duplicated from src/gt/depth.py's
    depth_members (not imported -- see module docstring)."""
    out = {}
    for n in zf.namelist():
        if n.endswith("_depth.tiff") and ("/depth/" in n or n.startswith("depth/")):
            stem = n.split("/")[-1]
            idx = int(stem.split("_")[0])
            out[idx] = n
    return out


def load_gt_depth_mm(zf: zipfile.ZipFile, member: str) -> tuple[np.ndarray, np.ndarray]:
    """raw uint16 -> (depth_mm float64, valid bool). Duplicated conversion,
    CLAUDE.md's frozen convention: depth_mm = raw/65535*100, mask
    raw==0 (no hit) and raw==65535 (clamped)."""
    raw = tifffile.imread(io.BytesIO(zf.read(member)))
    if raw.ndim == 3:
        raw = raw[:, :, 0]
    valid = (raw != 0) & (raw != 65535)
    depth_mm = raw.astype(np.float64) * (100.0 / 65535.0)
    return depth_mm, valid


def compute_depth_scale(sequence: str, m: dict, n_frames: int) -> dict:
    out_dir = FULL_RUN_ROOT / sequence
    depth_dir = out_dir / "depth"
    zpath = REGISTERED_DIR / f"{sequence}.zip"

    ratios = []
    with zipfile.ZipFile(zpath) as zf:
        d_members = depth_members(zf)
        for frame_idx in range(n_frames):
            npz_path = depth_dir / f"{frame_idx:04d}.npz"
            if not npz_path.exists() or frame_idx not in d_members:
                continue
            gt_mm, gt_valid = load_gt_depth_mm(zf, d_members[frame_idx])
            gt_ys, gt_xs = np.nonzero(gt_valid)
            if len(gt_ys) == 0:
                continue

            px, py = original_to_model(gt_xs.astype(np.float64), gt_ys.astype(np.float64), m)
            px_r, py_r = np.round(px).astype(np.int64), np.round(py).astype(np.int64)
            in_bounds = (px_r >= 0) & (px_r < m["model_grid_w"]) & (py_r >= 0) & (py_r < m["model_grid_h"])
            if not in_bounds.any():
                continue

            npz = np.load(npz_path)
            z = npz["z"]
            pred_vals = z[py_r[in_bounds], px_r[in_bounds]]
            gt_vals = gt_mm[gt_ys[in_bounds], gt_xs[in_bounds]]
            pred_finite = np.isfinite(pred_vals) & (pred_vals != 0)
            if not pred_finite.any():
                continue
            pred_median = float(np.median(pred_vals[pred_finite]))
            if pred_median == 0:
                continue
            gt_median = float(np.median(gt_vals[pred_finite]))
            ratios.append(gt_median / pred_median)

    if not ratios:
        return {"depth_scale_median": None, "depth_scale_relative_iqr": None, "n_frames_used": 0}
    ratios = np.array(ratios)
    median = float(np.median(ratios))
    q1, q3 = np.percentile(ratios, [25, 75])
    return {
        "depth_scale_median": median,
        "depth_scale_relative_iqr": float((q3 - q1) / median) if median else None,
        "n_frames_used": int(len(ratios)),
    }


def compute_longest_keyframe_gap(sequence: str, gt_poses: list[np.ndarray], n_frames: int) -> dict:
    """Longest run of consecutive frames with no new keyframe. Includes
    the TAIL gap (last keyframe -> final frame of the sequence) as a
    candidate even though the final frame is never itself a keyframe --
    this is exactly the frame 168-217 gap Stage 2 diagnosed for
    c1_cecum_t1_v1 (49 frames, longer than any actual inter-keyframe
    interval, 30 frames), so omitting it would silently miss the
    sequence's real longest tracked-without-a-keyframe run."""
    out_dir = FULL_RUN_ROOT / sequence
    ids = []
    with open(out_dir / "keyframes_final.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids.append(int(row["frame_id"]))
    ids = sorted(ids)
    n_keyframes = len(ids)
    last_frame_idx = n_frames - 1
    if ids and ids[-1] != last_frame_idx:
        ids = ids + [last_frame_idx]
    if len(ids) < 2:
        return {"n_keyframes": n_keyframes, "longest_gap_frames": None, "longest_gap_range": None,
                "longest_gap_gt_motion": None}

    gaps = [(j - i, i, j) for i, j in zip(ids[:-1], ids[1:])]
    gaps.sort(key=lambda g: -g[0])
    gap_len, i, j = gaps[0]
    motion = gt_motion_over_range(gt_poses, list(range(i, j + 1)))
    return {
        "n_keyframes": n_keyframes, "longest_gap_frames": gap_len, "longest_gap_range": [i, j],
        "longest_gap_gt_motion": motion,
    }


def check_completion(sequence: str) -> dict:
    """D1.1 operational definition (docs/success_criteria.md, 2026-09-23
    clarification, read not modified): every frame has finite predicted
    depth and finite predicted pose, and Sim(3) alignment succeeds."""
    out_dir = FULL_RUN_ROOT / sequence
    poses_path = out_dir / "poses_per_frame.csv"
    n_total = n_reconstructed = n_finite_pose = 0
    with open(poses_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_total += 1
            if row["reconstructed"] == "True":
                n_reconstructed += 1
                vals = [row["tx"], row["ty"], row["tz"], row["qx"], row["qy"], row["qz"], row["qw"]]
                if all(np.isfinite(float(v)) for v in vals):
                    n_finite_pose += 1

    n_finite_depth = 0
    n_depth_checked = 0
    depth_dir = out_dir / "depth"
    for npz_path in sorted(depth_dir.glob("*.npz")):
        n_depth_checked += 1
        npz = np.load(npz_path)
        if np.isfinite(npz["z"]).all() and np.isfinite(npz["conf"]).all():
            n_finite_depth += 1

    traj_path = out_dir / "trajectory_quality_perframe.json"
    sim3_ok = traj_path.exists() and json.loads(traj_path.read_text()).get("ate_rmse_mm") is not None

    complete = (n_total > 0 and n_reconstructed == n_total and n_finite_pose == n_total
                and n_finite_depth == n_depth_checked and sim3_ok)
    return {
        "n_total_frames": n_total, "n_reconstructed": n_reconstructed,
        "n_finite_pose": n_finite_pose, "n_depth_checked": n_depth_checked,
        "n_finite_depth": n_finite_depth, "sim3_alignment_succeeded": bool(sim3_ok),
        "d1_1_complete": bool(complete),
    }


def compute(sequence: str) -> dict:
    out_dir = FULL_RUN_ROOT / sequence
    manifest = json.loads((out_dir / "MANIFEST.json").read_text())
    n_frames = manifest["n_frames_total"]

    m = load_resolution_mapping()

    gt_poses = load_gt_poses(sequence)

    completion = check_completion(sequence)
    traj = json.loads((out_dir / "trajectory_quality_perframe.json").read_text())
    depth_scale = compute_depth_scale(sequence, m, n_frames)
    kf_gap = compute_longest_keyframe_gap(sequence, gt_poses, n_frames)

    result = {
        "sequence": sequence,
        "completion": completion,
        "ate_rmse_mm": traj["ate_rmse_mm"],
        "s_pose": traj["sim3_scale"],
        "endpoint_drift_frac_of_gt_path": traj["endpoint_drift_frac_of_gt_path"],
        "gt_path_length_mm": traj["gt_path_length_mm"],
        "depth_scale": depth_scale,
        "keyframe_gap": kf_gap,
    }
    out_path = out_dir / "descriptives.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence", required=True)
    args = ap.parse_args()
    r = compute(args.sequence)
    print(json.dumps(r, indent=2))
