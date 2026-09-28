#!/usr/bin/env python
"""MASt3R-SLAM Stage 2 step 6 (docs/pipelines/mast3r_slam.md): diagnose
why keyframe selection stopped at frame 168/217 in Stage 1 (flagged
UNKNOWN there). Two independent signals, kept separate:

  1. GT-only camera motion (no MASt3R data needed): total path length and
     total rotation over frames 169-217, compared against the same two
     numbers for each inter-keyframe interval within frames 0-168.
  2. MASt3R-SLAM's own per-frame tracking signal for frames 169-217
     (match_frac, match_frac_k, skipped), from scripts/
     mast3r_slam_run_perframe.py's per_frame_tracking.csv.

Computes no evaluation metric, imports nothing from src/eval/ or src/gt/.

Usage:
    python scripts/mast3r_slam_diagnose_gap.py --sequence c1_cecum_t1_v1
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


def load_gt_poses(sequence: str) -> list[np.ndarray]:
    zpath = REGISTERED_DIR / f"{sequence}.zip"
    with zipfile.ZipFile(zpath) as zf:
        lines = zf.read("pose.txt").decode().strip().split("\n")
    return [np.array([float(x) for x in line.split(",")]).reshape(4, 4).T for line in lines]


def rot_angle_deg(R: np.ndarray) -> float:
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    return float(np.degrees(np.arccos(c)))


def gt_motion(gt_poses: list[np.ndarray], frame_indices: list[int]) -> dict:
    """Total path length (mm) and total rotation (deg) across consecutive
    pairs within frame_indices (assumed sorted, may be non-contiguous --
    e.g. a keyframe interval spans two indices directly, a frame-range
    spans every consecutive integer pair)."""
    total_path = 0.0
    total_rot = 0.0
    for i, j in zip(frame_indices[:-1], frame_indices[1:]):
        Ci, Cj = gt_poses[i], gt_poses[j]
        total_path += float(np.linalg.norm(Cj[:3, 3] - Ci[:3, 3]))
        G = np.linalg.inv(Ci) @ Cj
        total_rot += rot_angle_deg(G[:3, :3])
    return {
        "frame_range": [frame_indices[0], frame_indices[-1]],
        "n_steps": len(frame_indices) - 1,
        "total_path_length_mm": total_path,
        "total_rotation_deg": total_rot,
        "path_length_per_step_mm": total_path / (len(frame_indices) - 1) if len(frame_indices) > 1 else float("nan"),
        "rotation_per_step_deg": total_rot / (len(frame_indices) - 1) if len(frame_indices) > 1 else float("nan"),
    }


def load_keyframe_indices(sequence: str) -> list[int]:
    out_dir = PERFRAME_ROOT / sequence
    ids = []
    with open(out_dir / "keyframes_final.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids.append(int(row["frame_id"]))
    return sorted(ids)


def load_tracking_rows(sequence: str, lo: int, hi: int) -> list[dict]:
    out_dir = PERFRAME_ROOT / sequence
    rows = []
    with open(out_dir / "per_frame_tracking.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fid = int(row["frame_id"])
            if lo <= fid <= hi:
                rows.append({
                    "frame_id": fid,
                    "ref_keyframe_frame_id": int(row["ref_keyframe_frame_id"]),
                    "match_frac": None if row["match_frac"] == "" else float(row["match_frac"]),
                    "match_frac_k": None if row["match_frac_k"] == "" else float(row["match_frac_k"]),
                    "skipped": row["skipped"] == "True",
                })
    return sorted(rows, key=lambda r: r["frame_id"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence", required=True)
    ap.add_argument("--gap-start", type=int, default=169)
    ap.add_argument("--gap-end", type=int, default=217)
    args = ap.parse_args()

    gt_poses = load_gt_poses(args.sequence)
    n_frames = len(gt_poses)
    keyframe_ids = load_keyframe_indices(args.sequence)
    print(f"keyframe frame indices ({len(keyframe_ids)}): {keyframe_ids}")

    gap_range = list(range(args.gap_start, args.gap_end + 1))
    gap_motion = gt_motion(gt_poses, gap_range)
    print(f"gap ({args.gap_start}-{args.gap_end}) GT motion: {json.dumps(gap_motion, indent=2)}")

    interval_motions = []
    for i, j in zip(keyframe_ids[:-1], keyframe_ids[1:]):
        if j > args.gap_start:
            break
        interval_motions.append(gt_motion(gt_poses, [i, j]))
    print(f"inter-keyframe intervals within 0-{args.gap_start - 1}, n={len(interval_motions)}:")
    for m in interval_motions:
        print(f"  {m}")

    path_per_step_vals = [m["path_length_per_step_mm"] for m in interval_motions]
    rot_per_step_vals = [m["rotation_per_step_deg"] for m in interval_motions]
    interval_summary = {
        "n_intervals": len(interval_motions),
        "path_length_per_step_mm": {
            "median": float(np.median(path_per_step_vals)), "min": float(np.min(path_per_step_vals)),
            "max": float(np.max(path_per_step_vals)),
        },
        "rotation_per_step_deg": {
            "median": float(np.median(rot_per_step_vals)), "min": float(np.min(rot_per_step_vals)),
            "max": float(np.max(rot_per_step_vals)),
        },
    }
    print(f"inter-keyframe interval summary (0-{args.gap_start - 1}): {json.dumps(interval_summary, indent=2)}")

    tracking_rows = load_tracking_rows(args.sequence, args.gap_start, args.gap_end)
    n_skipped = sum(1 for r in tracking_rows if r["skipped"])
    match_frac_vals = [r["match_frac"] for r in tracking_rows if r["match_frac"] is not None]
    match_frac_k_vals = [r["match_frac_k"] for r in tracking_rows if r["match_frac_k"] is not None]
    tracking_summary = {
        "n_frames_in_gap_with_tracking_record": len(tracking_rows),
        "n_skipped": n_skipped,
        "match_frac": {
            "median": float(np.median(match_frac_vals)) if match_frac_vals else None,
            "min": float(np.min(match_frac_vals)) if match_frac_vals else None,
        },
        "match_frac_k": {
            "median": float(np.median(match_frac_k_vals)) if match_frac_k_vals else None,
            "min": float(np.min(match_frac_k_vals)) if match_frac_k_vals else None,
        },
    }
    print(f"tracking signal in gap: {json.dumps(tracking_summary, indent=2)}")

    result = {
        "sequence": args.sequence, "n_frames": n_frames, "keyframe_frame_indices": keyframe_ids,
        "gap_gt_motion": gap_motion, "inter_keyframe_interval_motions": interval_motions,
        "inter_keyframe_interval_summary": interval_summary,
        "gap_tracking_rows": tracking_rows, "gap_tracking_summary": tracking_summary,
    }
    out_dir = PERFRAME_ROOT / args.sequence
    with open(out_dir / "gap_diagnosis.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {out_dir / 'gap_diagnosis.json'}")


if __name__ == "__main__":
    main()
