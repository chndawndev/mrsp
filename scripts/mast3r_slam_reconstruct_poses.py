#!/usr/bin/env python
"""MASt3R-SLAM Stage 2 step 4 (docs/pipelines/mast3r_slam.md): reconstruct
every frame's final camera-to-world pose from scripts/
mast3r_slam_run_perframe.py's per-frame tracking records.

For a keyframe: use its own final Sim3 pose directly (keyframes_final.csv)
-- per the task's own instruction, no re-anchoring needed since that IS
its final pose.

For a non-keyframe frame: T_WC_frame_final = T_WC_ref_final * T_ref_to_frame
(lietorch Sim3 composition, the SAME operator mast3r_slam/tracker.py's
own `track()` uses -- T_WCf = T_WCk * T_CkCf -- substituting the
reference keyframe's FINAL globally-optimized pose for the tracking-time
one it used when computing T_ref_to_frame; mathematically exact because
T_ref_to_frame doesn't depend on which T_WC_ref was used to derive it,
see docs/pipelines/mast3r_slam.md Stage 2's "Key technical findings").
Scale is preserved through the Sim3 composition and only dropped at the
very end via `mast3r_slam.lietorch_utils.as_SE3` -- the same helper
`mast3r_slam/evaluate.py::save_traj` uses for the keyframe-only TUM file,
so this is the same convention, not a new one.

Does NOT compute any evaluation metric and does NOT import from
src/eval/ or src/gt/.

Usage (mast3r-slam conda env, no GPU needed -- pure pose composition):
    python scripts/mast3r_slam_reconstruct_poses.py --sequence c1_cecum_t1_v1
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/data1_ycao/chua/projects/mrsp")
MAST3R_REPO = REPO / "scratch/pipelines/MASt3R-SLAM"
PERFRAME_ROOT = REPO / "results/pipelines/mast3r_slam_perframe"

sys.path.insert(0, str(MAST3R_REPO))
import lietorch  # noqa: E402
from mast3r_slam.lietorch_utils import as_SE3  # noqa: E402


def load_keyframes_final(out_dir: Path) -> dict[int, list[float]]:
    kf = {}
    with open(out_dir / "keyframes_final.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_id = int(row["frame_id"])
            vec = [float(row[k]) for k in ["tx", "ty", "tz", "qx", "qy", "qz", "qw", "s"]]
            kf[frame_id] = vec
    return kf


def load_per_frame_tracking(out_dir: Path) -> list[dict]:
    rows = []
    with open(out_dir / "per_frame_tracking.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t_ref_to_frame = json.loads(row["T_ref_to_frame_json"])
            rows.append({
                "frame_id": int(row["frame_id"]),
                "ref_keyframe_frame_id": int(row["ref_keyframe_frame_id"]),
                "T_ref_to_frame": t_ref_to_frame,
                "match_frac": None if row["match_frac"] == "" else float(row["match_frac"]),
                "match_frac_k": None if row["match_frac_k"] == "" else float(row["match_frac_k"]),
                "skipped": row["skipped"] == "True",
            })
    return rows


def sim3_from_vec(vec: list[float]) -> "lietorch.Sim3":
    import torch
    return lietorch.Sim3(torch.tensor(vec, dtype=torch.float32).reshape(1, 8))


def reconstruct(sequence: str, out_root: Path = PERFRAME_ROOT) -> list[dict]:
    out_dir = out_root / sequence
    kf_final = load_keyframes_final(out_dir)
    tracking_rows = load_per_frame_tracking(out_dir)
    keyframe_ids = set(kf_final.keys())

    results = []
    # frame 0: always a keyframe (the run's anchor), handled by the keyframe branch below
    for row in tracking_rows:
        frame_id = row["frame_id"]
        if frame_id in keyframe_ids:
            continue  # handled below, directly from keyframes_final.csv
        if row["skipped"] or row["T_ref_to_frame"] is None:
            results.append({
                "frame_id": frame_id, "is_keyframe": False,
                "ref_keyframe_frame_id": row["ref_keyframe_frame_id"],
                "tx": None, "ty": None, "tz": None, "qx": None, "qy": None, "qz": None, "qw": None,
                "reconstructed": False, "reason": "skipped_frame_no_valid_relative_pose",
            })
            continue
        ref_id = row["ref_keyframe_frame_id"]
        if ref_id not in kf_final:
            results.append({
                "frame_id": frame_id, "is_keyframe": False,
                "ref_keyframe_frame_id": ref_id,
                "tx": None, "ty": None, "tz": None, "qx": None, "qy": None, "qz": None, "qw": None,
                "reconstructed": False, "reason": f"reference_keyframe_{ref_id}_never_became_final_keyframe",
            })
            continue
        T_WCk_final = sim3_from_vec(kf_final[ref_id])
        T_ref_to_frame = sim3_from_vec(row["T_ref_to_frame"])
        T_WCf_final = T_WCk_final * T_ref_to_frame
        se3 = as_SE3(T_WCf_final)
        x, y, z, qx, qy, qz, qw = se3.data.numpy().reshape(-1)
        results.append({
            "frame_id": frame_id, "is_keyframe": False, "ref_keyframe_frame_id": ref_id,
            "tx": float(x), "ty": float(y), "tz": float(z),
            "qx": float(qx), "qy": float(qy), "qz": float(qz), "qw": float(qw),
            "reconstructed": True, "reason": "",
        })

    for frame_id, vec in kf_final.items():
        T_WC_final = sim3_from_vec(vec)
        se3 = as_SE3(T_WC_final)
        x, y, z, qx, qy, qz, qw = se3.data.numpy().reshape(-1)
        results.append({
            "frame_id": frame_id, "is_keyframe": True, "ref_keyframe_frame_id": frame_id,
            "tx": float(x), "ty": float(y), "tz": float(z),
            "qx": float(qx), "qy": float(qy), "qz": float(qz), "qw": float(qw),
            "reconstructed": True, "reason": "",
        })

    results.sort(key=lambda r: r["frame_id"])
    return results


def sanity_check_against_tum(sequence: str, results: list[dict], save_as: str) -> dict:
    """Keyframes' reconstructed poses (direct passthrough) should match
    the already-saved TUM file (logs/<save_as>/<sequence>.txt) exactly --
    a correctness check on THIS script, not a new metric."""
    tum_path = MAST3R_REPO / "logs" / save_as / f"{sequence}.txt"
    tum_by_frame = {}
    fps = 30.0
    with open(tum_path) as f:
        for line in f:
            vals = [float(x) for x in line.strip().split()]
            t, tx, ty, tz, qx, qy, qz, qw = vals[:8]
            tum_by_frame[round(t * fps)] = (tx, ty, tz, qx, qy, qz, qw)

    max_diff = 0.0
    n_checked = 0
    for r in results:
        if not r["is_keyframe"]:
            continue
        fid = r["frame_id"]
        if fid not in tum_by_frame:
            continue
        tum_vals = np.array(tum_by_frame[fid])
        recon_vals = np.array([r["tx"], r["ty"], r["tz"], r["qx"], r["qy"], r["qz"], r["qw"]])
        diff = float(np.max(np.abs(tum_vals - recon_vals)))
        max_diff = max(max_diff, diff)
        n_checked += 1
    return {"n_keyframes_checked": n_checked, "max_abs_diff": max_diff}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence", required=True)
    ap.add_argument("--out-root", default=str(PERFRAME_ROOT))
    ap.add_argument("--save-as", default=None, help="defaults to <sequence>_perframe, matching Stage 2's convention")
    args = ap.parse_args()
    out_root = Path(args.out_root)
    save_as = args.save_as or f"{args.sequence}_perframe"

    results = reconstruct(args.sequence, out_root)
    out_dir = out_root / args.sequence
    with open(out_dir / "poses_per_frame.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "frame_id", "is_keyframe", "ref_keyframe_frame_id",
            "tx", "ty", "tz", "qx", "qy", "qz", "qw", "reconstructed", "reason",
        ])
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    n_total = len(results)
    n_keyframes = sum(1 for r in results if r["is_keyframe"])
    n_reconstructed = sum(1 for r in results if r["reconstructed"])
    n_failed = n_total - n_reconstructed
    print(f"n_total_frames={n_total} n_keyframes={n_keyframes} n_reconstructed={n_reconstructed} n_failed={n_failed}")
    for r in results:
        if not r["reconstructed"]:
            print(f"  NOT RECONSTRUCTED: frame {r['frame_id']}: {r['reason']}")

    sanity = sanity_check_against_tum(args.sequence, results, save_as)
    print(f"sanity check vs TUM file: {sanity}")
    with open(out_dir / "reconstruction_manifest.json", "w") as f:
        json.dump({
            "n_total_frames": n_total, "n_keyframes": n_keyframes,
            "n_reconstructed": n_reconstructed, "n_failed": n_failed,
            "sanity_check_vs_tum": sanity,
        }, f, indent=2)


if __name__ == "__main__":
    main()
