#!/usr/bin/env python
"""CUT3R Stage 2 diagnosis: trajectory-quality and depth-scale
descriptives (NOT evaluation metrics; same definitions as
scripts/cut3r_stage1_checks.py section 7) for several runs of the same
sequence that differ only in inference path or numerical setting, to see
how far those descriptives move between runs. Nothing from src/eval/ or
src/eval_ext/ is imported.

Usage: scratch/.venv/bin/python scripts/cut3r_variant_descriptives.py --sequence NAME --tags TAG [TAG ...]
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

REPO = Path("/data1_ycao/chua/projects/mrsp")
sys.path.insert(0, str(REPO / "scripts"))

from mast3r_slam_sequence_descriptives import depth_members, load_gt_depth_mm, load_gt_poses  # noqa: E402
from mast3r_slam_trajectory_quality_perframe import path_length, umeyama  # noqa: E402
from pipeline_adapters import BilinearGridMap  # noqa: E402

OUT_ROOT = REPO / "results/pipelines/cut3r_stage1"
REGISTERED_DIR = Path("/data1_ycao/chua/datasets/C3VDv2/registered_videos")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence", required=True)
    ap.add_argument("--tags", nargs="+", required=True)
    args = ap.parse_args()
    m = json.loads((OUT_ROOT / "grid_alignment" / "grid_alignment_test.json").read_text())["mapping_parameters"]
    gm = BilinearGridMap(m)
    gt = load_gt_poses(args.sequence)
    gt_pos = np.array([g[:3, 3] for g in gt])
    gl = path_length(gt_pos)
    rows = {}
    with zipfile.ZipFile(REGISTERED_DIR / f"{args.sequence}.zip") as zf:
        members = depth_members(zf)
        gt_depth = [load_gt_depth_mm(zf, members[i]) for i in range(len(gt))]
    for tag in args.tags:
        d = OUT_ROOT / tag
        man = json.loads((d / "MANIFEST.json").read_text())
        pred = np.load(d / "poses_c2w.npy")
        if len(pred) != len(gt):
            raise RuntimeError(f"{tag}: {len(pred)} poses for {len(gt)} GT frames")
        _, _, c, aligned = umeyama(pred[:, :3, 3], gt_pos)
        err = np.linalg.norm(aligned - gt_pos, axis=1)
        ratios = []
        for i, (gt_mm, gt_valid) in enumerate(gt_depth):
            with np.load(d / "depth" / f"{i:04d}.npz") as npz:
                z = gm.sample(npz["z"])
            use = gt_valid.ravel() & gm.available
            ratios.append(float(np.median(gt_mm.ravel()[use])) / float(np.median(z[use])))
        ratios = np.array(ratios)
        q1, q3 = np.percentile(ratios, [25, 75])
        rows[tag] = {
            "path": man.get("path", "parallel"), "tf32": man.get("tf32", {"matmul": True, "cudnn": True}),
            "input_noise_sigma": man.get("input_noise_sigma", 0.0), "n_frames": int(len(pred)),
            "ate_rmse_mm": float(np.sqrt(np.mean(err**2))), "s_pose": float(c),
            "endpoint_drift_frac_of_gt_path": float(err[-1] / gl),
            "depth_scale_median": float(np.median(ratios)), "depth_scale_relative_iqr": float((q3 - q1) / np.median(ratios)),
            "peak_gpu_allocated_mib": man["peak_gpu_allocated_mib"], "inference_seconds": man["inference_seconds"],
        }
        r = rows[tag]
        print(f"{tag:45s} ATE {r['ate_rmse_mm']:6.2f} mm  drift {100 * r['endpoint_drift_frac_of_gt_path']:5.2f}%  "
              f"s_pose {r['s_pose']:6.2f}  depth scale {r['depth_scale_median']:6.2f}  rel IQR {r['depth_scale_relative_iqr']:.3f}")
    (OUT_ROOT / f"variant_descriptives__{args.sequence}.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
