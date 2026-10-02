#!/usr/bin/env python
"""CUT3R Stage 3 (docs/pipelines/cut3r.md): per-sequence descriptives of
the pinned primary run and the corpus summary table. No thresholds, NO
evaluation metric; nothing from src/eval/ or src/eval_ext/ is imported.

Same definitions as MASt3R-SLAM Stage 3
(scripts/mast3r_slam_sequence_descriptives.py,
scripts/mast3r_slam_trajectory_quality_perframe.py, whose Umeyama and GT
loaders are imported):
  - completion, D1.1 operational definition (docs/success_criteria.md,
    2026-09-23 clarification, read not modified): every GT frame has a
    finite predicted depth and a finite predicted pose, and the Sim(3)
    alignment succeeds;
  - ATE RMSE after position-only Umeyama, endpoint drift / GT path length,
    s_pose;
  - depth scale: per frame median(GT mm) / median(pred) over GT-valid
    pixels that have a prediction, sequence value = median over frames,
    relative IQR = (q75 - q25) / median. Predicted depth is mapped from
    the 512x400 grid with the pixel-center bilinear mapping
    (scripts/pipeline_adapters.py::BilinearGridMap), not the
    nearest-neighbour corner-aligned lookup of MASt3R-SLAM Stage 3;
  - missing frames.
Stage A covariates are joined from results/d1/stage_a_summary.csv.

Usage: scratch/.venv/bin/python scripts/cut3r_sequence_descriptives.py [--workers 12]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/data1_ycao/chua/projects/mrsp")
sys.path.insert(0, str(REPO / "scripts"))

from cut3r_run_corpus import FULL_RUN_ROOT, REGISTERED_DIR, discover_sequences  # noqa: E402
from mast3r_slam_aggregate_corpus import COVARIATE_COLS, STAGE_A_CSV  # noqa: E402
from mast3r_slam_sequence_descriptives import depth_members, load_gt_depth_mm, load_gt_poses  # noqa: E402
from mast3r_slam_trajectory_quality_perframe import path_length, umeyama  # noqa: E402
from pipeline_adapters import BilinearGridMap, load_mast3r_mapping  # noqa: E402

OUT_CSV = REPO / "results/cut3r/stage3_summary.csv"
_GRID_MAP = None


def grid_map() -> BilinearGridMap:
    global _GRID_MAP
    if _GRID_MAP is None:
        m = load_mast3r_mapping()  # 512x400 grid, same loader and mapping as CUT3R (docs/pipelines/cut3r.md section 4)
        _GRID_MAP = BilinearGridMap(m)
    return _GRID_MAP


def compute(seq: str) -> dict:
    t0 = time.time()
    out_dir = FULL_RUN_ROOT / seq
    mpath = out_dir / "MANIFEST.json"
    if not mpath.exists():
        return {"sequence": seq, "status": "not_run"}
    man = json.loads(mpath.read_text())
    row = {"sequence": seq, "status": man["status"], "error": (man.get("error") or "")[-300:] or None,
           "gpu_index": man.get("gpu_index"), "gpu_name": man.get("gpu_name"),
           "runtime_seconds_total": man.get("runtime_seconds_total"),
           "inference_seconds": man.get("inference_seconds"),
           "peak_gpu_allocated_mib": man.get("peak_gpu_allocated_mib"), "repo_commit": man.get("repo_commit"),
           "depth_dir_bytes": man.get("depth_dir_bytes")}
    if man["status"] != "ok":
        row["d1_1_complete"] = False
        return row
    gt = load_gt_poses(seq)
    n_gt = len(gt)
    poses = np.load(out_dir / "poses_c2w.npy")
    gm = grid_map()
    if (man["grid_h"], man["grid_w"]) != (gm.grid_h, gm.grid_w):
        raise RuntimeError(f"{seq}: grid {man['grid_h']}x{man['grid_w']} differs from the mapping's")
    depth_frames = sorted(int(p.stem) for p in (out_dir / "depth").glob("*.npz"))
    missing_depth = sorted(set(range(n_gt)) - set(depth_frames))
    n_missing_pose = max(n_gt - len(poses), 0)
    finite_pose = np.isfinite(poses).all(axis=(1, 2))

    ratios, n_finite_depth = [], 0
    with zipfile.ZipFile(REGISTERED_DIR / f"{seq}.zip") as zf:
        members = depth_members(zf)
        for i in depth_frames:
            with np.load(out_dir / "depth" / f"{i:04d}.npz") as npz:
                z = npz["z"]
            if not np.isfinite(z).all():
                continue
            n_finite_depth += 1
            gt_mm, gt_valid = load_gt_depth_mm(zf, members[i])
            use = gt_valid.ravel() & gm.available
            if not use.any():
                continue
            pm = float(np.median(gm.sample(z)[use]))
            if pm != 0.0:
                ratios.append(float(np.median(gt_mm.ravel()[use])) / pm)
    ratios = np.array(ratios)

    sim3_ok, traj = False, {}
    if len(poses) == n_gt and finite_pose.all():
        try:
            gt_pos = np.array([g[:3, 3] for g in gt])
            _, _, c, aligned = umeyama(poses[:, :3, 3], gt_pos)
            err = np.linalg.norm(aligned - gt_pos, axis=1)
            gl = path_length(gt_pos)
            if np.isfinite(c) and np.isfinite(err).all():
                sim3_ok = True
                traj = {"ate_rmse_mm": float(np.sqrt(np.mean(err**2))), "s_pose": float(c), "gt_path_length_mm": gl,
                        "endpoint_drift_frac_of_gt_path": float(err[-1] / gl)}
        except np.linalg.LinAlgError:
            sim3_ok = False
    row.update({
        "n_total_frames": n_gt, "n_frames_with_pose": int(len(poses)), "n_frames_with_depth": len(depth_frames),
        "n_missing_pose": n_missing_pose, "n_missing_depth": len(missing_depth),
        "n_finite_pose": int(finite_pose.sum()), "n_finite_depth": n_finite_depth,
        "n_nonpositive_z_pixels": man.get("n_nonpositive_z_pixels"),
        "sim3_alignment_succeeded": sim3_ok,
        "d1_1_complete": bool(len(poses) == n_gt and len(depth_frames) == n_gt and finite_pose.all()
                              and n_finite_depth == n_gt and sim3_ok),
        **traj,
        "depth_scale_median": float(np.median(ratios)) if len(ratios) else None,
        "depth_scale_relative_iqr": (float((np.percentile(ratios, 75) - np.percentile(ratios, 25)) / np.median(ratios))
                                     if len(ratios) else None),
        "depth_scale_n_frames": int(len(ratios)),
        "depth_scale_n_frames_nonpositive": int((ratios <= 0).sum()),
    })
    (out_dir / "descriptives.json").write_text(json.dumps(row, indent=2))
    print(f"{seq}: {time.time() - t0:.0f}s", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    seqs = discover_sequences()
    with Pool(args.workers) as pool:
        rows = pool.map(compute, seqs, chunksize=1)
    df = pd.DataFrame(rows)
    cov = pd.read_csv(STAGE_A_CSV)
    df = df.merge(cov[[c for c in COVARIATE_COLS if c in cov.columns]], on="sequence", how="left", validate="one_to_one")
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV} ({len(df)} rows); status: {df['status'].value_counts().to_dict()}; "
          f"D1.1 complete: {int(df['d1_1_complete'].fillna(False).sum())}/{len(df)}")


if __name__ == "__main__":
    main()
