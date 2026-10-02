#!/usr/bin/env python
"""Noise sensitivity of the three pipelines on one sequence
(docs/noise_sensitivity.md): each pipeline's primary (pinned) run against
(a) an unperturbed rerun and (b) a run with i.i.d. Gaussian noise of std
1e-6 (0-1 image scale) added to the input, seed 1.

Descriptive. NO evaluation metric is computed and nothing from src/eval/
or src/eval_ext/ is imported.

Per frame, run b against run a (the primary):
  - translation difference relative to the trajectory extent of run a
    (max distance of any frame from frame 0), pipeline's own units;
  - rotation difference in degrees (angle of R_a^T R_b);
  - Z-depth max and median relative difference |z_a - z_b| / |z_a| over
    the pipeline's own depth grid (EndoDAC 1350x1080, MASt3R-SLAM and
    CUT3R 512x400).
Per run, sequence level (definitions of docs/pipelines/cut3r.md section
7): ATE RMSE after position-only Umeyama, endpoint drift / GT path
length, depth scale median and relative IQR (per frame
median(GT) / median(pred) over GT-valid pixels with a prediction; grid
depth mapped by the pixel-center bilinear mapping).

Usage: scratch/.venv/bin/python scripts/noise_sensitivity_compare.py [--sequence c1_cecum_t1_v1]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

REPO = Path("/data1_ycao/chua/projects/mrsp")
sys.path.insert(0, str(REPO / "scripts"))

from mast3r_slam_sequence_descriptives import depth_members, load_gt_depth_mm, load_gt_poses  # noqa: E402
from mast3r_slam_trajectory_quality_perframe import path_length, umeyama  # noqa: E402
from pipeline_adapters import BilinearGridMap, load_mast3r_mapping  # noqa: E402

P = REPO / "results/pipelines"
V = P / "variability"
OUT_DIR = REPO / "results/noise_sensitivity"
REGISTERED_DIR = Path("/data1_ycao/chua/datasets/C3VDv2/registered_videos")
RUNS = {  # pipeline -> {run label -> root}
    "endodac": {"primary": P / "endodac_full_run", "rerun_unperturbed": V / "endodac/rerun_unperturbed",
                "noise_1e-6_seed1": V / "endodac/seed1"},
    "mast3r_slam": {"primary": P / "mast3r_slam_full_run", "rerun_unperturbed": V / "mast3r_slam/rerun_unperturbed",
                    "noise_1e-6_seed1": V / "mast3r_slam/seed1"},
    "cut3r": {"primary": P / "cut3r_full_run", "rerun_unperturbed": P / "cut3r_stage1/{seq}_recurrent",
              "noise_1e-6_seed1": V / "cut3r/seed1"},
}


class Run:
    """Poses (camera-to-world, pipeline units) and per-frame Z on the
    pipeline's own grid, for one run of one sequence."""

    def __init__(self, pipeline: str, root: Path, seq: str, n_frames: int):
        self.pipeline = pipeline
        self.dir = Path(str(root).format(seq=seq)) if "{seq}" in str(root) else root / seq
        if pipeline == "endodac":
            self.poses = np.load(self.dir / "poses_pred.npy").astype(np.float64)
        elif pipeline == "cut3r":
            self.poses = np.load(self.dir / "poses_c2w.npy").astype(np.float64)
        else:
            poses = {}
            with open(self.dir / "poses_per_frame.csv") as f:
                for row in csv.DictReader(f):
                    if row["reconstructed"] != "True":
                        continue
                    T = np.eye(4)
                    T[:3, :3] = Rotation.from_quat([float(row[k]) for k in ("qx", "qy", "qz", "qw")]).as_matrix()
                    T[:3, 3] = [float(row[k]) for k in ("tx", "ty", "tz")]
                    poses[int(row["frame_id"])] = T
            if sorted(poses) != list(range(n_frames)):
                raise RuntimeError(f"{self.dir}: {len(poses)} reconstructed poses for {n_frames} frames")
            self.poses = np.stack([poses[i] for i in range(n_frames)])
        if len(self.poses) != n_frames:
            raise RuntimeError(f"{self.dir}: {len(self.poses)} poses for {n_frames} frames")

    def z(self, i: int) -> np.ndarray:
        if self.pipeline == "endodac":
            return np.load(self.dir / "depth" / f"{i:04d}_pred_depth.npy").astype(np.float64)
        with np.load(self.dir / "depth" / f"{i:04d}.npz") as npz:
            return npz["z"].astype(np.float64)


def dist(x: np.ndarray) -> dict:
    q = np.percentile(x, [50, 95])
    return {"min": float(x.min()), "median": float(q[0]), "p95": float(q[1]), "max": float(x.max())}


def per_frame(a: Run, b: Run) -> dict:
    Pa, Pb = a.poses, b.poses
    n = len(Pa)
    extent = float(np.linalg.norm(Pa[:, :3, 3] - Pa[0, :3, 3], axis=1).max())
    dt = np.linalg.norm(Pa[:, :3, 3] - Pb[:, :3, 3], axis=1) / extent
    cosang = (np.einsum("nij,nij->n", Pa[:, :3, :3], Pb[:, :3, :3]) - 1) / 2
    rot = np.arccos(np.clip(cosang, -1, 1))
    chord = np.linalg.norm(Pa[:, :3, :3] - Pb[:, :3, :3], axis=(1, 2)) / np.sqrt(2)
    rot = np.degrees(np.where(rot < 1e-3, chord, rot))
    zmax, zmed = np.zeros(n), np.zeros(n)
    for i in range(n):
        za, zb = a.z(i), b.z(i)
        rel = np.abs(za - zb) / np.abs(za)
        zmax[i], zmed[i] = rel.max(), np.median(rel)
    return {"n_frames": n, "trajectory_extent": extent,
            "translation_diff_relative_to_extent": dist(dt), "rotation_diff_deg": dist(rot),
            "z_max_relative_diff": dist(zmax), "z_median_relative_diff": dist(zmed),
            "bit_identical": bool((dt == 0).all() and (rot == 0).all() and (zmax == 0).all())}


def sequence_level(run: Run, gt_pos: np.ndarray, gt_depth: list, grid_map) -> dict:
    _, _, c, aligned = umeyama(run.poses[:, :3, 3], gt_pos)
    err = np.linalg.norm(aligned - gt_pos, axis=1)
    ratios = []
    for i, (gt_mm, gt_valid) in enumerate(gt_depth):
        z = run.z(i)
        if grid_map is None:
            d, use = z.ravel(), gt_valid.ravel()
        else:
            d, use = grid_map.sample(z), gt_valid.ravel() & grid_map.available
        ratios.append(float(np.median(gt_mm.ravel()[use])) / float(np.median(d[use])))
    ratios = np.array(ratios)
    q1, q3 = np.percentile(ratios, [25, 75])
    return {"ate_rmse_mm": float(np.sqrt(np.mean(err**2))), "s_pose": float(c),
            "endpoint_drift_frac_of_gt_path": float(err[-1] / path_length(gt_pos)),
            "depth_scale_median": float(np.median(ratios)),
            "depth_scale_relative_iqr": float((q3 - q1) / np.median(ratios))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence", default="c1_cecum_t1_v1")
    args = ap.parse_args()
    seq = args.sequence
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gt = load_gt_poses(seq)
    gt_pos = np.array([g[:3, 3] for g in gt])
    with zipfile.ZipFile(REGISTERED_DIR / f"{seq}.zip") as zf:
        members = depth_members(zf)
        gt_depth = [load_gt_depth_mm(zf, members[i]) for i in range(len(gt))]
    grid_map = BilinearGridMap(load_mast3r_mapping())  # same 512x400 grid and mapping for MASt3R-SLAM and CUT3R

    res = {"sequence": seq, "n_frames": len(gt), "pipelines": {}}
    for pipeline, roots in RUNS.items():
        runs = {label: Run(pipeline, root, seq, len(gt)) for label, root in roots.items()}
        gm = None if pipeline == "endodac" else grid_map
        res["pipelines"][pipeline] = {
            "run_dirs": {label: str(r.dir.relative_to(REPO)) for label, r in runs.items()},
            "per_frame_vs_primary": {label: per_frame(runs["primary"], r) for label, r in runs.items() if label != "primary"},
            "sequence_level": {label: sequence_level(r, gt_pos, gt_depth, gm) for label, r in runs.items()},
        }
        print(pipeline, json.dumps(res["pipelines"][pipeline], indent=1))
    (OUT_DIR / f"noise_sensitivity__{seq}.json").write_text(json.dumps(res, indent=2))

    names = {"endodac": "EndoDAC", "mast3r_slam": "MASt3R-SLAM", "cut3r": "CUT3R"}
    L = []
    for label, title in [("rerun_unperturbed", "Unperturbed rerun vs primary run"),
                         ("noise_1e-6_seed1", "Noise std 1e-6 (seed 1) vs primary run")]:
        L += [f"**{title}**, per frame, `{seq}` ({len(gt)} frames)", "",
              "| quantity | statistic | " + " | ".join(names.values()) + " |", "|---|---|---|---|---|"]
        for key, qn in [("translation_diff_relative_to_extent", "translation difference / extent"),
                        ("rotation_diff_deg", "rotation difference (deg)"),
                        ("z_max_relative_diff", "Z, max relative difference"),
                        ("z_median_relative_diff", "Z, median relative difference")]:
            for st in ["min", "median", "p95", "max"]:
                L.append(f"| {qn} | {st} | " + " | ".join(
                    f"{res['pipelines'][p]['per_frame_vs_primary'][label][key][st]:.2g}" for p in names) + " |")
        L.append("| bit-identical | | " + " | ".join(
            "yes" if res["pipelines"][p]["per_frame_vs_primary"][label]["bit_identical"] else "no" for p in names) + " |")
        L.append("")
    L += [f"**Sequence level**, `{seq}` (not evaluation metrics)", "",
          "| pipeline | run | ATE (mm) | endpoint drift | `s_pose` | depth scale median | relative IQR |", "|---|---|---|---|---|---|---|"]
    for p, nm in names.items():
        for label, s in res["pipelines"][p]["sequence_level"].items():
            L.append(f"| {nm} | {label} | {s['ate_rmse_mm']:.3f} | {100 * s['endpoint_drift_frac_of_gt_path']:.2f}% | "
                     f"{s['s_pose']:.2f} | {s['depth_scale_median']:.3f} | {s['depth_scale_relative_iqr']:.3f} |")
    md = "\n".join(L) + "\n"
    (OUT_DIR / f"noise_sensitivity__{seq}.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
