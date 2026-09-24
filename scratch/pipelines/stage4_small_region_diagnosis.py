#!/usr/bin/env python
"""Diagnose the small-region recall inversion at tau=0.25:
pred_pose_only scores 0.50, fully_predicted scores 1.00
(results/eval_stage4/summary.json, c1_cecum_t1_v1).

Reruns the identical Stage 4 pipeline (scripts/eval_stage4.py) -- same
sequence, no new sequences -- but keeps every configuration's
predicted_observed[tau] arrays in memory (the original driver discards
them after computing aggregate stats) so the two small GT regions' exact
face-level coverage and the pred_pose_only vs fully_predicted face-level
diff can be reported.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

REPO = Path("/data1_ycao/chua/projects/mrsp")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from geometry.camera import CameraIntrinsics, unproject  # noqa: E402
from geometry.coverage_mesh import stream_coverage_mesh_from_zip  # noqa: E402
from geometry.mesh_stats import face_adjacency, face_areas  # noqa: E402
from geometry.pose import stream_poses_from_zip  # noqa: E402
from gt.depth import depth_members, depth_to_mm, stream_raw_depth_frames_from_zip  # noqa: E402
from eval.evaluable import load_vignette_mask  # noqa: E402
from eval.fusion import run_pose_variant_sequence  # noqa: E402
from eval.ignore_set import compute_ignore_set  # noqa: E402
from eval.oracle import run_oracle_sequence  # noqa: E402
from eval.regions import compute_regions  # noqa: E402
from eval.region_metrics import region_coverage_fraction  # noqa: E402
from eval.scale_recovery import aligned_predicted_pose, compute_depth_scale, compute_pose_alignment  # noqa: E402
from gpu_status import get_gpu_stats  # noqa: E402

SEQ_NAME = "c1_cecum_t1_v1"
DATASET_ROOT = Path("/data1_ycao/chua/datasets/C3VDv2/registered_videos")
INTRINSICS_PATH = "/data1_ycao/chua/datasets/C3VDv2/camera_intrinsics.txt"
PRED_DIR = REPO / "results/pipelines/endodac_full_run" / SEQ_NAME
TAUS = [0.15, 0.25, 0.35, 0.50]
OUT_DIR = REPO / "results/stage4_small_region_diagnosis"
LOG_PATH = REPO / "logs/stage4_small_region_diagnosis.log"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def select_gpu() -> int:
    gpus = get_gpu_stats()
    freest = max(gpus, key=lambda g: g["memory_free_mib"])
    log(f"Selected GPU: index {freest['index']} ({freest['name']}), {freest['memory_free_mib']/1000:.1f} GB free")
    if freest["memory_free_mib"] < 8000:
        raise RuntimeError("no GPU with >= 8GB free")
    return freest["index"]


def main():
    log("=== Stage 4 small-region recall-inversion diagnosis ===")
    gpu_index = select_gpu()
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    import warp as wp

    wp.init()
    device = "cuda:0"

    intr = CameraIntrinsics.from_file(INTRINSICS_PATH)
    cols, rows = np.meshgrid(np.arange(intr.width), np.arange(intr.height))
    px_grid = np.stack([cols.ravel(), rows.ravel()], axis=-1).astype(np.float64)
    cam_rays = unproject(px_grid, intr).astype(np.float32)
    vignette_mask = load_vignette_mask(intr.width, intr.height)

    zpath = DATASET_ROOT / f"{SEQ_NAME}.zip"
    mesh_data = stream_coverage_mesh_from_zip(zpath)
    gt_poses = stream_poses_from_zip(zpath)
    n_frames = len(gt_poses)
    with zipfile.ZipFile(zpath) as zf:
        assert len(depth_members(zf)) == n_frames
    log(f"streamed mesh ({len(mesh_data.faces)} faces) + poses ({n_frames} frames)")

    n_faces = len(mesh_data.faces)
    gt_observed = mesh_data.face_observed
    gt_unobserved = ~gt_observed
    areas = face_areas(mesh_data.vertices, mesh_data.faces)
    adjacency = face_adjacency(mesh_data.vertices, mesh_data.faces)

    gt_depth_mm_frames, gt_depth_valid_frames = [], []
    for raw in stream_raw_depth_frames_from_zip(zpath):
        d_mm, valid = depth_to_mm(raw.ravel())
        gt_depth_mm_frames.append(d_mm)
        gt_depth_valid_frames.append(valid)

    pred_depth_native_frames = [
        np.load(PRED_DIR / "depth" / f"{i:04d}_pred_depth.npy").ravel().astype(np.float64) for i in range(n_frames)
    ]
    pred_poses = np.load(PRED_DIR / "poses_pred.npy")

    depth_scale = compute_depth_scale(gt_depth_mm_frames, gt_depth_valid_frames, pred_depth_native_frames)
    gt_positions = np.array([M[3, :3] for M in gt_poses])
    pred_positions = pred_poses[:, :3, 3]
    alignment = compute_pose_alignment(pred_positions, gt_positions)
    log(f"depth_scale.median={depth_scale.median!r} s_pose={alignment.s_pose!r}")

    def gt_pose_provider(i):
        M = gt_poses[i]
        return M[:3, :3].T, M[3, :3]

    def pred_pose_provider(i):
        return aligned_predicted_pose(alignment, pred_poses[i, :3, :3], pred_poses[i, :3, 3])

    def gt_depth_provider(i):
        return gt_depth_mm_frames[i], gt_depth_valid_frames[i]

    def pred_depth_provider(i):
        d_mm = pred_depth_native_frames[i] * depth_scale.median
        return d_mm, np.ones(len(d_mm), dtype=bool)

    t0 = time.time()
    results_gt_pose = run_pose_variant_sequence(
        wp, device, mesh_data.vertices, mesh_data.faces, n_frames,
        gt_pose_provider, {"gt_depth": gt_depth_provider, "pred_depth": pred_depth_provider},
        cam_rays, vignette_mask, TAUS,
    )
    log(f"pose variant GT done in {time.time()-t0:.1f}s")

    t0 = time.time()
    results_pred_pose = run_pose_variant_sequence(
        wp, device, mesh_data.vertices, mesh_data.faces, n_frames,
        pred_pose_provider, {"gt_depth": gt_depth_provider, "pred_depth": pred_depth_provider},
        cam_rays, vignette_mask, TAUS,
    )
    log(f"pose variant pred done in {time.time()-t0:.1f}s")

    configs = {
        "oracle": results_gt_pose["gt_depth"],
        "pred_depth_only": results_gt_pose["pred_depth"],
        "pred_pose_only": results_pred_pose["gt_depth"],
        "fully_predicted": results_pred_pose["pred_depth"],
    }

    t0 = time.time()
    oracle_reference = run_oracle_sequence(
        wp, device, mesh_data.vertices, mesh_data.faces, gt_poses,
        stream_raw_depth_frames_from_zip(zpath), cam_rays, vignette_mask, TAUS,
    )
    log(f"oracle.py cross-check done in {time.time()-t0:.1f}s")
    for tau in TAUS:
        assert np.array_equal(configs["oracle"].predicted_observed[tau], oracle_reference.predicted_observed[tau])
    log("cross-check: fusion.py oracle config == oracle.py, confirmed again")

    ignore_set = compute_ignore_set(gt_observed, oracle_reference.ever_evaluable_hit)

    gt_regions = compute_regions(n_faces, adjacency, gt_unobserved, areas)
    small_regions = [r for r in gt_regions if r.size_class == "small"]
    log(f"found {len(small_regions)} small GT regions")
    for i, r in enumerate(small_regions):
        log(f"  small region {i}: {len(r.faces)} faces, area={r.area:.4f}mm^2, diameter={r.diameter:.4f}mm, "
            f"face_ids={sorted(r.faces.tolist())}")

    out = {"small_regions": [], "per_config_per_tau_coverage": {}, "tau25_face_diff": {}}
    for i, r in enumerate(small_regions):
        out["small_regions"].append({
            "region_index": i, "n_faces": len(r.faces), "area_mm2": r.area, "diameter_mm": r.diameter,
            "face_ids": sorted(r.faces.tolist()),
        })

    for config_name, result in configs.items():
        out["per_config_per_tau_coverage"][config_name] = {}
        for tau in TAUS:
            predicted_unobserved = ~result.predicted_observed[tau]
            coverages = [region_coverage_fraction(r, predicted_unobserved, areas) for r in small_regions]
            out["per_config_per_tau_coverage"][config_name][str(tau)] = coverages
            log(f"  [{config_name}, tau={tau}] small region coverage fractions: {coverages}")

    # face-level diff at tau=0.25 between pred_pose_only and fully_predicted, restricted to the small regions
    tau = 0.25
    pu_pred_pose_only = ~configs["pred_pose_only"].predicted_observed[tau]
    pu_fully_predicted = ~configs["fully_predicted"].predicted_observed[tau]
    for i, r in enumerate(small_regions):
        region_faces = r.faces
        only_in_pred_pose_only = region_faces[pu_pred_pose_only[region_faces] & ~pu_fully_predicted[region_faces]]
        only_in_fully_predicted = region_faces[pu_fully_predicted[region_faces] & ~pu_pred_pose_only[region_faces]]
        both_unobserved = region_faces[pu_pred_pose_only[region_faces] & pu_fully_predicted[region_faces]]
        neither = region_faces[~pu_pred_pose_only[region_faces] & ~pu_fully_predicted[region_faces]]
        out["tau25_face_diff"][f"region_{i}"] = {
            "n_faces_total": len(region_faces),
            "predicted_unobserved_in_both": len(both_unobserved),
            "predicted_unobserved_only_in_pred_pose_only": len(only_in_pred_pose_only),
            "predicted_unobserved_only_in_fully_predicted": len(only_in_fully_predicted),
            "predicted_observed_in_both": len(neither),
        }
        log(f"  region {i} @ tau=0.25 face diff: total={len(region_faces)} "
            f"unobserved_in_both={len(both_unobserved)} "
            f"only_pred_pose_only={len(only_in_pred_pose_only)} "
            f"only_fully_predicted={len(only_in_fully_predicted)} "
            f"observed_in_both={len(neither)}")

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(out, f, indent=2)
    log(f"wrote {OUT_DIR / 'summary.json'}")
    log("ALL DONE")


if __name__ == "__main__":
    main()
