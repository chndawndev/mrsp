#!/usr/bin/env python
"""Stage 4 driver: the predicted path, one sequence (c1_cecum_t1_v1), all
four configurations, docs/eval_protocol.md sections 1 (scale recovery), 4
(post-hoc substitution), and 5 (outputs).

Scale factors (depth median-of-ratios, pose Umeyama) are fit ONCE per
sequence and reused across every configuration that needs them (section
4's explicit requirement -- not refit per configuration).

Ray-casting: two pose variants (GT pose; aligned predicted pose), each
cast once per frame and shared across both depth sources at that pose
variant (section 7's design point) -- scripts/eval_stage4.py therefore
calls src/eval/fusion.py's run_pose_variant_sequence exactly twice.

Cross-check before trusting anything: the "oracle" configuration (GT
depth + GT pose), produced here through the NEW, general fusion.py code
path, is compared bit-for-bit against src/eval/oracle.py's own (locked,
Stage-2-validated) run_oracle_sequence on the SAME real sequence -- not
just the synthetic unit test in tests/eval/test_fusion.py.

Reports the full table: every metric, every configuration, every tau,
area fraction AND ignore-set fraction next to every recall number. Does
NOT compute or comment on D1 pass/fail -- one sequence is not D1, whose
trigger is all 169 (docs/success_criteria.md section 2).
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path

import numpy as np
import trimesh

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
from eval.region_metrics import (  # noqa: E402
    area_fraction_and_calibration,
    build_face_to_component_id,
    detection_sweep,
    false_alarm_rate,
    false_reassurance_rate,
    localization_error,
    matching_predicted_component,
    region_centroid,
    region_recall_by_size_class,
    segment_intersects_mesh,
)
from eval.scale_recovery import aligned_predicted_pose, compute_depth_scale, compute_pose_alignment  # noqa: E402
from gpu_status import get_gpu_stats  # noqa: E402

SEQ_NAME = "c1_cecum_t1_v1"
DATASET_ROOT = Path("/data1_ycao/chua/datasets/C3VDv2/registered_videos")
INTRINSICS_PATH = "/data1_ycao/chua/datasets/C3VDv2/camera_intrinsics.txt"
PRED_DIR = REPO / "results/pipelines/endodac_full_run" / SEQ_NAME
SAVED_SCALE_PATH = REPO / "results/pipelines/endodac_scale_analysis.json"
TAUS = [0.15, 0.25, 0.35, 0.50]
OUT_DIR = REPO / "results/eval_stage4"
LOG_PATH = REPO / "logs/eval_stage4.log"

OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def select_gpu() -> int:
    log("=== GPU availability check (scripts/gpu_status.py) ===")
    table = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
    log(table)
    gpus = get_gpu_stats()
    freest = max(gpus, key=lambda g: g["memory_free_mib"])
    log(
        f"Selected GPU: index {freest['index']} ({freest['name']}), "
        f"{freest['memory_free_mib']/1000:.1f} GB free, {freest['utilization_pct']}% utilized"
    )
    if freest["memory_free_mib"] < 8000:
        raise RuntimeError(f"no GPU with >= 8GB free (best: {freest['memory_free_mib']}MiB) -- aborting")
    return freest["index"]


def run():
    log(f"=== Stage 4: predicted path, {SEQ_NAME}, all 4 configurations ===")
    gpu_index = select_gpu()
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    log(f"CUDA_VISIBLE_DEVICES set to '{gpu_index}' explicitly")

    import warp as wp

    wp.init()
    device = "cuda:0"
    log(f"warp version {wp.config.version}, device {device}")

    intr = CameraIntrinsics.from_file(INTRINSICS_PATH)
    cols, rows = np.meshgrid(np.arange(intr.width), np.arange(intr.height))
    px_grid = np.stack([cols.ravel(), rows.ravel()], axis=-1).astype(np.float64)
    cam_rays = unproject(px_grid, intr).astype(np.float32)
    vignette_mask = load_vignette_mask(intr.width, intr.height)

    zpath = DATASET_ROOT / f"{SEQ_NAME}.zip"
    t0 = time.time()
    mesh_data = stream_coverage_mesh_from_zip(zpath)
    gt_poses = stream_poses_from_zip(zpath)  # (n_frames, 4, 4) raw
    n_frames = len(gt_poses)
    with zipfile.ZipFile(zpath) as zf:
        n_depth = len(depth_members(zf))
    if n_depth != n_frames:
        raise RuntimeError(f"depth frame count {n_depth} != pose count {n_frames}")
    log(f"streamed mesh ({len(mesh_data.faces)} faces) + poses ({n_frames} frames) in {time.time()-t0:.1f}s")

    n_faces = len(mesh_data.faces)
    gt_observed = mesh_data.face_observed
    gt_unobserved = ~gt_observed
    areas = face_areas(mesh_data.vertices, mesh_data.faces)
    adjacency = face_adjacency(mesh_data.vertices, mesh_data.faces)
    total_area = float(areas.sum())
    mesh_trimesh = trimesh.Trimesh(vertices=mesh_data.vertices, faces=mesh_data.faces, process=False)

    # ---------------- load all GT depth + predicted depth/pose ----------------
    t0 = time.time()
    gt_depth_mm_frames = []
    gt_depth_valid_frames = []
    for raw in stream_raw_depth_frames_from_zip(zpath):
        d_mm, valid = depth_to_mm(raw.ravel())
        gt_depth_mm_frames.append(d_mm)
        gt_depth_valid_frames.append(valid)
    log(f"loaded {len(gt_depth_mm_frames)} GT depth frames in {time.time()-t0:.1f}s")

    t0 = time.time()
    pred_depth_native_frames = [
        np.load(PRED_DIR / "depth" / f"{i:04d}_pred_depth.npy").ravel().astype(np.float64) for i in range(n_frames)
    ]
    pred_poses = np.load(PRED_DIR / "poses_pred.npy")  # (n_frames, 4, 4) already camera-to-world
    log(f"loaded predicted depth + poses in {time.time()-t0:.1f}s")

    # ---------------- scale recovery (section 1) ----------------
    depth_scale = compute_depth_scale(gt_depth_mm_frames, gt_depth_valid_frames, pred_depth_native_frames)
    log(f"depth scale: median={depth_scale.median!r}")

    gt_positions = np.array([M[3, :3] for M in gt_poses])
    pred_positions = pred_poses[:, :3, 3]
    alignment = compute_pose_alignment(pred_positions, gt_positions)
    log(f"pose alignment: s_pose={alignment.s_pose!r}")

    if SAVED_SCALE_PATH.exists():
        saved = json.loads(SAVED_SCALE_PATH.read_text())
        saved_depth_median = saved["depth_scale"]["median"]
        saved_s_pose = saved["pose_scale"]["s_pose"]
        log(
            f"cross-check vs already-measured values ({SAVED_SCALE_PATH}): "
            f"depth_scale.median {depth_scale.median!r} vs saved {saved_depth_median!r} "
            f"(diff {abs(depth_scale.median - saved_depth_median):.6e}); "
            f"s_pose {alignment.s_pose!r} vs saved {saved_s_pose!r} "
            f"(diff {abs(alignment.s_pose - saved_s_pose):.6e})"
        )
    else:
        log(f"no saved scale reference found at {SAVED_SCALE_PATH} -- skipping cross-check")

    # ---------------- providers ----------------
    def gt_pose_provider(i):
        M = gt_poses[i]
        return M[:3, :3].T, M[3, :3]

    def pred_pose_provider(i):
        R_i = pred_poses[i, :3, :3]
        t_i = pred_poses[i, :3, 3]
        return aligned_predicted_pose(alignment, R_i, t_i)

    def gt_depth_provider(i):
        return gt_depth_mm_frames[i], gt_depth_valid_frames[i]

    def pred_depth_provider(i):
        d_mm = pred_depth_native_frames[i] * depth_scale.median
        valid = np.ones(len(d_mm), dtype=bool)
        return d_mm, valid

    # ---------------- pose variant 1: GT pose (-> oracle, pred_depth_only) ----------------
    t0 = time.time()
    results_gt_pose = run_pose_variant_sequence(
        wp, device, mesh_data.vertices, mesh_data.faces, n_frames,
        gt_pose_provider, {"gt_depth": gt_depth_provider, "pred_depth": pred_depth_provider},
        cam_rays, vignette_mask, TAUS,
    )
    log(f"pose variant GT: ray-cast + fusion over {n_frames} frames, 2 depth sources, in {time.time()-t0:.1f}s")

    # ---------------- pose variant 2: aligned predicted pose (-> pred_pose_only, fully_predicted) ----------------
    t0 = time.time()
    results_pred_pose = run_pose_variant_sequence(
        wp, device, mesh_data.vertices, mesh_data.faces, n_frames,
        pred_pose_provider, {"gt_depth": gt_depth_provider, "pred_depth": pred_depth_provider},
        cam_rays, vignette_mask, TAUS,
    )
    log(f"pose variant pred: ray-cast + fusion over {n_frames} frames, 2 depth sources, in {time.time()-t0:.1f}s")

    configs = {
        "oracle": results_gt_pose["gt_depth"],
        "pred_depth_only": results_gt_pose["pred_depth"],
        "pred_pose_only": results_pred_pose["gt_depth"],
        "fully_predicted": results_pred_pose["pred_depth"],
    }

    # ---------------- cross-check: oracle config vs src/eval/oracle.py on real data ----------------
    t0 = time.time()
    oracle_reference = run_oracle_sequence(
        wp, device, mesh_data.vertices, mesh_data.faces, gt_poses,
        stream_raw_depth_frames_from_zip(zpath), cam_rays, vignette_mask, TAUS,
    )
    log(f"cross-check: ran src/eval/oracle.py's run_oracle_sequence fresh in {time.time()-t0:.1f}s")

    oracle_matches = True
    for tau in TAUS:
        same = np.array_equal(configs["oracle"].predicted_observed[tau], oracle_reference.predicted_observed[tau])
        oracle_matches = oracle_matches and same
        if not same:
            log(f"  *** MISMATCH at tau={tau}: fusion.py's oracle config != oracle.py's run_oracle_sequence ***")
    oracle_matches = (
        oracle_matches
        and configs["oracle"].ray_miss_count == oracle_reference.ray_miss_count
        and configs["oracle"].evaluable_count == oracle_reference.evaluable_count
    )
    log(f"cross-check result: fusion.py's oracle config bit-for-bit identical to oracle.py = {oracle_matches}")
    if not oracle_matches:
        raise RuntimeError("Stage 4's oracle configuration does not match src/eval/oracle.py -- stopping, not trusting any config's numbers")

    ignore_set = compute_ignore_set(gt_observed, oracle_reference.ever_evaluable_hit)
    ignore_set_frac_faces = ignore_set.sum() / n_faces
    log(f"ignore_set: {int(ignore_set.sum())}/{n_faces} faces ({ignore_set_frac_faces*100:.4f}%)")

    gt_regions = compute_regions(n_faces, adjacency, gt_unobserved, areas)
    gt_regions_headline = [r for r in gt_regions if r.size_class != "below_headline"]
    log(f"GT regions: {len(gt_regions)} total, {len(gt_regions_headline)} headline (d>=5mm)")

    # ---------------- per-configuration, per-tau metrics ----------------
    all_results = {
        "sequence": SEQ_NAME,
        "n_faces": n_faces,
        "n_frames": n_frames,
        "depth_scale_median": depth_scale.median,
        "pose_alignment_s_pose": alignment.s_pose,
        "ignore_set_frac_faces": ignore_set_frac_faces,
        "n_gt_regions_total": len(gt_regions),
        "n_gt_regions_headline": len(gt_regions_headline),
        "configurations": {},
    }

    for config_name, result in configs.items():
        log(f"\n=== configuration: {config_name} ===")
        ray_miss_frac = result.ray_miss_count / result.n_rays_total
        evaluable_frac = result.evaluable_count / result.n_rays_total
        log(f"  ray_miss_frac={ray_miss_frac*100:.5f}% evaluable_frac={evaluable_frac*100:.5f}% "
            f"d_pred_unavailable={result.d_pred_unavailable_count}")

        config_out = {"ray_miss_frac": ray_miss_frac, "evaluable_frac": evaluable_frac,
                      "d_pred_unavailable_count": result.d_pred_unavailable_count, "by_tau": {}}

        for tau in TAUS:
            predicted_observed = result.predicted_observed[tau]
            predicted_unobserved = ~predicted_observed

            predicted_components = compute_regions(n_faces, adjacency, predicted_unobserved & ~ignore_set, areas)
            face_to_component_id = build_face_to_component_id(n_faces, predicted_components)

            sweep = detection_sweep(gt_regions, predicted_unobserved, areas)
            f_reassur = false_reassurance_rate(gt_unobserved, predicted_observed, areas)
            f_alarm = false_alarm_rate(predicted_unobserved, gt_observed, ignore_set, areas)
            area_frac, calib_ratio = area_fraction_and_calibration(predicted_unobserved, gt_unobserved, areas, total_area)

            loc_errors = []
            n_undetected = 0
            n_seg_checked = 0
            n_seg_intersect = 0
            for r in gt_regions_headline:
                err = localization_error(r, predicted_components, face_to_component_id, mesh_data.vertices, mesh_data.faces, areas)
                if err is None:
                    n_undetected += 1
                else:
                    loc_errors.append(err)
                    comp = matching_predicted_component(r, predicted_components, face_to_component_id)
                    c_gt = region_centroid(r, mesh_data.vertices, mesh_data.faces, areas)
                    c_pred = region_centroid(comp, mesh_data.vertices, mesh_data.faces, areas)
                    n_seg_checked += 1
                    if segment_intersects_mesh(mesh_trimesh, c_gt, c_pred):
                        n_seg_intersect += 1

            recall_50 = region_recall_by_size_class(gt_regions, predicted_unobserved, areas, 0.50)

            tau_out = {
                "ignore_set_frac_faces": ignore_set_frac_faces,  # repeated alongside every recall number, per request
                "area_fraction": area_frac,
                "calibration_ratio": calib_ratio,
                "detection_sweep": sweep,
                "recall_at_50pct": {cls: v for cls, v in recall_50.items()},
                "false_reassurance_rate": f_reassur,
                "false_alarm_rate": f_alarm,
                "tau_reject_count": result.tau_reject_count[tau],
                "n_predicted_unobserved_faces": int(predicted_unobserved.sum()),
                "n_headline_regions_with_localization_error": len(loc_errors),
                "n_headline_regions_undetected": n_undetected,
                "localization_error_median_mm": float(np.median(loc_errors)) if loc_errors else None,
                "localization_error_mean_mm": float(np.mean(loc_errors)) if loc_errors else None,
                "segment_intersect_fraction": (n_seg_intersect / n_seg_checked) if n_seg_checked else None,
            }
            config_out["by_tau"][str(tau)] = tau_out

            loc_med = tau_out["localization_error_median_mm"]
            log(
                f"  [tau={tau}] ignore_set_frac={ignore_set_frac_faces*100:.4f}% "
                f"area_fraction={area_frac:.6f} calib_ratio={calib_ratio:.6f} | "
                f"recall@50%: small={recall_50['small']['recall']:.4f}({recall_50['small']['n_regions']}) "
                f"medium={recall_50['medium']['recall']:.4f}({recall_50['medium']['n_regions']}) "
                f"large={recall_50['large']['recall']:.4f}({recall_50['large']['n_regions']}) | "
                f"false_reassur={f_reassur:.6f} false_alarm={f_alarm:.6f} | "
                f"loc_err_median={loc_med if loc_med is None else f'{loc_med:.4f}mm'} "
                f"undetected={n_undetected} | tau_reject={result.tau_reject_count[tau]}"
            )

        all_results["configurations"][config_name] = config_out

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2)
    log(f"\nwrote {OUT_DIR / 'summary.json'}")
    log("ALL DONE (no D1 pass/fail commentary -- one sequence is not D1, per instructions)")


def main():
    try:
        run()
    except Exception:
        log(f"*** FATAL ERROR at {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} ***")
        nvidia_smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
        log(f"nvidia-smi at failure time:\n{nvidia_smi}")
        log(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
