#!/usr/bin/env python
"""Stage 3 driver: region-level metrics on the oracle configuration,
docs/success_criteria.md section 1 + docs/eval_protocol.md section 5, on
the same 3 sequences validated in Stage 2 (c1_cecum_t1_v1, c2_rectum_t1_v1,
c1_descending_t1_v1).

For each sequence, at the primary tau=0.25 (docs/eval_protocol.md section 2):
region recall by size class + the 25/50/75% detection sweep, false
reassurance rate, false alarm rate (ignore-set-excluded, section 6),
localization error, predicted-unobserved area fraction and calibration
ratio. The ignore-set fraction (measured per sequence -- 2.87% cecum,
3.28% rectum, 1.35% descending, narrow gt_observed-restricted definition,
docs/eval_protocol.md section 6) is reported alongside every metric per
sequence, per request -- metrics are not comparable across sequences
without it.

Also reports the localization-error diagnostic (docs/eval_protocol.md
section 5, "Localization error: Euclidean, decided" -- 2026-09-23,
approved by Chen): for each matched GT-region/predicted-component pair,
whether the straight segment between their centroids tunnels through the
mesh (src/eval/region_metrics.py::segment_intersects_mesh), reported as a
fraction of matched pairs per sequence, NOT used to adjust the metric.

Validates (Stage 3's explicit ask): with the ignore set applied, false
alarm rate should be ~0 and predicted-unobserved components should match
GT regions -- checked at EVERY tau in the sweep, not just primary, both by
component COUNT (the level docs/eval_protocol.md section 6's own evidence
table reports) and by exact face-set equality (a stricter secondary
check) -- reports whether each holds, and diagnoses any gap before
touching anything, per docs/oracle_check.md's precedent.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import traceback
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
from gt.depth import depth_members, stream_raw_depth_frames_from_zip  # noqa: E402
from eval.evaluable import load_vignette_mask  # noqa: E402
from eval.ignore_set import compute_ignore_set  # noqa: E402
from eval.oracle import run_oracle_sequence  # noqa: E402
from eval.regions import compute_regions  # noqa: E402
from eval.region_metrics import (  # noqa: E402
    DETECTION_PRIMARY,
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
from gpu_status import get_gpu_stats  # noqa: E402

SEQUENCES = ["c1_cecum_t1_v1", "c2_rectum_t1_v1", "c1_descending_t1_v1"]
DATASET_ROOT = Path("/data1_ycao/chua/datasets/C3VDv2/registered_videos")
INTRINSICS_PATH = "/data1_ycao/chua/datasets/C3VDv2/camera_intrinsics.txt"
TAUS = [0.15, 0.25, 0.35, 0.50]
PRIMARY_TAU = 0.25
OUT_DIR = REPO / "results/eval_region_metrics_stage3"
LOG_PATH = REPO / "logs/eval_region_metrics_stage3.log"

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


def facesets(regions) -> set:
    return {frozenset(r.faces.tolist()) for r in regions}


def run():
    log("=== Stage 3: region-level metrics, oracle configuration, 3 sequences ===")
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
    log(f"vignette mask: {int(vignette_mask.sum())} px ({vignette_mask.mean()*100:.4f}%)")

    all_results = {}
    for seq in SEQUENCES:
        log(f"\n=== {seq} ===")
        zpath = DATASET_ROOT / f"{seq}.zip"

        t0 = time.time()
        mesh_data = stream_coverage_mesh_from_zip(zpath)
        poses = stream_poses_from_zip(zpath)
        import zipfile

        with zipfile.ZipFile(zpath) as zf:
            n_depth = len(depth_members(zf))
        if n_depth != len(poses):
            raise RuntimeError(f"{seq}: depth frame count {n_depth} != pose count {len(poses)}")
        log(f"  streamed mesh ({len(mesh_data.faces)} faces) + poses ({len(poses)} frames) in {time.time()-t0:.1f}s")

        t0 = time.time()
        result = run_oracle_sequence(
            wp, device,
            mesh_data.vertices, mesh_data.faces,
            poses, stream_raw_depth_frames_from_zip(zpath), cam_rays, vignette_mask, TAUS,
        )
        log(f"  oracle run over {result.n_frames} frames in {time.time()-t0:.1f}s")

        n_faces = result.n_faces
        gt_observed = mesh_data.face_observed
        gt_unobserved = ~gt_observed
        t0 = time.time()
        areas = face_areas(mesh_data.vertices, mesh_data.faces)
        log(f"  face_areas: {time.time()-t0:.1f}s")
        t0 = time.time()
        adjacency = face_adjacency(mesh_data.vertices, mesh_data.faces)
        log(f"  face_adjacency: {time.time()-t0:.1f}s ({len(adjacency)} edge pairs)")
        total_area = float(areas.sum())

        ignore_set = compute_ignore_set(gt_observed, result.ever_evaluable_hit)
        ignore_set_frac_faces = ignore_set.sum() / n_faces
        ignore_set_frac_area = float(areas[ignore_set].sum() / total_area)
        log(f"  ignore_set: {int(ignore_set.sum())}/{n_faces} faces ({ignore_set_frac_faces*100:.4f}%), "
            f"{ignore_set_frac_area*100:.4f}% of mesh area")

        t0 = time.time()
        gt_regions = compute_regions(n_faces, adjacency, gt_unobserved, areas)
        gt_regions_headline = [r for r in gt_regions if r.size_class != "below_headline"]
        log(f"  GT regions: {len(gt_regions)} total, {len(gt_regions_headline)} headline (d>=5mm) "
            f"in {time.time()-t0:.1f}s")

        seq_out = {
            "n_faces": n_faces,
            "ignore_set_frac_faces": ignore_set_frac_faces,
            "ignore_set_frac_area": ignore_set_frac_area,
            "n_gt_regions_total": len(gt_regions),
            "n_gt_regions_headline": len(gt_regions_headline),
            "primary_tau": {},
            "ignore_set_validation_by_tau": {},
        }

        # ---------------- primary-tau region metrics ----------------
        predicted_observed = result.predicted_observed[PRIMARY_TAU]
        predicted_unobserved = ~predicted_observed
        t0 = time.time()
        predicted_components = compute_regions(n_faces, adjacency, predicted_unobserved & ~ignore_set, areas)
        face_to_component_id = build_face_to_component_id(n_faces, predicted_components)
        log(f"  predicted components (tau={PRIMARY_TAU}): {len(predicted_components)} in {time.time()-t0:.1f}s")

        sweep = detection_sweep(gt_regions, predicted_unobserved, areas)
        f_reassur = false_reassurance_rate(gt_unobserved, predicted_observed, areas)
        f_alarm = false_alarm_rate(predicted_unobserved, gt_observed, ignore_set, areas)
        f_alarm_no_ignore = false_alarm_rate(predicted_unobserved, gt_observed, np.zeros(n_faces, dtype=bool), areas)
        area_frac, calib_ratio = area_fraction_and_calibration(predicted_unobserved, gt_unobserved, areas, total_area)

        t0 = time.time()
        mesh_trimesh = trimesh.Trimesh(vertices=mesh_data.vertices, faces=mesh_data.faces, process=False)
        loc_errors = []
        n_headline_undetected = 0
        n_segment_checked = 0
        n_segment_intersects = 0
        for r in gt_regions_headline:
            err = localization_error(r, predicted_components, face_to_component_id, mesh_data.vertices, mesh_data.faces, areas)
            if err is None:
                n_headline_undetected += 1
            else:
                loc_errors.append(err)
                comp = matching_predicted_component(r, predicted_components, face_to_component_id)
                c_gt = region_centroid(r, mesh_data.vertices, mesh_data.faces, areas)
                c_pred = region_centroid(comp, mesh_data.vertices, mesh_data.faces, areas)
                n_segment_checked += 1
                if segment_intersects_mesh(mesh_trimesh, c_gt, c_pred):
                    n_segment_intersects += 1
        log(f"  localization_error over {len(gt_regions_headline)} headline regions: {time.time()-t0:.1f}s")

        segment_intersect_frac = n_segment_intersects / n_segment_checked if n_segment_checked else float("nan")

        seq_out["primary_tau"] = {
            "tau": PRIMARY_TAU,
            "ignore_set_frac_faces": ignore_set_frac_faces,  # repeated alongside, per request
            "detection_sweep": sweep,
            "false_reassurance_rate": f_reassur,
            "false_alarm_rate_with_ignore_set": f_alarm,
            "false_alarm_rate_without_ignore_set": f_alarm_no_ignore,
            "area_fraction": area_frac,
            "calibration_ratio": calib_ratio,
            "n_headline_regions_with_localization_error": len(loc_errors),
            "n_headline_regions_undetected": n_headline_undetected,
            "localization_error_median_mm": float(np.median(loc_errors)) if loc_errors else None,
            "localization_error_mean_mm": float(np.mean(loc_errors)) if loc_errors else None,
            "localization_error_max_mm": float(np.max(loc_errors)) if loc_errors else None,
            "n_segment_pairs_checked": n_segment_checked,
            "n_segment_pairs_intersect_mesh": n_segment_intersects,
            "segment_intersect_fraction": segment_intersect_frac,
        }

        recall_50 = region_recall_by_size_class(gt_regions, predicted_unobserved, areas, DETECTION_PRIMARY)
        log(f"  [tau={PRIMARY_TAU}, ignore_set_frac_faces={ignore_set_frac_faces*100:.4f}%] "
            f"recall@50%: small={recall_50['small']['recall']:.4f} ({recall_50['small']['n_regions']}) "
            f"medium={recall_50['medium']['recall']:.4f} ({recall_50['medium']['n_regions']}) "
            f"large={recall_50['large']['recall']:.4f} ({recall_50['large']['n_regions']})")
        log(f"  [tau={PRIMARY_TAU}] false_reassurance={f_reassur:.6f} "
            f"false_alarm(with ignore)={f_alarm:.6f} false_alarm(no ignore)={f_alarm_no_ignore:.6f}")
        log(f"  [tau={PRIMARY_TAU}] area_fraction={area_frac:.6f} calibration_ratio={calib_ratio:.6f}")
        loc_med = seq_out["primary_tau"]["localization_error_median_mm"]
        log(f"  [tau={PRIMARY_TAU}] localization_error: n={len(loc_errors)} undetected={n_headline_undetected} "
            f"median={loc_med if loc_med is None else f'{loc_med:.4f}mm'}")
        log(f"  [tau={PRIMARY_TAU}] segment-intersects-mesh diagnostic: {n_segment_intersects}/{n_segment_checked} "
            f"matched pairs ({segment_intersect_frac*100:.2f}%) -- straight centroid line tunnels through the mesh")

        # ---------------- ignore-set validation, every tau ----------------
        t0 = time.time()
        gt_facesets = facesets(gt_regions)
        for tau in TAUS:
            t_tau = time.time()
            pu = ~result.predicted_observed[tau]
            fa_with = false_alarm_rate(pu, gt_observed, ignore_set, areas)
            fa_without = false_alarm_rate(pu, gt_observed, np.zeros(n_faces, dtype=bool), areas)
            pred_comps_tau = compute_regions(n_faces, adjacency, pu & ~ignore_set, areas)
            pred_comps_tau_headline = [r for r in pred_comps_tau if r.size_class != "below_headline"]
            log(f"  [ignore-set validation, tau={tau}] {len(pred_comps_tau)} predicted components "
                f"computed in {time.time()-t_tau:.1f}s")

            count_match_all = len(pred_comps_tau) == len(gt_regions)
            count_match_headline = len(pred_comps_tau_headline) == len(gt_regions_headline)
            pred_facesets = facesets(pred_comps_tau)
            exact_match = pred_facesets == gt_facesets
            n_extra_predicted = len(pred_facesets - gt_facesets)
            n_missing_predicted = len(gt_facesets - pred_facesets)

            entry = {
                "false_alarm_rate_with_ignore_set": fa_with,
                "false_alarm_rate_without_ignore_set": fa_without,
                "n_predicted_components_all": len(pred_comps_tau),
                "n_gt_regions_all": len(gt_regions),
                "component_count_match_all": count_match_all,
                "n_predicted_components_headline": len(pred_comps_tau_headline),
                "n_gt_regions_headline": len(gt_regions_headline),
                "component_count_match_headline": count_match_headline,
                "exact_faceset_match_all": exact_match,
                "n_predicted_facesets_not_in_gt": n_extra_predicted,
                "n_gt_facesets_not_in_predicted": n_missing_predicted,
            }
            seq_out["ignore_set_validation_by_tau"][str(tau)] = entry

            log(
                f"  [ignore-set validation, tau={tau}] false_alarm(with ignore)={fa_with:.6f} "
                f"(no ignore)={fa_without:.6f} | components: all {len(pred_comps_tau)}/{len(gt_regions)} "
                f"(match={count_match_all}) headline {len(pred_comps_tau_headline)}/{len(gt_regions_headline)} "
                f"(match={count_match_headline}) | exact_faceset_match={exact_match} "
                f"(extra={n_extra_predicted}, missing={n_missing_predicted})"
            )

        all_results[seq] = seq_out

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2)
    log(f"\nwrote {OUT_DIR / 'summary.json'}")
    log("ALL DONE")


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
