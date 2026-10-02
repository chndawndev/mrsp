#!/usr/bin/env python
"""Pipeline-generic evaluation of one sequence, all 4 configurations x 4
taus: scripts/eval_d1_sequence.py (D1 Stage B) with the prediction loading
moved behind a pipeline adapter (scripts/pipeline_adapters.py), plus the
missing-prediction handling of docs/eval_protocol.md "2026-09-29: Missing
predictions, internal crops, confidence maps".

Same procedure as D1 Stage B: scale recovery fit once per sequence and
reused across configurations; two pose variants, each ray-cast once per
frame via src/eval/fusion.py; the oracle configuration cross-checked
bit-for-bit against src/eval/oracle.py on every sequence (also the source
of ever_evaluable_hit for the ignore set); same per-region rows, same
metrics.json schema, same diagnostics a-c.

Missing predictions (applied here, src/eval/ untouched):
  - pred_depth_only: GT-pose ray-cast over every GT frame; a frame with no
    predicted depth gets d_pred unavailable at every pixel, so it marks
    nothing observed.
  - pred_pose_only and fully_predicted: the predicted-pose ray-cast runs
    only over frames that have a predicted pose; within it, a frame with no
    predicted depth marks nothing observed under fully_predicted.
  - Scale recovery: depth scale from frames that have predicted depth,
    over pixels that are GT-valid AND have a prediction; Umeyama over
    frames that have a predicted pose.
  Frames are never filled or interpolated.

For EndoDAC (every frame and pixel predicted) every one of these reduces
to exactly D1 Stage B's computation; pre-flight 1 verifies bit-identity
against results/eval_stage4/summary.json.

CLI:
    scratch/.venv/bin/python scripts/eval_pipeline_sequence.py --pipeline mast3r_slam --sequence NAME
    scratch/.venv/bin/python scripts/eval_pipeline_sequence.py --pipeline mast3r_slam --all
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
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
    detection_at_threshold,
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
from pipeline_adapters import ADAPTERS  # noqa: E402

DATASET_ROOT = Path("/data1_ycao/chua/datasets/C3VDv2/registered_videos")
INTRINSICS_PATH = "/data1_ycao/chua/datasets/C3VDv2/camera_intrinsics.txt"
TAUS = [0.15, 0.25, 0.35, 0.50]
CONFIG_NAMES = ["oracle", "pred_depth_only", "pred_pose_only", "fully_predicted"]
DEFAULT_OUT_ROOT = REPO / "results/pipeline2_eval/per_sequence"
D1_STAGE_B_ROOT = REPO / "results/d1/per_sequence"
D1_IGNORE_SET_DIR = REPO / "results/d1/diagnosis/ignore_set"
EXPECTED_N_SEQUENCES = 169
DETECTION_THRESHOLDS = [0.25, 0.50, 0.75]
MIN_FREE_DISK_GB = 50
DISK_PAUSE_SECONDS = 600


def log(msg: str, tag: str = "") -> None:
    prefix = f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}]"
    if tag:
        prefix += f"[{tag}]"
    print(f"{prefix} {msg}", flush=True)


def discover_sequences() -> list[tuple[str, Path]]:
    seen: dict[str, Path] = {}
    for zpath in sorted(DATASET_ROOT.glob("*.zip")):
        name = zpath.stem
        if name.startswith("c0_"):
            continue
        seen[name] = zpath
    seqs = sorted(seen.items())
    assert len(seqs) == EXPECTED_N_SEQUENCES, (
        f"expected {EXPECTED_N_SEQUENCES} c1/c2 registered_videos sequences, found {len(seqs)}"
    )
    return seqs


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()


def is_complete(out_dir: Path) -> bool:
    manifest_path = out_dir / "MANIFEST.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    if manifest.get("status") != "ok":
        return False
    return all((out_dir / f).exists() for f in ["predicted_observed_packed.bin", "regions.csv", "metrics.json"])


def free_disk_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


def rotation_angle_deg(R: np.ndarray) -> float:
    c = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


# --------------------------------------------------------------------------

def evaluate_sequence(pipeline: str, name: str, out_dir: Path, wp, device: str, gpu_index: int,
                      pred_root: Path | None = None) -> dict:
    t_start = time.time()
    timings = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    zpath = DATASET_ROOT / f"{name}.zip"

    intr = CameraIntrinsics.from_file(INTRINSICS_PATH)
    cols, rows = np.meshgrid(np.arange(intr.width), np.arange(intr.height))
    px_grid = np.stack([cols.ravel(), rows.ravel()], axis=-1).astype(np.float64)
    cam_rays = unproject(px_grid, intr).astype(np.float32)
    vignette_mask = load_vignette_mask(intr.width, intr.height)
    n_rays = len(cam_rays)

    mesh_data = stream_coverage_mesh_from_zip(zpath)
    gt_poses = stream_poses_from_zip(zpath)
    n_frames = len(gt_poses)
    with zipfile.ZipFile(zpath) as zf:
        n_depth = len(depth_members(zf))
    if n_depth != n_frames:
        raise RuntimeError(f"depth frame count {n_depth} != pose count {n_frames}")

    n_faces = len(mesh_data.faces)
    gt_observed = mesh_data.face_observed
    gt_unobserved = ~gt_observed
    areas = face_areas(mesh_data.vertices, mesh_data.faces)
    adjacency = face_adjacency(mesh_data.vertices, mesh_data.faces)
    total_area = float(areas.sum())
    mesh_trimesh = trimesh.Trimesh(vertices=mesh_data.vertices, faces=mesh_data.faces, process=False)

    gt_depth_mm_frames, gt_depth_valid_frames = [], []
    for raw in stream_raw_depth_frames_from_zip(zpath):
        d_mm, valid = depth_to_mm(raw.ravel())
        gt_depth_mm_frames.append(d_mm)
        gt_depth_valid_frames.append(valid)

    adapter = ADAPTERS[pipeline](name, n_frames) if pred_root is None else ADAPTERS[pipeline](name, n_frames, root=pred_root)
    frames_with_depth = list(adapter.frames_with_depth)
    frames_with_pose = list(adapter.frames_with_pose)
    depth_set, pose_set = set(frames_with_depth), set(frames_with_pose)
    missing_depth = [i for i in range(n_frames) if i not in depth_set]
    missing_pose = [i for i in range(n_frames) if i not in pose_set]
    missing_either = [i for i in range(n_frames) if i not in depth_set or i not in pose_set]
    if len(frames_with_pose) < 3:
        raise RuntimeError(f"{name}: only {len(frames_with_pose)} frames with a predicted pose -- Umeyama undefined")
    timings["load_s"] = time.time() - t_start

    # ---------------- scale recovery (section 1), only frames/pixels with predictions ----------------
    t0 = time.time()
    scale_gt, scale_valid, scale_pred = [], [], []
    for i in frames_with_depth:
        d_native, available = adapter.depth_native(i)
        scale_gt.append(gt_depth_mm_frames[i])
        scale_valid.append(gt_depth_valid_frames[i] & available)
        scale_pred.append(d_native)
    depth_scale = compute_depth_scale(scale_gt, scale_valid, scale_pred)
    del scale_gt, scale_valid, scale_pred

    gt_positions = np.array([M[3, :3] for M in gt_poses])
    pred_positions = np.array([adapter.pose(i)[1] for i in frames_with_pose])
    alignment = compute_pose_alignment(pred_positions, gt_positions[frames_with_pose])
    timings["scale_recovery_s"] = time.time() - t0

    # descriptive convention check: rotation residual after alignment
    rot_resid = []
    for i in frames_with_pose:
        R_i, t_i = adapter.pose(i)
        R_w, _ = aligned_predicted_pose(alignment, R_i, t_i)
        rot_resid.append(rotation_angle_deg(R_w.T @ gt_poses[i][:3, :3].T))
    aligned_pos = np.array([aligned_predicted_pose(alignment, *adapter.pose(i))[1] for i in frames_with_pose])
    ate_mm = float(np.sqrt(np.mean(np.sum((aligned_pos - gt_positions[frames_with_pose]) ** 2, axis=1))))

    def gt_pose_provider(i):
        M = gt_poses[i]
        return M[:3, :3].T, M[3, :3]

    def gt_depth_provider(i):
        return gt_depth_mm_frames[i], gt_depth_valid_frames[i]

    no_pred = (np.full(n_rays, np.nan), np.zeros(n_rays, dtype=bool))

    def pred_depth_provider(i):
        out = adapter.depth_native(i)
        if out is None:
            return no_pred
        d_native, available = out
        d_mm = d_native * depth_scale.median
        return d_mm, available

    # pose variant 1: GT pose, every GT frame
    t0 = time.time()
    results_gt_pose = run_pose_variant_sequence(
        wp, device, mesh_data.vertices, mesh_data.faces, n_frames,
        gt_pose_provider, {"gt_depth": gt_depth_provider, "pred_depth": pred_depth_provider},
        cam_rays, vignette_mask, TAUS,
    )
    timings["raycast_gt_pose_s"] = time.time() - t0

    # pose variant 2: aligned predicted pose, only frames that have a predicted pose
    def pred_pose_provider_sub(j):
        i = frames_with_pose[j]
        R_i, t_i = adapter.pose(i)
        return aligned_predicted_pose(alignment, R_i, t_i)

    t0 = time.time()
    results_pred_pose = run_pose_variant_sequence(
        wp, device, mesh_data.vertices, mesh_data.faces, len(frames_with_pose),
        pred_pose_provider_sub,
        {"gt_depth": lambda j: gt_depth_provider(frames_with_pose[j]),
         "pred_depth": lambda j: pred_depth_provider(frames_with_pose[j])},
        cam_rays, vignette_mask, TAUS,
    )
    timings["raycast_pred_pose_s"] = time.time() - t0
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
    timings["raycast_oracle_crosscheck_s"] = time.time() - t0
    for tau in TAUS:
        if not np.array_equal(configs["oracle"].predicted_observed[tau], oracle_reference.predicted_observed[tau]):
            raise RuntimeError(f"{name}: oracle cross-check MISMATCH at tau={tau}")

    ignore_set = compute_ignore_set(gt_observed, oracle_reference.ever_evaluable_hit)
    ignore_set_frac_faces = ignore_set.sum() / n_faces

    # extra recorded checks (not used by any metric): oracle bits and ignore set
    # identical to D1 Stage B's saved outputs for the same sequence
    cross_checks = {}
    n_bytes_per_row = (n_faces + 7) // 8
    stage_b_bits = D1_STAGE_B_ROOT / name / "predicted_observed_packed.bin"
    if stage_b_bits.exists():
        sb = np.fromfile(stage_b_bits, dtype=np.uint8).reshape(len(CONFIG_NAMES), len(TAUS), n_bytes_per_row)
        cross_checks["oracle_bits_n_faces_differing_from_d1_stage_b_by_tau"] = {
            str(tau): int((np.unpackbits(sb[0, ti], bitorder="little")[:n_faces].astype(bool)
                           != configs["oracle"].predicted_observed[tau]).sum())
            for ti, tau in enumerate(TAUS)
        }
    ign_path = D1_IGNORE_SET_DIR / f"{name}.bin"
    if ign_path.exists():
        ign = np.unpackbits(np.fromfile(ign_path, dtype=np.uint8), bitorder="little")[:n_faces].astype(bool)
        cross_checks["ignore_set_n_faces_differing_from_d1_diagnosis"] = int((ign != ignore_set).sum())

    gt_regions = compute_regions(n_faces, adjacency, gt_unobserved, areas)
    gt_regions_headline = [r for r in gt_regions if r.size_class != "below_headline"]

    # ---------------- packed predicted_observed bits ----------------
    packed = np.zeros((len(CONFIG_NAMES), len(TAUS), n_bytes_per_row), dtype=np.uint8)
    for ci, cname in enumerate(CONFIG_NAMES):
        for ti, tau in enumerate(TAUS):
            packed[ci, ti] = np.packbits(configs[cname].predicted_observed[tau], bitorder="little")
    packed.tofile(out_dir / "predicted_observed_packed.bin")

    # ---------------- missing frames and no-depth pixels ----------------
    valid_px = ~vignette_mask
    n_valid_px = int(valid_px.sum())
    no_depth_px_frames_with_depth = 0
    for i in frames_with_depth:
        _, available = adapter.depth_native(i)
        no_depth_px_frames_with_depth += int((valid_px & ~available).sum())
    missing_info = {
        "n_gt_frames": n_frames,
        "n_frames_with_depth": len(frames_with_depth),
        "n_frames_with_pose": len(frames_with_pose),
        "n_frames_with_both": n_frames - len(missing_either),
        "n_frames_missing_depth": len(missing_depth),
        "n_frames_missing_pose": len(missing_pose),
        "n_frames_missing_either": len(missing_either),
        "frames_missing_depth": missing_depth,
        "frames_missing_pose": missing_pose,
        "n_valid_pixels_per_frame": n_valid_px,
        # no-depth fraction over valid (non-vignette) pixels of frames that HAVE a depth map
        "no_depth_frac_of_valid_pixels_frames_with_depth": (
            no_depth_px_frames_with_depth / (n_valid_px * len(frames_with_depth)) if frames_with_depth else float("nan")
        ),
        # same, counting every pixel of a missing-depth frame as no depth, over all GT frames
        "no_depth_frac_of_valid_pixels_all_gt_frames": (
            (no_depth_px_frames_with_depth + n_valid_px * len(missing_depth)) / (n_valid_px * n_frames)
        ),
        "adapter": adapter.descriptives(),
        "pred_root": None if pred_root is None else str(pred_root),
    }

    # ---------------- per config x tau metrics + region rows ----------------
    metrics_out = {
        "sequence": name, "pipeline": pipeline, "n_faces": n_faces, "n_frames": n_frames,
        "depth_scale_median": depth_scale.median, "pose_alignment_s_pose": alignment.s_pose,
        "depth_scale_n_frames_used": len(depth_scale.per_frame_ratios),
        "pose_alignment_n_frames_used": len(frames_with_pose),
        "ate_after_alignment_mm": ate_mm,
        "rotation_residual_after_alignment_deg": {
            "median": float(np.median(rot_resid)), "p95": float(np.percentile(rot_resid, 95)),
            "max": float(np.max(rot_resid)),
        },
        "ignore_set_frac_faces": ignore_set_frac_faces,
        "n_gt_regions_total": len(gt_regions), "n_gt_regions_headline": len(gt_regions_headline),
        "missing_predictions": missing_info,
        "cross_checks": cross_checks,
        "configurations": {},
    }
    region_rows = []

    t0 = time.time()
    for config_name, result in configs.items():
        ray_miss_frac = result.ray_miss_count / result.n_rays_total
        evaluable_frac = result.evaluable_count / result.n_rays_total
        d_pred_unavailable_frac = (
            result.d_pred_unavailable_count / result.evaluable_count if result.evaluable_count else float("nan")
        )
        config_out = {
            "n_frames_raycast": result.n_frames,
            "ray_miss_frac": ray_miss_frac, "evaluable_frac": evaluable_frac,
            "d_pred_unavailable_count": result.d_pred_unavailable_count,
            "d_pred_unavailable_frac_of_evaluable": d_pred_unavailable_frac,
            "by_tau": {},
        }

        for tau in TAUS:
            predicted_observed = result.predicted_observed[tau]
            predicted_unobserved = ~predicted_observed

            predicted_components = compute_regions(n_faces, adjacency, predicted_unobserved & ~ignore_set, areas)
            face_to_component_id = build_face_to_component_id(n_faces, predicted_components)

            sweep = detection_sweep(gt_regions, predicted_unobserved, areas)
            f_reassur = false_reassurance_rate(gt_unobserved, predicted_observed, areas)
            f_alarm = false_alarm_rate(predicted_unobserved, gt_observed, ignore_set, areas)
            area_frac, calib_ratio = area_fraction_and_calibration(predicted_unobserved, gt_unobserved, areas, total_area)

            # raw numerator/denominator areas for corpus pooling (same as D1 Stage B)
            gt_unobserved_area = float(areas[gt_unobserved].sum())
            false_reassurance_numerator_area_mm2 = float(areas[gt_unobserved & predicted_observed].sum())
            scored = predicted_unobserved & ~ignore_set
            false_alarm_denominator_area_mm2 = float(areas[scored].sum())
            false_alarm_numerator_area_mm2 = float(areas[scored & gt_observed].sum())
            pred_area = float(areas[predicted_unobserved].sum())

            # diagnostic (a): calibration ratio with ignore set excluded from both num/denom
            pred_area_excl = float(areas[predicted_unobserved & ~ignore_set].sum())
            gt_area_excl = float(areas[gt_unobserved & ~ignore_set].sum())
            calib_ratio_ignore_excluded = pred_area_excl / gt_area_excl if gt_area_excl > 0 else float("nan")

            recall_50 = region_recall_by_size_class(gt_regions, predicted_unobserved, areas, 0.50)
            per_threshold = {
                t: detection_at_threshold(gt_regions, predicted_unobserved, areas, t) for t in DETECTION_THRESHOLDS
            }

            loc_errors, n_undetected, n_seg_checked, n_seg_intersect = [], 0, 0, 0
            for region_id, r in enumerate(gt_regions):
                is_headline = r.size_class != "below_headline"
                coverage = per_threshold[0.25].coverage[region_id]
                row = {
                    "sequence": name, "config": config_name, "tau": tau, "region_id": region_id,
                    "n_faces": len(r.faces), "size_class": r.size_class, "headline": is_headline,
                    "coverage_fraction": coverage,
                    "detected_at_0.25": per_threshold[0.25].detected[region_id],
                    "detected_at_0.5": per_threshold[0.50].detected[region_id],
                    "detected_at_0.75": per_threshold[0.75].detected[region_id],
                    "localization_error_mm": None, "segment_intersects_mesh": None,
                }
                if is_headline:
                    err = localization_error(r, predicted_components, face_to_component_id, mesh_data.vertices, mesh_data.faces, areas)
                    if err is None:
                        n_undetected += 1
                    else:
                        row["localization_error_mm"] = err
                        loc_errors.append(err)
                        comp = matching_predicted_component(r, predicted_components, face_to_component_id)
                        c_gt = region_centroid(r, mesh_data.vertices, mesh_data.faces, areas)
                        c_pred = region_centroid(comp, mesh_data.vertices, mesh_data.faces, areas)
                        n_seg_checked += 1
                        seg_x = bool(segment_intersects_mesh(mesh_trimesh, c_gt, c_pred))
                        row["segment_intersects_mesh"] = seg_x
                        if seg_x:
                            n_seg_intersect += 1
                region_rows.append(row)

            tau_out = {
                "ignore_set_frac_faces": ignore_set_frac_faces,
                "area_fraction": area_frac, "calibration_ratio": calib_ratio,
                "calibration_ratio_ignore_excluded": calib_ratio_ignore_excluded,
                "detection_sweep": sweep,
                "recall_at_50pct": {cls: v for cls, v in recall_50.items()},
                "false_reassurance_rate": f_reassur, "false_alarm_rate": f_alarm,
                "pred_unobserved_area_mm2": pred_area,
                "total_mesh_area_mm2": total_area,
                "gt_unobserved_area_mm2": gt_unobserved_area,
                "false_reassurance_numerator_area_mm2": false_reassurance_numerator_area_mm2,
                "false_alarm_numerator_area_mm2": false_alarm_numerator_area_mm2,
                "false_alarm_denominator_area_mm2": false_alarm_denominator_area_mm2,
                "pred_unobserved_area_ignore_excluded_mm2": pred_area_excl,
                "gt_unobserved_area_ignore_excluded_mm2": gt_area_excl,
                "tau_reject_count": result.tau_reject_count[tau],
                "n_predicted_unobserved_faces": int(predicted_unobserved.sum()),
                "n_headline_regions_with_localization_error": len(loc_errors),
                "n_headline_regions_undetected": n_undetected,
                "localization_error_median_mm": float(np.median(loc_errors)) if loc_errors else None,
                "localization_error_mean_mm": float(np.mean(loc_errors)) if loc_errors else None,
                "localization_error_iqr_mm": (
                    float(np.percentile(loc_errors, 75) - np.percentile(loc_errors, 25)) if loc_errors else None
                ),
                "segment_intersect_fraction": (n_seg_intersect / n_seg_checked) if n_seg_checked else None,
            }
            config_out["by_tau"][str(tau)] = tau_out

        metrics_out["configurations"][config_name] = config_out
    timings["metrics_s"] = time.time() - t0

    # ---------------- diagnostic (c): face diff, fully_predicted vs pred_pose_only ----------------
    diag_c_rows = []
    for tau in TAUS:
        pu_fp = ~configs["fully_predicted"].predicted_observed[tau]
        pu_pp = ~configs["pred_pose_only"].predicted_observed[tau]
        pu_pd = ~configs["pred_depth_only"].predicted_observed[tau]
        only_fp = pu_fp & ~pu_pp
        n_only_fp = int(only_fp.sum())
        n_also_pd = int((only_fp & pu_pd).sum())
        diag_c_rows.append({
            "sequence": name, "tau": tau,
            "n_only_fully_predicted_unobserved": n_only_fp,
            "n_also_pred_depth_only_unobserved": n_also_pd,
            "n_not_pred_depth_only_unobserved": n_only_fp - n_also_pd,
        })
    metrics_out["diagnostic_c_face_diff"] = diag_c_rows
    metrics_out["timings_s"] = timings

    pd.DataFrame(region_rows).to_csv(out_dir / "regions.csv", index=False)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    manifest = {
        "sequence": name, "pipeline": pipeline, "n_frames": n_frames, "n_faces": n_faces,
        "n_gt_regions": len(gt_regions), "n_gt_regions_headline": len(gt_regions_headline),
        "git_commit": git_head(), "gpu_index": gpu_index,
        "runtime_seconds_total": time.time() - t_start, "status": "ok", "error": None,
    }
    with open(out_dir / "MANIFEST.json", "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def run_one(pipeline: str, name: str, out_dir: Path, wp, device, gpu_index: int, pred_root: Path | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        manifest = evaluate_sequence(pipeline, name, out_dir, wp, device, gpu_index, pred_root)
        log(f"{name}: DONE in {manifest['runtime_seconds_total']:.1f}s", tag=f"gpu{gpu_index}")
        return manifest
    except Exception as exc:
        err = traceback.format_exc()
        log(f"{name}: FAILED\n{err}", tag=f"gpu{gpu_index}")
        manifest = {
            "sequence": name, "pipeline": pipeline, "status": "failed", "error": err,
            "git_commit": git_head(), "gpu_index": gpu_index,
            "runtime_seconds_total": time.time() - t0,
        }
        is_cuda = "cuda" in err.lower() or "out of memory" in err.lower()
        if is_cuda:
            # CLAUDE.md: on CUDA error log timestamp, full nvidia-smi, peak memory, exception; exit non-zero
            log("CUDA error -- nvidia-smi at failure time:\n"
                + subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout, tag=f"gpu{gpu_index}")
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / "MANIFEST.json", "w") as f:
                json.dump(manifest, f, indent=2)
        except OSError:
            log(f"{name}: could not write failure MANIFEST.json either -- logging only", tag=f"gpu{gpu_index}")
        if is_cuda:
            raise SystemExit(f"CUDA error on {name}: {exc}")
        return manifest


def select_gpu() -> int:
    log("=== GPU availability check (scripts/gpu_status.py) ===")
    table = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
    log(table)
    gpus = get_gpu_stats()
    freest = max(gpus, key=lambda g: g["memory_free_mib"])
    log(f"Selected GPU: index {freest['index']} ({freest['name']}), "
        f"{freest['memory_free_mib']/1000:.1f} GB free, {freest['utilization_pct']}% utilized")
    if freest["memory_free_mib"] < 8000:
        raise RuntimeError(f"no GPU with >= 8GB free (best: {freest['memory_free_mib']}MiB)")
    return freest["index"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", required=True, choices=sorted(ADAPTERS))
    parser.add_argument("--sequence", action="append", help="repeatable; omit with --all")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0, help="with --all: this process's shard")
    parser.add_argument("--shard-total", type=int, default=1, help="with --all: number of shards (same GPU)")
    parser.add_argument("--out-root", default=None,
                        help="default: results/pipeline2_eval/per_sequence/<pipeline>")
    parser.add_argument("--pred-root", default=None,
                        help="read the pipeline's predictions from this directory instead of its primary run "
                             "(variability runs); requires --out-root")
    args = parser.parse_args()
    if args.pred_root and not args.out_root:
        raise SystemExit("--pred-root requires --out-root: a non-primary run never writes into the primary results")
    pred_root = Path(args.pred_root).resolve() if args.pred_root else None

    out_root = Path(args.out_root).resolve() if args.out_root else DEFAULT_OUT_ROOT / args.pipeline
    out_root.mkdir(parents=True, exist_ok=True)

    gpu_index = int(os.environ.get("CUDA_VISIBLE_DEVICES", "-1"))
    if gpu_index == -1:
        gpu_index = select_gpu()
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    else:
        log("CUDA_VISIBLE_DEVICES preset; nvidia-smi at start:\n"
            + subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)

    import warp as wp
    wp.init()
    device = "cuda:0"
    log(f"warp version {wp.config.version}, device {device}, CUDA_VISIBLE_DEVICES={gpu_index}, "
        f"pipeline={args.pipeline}, out_root={out_root}", tag=f"gpu{gpu_index}")

    sequences = [n for n, _ in discover_sequences()]
    if args.all:
        todo = [n for i, n in enumerate(sequences) if i % args.shard_total == args.shard_index]
    elif args.sequence:
        unknown = [s for s in args.sequence if s not in sequences]
        if unknown:
            raise SystemExit(f"not among the 169 registered sequences: {unknown}")
        todo = [n for i, n in enumerate(args.sequence) if i % args.shard_total == args.shard_index]
    else:
        raise SystemExit("provide --sequence NAME (repeatable) or --all")

    t_run = time.time()
    n_done_here, n_failed = 0, 0
    for k, name in enumerate(todo):
        out_dir = out_root / name
        if is_complete(out_dir):
            log(f"[{k+1}/{len(todo)}] {name}: already complete, skipping", tag=f"gpu{gpu_index}")
            continue
        while free_disk_gb(REPO) < MIN_FREE_DISK_GB:
            log(f"free disk {free_disk_gb(REPO):.1f}GB < {MIN_FREE_DISK_GB}GB -- pausing {DISK_PAUSE_SECONDS}s",
                tag=f"gpu{gpu_index}")
            time.sleep(DISK_PAUSE_SECONDS)
        log(f"[{k+1}/{len(todo)}] {name}: start (free disk {free_disk_gb(REPO):.0f}GB)", tag=f"gpu{gpu_index}")
        try:
            manifest = run_one(args.pipeline, name, out_dir, wp, device, gpu_index, pred_root)
            n_done_here += 1
            n_failed += manifest["status"] != "ok"
        except SystemExit:
            raise
        except Exception:
            log(f"{name}: run_one itself raised -- logging and continuing\n{traceback.format_exc()}",
                tag=f"gpu{gpu_index}")
            n_failed += 1
        elapsed = time.time() - t_run
        remaining = sum(1 for n in todo[k + 1:] if not is_complete(out_root / n))
        eta = elapsed / max(n_done_here, 1) * remaining
        log(f"progress: {k+1}/{len(todo)} visited, {n_done_here} run here, {n_failed} failed, "
            f"elapsed {elapsed/3600:.2f}h, ETA {eta/3600:.2f}h ({remaining} remaining)", tag=f"gpu{gpu_index}")
    log(f"all assigned sequences visited; failed: {n_failed}", tag=f"gpu{gpu_index}")
    if args.sequence and not args.all and n_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
