#!/usr/bin/env python
"""Viewer data export, Stage 1 of 2: everything a 3D results viewer needs
for ONE sequence (c1_cecum_t1_v1), plus a from-scratch verification that
the export reproduces every metric in results/eval_stage4/summary.json.

Two modes, run as separate processes so verification really starts cold:
    python scripts/export_viewer_data.py --export
    python scripts/export_viewer_data.py --verify

--export repeats scripts/eval_stage4.py's call sequence (mesh/pose/depth
loading, scale recovery, the two run_pose_variant_sequence calls, the
oracle.py cross-check, ignore set, GT regions) via the SAME src/eval,
src/gt, src/geometry imports -- no logic is reimplemented, and nothing
under src/eval/, src/gt/, or docs/success_criteria.md is modified. It
cannot import eval_stage4.run() directly: that function writes
results/eval_stage4/summary.json and logs/eval_stage4.log itself, and
calling it here would overwrite the very reference file --verify checks
against. So the small amount of provider-callable glue (~20 lines,
eval_stage4.py:183-198) is duplicated here, unchanged.

--verify loads ONLY the exported files (plus the reference summary.json
and the small-region diagnosis summary, for the region-id sanity check in
check C) and recomputes every metric from them via the same locked
src/eval functions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path

import numpy as np

REPO = Path("/data1_ycao/chua/projects/mrsp")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

SEQ_NAME = "c1_cecum_t1_v1"
DATASET_ROOT = Path("/data1_ycao/chua/datasets/C3VDv2/registered_videos")
INTRINSICS_PATH = "/data1_ycao/chua/datasets/C3VDv2/camera_intrinsics.txt"
PRED_DIR = REPO / "results/pipelines/endodac_full_run" / SEQ_NAME
SAVED_SCALE_PATH = REPO / "results/pipelines/endodac_scale_analysis.json"
TAUS = [0.15, 0.25, 0.35, 0.50]
CONFIG_NAMES = ["oracle", "pred_depth_only", "pred_pose_only", "fully_predicted"]
OUT_DIR = REPO / "results/viewer" / SEQ_NAME
REFERENCE_SUMMARY = REPO / "results/eval_stage4/summary.json"
DIAGNOSIS_SUMMARY = REPO / "results/stage4_small_region_diagnosis/summary.json"
LOG_PATH = REPO / "logs/export_viewer_data.log"
CHECK_C_GT_REGION_ID = 45  # see docs/viewer_export.md "region-id mapping"
CHECK_C_EXPECTED_COUNT = 73
DETECTION_THRESHOLDS = [0.25, 0.50, 0.75]

OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def select_gpu() -> int:
    from gpu_status import get_gpu_stats

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


def git_commit_info() -> dict:
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    return {"commit": sha, "dirty": bool(dirty)}


def write_bin(path: Path, arr: np.ndarray, manifest: dict, key: str) -> None:
    arr = np.ascontiguousarray(arr)
    if arr.dtype.byteorder not in ("<", "=", "|"):
        arr = arr.astype(arr.dtype.newbyteorder("<"))
    arr.tofile(path)
    manifest[key] = {
        "path": str(path.relative_to(OUT_DIR)),
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
        "byte_size": arr.nbytes,
        "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
    }


# --------------------------------------------------------------------------
# --export
# --------------------------------------------------------------------------

def run_export():
    from geometry.camera import CameraIntrinsics, unproject
    from geometry.coverage_mesh import stream_coverage_mesh_from_zip
    from geometry.mesh_stats import face_adjacency, face_areas
    from geometry.pose import stream_poses_from_zip
    from gt.depth import depth_members, depth_to_mm, stream_raw_depth_frames_from_zip
    from eval.evaluable import load_vignette_mask
    from eval.fusion import run_pose_variant_sequence
    from eval.ignore_set import compute_ignore_set
    from eval.oracle import run_oracle_sequence
    from eval.regions import compute_regions
    from eval.region_metrics import (
        area_fraction_and_calibration,
        build_face_to_component_id,
        detection_at_threshold,
        detection_sweep,
        false_alarm_rate,
        false_reassurance_rate,
        localization_error,
        matching_predicted_component,
        region_centroid,
        region_coverage_fraction,
        region_recall_by_size_class,
        segment_intersects_mesh,
    )
    from eval.scale_recovery import aligned_predicted_pose, compute_depth_scale, compute_pose_alignment

    log(f"=== Viewer export: {SEQ_NAME} ===")
    gpu_index = select_gpu()
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    log(f"CUDA_VISIBLE_DEVICES set to '{gpu_index}' explicitly")

    import trimesh
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
    gt_poses = stream_poses_from_zip(zpath)
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
    pred_poses = np.load(PRED_DIR / "poses_pred.npy")
    log(f"loaded predicted depth + poses in {time.time()-t0:.1f}s")

    depth_scale = compute_depth_scale(gt_depth_mm_frames, gt_depth_valid_frames, pred_depth_native_frames)
    log(f"depth scale: median={depth_scale.median!r}")

    gt_positions = np.array([M[3, :3] for M in gt_poses])
    pred_positions = pred_poses[:, :3, 3]
    alignment = compute_pose_alignment(pred_positions, gt_positions)
    log(f"pose alignment: s_pose={alignment.s_pose!r}")

    if SAVED_SCALE_PATH.exists():
        saved = json.loads(SAVED_SCALE_PATH.read_text())
        log(
            f"cross-check vs saved values: depth_scale.median diff "
            f"{abs(depth_scale.median - saved['depth_scale']['median']):.6e}; "
            f"s_pose diff {abs(alignment.s_pose - saved['pose_scale']['s_pose']):.6e}"
        )

    # ---- providers (duplicated from eval_stage4.py:183-198, unchanged) ----
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

    t0 = time.time()
    results_gt_pose = run_pose_variant_sequence(
        wp, device, mesh_data.vertices, mesh_data.faces, n_frames,
        gt_pose_provider, {"gt_depth": gt_depth_provider, "pred_depth": pred_depth_provider},
        cam_rays, vignette_mask, TAUS,
    )
    log(f"pose variant GT: {time.time()-t0:.1f}s")

    t0 = time.time()
    results_pred_pose = run_pose_variant_sequence(
        wp, device, mesh_data.vertices, mesh_data.faces, n_frames,
        pred_pose_provider, {"gt_depth": gt_depth_provider, "pred_depth": pred_depth_provider},
        cam_rays, vignette_mask, TAUS,
    )
    log(f"pose variant pred: {time.time()-t0:.1f}s")

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
    log(f"oracle.py cross-check: {time.time()-t0:.1f}s")
    for tau in TAUS:
        if not np.array_equal(configs["oracle"].predicted_observed[tau], oracle_reference.predicted_observed[tau]):
            raise RuntimeError(f"oracle cross-check MISMATCH at tau={tau} -- stopping, not exporting")
    log("cross-check: fusion.py oracle config == oracle.py, confirmed")

    ignore_set = compute_ignore_set(gt_observed, oracle_reference.ever_evaluable_hit)
    ignore_set_frac_faces = ignore_set.sum() / n_faces
    log(f"ignore_set: {int(ignore_set.sum())}/{n_faces} faces ({ignore_set_frac_faces*100:.4f}%)")

    gt_regions = compute_regions(n_faces, adjacency, gt_unobserved, areas)
    gt_regions_headline = [r for r in gt_regions if r.size_class != "below_headline"]
    log(f"GT regions: {len(gt_regions)} total, {len(gt_regions_headline)} headline (d>=5mm)")

    gt_region_id = np.full(n_faces, -1, dtype=np.int32)
    for i, r in enumerate(gt_regions):
        gt_region_id[r.faces] = i

    # ---------------- write mesh / static per-face arrays ----------------
    manifest = {"files": {}}
    write_bin(OUT_DIR / "vertices_f32.bin", mesh_data.vertices.astype(np.float32), manifest["files"], "vertices_f32")
    write_bin(OUT_DIR / "vertices_f64.bin", mesh_data.vertices.astype(np.float64), manifest["files"], "vertices_f64")
    write_bin(OUT_DIR / "faces_i32.bin", mesh_data.faces.astype(np.int32), manifest["files"], "faces_i32")
    write_bin(OUT_DIR / "gt_observed_u8.bin", gt_observed.astype(np.uint8), manifest["files"], "gt_observed_u8")
    write_bin(OUT_DIR / "ignore_set_u8.bin", ignore_set.astype(np.uint8), manifest["files"], "ignore_set_u8")
    write_bin(OUT_DIR / "face_area_mm2_f32.bin", areas.astype(np.float32), manifest["files"], "face_area_mm2_f32")
    write_bin(OUT_DIR / "face_area_mm2_f64.bin", areas.astype(np.float64), manifest["files"], "face_area_mm2_f64")
    write_bin(OUT_DIR / "gt_region_id_i32.bin", gt_region_id, manifest["files"], "gt_region_id_i32")

    # ---------------- per-config x per-tau predicted_observed, packed ----
    n_bytes_per_row = (n_faces + 7) // 8
    packed = np.zeros((len(CONFIG_NAMES), len(TAUS), n_bytes_per_row), dtype=np.uint8)
    for ci, cname in enumerate(CONFIG_NAMES):
        for ti, tau in enumerate(TAUS):
            bits = configs[cname].predicted_observed[tau]
            packed[ci, ti] = np.packbits(bits, bitorder="little")
    write_bin(OUT_DIR / "predicted_observed_packed.bin", packed, manifest["files"], "predicted_observed_packed")

    # ---------------- metrics.json + regions.json (locked functions) -----
    all_metrics = {
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

    predicted_components_by_config_tau = {}  # (config, tau) -> (components, face_to_component_id)

    for config_name, result in configs.items():
        log(f"=== configuration: {config_name} (metrics) ===")
        ray_miss_frac = result.ray_miss_count / result.n_rays_total
        evaluable_frac = result.evaluable_count / result.n_rays_total
        config_out = {"ray_miss_frac": ray_miss_frac, "evaluable_frac": evaluable_frac,
                      "d_pred_unavailable_count": result.d_pred_unavailable_count, "by_tau": {}}

        for tau in TAUS:
            predicted_observed = result.predicted_observed[tau]
            predicted_unobserved = ~predicted_observed

            predicted_components = compute_regions(n_faces, adjacency, predicted_unobserved & ~ignore_set, areas)
            face_to_component_id = build_face_to_component_id(n_faces, predicted_components)
            predicted_components_by_config_tau[(config_name, tau)] = (predicted_components, face_to_component_id)

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
                "ignore_set_frac_faces": ignore_set_frac_faces,
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

        all_metrics["configurations"][config_name] = config_out

    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    log(f"wrote {OUT_DIR / 'metrics.json'}")

    # ---------------- regions.json ----------------
    regions_out = []
    for i, r in enumerate(gt_regions):
        headline = r.size_class != "below_headline"
        c_gt = region_centroid(r, mesh_data.vertices, mesh_data.faces, areas)
        entry = {
            "id": i,
            "n_faces": len(r.faces),
            "area_mm2": r.area,
            "diameter_mm": r.diameter,
            "size_class": r.size_class,
            "headline": headline,
            "centroid_mm": c_gt.tolist(),
        }
        if headline:
            per_config = {}
            for config_name in CONFIG_NAMES:
                per_tau = {}
                for tau in TAUS:
                    predicted_unobserved = ~configs[config_name].predicted_observed[tau]
                    components, face_to_component_id = predicted_components_by_config_tau[(config_name, tau)]
                    coverage = region_coverage_fraction(r, predicted_unobserved, areas)
                    detected_at = {
                        str(t): detection_at_threshold([r], predicted_unobserved, areas, t).detected[0]
                        for t in DETECTION_THRESHOLDS
                    }
                    comp = matching_predicted_component(r, components, face_to_component_id)
                    err = localization_error(r, components, face_to_component_id, mesh_data.vertices, mesh_data.faces, areas)
                    matched_centroid = None
                    seg_intersects = None
                    if comp is not None:
                        c_pred = region_centroid(comp, mesh_data.vertices, mesh_data.faces, areas)
                        matched_centroid = c_pred.tolist()
                        seg_intersects = bool(segment_intersects_mesh(mesh_trimesh, c_gt, c_pred))
                    per_tau[str(tau)] = {
                        "coverage_fraction": coverage,
                        "detected_at": detected_at,
                        "matched_predicted_component_centroid_mm": matched_centroid,
                        "localization_error_mm": err,
                        "segment_intersects_mesh": seg_intersects,
                    }
                per_config[config_name] = per_tau
            entry["by_config_by_tau"] = per_config
        regions_out.append(entry)

    with open(OUT_DIR / "regions.json", "w") as f:
        json.dump(regions_out, f, indent=2)
    log(f"wrote {OUT_DIR / 'regions.json'}")

    # ---------------- trajectory.json ----------------
    trajectory = []
    for i in range(n_frames):
        M = gt_poses[i]
        gt_R = M[:3, :3].T
        gt_pos = M[3, :3]
        pred_R, pred_pos = aligned_predicted_pose(alignment, pred_poses[i, :3, :3], pred_poses[i, :3, 3])
        trajectory.append({
            "frame": i,
            "gt_position_mm": gt_pos.tolist(),
            "gt_direction": gt_R[:, 2].tolist(),
            "pred_position_mm": pred_pos.tolist(),
            "pred_direction": pred_R[:, 2].tolist(),
        })
    traj_out = {
        "s_pose": alignment.s_pose,
        "depth_scale_median": depth_scale.median,
        "R_u": alignment.R_u.tolist(),
        "t_u": alignment.t_u.tolist(),
        "frames": trajectory,
    }
    with open(OUT_DIR / "trajectory.json", "w") as f:
        json.dump(traj_out, f, indent=2)
    log(f"wrote {OUT_DIR / 'trajectory.json'}")

    # ---------------- manifest.json ----------------
    manifest["git"] = git_commit_info()
    manifest["taus"] = TAUS
    manifest["configurations"] = CONFIG_NAMES
    manifest["predicted_observed_packed_axes"] = ["config", "tau", "byte"]
    manifest["predicted_observed_packed_bitorder"] = "little"
    manifest["n_faces"] = n_faces
    manifest["n_vertices"] = len(mesh_data.vertices)
    manifest["n_frames"] = n_frames
    manifest["python_version"] = sys.version
    manifest["numpy_version"] = np.__version__
    manifest["json_files"] = ["metrics.json", "regions.json", "trajectory.json"]
    with open(OUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    log(f"wrote {OUT_DIR / 'manifest.json'}")
    log("EXPORT DONE")


# --------------------------------------------------------------------------
# --verify
# --------------------------------------------------------------------------

def _nan_aware_diff(a, b) -> float:
    a = float("nan") if a is None else float(a)
    b = float("nan") if b is None else float(b)
    if np.isnan(a) and np.isnan(b):
        return 0.0
    if np.isnan(a) or np.isnan(b):
        return float("inf")
    return abs(a - b)


def _load_bin(manifest: dict, key: str) -> np.ndarray:
    info = manifest["files"][key]
    arr = np.fromfile(OUT_DIR / info["path"], dtype=info["dtype"])
    return arr.reshape(info["shape"])


def run_verify():
    from geometry.mesh_stats import face_adjacency, face_areas
    from eval.ignore_set import compute_ignore_set  # noqa: F401 (kept for symmetry / future use)
    from eval.regions import compute_regions
    from eval.region_metrics import (
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
    import trimesh

    log(f"=== Viewer export verification: {SEQ_NAME} ===")
    manifest = json.loads((OUT_DIR / "manifest.json").read_text())
    reference = json.loads(REFERENCE_SUMMARY.read_text())
    diagnosis = json.loads(DIAGNOSIS_SUMMARY.read_text())

    report = {"check_A": {}, "check_B": {}, "check_C": {}}

    for label, vdtype_key, adtype_key in [("f64", "vertices_f64", "face_area_mm2_f64"),
                                            ("f32", "vertices_f32", "face_area_mm2_f32")]:
        vertices = _load_bin(manifest, vdtype_key).astype(np.float64 if label == "f64" else np.float32)
        faces = _load_bin(manifest, "faces_i32")
        exported_areas = _load_bin(manifest, adtype_key)
        recomputed_areas = face_areas(vertices.astype(np.float64), faces).astype(exported_areas.dtype)
        max_area_diff = float(np.abs(recomputed_areas.astype(np.float64) - exported_areas.astype(np.float64)).max())
        report["check_A"][f"area_recompute_max_diff_{label}"] = max_area_diff
        log(f"check A [{label}]: max |recomputed_area - exported_area| = {max_area_diff:.3e}")
        if label == "f64":
            vertices64, faces64, areas64 = vertices, faces, exported_areas

    n_faces = manifest["n_faces"]
    gt_observed = _load_bin(manifest, "gt_observed_u8").astype(bool)
    gt_unobserved = ~gt_observed
    ignore_set = _load_bin(manifest, "ignore_set_u8").astype(bool)
    gt_region_id = _load_bin(manifest, "gt_region_id_i32")
    total_area = float(areas64.sum())

    adjacency = face_adjacency(vertices64, faces64)
    gt_regions = compute_regions(n_faces, adjacency, gt_unobserved, areas64)
    gt_regions_headline = [r for r in gt_regions if r.size_class != "below_headline"]

    recomputed_gt_region_id = np.full(n_faces, -1, dtype=np.int32)
    for i, r in enumerate(gt_regions):
        recomputed_gt_region_id[r.faces] = i
    region_id_matches = bool(np.array_equal(recomputed_gt_region_id, gt_region_id))
    report["check_A"]["gt_region_id_matches_recomputed_order"] = region_id_matches
    log(f"check A: recomputed GT region ids match exported gt_region_id = {region_id_matches}")

    ignore_set_frac_faces = ignore_set.sum() / n_faces
    diff_ignore_frac = abs(ignore_set_frac_faces - reference["ignore_set_frac_faces"])
    report["check_A"]["ignore_set_frac_faces_diff"] = diff_ignore_frac
    log(f"check A: ignore_set_frac_faces diff = {diff_ignore_frac:.3e}")

    mesh_trimesh = trimesh.Trimesh(vertices=vertices64, faces=faces64, process=False)

    packed = _load_bin(manifest, "predicted_observed_packed")
    config_names = manifest["configurations"]
    taus = manifest["taus"]

    metrics = json.loads((OUT_DIR / "metrics.json").read_text())

    max_diffs = {}
    worst = 0.0
    worst_key = None
    for ci, config_name in enumerate(config_names):
        ref_config = reference["configurations"][config_name]
        for ti, tau in enumerate(taus):
            n_bits = n_faces
            predicted_observed = np.unpackbits(packed[ci, ti], bitorder="little")[:n_bits].astype(bool)
            predicted_unobserved = ~predicted_observed

            predicted_components = compute_regions(n_faces, adjacency, predicted_unobserved & ~ignore_set, areas64)
            face_to_component_id = build_face_to_component_id(n_faces, predicted_components)

            sweep = detection_sweep(gt_regions, predicted_unobserved, areas64)
            f_reassur = false_reassurance_rate(gt_unobserved, predicted_observed, areas64)
            f_alarm = false_alarm_rate(predicted_unobserved, gt_observed, ignore_set, areas64)
            area_frac, calib_ratio = area_fraction_and_calibration(predicted_unobserved, gt_unobserved, areas64, total_area)
            recall_50 = region_recall_by_size_class(gt_regions, predicted_unobserved, areas64, 0.50)

            loc_errors = []
            n_undetected = 0
            n_seg_checked = 0
            n_seg_intersect = 0
            for r in gt_regions_headline:
                err = localization_error(r, predicted_components, face_to_component_id, vertices64, faces64, areas64)
                if err is None:
                    n_undetected += 1
                else:
                    loc_errors.append(err)
                    comp = matching_predicted_component(r, predicted_components, face_to_component_id)
                    c_gt = region_centroid(r, vertices64, faces64, areas64)
                    c_pred = region_centroid(comp, vertices64, faces64, areas64)
                    n_seg_checked += 1
                    if segment_intersects_mesh(mesh_trimesh, c_gt, c_pred):
                        n_seg_intersect += 1

            recomputed = {
                "ignore_set_frac_faces": ignore_set_frac_faces,
                "area_fraction": area_frac,
                "calibration_ratio": calib_ratio,
                "false_reassurance_rate": f_reassur,
                "false_alarm_rate": f_alarm,
                "n_predicted_unobserved_faces": int(predicted_unobserved.sum()),
                "n_headline_regions_with_localization_error": len(loc_errors),
                "n_headline_regions_undetected": n_undetected,
                "localization_error_median_mm": float(np.median(loc_errors)) if loc_errors else None,
                "localization_error_mean_mm": float(np.mean(loc_errors)) if loc_errors else None,
                "segment_intersect_fraction": (n_seg_intersect / n_seg_checked) if n_seg_checked else None,
            }
            ref_tau = ref_config["by_tau"][str(tau)]
            for key, val in recomputed.items():
                d = _nan_aware_diff(val, ref_tau[key])
                mkey = f"{config_name}/tau={tau}/{key}"
                max_diffs[mkey] = d
                if d > worst:
                    worst = d
                    worst_key = mkey
            for size_class in ["below_headline", "small", "medium", "large"]:
                d = _nan_aware_diff(recall_50[size_class]["recall"], ref_tau["recall_at_50pct"][size_class]["recall"])
                mkey = f"{config_name}/tau={tau}/recall_at_50pct/{size_class}"
                max_diffs[mkey] = d
                if d > worst:
                    worst = d
                    worst_key = mkey
            for t_str, by_class in sweep.items():
                ref_sweep = ref_tau["detection_sweep"][str(t_str)]
                for size_class, v in by_class.items():
                    d = _nan_aware_diff(v["recall"], ref_sweep[size_class]["recall"])
                    mkey = f"{config_name}/tau={tau}/detection_sweep/{t_str}/{size_class}"
                    max_diffs[mkey] = d
                    if d > worst:
                        worst = d
                        worst_key = mkey

    report["check_A"]["metric_max_diffs"] = max_diffs
    report["check_A"]["worst_diff"] = worst
    report["check_A"]["worst_diff_key"] = worst_key
    report["check_A"]["pass"] = worst <= 1e-9
    log(f"check A: worst metric diff = {worst:.3e} ({worst_key}) -- {'PASS' if worst <= 1e-9 else 'FAIL'}")

    for key in ["ray_miss_frac", "evaluable_frac", "d_pred_unavailable_count"]:
        note_diffs = {}
        for config_name in config_names:
            note_diffs[config_name] = _nan_aware_diff(metrics["configurations"][config_name][key],
                                                        reference["configurations"][config_name][key])
        report["check_A"][f"not_recomputable_{key}_copy_diff"] = note_diffs
    for key in ["depth_scale_median", "pose_alignment_s_pose"]:
        report["check_A"][f"not_recomputable_{key}_copy_diff"] = _nan_aware_diff(metrics[key], reference[key])

    if not report["check_A"]["pass"]:
        log("*** CHECK A FAILED -- stopping, not running checks B/C ***")
        with open(OUT_DIR / "verification.json", "w") as f:
            json.dump(report, f, indent=2)
        sys.exit(1)

    # ---------------- check B ----------------
    n_faces_ok = n_faces == 699908
    ignore_count = int(ignore_set.sum())
    ignore_ok = ignore_count == 20094
    headline_counts = {"small": 0, "medium": 0, "large": 0}
    for r in gt_regions_headline:
        headline_counts[r.size_class] += 1
    headline_total = len(gt_regions_headline)
    headline_ok = headline_total == 3 and headline_counts["small"] == 2 and headline_counts["medium"] == 0 and headline_counts["large"] == 1
    report["check_B"] = {
        "n_faces": n_faces, "n_faces_ok": n_faces_ok,
        "ignore_set_count": ignore_count, "ignore_set_ok": ignore_ok,
        "headline_count": headline_total, "headline_counts_by_class": headline_counts, "headline_ok": headline_ok,
        "pass": n_faces_ok and ignore_ok and headline_ok,
    }
    log(f"check B: n_faces={n_faces} ({n_faces_ok}), ignore_set={ignore_count} ({ignore_ok}), "
        f"headline={headline_total} small={headline_counts['small']} medium={headline_counts['medium']} "
        f"large={headline_counts['large']} ({headline_ok})")
    if not report["check_B"]["pass"]:
        log("*** CHECK B FAILED ***")

    # ---------------- check C ----------------
    diag_faces = set(diagnosis["small_regions"][0]["face_ids"])
    matched_id, region = next(
        (i, r) for i, r in enumerate(gt_regions) if len(r.faces) == len(diag_faces) and set(r.faces.tolist()) == diag_faces
    )
    log(f"check C: diagnosis small-region-0 face set matches gt_region_id {matched_id} "
        f"(expected {CHECK_C_GT_REGION_ID})")

    ci_fp = config_names.index("fully_predicted")
    ci_pp = config_names.index("pred_pose_only")
    ci_pd = config_names.index("pred_depth_only")
    ti = taus.index(0.25)
    pu_fp = ~np.unpackbits(packed[ci_fp, ti], bitorder="little")[:n_faces].astype(bool)
    pu_pp = ~np.unpackbits(packed[ci_pp, ti], bitorder="little")[:n_faces].astype(bool)
    pu_pd = ~np.unpackbits(packed[ci_pd, ti], bitorder="little")[:n_faces].astype(bool)

    region_faces = region.faces
    only_fp = region_faces[pu_fp[region_faces] & ~pu_pp[region_faces]]
    count = len(only_fp)
    count_ok = count == CHECK_C_EXPECTED_COUNT
    report["check_C"] = {
        "matched_gt_region_id": int(matched_id),
        "expected_gt_region_id": CHECK_C_GT_REGION_ID,
        "region_id_mapping_ok": matched_id == CHECK_C_GT_REGION_ID,
        "n_only_fully_predicted": count,
        "expected_count": CHECK_C_EXPECTED_COUNT,
        "count_ok": count_ok,
    }
    if not count_ok:
        log(f"*** CHECK C: expected {CHECK_C_EXPECTED_COUNT} faces, got {count} -- stopping ***")
        with open(OUT_DIR / "verification.json", "w") as f:
            json.dump(report, f, indent=2)
        sys.exit(1)

    also_pred_depth = only_fp[pu_pd[only_fp]]
    not_pred_depth = only_fp[~pu_pd[only_fp]]
    report["check_C"]["n_also_unobserved_under_pred_depth_only"] = len(also_pred_depth)
    report["check_C"]["n_not_unobserved_under_pred_depth_only"] = len(not_pred_depth)
    log(f"check C: {count} faces only-unobserved under fully_predicted (tau=0.25, region {matched_id}); "
        f"{len(also_pred_depth)} also unobserved under pred_depth_only, {len(not_pred_depth)} not")

    with open(OUT_DIR / "verification.json", "w") as f:
        json.dump(report, f, indent=2)
    log(f"wrote {OUT_DIR / 'verification.json'}")
    log("VERIFY DONE")


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--export", action="store_true")
    mode.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    try:
        if args.export:
            run_export()
        else:
            run_verify()
    except Exception:
        log(f"*** FATAL ERROR at {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} ***")
        nvidia_smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
        log(f"nvidia-smi at failure time:\n{nvidia_smi}")
        log(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
