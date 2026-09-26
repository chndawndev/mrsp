#!/usr/bin/env python
"""D1 Stage B: full evaluation of one sequence, all 4 configurations x 4
taus. Generalizes scripts/eval_stage4.py's exact procedure (mesh/pose/
depth loading, scale recovery, the two run_pose_variant_sequence calls,
the oracle.py cross-check, ignore set, GT regions, every region_metrics.py
function) to any of the 169 registered sequences, reading Stage A's
predictions from results/pipelines/endodac_full_run/<seq>/.

Runs the oracle.py cross-check for EVERY sequence, not just a pilot:
oracle.py's run_oracle_sequence is the only source of ever_evaluable_hit
(needed for the ignore set) without modifying src/eval/fusion.py, which
doesn't expose it. This also re-validates the oracle configuration
bit-for-bit on every sequence, not just once.

CLI:
    scratch/.venv/bin/python scripts/eval_d1_sequence.py --sequence NAME
    scratch/.venv/bin/python scripts/eval_d1_sequence.py --shard-index I --shard-total K
"""
from __future__ import annotations

import argparse
import json
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

DATASET_ROOT = Path("/data1_ycao/chua/datasets/C3VDv2/registered_videos")
INTRINSICS_PATH = "/data1_ycao/chua/datasets/C3VDv2/camera_intrinsics.txt"
PRED_ROOT = REPO / "results/pipelines/endodac_full_run"
TAUS = [0.15, 0.25, 0.35, 0.50]
CONFIG_NAMES = ["oracle", "pred_depth_only", "pred_pose_only", "fully_predicted"]
OUT_ROOT = REPO / "results/d1/per_sequence"
EXPECTED_N_SEQUENCES = 169
DETECTION_THRESHOLDS = [0.25, 0.50, 0.75]
MIN_FREE_DISK_GB = 50


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


def is_complete(name: str) -> bool:
    out_dir = OUT_ROOT / name
    manifest_path = out_dir / "MANIFEST.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    if manifest.get("status") != "ok":
        return False
    for f in ["predicted_observed_packed.bin", "regions.csv", "metrics.json"]:
        if not (out_dir / f).exists():
            return False
    return True


def free_disk_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


# --------------------------------------------------------------------------

def evaluate_sequence(name: str, wp, device: str, gpu_index: int) -> dict:
    t_start = time.time()
    out_dir = OUT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    zpath = DATASET_ROOT / f"{name}.zip"
    pred_dir = PRED_ROOT / name

    intr = CameraIntrinsics.from_file(INTRINSICS_PATH)
    cols, rows = np.meshgrid(np.arange(intr.width), np.arange(intr.height))
    px_grid = np.stack([cols.ravel(), rows.ravel()], axis=-1).astype(np.float64)
    cam_rays = unproject(px_grid, intr).astype(np.float32)
    vignette_mask = load_vignette_mask(intr.width, intr.height)

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

    pred_depth_native_frames = [
        np.load(pred_dir / "depth" / f"{i:04d}_pred_depth.npy").ravel().astype(np.float64) for i in range(n_frames)
    ]
    pred_poses = np.load(pred_dir / "poses_pred.npy")

    depth_scale = compute_depth_scale(gt_depth_mm_frames, gt_depth_valid_frames, pred_depth_native_frames)
    gt_positions = np.array([M[3, :3] for M in gt_poses])
    pred_positions = pred_poses[:, :3, 3]
    alignment = compute_pose_alignment(pred_positions, gt_positions)

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

    results_gt_pose = run_pose_variant_sequence(
        wp, device, mesh_data.vertices, mesh_data.faces, n_frames,
        gt_pose_provider, {"gt_depth": gt_depth_provider, "pred_depth": pred_depth_provider},
        cam_rays, vignette_mask, TAUS,
    )
    results_pred_pose = run_pose_variant_sequence(
        wp, device, mesh_data.vertices, mesh_data.faces, n_frames,
        pred_pose_provider, {"gt_depth": gt_depth_provider, "pred_depth": pred_depth_provider},
        cam_rays, vignette_mask, TAUS,
    )
    configs = {
        "oracle": results_gt_pose["gt_depth"],
        "pred_depth_only": results_gt_pose["pred_depth"],
        "pred_pose_only": results_pred_pose["gt_depth"],
        "fully_predicted": results_pred_pose["pred_depth"],
    }

    oracle_reference = run_oracle_sequence(
        wp, device, mesh_data.vertices, mesh_data.faces, gt_poses,
        stream_raw_depth_frames_from_zip(zpath), cam_rays, vignette_mask, TAUS,
    )
    for tau in TAUS:
        if not np.array_equal(configs["oracle"].predicted_observed[tau], oracle_reference.predicted_observed[tau]):
            raise RuntimeError(f"{name}: oracle cross-check MISMATCH at tau={tau}")

    ignore_set = compute_ignore_set(gt_observed, oracle_reference.ever_evaluable_hit)
    ignore_set_frac_faces = ignore_set.sum() / n_faces

    gt_regions = compute_regions(n_faces, adjacency, gt_unobserved, areas)
    gt_regions_headline = [r for r in gt_regions if r.size_class != "below_headline"]

    # ---------------- packed predicted_observed bits ----------------
    n_bytes_per_row = (n_faces + 7) // 8
    packed = np.zeros((len(CONFIG_NAMES), len(TAUS), n_bytes_per_row), dtype=np.uint8)
    for ci, cname in enumerate(CONFIG_NAMES):
        for ti, tau in enumerate(TAUS):
            packed[ci, ti] = np.packbits(configs[cname].predicted_observed[tau], bitorder="little")
    packed.tofile(out_dir / "predicted_observed_packed.bin")

    # ---------------- per config x tau metrics + region rows ----------------
    metrics_out = {
        "sequence": name, "n_faces": n_faces, "n_frames": n_frames,
        "depth_scale_median": depth_scale.median, "pose_alignment_s_pose": alignment.s_pose,
        "ignore_set_frac_faces": ignore_set_frac_faces,
        "n_gt_regions_total": len(gt_regions), "n_gt_regions_headline": len(gt_regions_headline),
        "configurations": {},
    }
    region_rows = []

    # precompute per-region (id, coverage-independent) fields once -- region id is
    # this region's position in compute_regions' own output order, matching the
    # convention already established for the Stage 1 viewer export.
    for config_name, result in configs.items():
        ray_miss_frac = result.ray_miss_count / result.n_rays_total
        evaluable_frac = result.evaluable_count / result.n_rays_total
        d_pred_unavailable_frac = (
            result.d_pred_unavailable_count / result.evaluable_count if result.evaluable_count else float("nan")
        )
        config_out = {
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

            # Raw numerator/denominator areas for every area-based metric, in ADDITION
            # to the ratios the locked functions above return. Needed because
            # docs/eval_protocol.md's aggregation rule pools area-based metrics as
            # "summed numerator area over summed denominator area across sequences"
            # -- a corpus-level number the per-sequence ratio alone can't reconstruct
            # (pooling ratios directly is a different, wrong computation). These
            # re-derive the exact same intermediate areas region_metrics.py's own
            # functions compute internally, via the same face-level arrays already in
            # scope here -- not a different formula, just exposing the intermediate
            # values those functions don't return.
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

            # ---------------- single pass over every GT region ----------------
            loc_errors, n_undetected, n_seg_checked, n_seg_intersect = [], 0, 0, 0
            for region_id, r in enumerate(gt_regions):
                is_headline = r.size_class != "below_headline"
                coverage = per_threshold[0.25].coverage[region_id]  # coverage is threshold-independent
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
                # raw areas (mm^2) for corpus-level pooling -- see comment above
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

    pd.DataFrame(region_rows).to_csv(out_dir / "regions.csv", index=False)

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    manifest = {
        "sequence": name, "n_frames": n_frames, "n_faces": n_faces,
        "n_gt_regions": len(gt_regions), "n_gt_regions_headline": len(gt_regions_headline),
        "git_commit": git_head(), "gpu_index": gpu_index,
        "runtime_seconds_total": time.time() - t_start, "status": "ok", "error": None,
    }
    with open(out_dir / "MANIFEST.json", "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def run_one(name: str, wp, device, gpu_index: int) -> dict:
    out_dir = OUT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        manifest = evaluate_sequence(name, wp, device, gpu_index)
        log(f"{name}: DONE in {manifest['runtime_seconds_total']:.1f}s", tag=f"gpu{gpu_index}")
        return manifest
    except Exception:
        err = traceback.format_exc()
        log(f"{name}: FAILED\n{err}", tag=f"gpu{gpu_index}")
        manifest = {
            "sequence": name, "status": "failed", "error": err,
            "git_commit": git_head(), "gpu_index": gpu_index,
            "runtime_seconds_total": time.time() - t0,
        }
        try:
            out_dir.mkdir(parents=True, exist_ok=True)  # defensive: out_dir may have vanished mid-run
            with open(out_dir / "MANIFEST.json", "w") as f:
                json.dump(manifest, f, indent=2)
        except OSError:
            log(f"{name}: could not write failure MANIFEST.json either -- logging only", tag=f"gpu{gpu_index}")
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
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence")
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-total", type=int)
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    gpu_index = int(os.environ.get("CUDA_VISIBLE_DEVICES", "-1"))
    if gpu_index == -1:
        gpu_index = select_gpu()
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)

    import warp as wp
    wp.init()
    device = "cuda:0"
    log(f"warp version {wp.config.version}, device {device}, CUDA_VISIBLE_DEVICES={gpu_index}", tag=f"gpu{gpu_index}")

    sequences = discover_sequences()

    if args.sequence:
        seq_dict = dict(sequences)
        if args.sequence not in seq_dict:
            raise SystemExit(f"{args.sequence!r} not among the 169 registered sequences")
        manifest = run_one(args.sequence, wp, device, gpu_index)
        if manifest["status"] != "ok":
            sys.exit(1)
        return

    if args.shard_index is None or args.shard_total is None:
        raise SystemExit("provide --sequence NAME, or both --shard-index and --shard-total")

    my_sequences = [name for i, (name, _) in enumerate(sequences) if i % args.shard_total == args.shard_index]
    log(f"shard {args.shard_index}/{args.shard_total}: {len(my_sequences)} sequences assigned", tag=f"gpu{gpu_index}")
    for name in my_sequences:
        if is_complete(name):
            log(f"{name}: already complete, skipping", tag=f"gpu{gpu_index}")
            continue
        free_gb = free_disk_gb(REPO)
        if free_gb < MIN_FREE_DISK_GB:
            log(f"free disk {free_gb:.1f}GB < {MIN_FREE_DISK_GB}GB -- stopping shard", tag=f"gpu{gpu_index}")
            return
        try:
            run_one(name, wp, device, gpu_index)
        except Exception:
            # run_one already catches evaluate_sequence's own exceptions and
            # records them; this is a last-resort guard so one sequence
            # truly crashing (e.g. its own failure-manifest write also
            # failing) can't take down the other 168 in this shard.
            log(f"{name}: run_one itself raised -- logging and continuing\n{traceback.format_exc()}", tag=f"gpu{gpu_index}")
    log(f"shard {args.shard_index}/{args.shard_total}: all assigned sequences done", tag=f"gpu{gpu_index}")


if __name__ == "__main__":
    main()
