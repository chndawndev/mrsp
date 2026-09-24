"""Stage 1 validation (docs/eval_protocol.md section 6 Step 0 + the
IoU-reproduction check), for the src/eval/ implementation task.

Two checks, both on c1_cecum_t1_v1, streamed directly from its zip archive
(no bulk extraction -- CLAUDE.md):

Part A -- confirm src/geometry/rasterizer.py's accumulation behaviour is
UNCHANGED from scripts/visibility_full.py's frozen, already-validated
result: run the new kernel across the full GT trajectory in accumulate
mode and compare face-set IoU (and false_observed/false_unobserved raw
counts) against the exact recorded values in
results/visibility_full/metrics.csv for this sequence.

Part B -- Step 0: validate the new hit-distance output itself. Sample
frames, compare the rasterizer's per-pixel camera-frame Z-depth against
GT depth TIFFs read from the same archive (an independent computation --
the TIFFs were rendered by a different renderer, DurrLab/C3VDv3's
Render.cu, not by this rasterizer). Report median/p95/max abs diff,
propose a tolerance with reasoning, state pass/fail.

Run under tmux per instructions if long; measured cost here is a few
seconds (scripts/visibility_full.py's own t_cast_s for this sequence was
2.87s for 218 frames), so this runs directly.
"""
from __future__ import annotations

import io
import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import tifffile

REPO = Path("/data1_ycao/chua/projects/mrsp")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from geometry.camera import CameraIntrinsics, unproject  # noqa: E402
from geometry.coverage_mesh import stream_coverage_mesh_from_zip  # noqa: E402
from geometry.pose import stream_poses_from_zip  # noqa: E402
from gt.rasterizer import build_mesh, cast_frame, make_raycast_frame_kernel  # noqa: E402
from gpu_status import get_gpu_stats  # noqa: E402

SEQ_NAME = "c1_cecum_t1_v1"
ZIP_PATH = Path("/data1_ycao/chua/datasets/C3VDv2/registered_videos") / f"{SEQ_NAME}.zip"
INTRINSICS_PATH = "/data1_ycao/chua/datasets/C3VDv2/camera_intrinsics.txt"
RECORDED_METRICS_CSV = REPO / "results/visibility_full/metrics.csv"
OUT_DIR = REPO / "results/eval_protocol_stage1"
LOG_PATH = REPO / "logs/eval_protocol_stage1_validation.log"
N_SAMPLE_FRAMES = 20  # matches the precedent sample size used throughout eval_protocol.md's tau derivation

OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def select_gpu() -> int:
    log("=== GPU availability check (scripts/gpu_status.py) ===")
    gpus = get_gpu_stats()
    freest = max(gpus, key=lambda g: g["memory_free_mib"])
    log(f"Selected GPU: index {freest['index']} ({freest['name']}), "
        f"{freest['memory_free_mib']/1000:.1f} GB free, {freest['utilization_pct']}% utilized")
    if freest["memory_free_mib"] < 8000:
        raise RuntimeError(f"no GPU with >= 8GB free (best: {freest['memory_free_mib']}MiB) -- aborting")
    return freest["index"]


def depth_members(zf: zipfile.ZipFile) -> list[str]:
    return sorted(
        n for n in zf.namelist()
        if n.endswith("_depth.tiff") and ("/depth/" in n or n.startswith("depth/"))
    )


def main():
    log(f"=== Stage 1 validation: {SEQ_NAME} ===")
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
    n_rays = len(cam_rays)
    log(f"camera rays: {n_rays} ({intr.width}x{intr.height})")

    t0 = time.time()
    mesh_data = stream_coverage_mesh_from_zip(ZIP_PATH)
    poses = stream_poses_from_zip(ZIP_PATH)
    log(f"streamed mesh ({len(mesh_data.faces)} faces) + poses ({len(poses)} frames) "
        f"from {ZIP_PATH} in {time.time()-t0:.1f}s")

    kernel = make_raycast_frame_kernel(wp)
    wp_mesh = build_mesh(wp, mesh_data.vertices, mesh_data.faces, device)

    # ---------------- Part A: accumulate-mode reproduction ----------------
    log("\n=== Part A: accumulate-mode IoU reproduction ===")
    n_faces = len(mesh_data.faces)
    observed_gpu = wp.zeros(n_faces, dtype=wp.int32, device=device)

    t0 = time.time()
    for M in poses:
        R_c2w = M[:3, :3].T
        T_c2w = M[3, :3]
        cast_frame(wp, device, kernel, wp_mesh, cam_rays, R_c2w, T_c2w, observed_gpu)
    wp.synchronize()
    t_cast = time.time() - t0
    log(f"accumulate-mode ray-cast over {len(poses)} frames: {t_cast:.2f}s")

    pred_observed = observed_gpu.numpy().astype(bool)
    gt_observed = mesh_data.face_observed

    inter = np.sum(pred_observed & gt_observed)
    union = np.sum(pred_observed | gt_observed)
    iou = float(inter / union) if union else float("nan")
    false_observed = int(np.sum(pred_observed & ~gt_observed))
    false_unobserved = int(np.sum(~pred_observed & gt_observed))
    n_pred_observed = int(pred_observed.sum())
    n_gt_observed = int(gt_observed.sum())

    import pandas as pd
    # float_precision="round_trip": pandas' default C float parser can
    # introduce a 1-ULP error vs. the exact decimal string (confirmed here:
    # default parser gives 0.9988272726701084, but the raw CSV text and
    # float('0.9988272726701083') both give ...83 -- round_trip matches the
    # text exactly).
    recorded = pd.read_csv(RECORDED_METRICS_CSV, float_precision="round_trip")
    recorded_row = recorded[recorded["Video Name"] == SEQ_NAME].iloc[0]

    log(f"NEW    : iou={iou!r} false_observed={false_observed} false_unobserved={false_unobserved} "
        f"n_gt_observed={n_gt_observed} n_pred_observed={n_pred_observed}")
    log(f"RECORDED: iou={recorded_row['iou']!r} false_observed={recorded_row['false_observed']} "
        f"false_unobserved={recorded_row['false_unobserved']} n_gt_observed={recorded_row['n_gt_observed']} "
        f"n_pred_observed={recorded_row['n_pred_observed']}")

    identical = (
        iou == float(recorded_row["iou"])
        and false_observed == int(recorded_row["false_observed"])
        and false_unobserved == int(recorded_row["false_unobserved"])
        and n_gt_observed == int(recorded_row["n_gt_observed"])
        and n_pred_observed == int(recorded_row["n_pred_observed"])
    )
    log(f"IDENTICAL to recorded (exact, not merely close): {identical}")
    if not identical:
        log("*** MISMATCH -- do not proceed; the extension changed the original behaviour. ***")

    part_a = {
        "iou": iou, "false_observed": false_observed, "false_unobserved": false_unobserved,
        "n_gt_observed": n_gt_observed, "n_pred_observed": n_pred_observed,
        "recorded_iou": float(recorded_row["iou"]),
        "recorded_false_observed": int(recorded_row["false_observed"]),
        "recorded_false_unobserved": int(recorded_row["false_unobserved"]),
        "recorded_n_gt_observed": int(recorded_row["n_gt_observed"]),
        "recorded_n_pred_observed": int(recorded_row["n_pred_observed"]),
        "identical": bool(identical),
        "t_cast_s": t_cast,
    }

    # ---------------- Part B: Step 0, hit-distance validation ----------------
    log(f"\n=== Part B: Step 0 hit-distance validation, {N_SAMPLE_FRAMES} sampled frames ===")
    frame_indices = np.linspace(0, len(poses) - 1, N_SAMPLE_FRAMES, dtype=int)
    frame_indices = sorted(set(frame_indices.tolist()))
    log(f"sampled frame indices: {frame_indices}")

    with zipfile.ZipFile(ZIP_PATH) as zf:
        members = depth_members(zf)
        assert len(members) == len(poses), (
            f"depth member count {len(members)} != pose count {len(poses)}"
        )

        all_diffs = []
        per_frame = []
        for idx in frame_indices:
            raw = tifffile.imread(io.BytesIO(zf.read(members[idx])))
            assert raw.shape == (intr.height, intr.width), raw.shape
            raw_flat = raw.ravel()
            depth_mm = raw_flat.astype(np.float64) / 65535.0 * 100.0
            gt_valid = (raw_flat != 0) & (raw_flat != 65535)

            M = poses[idx]
            R_c2w = M[:3, :3].T
            T_c2w = M[3, :3]
            hit_gpu, face_gpu, dist_gpu = cast_frame(
                wp, device, kernel, wp_mesh, cam_rays, R_c2w, T_c2w, observed_gpu
            )
            wp.synchronize()
            hit = hit_gpu.numpy().astype(bool)
            dist = dist_gpu.numpy().astype(np.float64)

            both = gt_valid & hit
            gt_valid_only = gt_valid & ~hit
            hit_only = hit & ~gt_valid

            diff = np.abs(dist[both] - depth_mm[both])
            all_diffs.append(diff)

            per_frame.append({
                "frame_idx": int(idx),
                "n_gt_valid": int(gt_valid.sum()),
                "n_hit": int(hit.sum()),
                "n_compared": int(both.sum()),
                "n_gt_valid_no_rasterizer_hit": int(gt_valid_only.sum()),
                "n_rasterizer_hit_no_gt_valid": int(hit_only.sum()),
                "median_abs_diff_mm": float(np.median(diff)) if len(diff) else None,
                "p95_abs_diff_mm": float(np.percentile(diff, 95)) if len(diff) else None,
                "max_abs_diff_mm": float(np.max(diff)) if len(diff) else None,
            })
            log(f"  frame {idx}: gt_valid={int(gt_valid.sum())} hit={int(hit.sum())} "
                f"compared={int(both.sum())} gt_valid_no_hit={int(gt_valid_only.sum())} "
                f"hit_no_gt_valid={int(hit_only.sum())} "
                f"median={per_frame[-1]['median_abs_diff_mm']:.5f}mm "
                f"p95={per_frame[-1]['p95_abs_diff_mm']:.5f}mm "
                f"max={per_frame[-1]['max_abs_diff_mm']:.5f}mm")

    all_diffs = np.concatenate(all_diffs)
    median_diff = float(np.median(all_diffs))
    p95_diff = float(np.percentile(all_diffs, 95))
    p99_diff = float(np.percentile(all_diffs, 99))
    p999_diff = float(np.percentile(all_diffs, 99.9))
    max_diff = float(np.max(all_diffs))
    n_total_compared = len(all_diffs)
    frac_gt_1mm = float(np.mean(all_diffs > 1.0))
    frac_gt_5mm = float(np.mean(all_diffs > 5.0))
    frac_gt_10mm = float(np.mean(all_diffs > 10.0))

    log(f"\n=== Part B aggregate over {n_total_compared} compared pixels, {len(frame_indices)} frames ===")
    log(f"median abs diff: {median_diff:.6f} mm")
    log(f"p95 abs diff:    {p95_diff:.6f} mm")
    log(f"p99 abs diff:    {p99_diff:.6f} mm")
    log(f"p99.9 abs diff:  {p999_diff:.6f} mm")
    log(f"max abs diff:    {max_diff:.6f} mm")
    log(f"fraction > 1mm:  {frac_gt_1mm*100:.5f}%  ({int(np.sum(all_diffs>1.0))} px)")
    log(f"fraction > 5mm:  {frac_gt_5mm*100:.5f}%  ({int(np.sum(all_diffs>5.0))} px)")
    log(f"fraction > 10mm: {frac_gt_10mm*100:.5f}% ({int(np.sum(all_diffs>10.0))} px)")

    part_b = {
        "n_sample_frames": len(frame_indices),
        "frame_indices": [int(i) for i in frame_indices],
        "n_total_compared": int(n_total_compared),
        "median_abs_diff_mm": median_diff,
        "p95_abs_diff_mm": p95_diff,
        "p99_abs_diff_mm": p99_diff,
        "p999_abs_diff_mm": p999_diff,
        "max_abs_diff_mm": max_diff,
        "frac_gt_1mm": frac_gt_1mm,
        "frac_gt_5mm": frac_gt_5mm,
        "frac_gt_10mm": frac_gt_10mm,
        "per_frame": per_frame,
    }

    summary = {"sequence": SEQ_NAME, "part_a": part_a, "part_b": part_b}
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(f"\nwrote {OUT_DIR / 'summary.json'}")
    log("ALL DONE")


if __name__ == "__main__":
    main()
