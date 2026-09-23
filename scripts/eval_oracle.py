#!/usr/bin/env python
"""Stage 2 driver: oracle configuration (GT depth + GT pose) end-to-end,
docs/eval_protocol.md, run on 3 sequences (c1_cecum_t1_v1, c2_rectum_t1_v1,
c1_descending_t1_v1), full pixel density, tau sweep {0.15, 0.25, 0.35, 0.50}.

Validates per section 6:
  - oracle vs our own GT rasterization restricted to evaluable pixels
    (target IoU >= 0.99, every tau in the sweep).
  - oracle vs released GT, reported alongside Part 1's already-measured
    structural ceiling (results/pipelines/oracle_gap_part1/summary.json,
    where available) as a cross-check, not a pass/fail target.
  - the tau-rejection diagnostic: how many evaluable pixels the tau test
    actually rejects in the oracle configuration, at each tau -- expected
    near zero, since d_pred and d_hit describe the same GT geometry; a
    nonzero rate especially at tau=0.15 is the empirical lower-bound
    evidence for tau, per the silhouette-aliasing tail already
    characterized in Stage 1 (results/eval_protocol_stage1/summary.json).

Streams mesh/poses/depth directly from each sequence's zip archive, no
bulk extraction (CLAUDE.md). Measured cost, see logs/eval_oracle_stage2.log.
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

REPO = Path("/data1_ycao/chua/projects/mrsp")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from geometry.camera import CameraIntrinsics, unproject  # noqa: E402
from geometry.coverage_mesh import stream_coverage_mesh_from_zip  # noqa: E402
from geometry.pose import stream_poses_from_zip  # noqa: E402
from gt.depth import depth_members, stream_raw_depth_frames_from_zip  # noqa: E402
from eval.evaluable import load_vignette_mask  # noqa: E402
from eval.ignore_set import compute_ignore_set  # noqa: E402
from eval.metrics_face import face_set_iou  # noqa: E402
from eval.oracle import run_oracle_sequence  # noqa: E402
from gpu_status import get_gpu_stats  # noqa: E402

SEQUENCES = ["c1_cecum_t1_v1", "c2_rectum_t1_v1", "c1_descending_t1_v1"]
DATASET_ROOT = Path("/data1_ycao/chua/datasets/C3VDv2/registered_videos")
INTRINSICS_PATH = "/data1_ycao/chua/datasets/C3VDv2/camera_intrinsics.txt"
TAUS = [0.15, 0.25, 0.35, 0.50]
IOU_TARGET = 0.99
OUT_DIR = REPO / "results/eval_oracle_stage2"
LOG_PATH = REPO / "logs/eval_oracle_stage2.log"
PART1_CEILING_PATH = REPO / "results/pipelines/oracle_gap_part1/summary.json"

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
    log("=== Stage 2: oracle configuration, 3 sequences ===")
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

    vignette_mask = load_vignette_mask(intr.width, intr.height)
    log(f"vignette mask: {int(vignette_mask.sum())} px ({vignette_mask.mean()*100:.4f}%)")

    part1 = {}
    if PART1_CEILING_PATH.exists():
        part1 = json.loads(PART1_CEILING_PATH.read_text())
        log(f"loaded Part 1 ceiling reference from {PART1_CEILING_PATH}")
    else:
        log(f"no Part 1 ceiling reference found at {PART1_CEILING_PATH}")

    all_results = {}
    for seq in SEQUENCES:
        log(f"\n=== {seq} ===")
        zpath = DATASET_ROOT / f"{seq}.zip"

        t0 = time.time()
        mesh_data = stream_coverage_mesh_from_zip(zpath)
        poses = stream_poses_from_zip(zpath)
        log(
            f"streamed mesh ({len(mesh_data.faces)} faces) + poses ({len(poses)} frames) "
            f"in {time.time()-t0:.1f}s"
        )

        with zipfile.ZipFile(zpath) as zf:
            n_depth = len(depth_members(zf))
        if n_depth != len(poses):
            raise RuntimeError(f"{seq}: depth frame count {n_depth} != pose count {len(poses)}")

        depth_iter = stream_raw_depth_frames_from_zip(zpath)

        t0 = time.time()
        result = run_oracle_sequence(
            wp, device,
            mesh_data.vertices, mesh_data.faces,
            poses, depth_iter, cam_rays, vignette_mask, TAUS,
        )
        t_run = time.time() - t0
        log(f"{seq}: oracle run over {result.n_frames} frames in {t_run:.1f}s")

        gt_observed = mesh_data.face_observed
        ignore_set = compute_ignore_set(gt_observed, result.ever_evaluable_hit)

        seq_out = {
            "n_faces": result.n_faces,
            "n_frames": result.n_frames,
            "n_rays_total": result.n_rays_total,
            "ray_miss_count": result.ray_miss_count,
            "ray_miss_frac": result.ray_miss_count / result.n_rays_total,
            "evaluable_count": result.evaluable_count,
            "evaluable_frac": result.evaluable_count / result.n_rays_total,
            "d_pred_unavailable_count": result.d_pred_unavailable_count,
            "ignore_set_size": int(ignore_set.sum()),
            "n_gt_observed": int(gt_observed.sum()),
            "by_tau": {},
        }

        for tau in TAUS:
            pred = result.predicted_observed[tau]
            iou_eval = face_set_iou(pred, result.ever_evaluable_hit)
            iou_released = face_set_iou(pred, gt_observed)
            reject = result.tau_reject_count[tau]
            usable = result.evaluable_count - result.d_pred_unavailable_count
            reject_frac = reject / usable if usable else float("nan")

            seq_out["by_tau"][str(tau)] = {
                "iou_vs_evaluable_gt_rasterization": iou_eval,
                "iou_vs_released_gt": iou_released,
                "tau_reject_count": reject,
                "tau_reject_frac_of_usable": reject_frac,
                "n_predicted_observed_faces": int(pred.sum()),
            }

            status = "PASS" if iou_eval >= IOU_TARGET else "FAIL"
            log(
                f"  tau={tau}: IoU vs evaluable-GT-rasterization={iou_eval:.6f} "
                f"(target >={IOU_TARGET}: {status}) | IoU vs released GT={iou_released:.6f} | "
                f"tau_reject={reject}/{usable} usable pixels ({reject_frac*100:.5f}%)"
            )

        ceiling = part1.get(seq, {}).get("max_face_iou")
        seq_out["part1_measured_ceiling"] = ceiling
        log(f"  Part 1 measured ceiling (oracle vs released GT, structural gap only): {ceiling}")
        log(
            f"  ray_miss_frac={seq_out['ray_miss_frac']*100:.5f}% "
            f"evaluable_frac={seq_out['evaluable_frac']*100:.5f}% "
            f"d_pred_unavailable={result.d_pred_unavailable_count} "
            f"ignore_set_size={seq_out['ignore_set_size']} ({seq_out['ignore_set_size']/result.n_faces*100:.4f}% of faces)"
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
        try:
            import torch

            if torch.cuda.is_available():
                log(f"peak allocated (torch, informational): {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
        except ImportError:
            pass
        log(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
