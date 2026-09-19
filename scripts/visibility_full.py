#!/usr/bin/env python
"""Full 169-sequence GPU visibility rasterization run.

For each sequence (discover_sequences() from scripts/coverage_stats.py, the
same zip-only, asserted-169 discovery used throughout): stream the mesh and
poses, build a GPU BVH (NVIDIA Warp), cast one ray per pixel at full
resolution (stride 1) for every frame, accumulate a per-face "observed"
boolean mask entirely on the GPU (only the final per-sequence mask is read
back, not per-frame images), and compare against the released
coverage_mesh.obj's vt-based observed set.

Validated in docs/gpu_validation.md before this run (backend equivalence,
GT miss-fraction investigation, MAX_DEPTH citation, accuracy vs released
mesh at strides 4/2/1, end-to-end per-sequence cost). Not modifying any
metric definition here -- same IoU/false_observed/false_unobserved formula
used throughout (scripts/oracle_check.py, scripts/validate_gpu_rasterizer.py).

Usage: run under tmux, per instructions -- see logs/visibility_full.log.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from coverage_stats import discover_sequences  # noqa: E402
from gpu_status import get_gpu_stats  # noqa: E402
from geometry.camera import CameraIntrinsics  # noqa: E402
from geometry.coverage_mesh import stream_coverage_mesh_from_zip  # noqa: E402
from geometry.pose import stream_poses_from_zip  # noqa: E402

OUT_DIR = Path("results/visibility_full")
LOG_PATH = Path("logs/visibility_full.log")
INTRINSICS_PATH = "/data1_ycao/chua/datasets/C3VDv2/camera_intrinsics.txt"
IOU_FLAG_THRESHOLD = 0.99


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def select_gpu() -> int:
    """Re-checks GPU availability via scripts/gpu_status.py at launch (its
    own recommendation logic, imported directly rather than re-implemented),
    logs the full table + selection."""
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


def record_environment(gpu_index: int, gpu_name: str) -> dict:
    import warp as wp

    commit_hash = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=Path(__file__).resolve().parent.parent
    ).stdout.strip()
    env = {
        "warp_version": wp.config.version,
        "gpu_index": gpu_index,
        "gpu_model": gpu_name,
        "commit_hash": commit_hash,
        "kernel_cache_note": (
            "Warp's compiled-kernel cache is machine-level "
            f"(this run's cache: ~/.cache/warp/{wp.config.version}). "
            "On a new machine the first sequence pays a one-time JIT "
            "compile cost (~1s observed); subsequent sequences load from cache."
        ),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    log(f"Environment: {json.dumps(env, indent=2)}")
    return env


def process_sequence(name: str, path: Path, wp, device: str, raycast_kernel, cam_rays: np.ndarray) -> dict:
    """cam_rays: (n_rays, 3) float32 unit camera-space ray directions, precomputed
    once for the whole run (depends only on the fixed camera_intrinsics.txt, not
    on the sequence). Per-frame math uses the rotation-submatrix shortcut,
    verified equivalent to the full transform_points()-based approach (see
    chat log): for the "transposed" pose interpretation, world_dir = cam_ray @
    M[:3,:3] and world_origin = M[3,:3] exactly (origin diff 0.0 verified;
    direction diff 3.4e-14 after normalizing both -- pure float roundoff).
    Skipping the post-rotation renormalization is safe here because
    ray-mesh intersection depends only on direction, not magnitude, and
    M[:3,:3] is very close to orthonormal (measured norm in [0.9993, 1.0009]).
    """
    t0 = time.time()
    mesh_data = stream_coverage_mesh_from_zip(path)
    poses = stream_poses_from_zip(path)
    t_stream = time.time() - t0

    n_faces = len(mesh_data.faces)
    gt_observed = mesh_data.face_observed
    n_rays = len(cam_rays)

    t0 = time.time()
    wp_points = wp.array(mesh_data.vertices.astype(np.float32), dtype=wp.vec3, device=device)
    wp_indices = wp.array(mesh_data.faces.astype(np.int32).ravel(), dtype=wp.int32, device=device)
    wp_mesh = wp.Mesh(points=wp_points, indices=wp_indices)
    observed_gpu = wp.zeros(n_faces, dtype=wp.int32, device=device)
    t_bvh = time.time() - t0

    t0 = time.time()
    for M in poses:
        world_dirs = (cam_rays @ M[:3, :3].astype(np.float32))
        origin = tuple(M[3, :3].astype(np.float32))
        dirs_wp = wp.array(world_dirs, dtype=wp.vec3, device=device)
        wp.launch(raycast_kernel, dim=n_rays, inputs=[wp_mesh.id, origin, dirs_wp, observed_gpu], device=device)
    wp.synchronize()
    t_cast = time.time() - t0

    pred_observed = observed_gpu.numpy().astype(bool)

    inter = np.sum(pred_observed & gt_observed)
    union = np.sum(pred_observed | gt_observed)
    iou = float(inter / union) if union else float("nan")
    false_observed = int(np.sum(pred_observed & ~gt_observed))
    false_unobserved = int(np.sum(~pred_observed & gt_observed))

    packed = np.packbits(pred_observed)

    return {
        "metrics": {
            "Video Name": name,
            "n_faces": n_faces,
            "n_frames": len(poses),
            "iou": iou,
            "false_observed": false_observed,
            "false_unobserved": false_unobserved,
            "n_gt_observed": int(gt_observed.sum()),
            "n_pred_observed": int(pred_observed.sum()),
            "t_stream_s": t_stream,
            "t_bvh_s": t_bvh,
            "t_cast_s": t_cast,
        },
        "packed_mask": packed,
        "n_faces_for_unpack": n_faces,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log("=== visibility_full.py starting: full 169-sequence GPU visibility rasterization ===")

    gpu_index = select_gpu()
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    log(f"CUDA_VISIBLE_DEVICES set to '{gpu_index}' explicitly")

    import warp as wp

    wp.init()
    device = "cuda:0"
    gpu_stats = get_gpu_stats()
    gpu_name = next(g["name"] for g in gpu_stats if g["index"] == gpu_index)
    env = record_environment(gpu_index, gpu_name)

    @wp.kernel
    def raycast_kernel(
        mesh_id: wp.uint64,
        origin: wp.vec3,
        dirs: wp.array(dtype=wp.vec3),
        observed: wp.array(dtype=wp.int32),
    ):
        tid = wp.tid()
        q = wp.mesh_query_ray(mesh_id, origin, dirs[tid], 1.0e6)
        if q.result:
            observed[q.face] = 1

    intr = CameraIntrinsics.from_file(INTRINSICS_PATH)
    from geometry.camera import unproject

    cols, rows = np.meshgrid(np.arange(intr.width), np.arange(intr.height))
    px = np.stack([cols.ravel(), rows.ravel()], axis=-1).astype(np.float64)
    cam_rays = unproject(px, intr).astype(np.float32)  # computed once for the whole run
    log(f"precomputed {len(cam_rays)} camera rays (fixed intrinsics, shared across all sequences)")

    seqs = discover_sequences()
    log(f"discovered {len(seqs)} sequences (asserted 169 inside discover_sequences())")

    all_metrics = []
    masks = {}
    t_start = time.time()

    for i, (name, kind, path) in enumerate(seqs):
        assert kind == "zip"
        result = process_sequence(name, path, wp, device, raycast_kernel, cam_rays)
        all_metrics.append(result["metrics"])
        masks[name] = result["packed_mask"]

        if (i + 1) % 10 == 0 or i == len(seqs) - 1:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta_sec = (len(seqs) - (i + 1)) / rate
            m = result["metrics"]
            log(
                f"[{i+1}/{len(seqs)}] {name}: IoU={m['iou']:.5f} false_obs={m['false_observed']} "
                f"false_unobs={m['false_unobserved']} (stream={m['t_stream_s']:.1f}s bvh={m['t_bvh_s']:.2f}s "
                f"cast={m['t_cast_s']:.1f}s) | elapsed={elapsed/60:.1f}min ETA={eta_sec/60:.1f}min"
            )

    metrics_df = pd.DataFrame(all_metrics)
    index_df = pd.read_csv("docs/release_v1.csv").rename(columns={"Video Number ": "Video Number"})
    id_cols = ["Video Name", "Colon", "Segment", "Phantom Number", "Video Number", "Debris", "Open End Visible", "Qualitative Score"]
    metrics_df = metrics_df.merge(index_df[id_cols], on="Video Name", how="left")

    metrics_path = OUT_DIR / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    log(f"wrote {metrics_path} ({len(metrics_df)} rows)")

    masks_path = OUT_DIR / "observed_masks.npz"
    np.savez_compressed(masks_path, **masks)
    log(f"wrote {masks_path}")

    env_path = OUT_DIR / "run_environment.json"
    env["n_sequences"] = len(seqs)
    env["total_elapsed_sec"] = time.time() - t_start
    env_path.write_text(json.dumps(env, indent=2))
    log(f"wrote {env_path}")

    log(f"\nTOTAL elapsed: {(time.time()-t_start)/60:.1f} min for {len(seqs)} sequences")
    log("=== ALL DONE ===")


if __name__ == "__main__":
    main()
