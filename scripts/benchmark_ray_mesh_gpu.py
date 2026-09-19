#!/usr/bin/env python
"""GPU ray-mesh intersection throughput test, for direct comparison against
scripts/benchmark_ray_mesh.py's CPU/embree numbers. Uses NVIDIA Warp
(wp.Mesh + mesh_query_ray), which builds a GPU BVH -- a fair comparison to
embree's CPU BVH, unlike a brute-force all-pairs approach (infeasible here:
~700k faces x 1.46M rays/frame).

Run only with the user's explicit go-ahead (COMPUTE POLICY). Implements every
required guard:
  - logs a timestamped nvidia-smi snapshot before touching the GPU
  - requires an explicit --gpu-index (never relies on the default device);
    sets CUDA_VISIBLE_DEVICES to exactly that one index
  - logs the chosen index and free memory at launch
  - on any CUDA error (including OOM), logs: timestamp, full nvidia-smi
    output at failure time, this process's own GPU memory (from
    nvidia-smi --query-compute-apps, filtered to our PID), and the
    exception -- then exits non-zero without retrying at a smaller size
  - touches exactly one GPU, never more
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

LOG_PATH = Path("logs/gpu_benchmark.log")


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def nvidia_smi_snapshot(label: str) -> str:
    out = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
    log(f"nvidia-smi snapshot ({label}):\n{out}")
    return out


def this_process_gpu_memory() -> str:
    """Our own process's GPU memory usage, per nvidia-smi's own per-process accounting."""
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv"],
        capture_output=True,
        text=True,
    ).stdout
    my_pid = os.getpid()
    lines = [l for l in out.splitlines() if l.strip().startswith(str(my_pid))]
    return f"pid={my_pid}: " + ("; ".join(lines) if lines else "no compute-app entry found for this pid")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-index", type=int, required=True, help="explicit physical GPU index to use (required, no default device)")
    ap.add_argument("--n-sample-frames", type=int, default=10)
    args = ap.parse_args()

    log("=== GPU ray-mesh benchmark starting ===")
    pre_snapshot = nvidia_smi_snapshot("before launch")

    # Confirm the chosen index actually has free memory (structured query, not table-scraping).
    free_mib = None
    query = subprocess.run(
        ["nvidia-smi", f"--query-gpu=index,memory.used,memory.total,utilization.gpu",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    ).stdout
    for line in query.strip().splitlines():
        idx, used, total, util = [p.strip() for p in line.split(",")]
        if int(idx) == args.gpu_index:
            free_mib = int(total) - int(used)
            log(f"chosen GPU index {args.gpu_index}: {free_mib} MiB free of {total} MiB, {util}% utilized (at launch)")
    if free_mib is None:
        log(f"ERROR: GPU index {args.gpu_index} not found in nvidia-smi output")
        return 1

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
    log(f"CUDA_VISIBLE_DEVICES set to '{args.gpu_index}' explicitly (single GPU only)")

    try:
        return run_benchmark(args)
    except Exception as exc:
        log("=== CUDA/GPU ERROR ===")
        log(f"exception: {type(exc).__name__}: {exc}")
        log("traceback:\n" + traceback.format_exc())
        nvidia_smi_snapshot("at failure time")
        log(f"this process's GPU memory at failure: {this_process_gpu_memory()}")
        log("NOT retrying with a smaller batch. Exiting non-zero for the user to decide next steps.")
        return 1


def run_benchmark(args) -> int:
    import warp as wp

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from geometry.camera import CameraIntrinsics, unproject
    from geometry.coverage_mesh import load_coverage_mesh
    from geometry.pose import load_poses, transform_points

    wp.init()
    device = "cuda:0"  # the only CUDA device visible, since CUDA_VISIBLE_DEVICES pins it
    log(f"warp devices visible: {wp.get_devices()}")

    sequence_dir = Path("scratch/c1_cecum_t1_v1")
    mesh_data = load_coverage_mesh(sequence_dir / "coverage_mesh.obj")
    log(f"mesh: {len(mesh_data.faces)} faces")

    points = wp.array(mesh_data.vertices.astype(np.float32), dtype=wp.vec3, device=device)
    indices = wp.array(mesh_data.faces.astype(np.int32).ravel(), dtype=wp.int32, device=device)
    mesh = wp.Mesh(points=points, indices=indices)
    log(f"GPU BVH built. free_memory now: {wp.get_device(device).free_memory/1e9:.2f} GB "
        f"(of {wp.get_device(device).total_memory/1e9:.2f} GB total)")

    @wp.kernel
    def raycast_kernel(
        mesh_id: wp.uint64,
        origins: wp.array(dtype=wp.vec3),
        dirs: wp.array(dtype=wp.vec3),
        out_face: wp.array(dtype=wp.int32),
    ):
        tid = wp.tid()
        query = wp.mesh_query_ray(mesh_id, origins[tid], dirs[tid], 1.0e6)
        if query.result:
            out_face[tid] = query.face
        else:
            out_face[tid] = -1

    intr = CameraIntrinsics.from_file("/data1_ycao/chua/datasets/C3VDv2/camera_intrinsics.txt")
    poses = load_poses(sequence_dir / "pose.txt")
    n_frames = len(poses)
    cols, rows = np.meshgrid(np.arange(intr.width), np.arange(intr.height))
    px = np.stack([cols.ravel(), rows.ravel()], axis=-1).astype(np.float64)
    cam_rays = unproject(px, intr)
    n_rays = len(px)

    out_face = wp.zeros(n_rays, dtype=wp.int32, device=device)

    sample_idx = np.linspace(0, n_frames - 1, args.n_sample_frames).astype(int)
    times = []
    for i in sample_idx:
        M = poses[i]
        world_rays = transform_points(cam_rays, M, "transposed") - transform_points(np.zeros((1, 3)), M, "transposed")
        world_rays = world_rays / np.linalg.norm(world_rays, axis=-1, keepdims=True)
        origin = transform_points(np.zeros((1, 3)), M, "transposed")
        origins_np = np.tile(origin, (n_rays, 1)).astype(np.float32)
        dirs_np = world_rays.astype(np.float32)

        # Timed region includes H2D transfer of the ray buffers (fair
        # comparison to the CPU benchmark, where rays are already in host
        # memory and only intersects_first() itself is timed).
        wp.synchronize()
        t0 = time.time()
        origins_wp = wp.array(origins_np, dtype=wp.vec3, device=device)
        dirs_wp = wp.array(dirs_np, dtype=wp.vec3, device=device)
        wp.launch(raycast_kernel, dim=n_rays, inputs=[mesh.id, origins_wp, dirs_wp, out_face], device=device)
        wp.synchronize()
        dt = time.time() - t0
        times.append(dt)

        hit_frac = float((out_face.numpy() >= 0).mean())
        log(f"  frame {i}: {dt*1000:.1f}ms, hit_frac={hit_frac:.4f}")

    fps = len(sample_idx) / sum(times)
    log(f"\nGPU (index {args.gpu_index}) throughput: {fps:.1f} frames/sec")

    import pandas as pd

    index_df = pd.read_csv("results/coverage_stats.csv")
    total_frames_169 = int(index_df["Total Frames"].sum())
    n_sequences = len(index_df)
    eta_sec = total_frames_169 / fps
    log(f"{n_sequences} sequences, {total_frames_169} total frames -> "
        f"est. {eta_sec/60:.1f} min ({eta_sec/3600:.2f} hr) for all sequences on this single GPU")

    log("=== GPU benchmark finished successfully ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
