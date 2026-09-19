#!/usr/bin/env python
"""CPU ray-mesh intersection throughput benchmark (per COMPUTE POLICY: default
to CPU/embree via trimesh, multi-process over frames, no GPU unless CPU is
shown to be clearly too slow and the user has agreed to switch).

Measures the core per-pixel ray-cast primitive (one ray per pixel, first-hit
triangle index -- the same primitive the original C3VDv3 renderer's
`primID != -1` coverage test uses, see docs/conventions.md sec 4) on one
representative sequence, single- and multi-process, and extrapolates wall
time for all 169 registered sequences.

Requires `embreex` (installed into scratch/.venv; trimesh's ray_pyembree
backend silently falls back to a much slower pure-Python intersector without
it -- confirmed this was NOT installed before this benchmark and fixed it,
see chat log).
"""
from __future__ import annotations

import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from geometry.camera import CameraIntrinsics, unproject  # noqa: E402
from geometry.coverage_mesh import load_coverage_mesh  # noqa: E402
from geometry.pose import load_poses, transform_points  # noqa: E402

SEQUENCE_DIR = Path("scratch/c1_cecum_t1_v1")
INTRINSICS_PATH = "/data1_ycao/chua/datasets/C3VDv2/camera_intrinsics.txt"

_INTERSECTOR = None  # set once in the parent process before forking workers


def build_intersector():
    global _INTERSECTOR
    mesh_data = load_coverage_mesh(SEQUENCE_DIR / "coverage_mesh.obj")
    mesh = trimesh.Trimesh(vertices=mesh_data.vertices, faces=mesh_data.faces, process=False)
    _INTERSECTOR = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)
    return _INTERSECTOR, mesh_data


def frame_rays(intr: CameraIntrinsics, pose_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """All-pixel ray origins + directions in world space for one frame."""
    cols, rows = np.meshgrid(np.arange(intr.width), np.arange(intr.height))
    px = np.stack([cols.ravel(), rows.ravel()], axis=-1).astype(np.float64)
    cam_rays = unproject(px, intr)  # (N,3) unit rays, camera space
    world_rays = transform_points(cam_rays, pose_matrix, "transposed") - transform_points(
        np.zeros((1, 3)), pose_matrix, "transposed"
    )
    world_rays = world_rays / np.linalg.norm(world_rays, axis=-1, keepdims=True)
    origin = transform_points(np.zeros((1, 3)), pose_matrix, "transposed")
    origins = np.tile(origin, (len(px), 1))
    return origins, world_rays


def cast_one_frame(pose_matrix: np.ndarray) -> float:
    """Casts rays for one frame, returns elapsed seconds."""
    intr = CameraIntrinsics.from_file(INTRINSICS_PATH)
    origins, dirs = frame_rays(intr, pose_matrix)
    t0 = time.time()
    tri_idx = _INTERSECTOR.intersects_first(origins, dirs)
    dt = time.time() - t0
    hit_frac = float((tri_idx >= 0).mean())
    return dt, hit_frac


def _worker_init():
    pass  # intersector inherited via fork copy-on-write, not rebuilt


def _worker_cast(pose_matrix):
    return cast_one_frame(pose_matrix)


def main():
    print(f"embree backend: {trimesh.ray.ray_pyembree}")
    print(f"CPU count available: {mp.cpu_count()}")

    print(f"\nbuilding intersector for {SEQUENCE_DIR} ...")
    t0 = time.time()
    intersector, mesh_data = build_intersector()
    print(f"  {len(mesh_data.faces)} faces, built in {time.time()-t0:.2f}s")

    poses = load_poses(SEQUENCE_DIR / "pose.txt")
    n_frames = len(poses)
    print(f"  {n_frames} frames in this sequence")

    # Single-process throughput: time N sample frames sequentially.
    n_sample = min(10, n_frames)
    sample_idx = np.linspace(0, n_frames - 1, n_sample).astype(int)
    times = []
    for i in sample_idx:
        dt, hit_frac = cast_one_frame(poses[i])
        times.append(dt)
    single_proc_fps = n_sample / sum(times)
    print(f"\nsingle-process: {n_sample} frames in {sum(times):.2f}s -> {single_proc_fps:.2f} frames/sec")
    print(f"  (last frame hit fraction: {hit_frac:.3f} -- sanity check, should be well under 1.0)")

    # Multi-process throughput: fork a pool, each worker inherits the
    # already-built embree scene via copy-on-write (no per-worker rebuild).
    n_workers = min(16, mp.cpu_count())
    t0 = time.time()
    with mp.get_context("fork").Pool(n_workers) as pool:
        results = pool.map(_worker_cast, [poses[i] for i in sample_idx])
    dt_total = time.time() - t0
    multi_proc_fps = n_sample / dt_total
    print(f"\nmulti-process ({n_workers} workers): {n_sample} frames in {dt_total:.2f}s "
          f"-> {multi_proc_fps:.2f} frames/sec (wall clock)")

    # Extrapolate.
    index_df = pd.read_csv("results/coverage_stats.csv")
    total_frames_169 = int(index_df["Total Frames"].sum())
    n_sequences = len(index_df)
    print(f"\n{n_sequences} registered sequences, {total_frames_169} total frames")

    for label, fps in [("single-process", single_proc_fps), (f"multi-process ({n_workers}w)", multi_proc_fps)]:
        eta_sec = total_frames_169 / fps
        print(f"  {label}: {fps:.1f} fps -> est. {eta_sec/60:.1f} min ({eta_sec/3600:.2f} hr) for all {n_sequences} sequences")


if __name__ == "__main__":
    main()
