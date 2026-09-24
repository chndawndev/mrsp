"""Benchmark the actual compute cost of docs/eval_protocol.md's ray-casting
step, before writing any eval code. Measures real embree-accelerated
ray-mesh intersection throughput using the project's own camera model and
coverage_mesh.obj -- not a guess. See docs/eval_protocol.md.

Key design point being validated: ray-casting only depends on (frame,
pose-variant), not on (frame, configuration) or (frame, configuration,
tau) -- GT-pose and predicted-pose configurations share their ray-cast,
and tau is applied post-hoc to the same cached hit distances. This
benchmark measures the actual number of ray-casts needed under that
sharing, not a naive 4x-configs x 4x-tau over-count.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import trimesh

REPO = Path("/data1_ycao/chua/projects/mrsp")
sys.path.insert(0, str(REPO / "src"))
from geometry.camera import CameraIntrinsics, unproject  # noqa: E402
from geometry.coverage_mesh import load_coverage_mesh  # noqa: E402
from geometry.pose import load_poses, transform_points  # noqa: E402

SEQ_DIR = REPO / "scratch" / "c1_cecum_t1_v1"
INTRINSICS_PATH = "/data1_ycao/chua/datasets/C3VDv2/camera_intrinsics.txt"
N_FRAMES = 218
N_SEQUENCES = 169


def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


log("loading intrinsics, mesh, poses ...")
intr = CameraIntrinsics.from_file(INTRINSICS_PATH)
mesh_data = load_coverage_mesh(SEQ_DIR / "coverage_mesh.obj")
mesh = trimesh.Trimesh(vertices=mesh_data.vertices, faces=mesh_data.faces, process=False)
log(f"mesh: {len(mesh_data.vertices)} verts, {len(mesh_data.faces)} faces, "
    f"ray intersector: {type(mesh.ray).__module__}")

gt_poses = load_poses(SEQ_DIR / "pose.txt")

W, H = intr.width, intr.height
n_pixels = W * H
log(f"frame resolution: {W}x{H} = {n_pixels} pixels/frame (full density)")

cols, rows = np.meshgrid(np.arange(W), np.arange(H))
px_grid = np.stack([cols.ravel(), rows.ravel()], axis=-1).astype(np.float64)  # (n_pixels, 2)

# Unit camera-space ray directions for every pixel -- computed once, reused for every frame
# (the camera model/pixel grid doesn't change frame to frame, only the pose does).
cam_rays = unproject(px_grid, intr)  # (n_pixels, 3)
log("computed per-pixel camera-space unit ray directions")


def cast_rays_for_frame(frame_idx: int) -> float:
    M = gt_poses[frame_idx]
    origin = transform_points(np.zeros((1, 3)), M, "transposed")[0]  # camera center, world coords
    # world-space ray directions: rotate cam_rays by the pose's rotation part.
    # transform_points applies the full affine transform; to get direction-only (no translation),
    # transform the ray endpoints and subtract the (already-known) origin.
    endpoints = transform_points(cam_rays, M, "transposed")
    directions = endpoints - origin
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    origins = np.broadcast_to(origin, directions.shape)

    t0 = time.time()
    locations, index_ray, index_tri = mesh.ray.intersects_location(
        origins, directions, multiple_hits=False
    )
    elapsed = time.time() - t0
    hit_fraction = len(index_ray) / n_pixels
    log(f"  frame {frame_idx}: {elapsed:.3f}s, {len(index_ray)}/{n_pixels} rays hit "
        f"({hit_fraction:.4f}), {n_pixels/elapsed/1e6:.2f}M rays/sec")
    return elapsed


log("\n=== timing single-frame full-density ray-cast (GT pose), 3 frames for stability ===")
times = []
for frame_idx in [0, 100, 217]:
    times.append(cast_rays_for_frame(frame_idx))
mean_time_per_frame = float(np.mean(times))
log(f"mean time per frame (full-density ray-cast, one pose variant): {mean_time_per_frame:.3f}s")

# ---- extrapolation ----
log("\n=== extrapolation ===")
# Ray-casting is needed once per (frame, pose-variant). Two pose variants: GT pose
# (shared by oracle + pred-depth-only configs) and aligned-predicted pose (shared by
# fully-predicted + pred-pose-only configs). tau is applied post-hoc to cached hit
# distances -- zero extra ray-casts for the 4-value tau sweep.
pose_variants_per_sequence = 2
raycast_seconds_per_sequence = mean_time_per_frame * N_FRAMES * pose_variants_per_sequence
log(f"per-sequence ray-casting time (218 frames x 2 pose variants): "
    f"{raycast_seconds_per_sequence:.1f}s = {raycast_seconds_per_sequence/60:.2f} min")

naive_4x_configs = mean_time_per_frame * N_FRAMES * 4  # if NOT sharing ray-casts across configs
log(f"(naive, if ray-casting were repeated per-configuration instead of per-pose-variant: "
    f"{naive_4x_configs:.1f}s = {naive_4x_configs/60:.2f} min -- "
    f"{naive_4x_configs/raycast_seconds_per_sequence:.1f}x more than the shared approach)")

corpus_seconds = raycast_seconds_per_sequence * N_SEQUENCES
log(f"corpus extrapolation (169 sequences, ray-casting only): "
    f"{corpus_seconds:.0f}s = {corpus_seconds/60:.1f} min = {corpus_seconds/3600:.2f} hours")

log(f"\nNote: this measures ray-casting only, the dominant cost. Backprojection, tau "
    f"comparison, face-set accumulation, and region-level metrics are cheap elementwise "
    f"numpy operations by comparison, not separately benchmarked here.")
