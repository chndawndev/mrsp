"""GPU (Warp) mesh ray-casting: per-frame, per-pixel hit/face/distance, for
an arbitrary camera-to-world pose.

Extends `scripts/visibility_full.py`'s accumulate-only rasterizer (GT pose
only, hit/miss + face index only, no distance output) with exactly the two
capabilities approved in `docs/eval_protocol.md` section 7 ("Option 1 ...
approved by Chen") and nothing else:

  (a) per-pixel hit **distance** -- camera-frame Z-depth of the hit, not
      Euclidean ray length, matching `docs/eval_protocol.md` section 2's
      `d_hit = hit_cam[2]` definition -- alongside hit/miss and face index.
  (b) an arbitrary per-frame camera-to-world pose `(R_c2w, T_c2w)`, not
      only the GT trajectory.

`scripts/visibility_full.py` itself is untouched -- this is a new module,
not an edit to that frozen script. The accumulation write inside the
kernel below (`observed[q.face] = 1`, gated on `q.result`) is
character-for-character the same logic as the original
`scripts/visibility_full.py::raycast_kernel`, so driving this kernel with
GT poses across a full sequence must reproduce that script's recorded
per-sequence IoU exactly -- this is the Stage 1 validation performed in
`scratch/pipelines/eval_protocol_stage1_validation.py`.

Pose convention: `(R_c2w, T_c2w)` is the camera-to-world rotation (3, 3)
and translation (3,) -- `CLAUDE.md`'s "transposed" `pose.txt` convention
(`np.array(vals).reshape(4,4).T`) for GT pose, or `docs/eval_protocol.md`
section 1's `R_world_i` / `T_world_i` for an aligned predicted pose. Both
are the same representation; this module does not special-case either,
which is exactly extension (b).

Not renormalized to orthonormal: GT pose rotation submatrices are only
*close* to orthonormal (measured column norm in [0.9993, 1.0009],
`scripts/visibility_full.py`'s docstring), and this module reuses that
submatrix as-is for both the direction transform and the z-axis, to stay
bit-consistent with the already-validated precedent in
`scratch/pipelines/oracle_gap_part1_steps234.py`
(`R = M[:3,:3].T; depth_cam_z = (rel @ R)[:, 2]`, median diff 0.0034mm
against GT depth on the CPU/trimesh path).
"""
from __future__ import annotations

import numpy as np


def build_mesh(wp, vertices: np.ndarray, faces: np.ndarray, device: str):
    """Uploads a coverage mesh to the GPU once; the returned `wp.Mesh` is
    reused across every frame and pose of a sequence."""
    wp_points = wp.array(vertices.astype(np.float32), dtype=wp.vec3, device=device)
    wp_indices = wp.array(faces.astype(np.int32).ravel(), dtype=wp.int32, device=device)
    return wp.Mesh(points=wp_points, indices=wp_indices)


def make_raycast_frame_kernel(wp):
    """Builds the extended ray-cast kernel. Warp compiles kernels at first
    use and caches by source; callers should build this once per process
    and reuse the returned kernel object across frames and sequences,
    exactly as `scripts/visibility_full.py::main` does for its own kernel.
    """

    @wp.kernel
    def raycast_frame_kernel(
        mesh_id: wp.uint64,
        origin: wp.vec3,
        z_axis: wp.vec3,
        dirs: wp.array(dtype=wp.vec3),
        observed: wp.array(dtype=wp.int32),
        hit: wp.array(dtype=wp.int32),
        face: wp.array(dtype=wp.int32),
        dist: wp.array(dtype=wp.float32),
    ):
        tid = wp.tid()
        q = wp.mesh_query_ray(mesh_id, origin, dirs[tid], 1.0e6)
        if q.result:
            # Unchanged from scripts/visibility_full.py::raycast_kernel.
            observed[q.face] = 1
            hit[tid] = 1
            face[tid] = q.face
            # hit_world = origin + q.t * dirs[tid] (Warp's ray parameter t is
            # defined w.r.t. dirs[tid] regardless of its magnitude, so this
            # holds even though dirs[tid] is not renormalized after the pose
            # transform below). Camera-frame Z-depth of that point is
            # dot(hit_world - origin, z_axis) = q.t * dot(dirs[tid], z_axis).
            dist[tid] = q.t * wp.dot(dirs[tid], z_axis)
        else:
            hit[tid] = 0
            face[tid] = -1
            dist[tid] = 0.0

    return raycast_frame_kernel


def pose_axes(R_c2w: np.ndarray, T_c2w: np.ndarray) -> tuple[np.ndarray, tuple, tuple]:
    """`(R_c2w, T_c2w)` -> `(world_dir_matrix, origin, z_axis)`.

    `world_dir_matrix` is applied as `cam_rays @ world_dir_matrix` to get
    world-frame ray directions. For GT pose, `R_c2w = M[:3, :3].T` (M the
    raw `pose.txt` reshape), so `world_dir_matrix = R_c2w.T == M[:3, :3]`,
    reproducing `scripts/visibility_full.py`'s already-verified
    `cam_rays @ M[:3, :3]` shortcut exactly. `z_axis = R_c2w[:, 2]` is the
    camera's world-frame Z axis (third column of R_c2w); for GT pose this
    equals `M[2, :3]`, matching the precedent cited in the module
    docstring. Neither is renormalized -- see module docstring.
    """
    R_c2w = np.asarray(R_c2w, dtype=np.float32)
    T_c2w = np.asarray(T_c2w, dtype=np.float32)
    world_dir_matrix = R_c2w.T
    origin = tuple(T_c2w)
    z_axis = tuple(R_c2w[:, 2])
    return world_dir_matrix, origin, z_axis


def cast_frame(
    wp,
    device: str,
    kernel,
    wp_mesh,
    cam_rays: np.ndarray,
    R_c2w: np.ndarray,
    T_c2w: np.ndarray,
    observed_gpu,
):
    """Launches one frame's ray-cast (one ray per row of `cam_rays`) at
    pose `(R_c2w, T_c2w)` against `wp_mesh`, OR-accumulating hit faces into
    `observed_gpu` (a persistent per-face `wp.array(dtype=wp.int32)`,
    reused across frames -- exactly `scripts/visibility_full.py`'s
    accumulation semantics, unaffected by this function's extra outputs).

    Async: does not synchronize or read back. Callers doing an
    accumulate-only run (many frames, only `observed_gpu` needed at the
    end) should call `wp.synchronize()` once after the whole frame loop,
    exactly as `scripts/visibility_full.py::process_sequence` does, and
    never call `.numpy()` on this call's `hit_gpu`/`face_gpu`/`dist_gpu`
    unless that frame's per-pixel output is actually needed -- avoids
    paying a host-transfer cost on frames where only the accumulated face
    set matters.

    Returns `(hit_gpu, face_gpu, dist_gpu)`, Warp device arrays for this
    frame only: `hit` (int32, 1/0), `face` (int32, -1 on miss), `dist`
    (float32, camera-frame Z-depth, 0.0 on miss).
    """
    n_rays = len(cam_rays)
    world_dir_matrix, origin, z_axis = pose_axes(R_c2w, T_c2w)
    world_dirs = cam_rays.astype(np.float32) @ world_dir_matrix

    dirs_wp = wp.array(world_dirs, dtype=wp.vec3, device=device)
    hit_gpu = wp.zeros(n_rays, dtype=wp.int32, device=device)
    face_gpu = wp.zeros(n_rays, dtype=wp.int32, device=device)
    dist_gpu = wp.zeros(n_rays, dtype=wp.float32, device=device)

    wp.launch(
        kernel,
        dim=n_rays,
        inputs=[wp_mesh.id, origin, z_axis, dirs_wp, observed_gpu, hit_gpu, face_gpu, dist_gpu],
        device=device,
    )
    return hit_gpu, face_gpu, dist_gpu
