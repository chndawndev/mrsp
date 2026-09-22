"""Unit tests for src/gt/rasterizer.py's camera-frame Z-depth output.

Runs on the CPU Warp device (no GPU required for CI) against small
synthetic meshes with known, hand-computable geometry -- not real C3VDv2
data. The real-data cross-check against GT depth (docs/eval_protocol.md
section 6 Step 0) lives in
scratch/pipelines/eval_protocol_stage1_validation.py, since that needs the
dataset archives and a GPU; this file only checks the coordinate
conversion itself (a "unit test for every coordinate or unit conversion",
per CLAUDE.md code style).
"""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from gt.rasterizer import build_mesh, cast_frame, make_raycast_frame_kernel, pose_axes  # noqa: E402

wp = pytest.importorskip("warp")

DEVICE = "cpu"


@pytest.fixture(scope="module")
def kernel():
    wp.init()
    return make_raycast_frame_kernel(wp)


def _big_z_plane_mesh(z: float, half_extent: float = 1000.0) -> tuple[np.ndarray, np.ndarray]:
    """A single large triangle in the world-frame plane Z=z, big enough
    that any ray aimed roughly at the origin's neighborhood hits it."""
    verts = np.array(
        [
            [-half_extent, -half_extent, z],
            [half_extent, -half_extent, z],
            [0.0, half_extent, z],
        ],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    return verts, faces


def test_identity_pose_straight_hit(kernel):
    """Camera at world origin, identity rotation, single ray straight down
    +Z. Hits a plane at world Z=10 -- camera-frame Z-depth must be exactly
    10 (t == 10, dir == (0,0,1) is already unit and axis-aligned, so this
    doesn't even exercise the dot-product formula, just the ray-mesh
    intersection + kernel wiring)."""
    verts, faces = _big_z_plane_mesh(z=10.0)
    wp_mesh = build_mesh(wp, verts, faces, DEVICE)

    cam_rays = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)
    R_c2w = np.eye(3)
    T_c2w = np.zeros(3)
    observed_gpu = wp.zeros(len(faces), dtype=wp.int32, device=DEVICE)

    hit_gpu, face_gpu, dist_gpu = cast_frame(
        wp, DEVICE, kernel, wp_mesh, cam_rays, R_c2w, T_c2w, observed_gpu
    )
    wp.synchronize()
    hit, face, dist = hit_gpu.numpy(), face_gpu.numpy(), dist_gpu.numpy()

    assert hit[0] == 1
    assert face[0] == 0
    assert dist[0] == pytest.approx(10.0, abs=1e-4)


def test_rotated_translated_pose_matches_independent_formula(kernel):
    """A non-trivial pose (90 deg rotation about world Y, plus a
    translation) and an off-axis ray. The mesh is a large triangle placed
    exactly perpendicular to the ray's own world-frame direction, passing
    through a point chosen `t_chosen` along that ray -- this pins down
    `hit_world` analytically (independent of the mesh/BVH), regardless of
    which way the rotated camera happens to be facing.

    The expected camera-frame Z-depth is computed two ways and must agree:
      (1) this test's own independent computation:
          depth = dot(hit_world - origin, R_c2w[:, 2]), built from scratch
          (not reusing rasterizer.pose_axes).
      (2) the kernel's reported dist.
    This is the algebraic identity (R.T @ rel)[2] == rel . R[:, 2], used
    to catch a wiring bug in the kernel (wrong row/column, sign, argument
    order, Warp API misuse) -- not a re-derivation of the identity itself,
    which docs/eval_protocol.md section 6 Step 0's real-GT-depth check
    (scratch/pipelines/eval_protocol_stage1_validation.py) validates
    end-to-end against real GT depth.
    """
    theta = np.pi / 2  # 90 degrees about world Y
    R_c2w = np.array(
        [
            [np.cos(theta), 0.0, np.sin(theta)],
            [0.0, 1.0, 0.0],
            [-np.sin(theta), 0.0, np.cos(theta)],
        ]
    )
    T_c2w = np.array([5.0, -3.0, 20.0])

    # Camera-space ray, slightly off-axis (not a pure +Z ray, so the dot
    # product actually does nontrivial work).
    cam_ray = np.array([0.15, -0.1, 1.0])
    cam_ray_unit = cam_ray / np.linalg.norm(cam_ray)

    world_dir = R_c2w @ cam_ray_unit
    origin = T_c2w

    # Pin hit_world exactly, independent of the kernel/BVH: a point
    # t_chosen along the ray's own world-frame direction.
    t_chosen = 25.0
    hit_world = origin + t_chosen * world_dir
    expected_depth = np.dot(hit_world - origin, R_c2w[:, 2])

    # A large triangle through hit_world, perpendicular to world_dir (so
    # the ray -- traveling exactly along world_dir -- meets it at exactly
    # hit_world, t=t_chosen, and nowhere else).
    arbitrary = np.array([1.0, 0.0, 0.0]) if abs(world_dir[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(world_dir, arbitrary)
    u /= np.linalg.norm(u)
    v = np.cross(world_dir, u)
    extent = 1000.0
    verts = np.array(
        [
            hit_world - extent * u - extent * v,
            hit_world + extent * u - extent * v,
            hit_world + extent * v,
        ]
    )
    faces = np.array([[0, 1, 2]], dtype=np.int64)

    wp_mesh = build_mesh(wp, verts, faces, DEVICE)
    cam_rays = cam_ray_unit[None, :]
    observed_gpu = wp.zeros(len(faces), dtype=wp.int32, device=DEVICE)

    hit_gpu, face_gpu, dist_gpu = cast_frame(
        wp, DEVICE, kernel, wp_mesh, cam_rays, R_c2w, T_c2w, observed_gpu
    )
    wp.synchronize()
    hit, dist = hit_gpu.numpy(), dist_gpu.numpy()

    assert hit[0] == 1
    assert dist[0] == pytest.approx(expected_depth, abs=1e-3)
    assert dist[0] == pytest.approx(t_chosen * np.dot(world_dir, R_c2w[:, 2]), abs=1e-3)


def test_miss_reports_no_hit(kernel):
    """A ray aimed away from all geometry: hit=False, face=-1, dist=0.0."""
    verts, faces = _big_z_plane_mesh(z=10.0, half_extent=1.0)
    wp_mesh = build_mesh(wp, verts, faces, DEVICE)

    cam_rays = np.array([[0.0, 0.0, -1.0]], dtype=np.float64)  # pointing away
    R_c2w = np.eye(3)
    T_c2w = np.zeros(3)
    observed_gpu = wp.zeros(len(faces), dtype=wp.int32, device=DEVICE)

    hit_gpu, face_gpu, dist_gpu = cast_frame(
        wp, DEVICE, kernel, wp_mesh, cam_rays, R_c2w, T_c2w, observed_gpu
    )
    wp.synchronize()
    hit, face, dist = hit_gpu.numpy(), face_gpu.numpy(), dist_gpu.numpy()

    assert hit[0] == 0
    assert face[0] == -1
    assert dist[0] == pytest.approx(0.0)


def test_pose_axes_matches_gt_raw_matrix_shortcut():
    """pose_axes(R_c2w, T_c2w) with R_c2w = M[:3,:3].T, T_c2w = M[3,:3]
    (the GT "transposed" convention, CLAUDE.md) must reproduce
    scripts/visibility_full.py's verified raw-matrix shortcut:
    world_dir_matrix == M[:3, :3] and z_axis == M[2, :3]."""
    rng = np.random.default_rng(0)
    M = rng.normal(size=(4, 4))

    R_c2w = M[:3, :3].T
    T_c2w = M[3, :3]

    world_dir_matrix, origin, z_axis = pose_axes(R_c2w, T_c2w)

    np.testing.assert_allclose(world_dir_matrix, M[:3, :3].astype(np.float32), atol=1e-6)
    np.testing.assert_allclose(np.array(origin), M[3, :3].astype(np.float32), atol=1e-6)
    np.testing.assert_allclose(np.array(z_axis), M[2, :3].astype(np.float32), atol=1e-6)
