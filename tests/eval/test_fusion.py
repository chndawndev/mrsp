"""Unit tests for src/eval/fusion.py's run_pose_variant_sequence: tau
boundary, ray-miss/d_pred-unavailable bookkeeping, sharing one ray-cast
across multiple depth sources, and -- critically -- a direct consistency
check against src/eval/oracle.py::run_oracle_sequence on the identical
synthetic scenario, since fusion.py duplicates that locked module's logic
rather than refactoring it (module docstring).
"""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from eval.fusion import run_pose_variant_sequence  # noqa: E402
from eval.oracle import run_oracle_sequence  # noqa: E402
from gt.depth import depth_to_mm  # noqa: E402

wp = pytest.importorskip("warp")

DEVICE = "cpu"


@pytest.fixture(scope="module")
def warp_ready():
    wp.init()


def _single_triangle_plane(z: float, half_extent: float = 1000.0):
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


def test_fusion_boundary_and_bookkeeping(warp_ready):
    cam_rays = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]], dtype=np.float64)
    vertices, faces = _single_triangle_plane(z=50.0)

    R_c2w = np.eye(3)
    T_c2w = np.zeros(3)

    def pose_provider(frame_idx):
        return R_c2w, T_c2w

    raw_near_50 = int(round(50.0 / 100.0 * 65535))
    raw_frame1 = int(round(57.0 / 100.0 * 65535))
    raw_unavailable = 0
    raw_by_frame = [raw_near_50, raw_frame1, raw_unavailable]

    def depth_provider(frame_idx):
        raw = np.array([raw_by_frame[frame_idx], 12345], dtype=np.uint16)
        return depth_to_mm(raw)

    d_hit_ray0 = 50.0
    d_pred_frame1, valid = depth_to_mm(np.array([raw_frame1], dtype=np.uint16))
    boundary_tau = float(abs(d_pred_frame1[0] - d_hit_ray0) / d_hit_ray0)
    taus = [boundary_tau, boundary_tau + 1e-6]

    vignette_mask = np.array([False, False])

    results = run_pose_variant_sequence(
        wp, DEVICE, vertices, faces, 3, pose_provider, {"depth": depth_provider}, cam_rays, vignette_mask, taus
    )
    result = results["depth"]

    assert result.n_faces == 1
    assert result.n_frames == 3
    assert result.ray_miss_count == 3
    assert result.evaluable_count == 3
    assert result.d_pred_unavailable_count == 1
    assert result.tau_reject_count[boundary_tau] == 1
    assert result.predicted_observed[boundary_tau][0] == True  # noqa: E712
    assert result.tau_reject_count[boundary_tau + 1e-6] == 0
    assert result.predicted_observed[boundary_tau + 1e-6][0] == True  # noqa: E712


def test_fusion_matches_oracle_on_identical_scenario(warp_ready):
    """The critical cross-check: feeding fusion.py GT-equivalent pose/depth
    providers on the exact same synthetic scenario as
    tests/eval/test_oracle.py::test_oracle_sequence_boundary_and_bookkeeping
    must reproduce oracle.py's results exactly."""
    cam_rays = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]], dtype=np.float64)
    vertices, faces = _single_triangle_plane(z=50.0)
    identity_pose = np.eye(4)
    poses = np.stack([identity_pose, identity_pose, identity_pose])

    raw_near_50 = int(round(50.0 / 100.0 * 65535))
    raw_frame1 = int(round(57.0 / 100.0 * 65535))
    raw_unavailable = 0
    depth_frames = [
        np.array([raw_near_50, 12345], dtype=np.uint16),
        np.array([raw_frame1, 12345], dtype=np.uint16),
        np.array([raw_unavailable, 12345], dtype=np.uint16),
    ]

    d_pred_frame1, _ = depth_to_mm(np.array([raw_frame1], dtype=np.uint16))
    boundary_tau = float(abs(d_pred_frame1[0] - 50.0) / 50.0)
    taus = [boundary_tau, boundary_tau + 1e-6]
    vignette_mask = np.array([False, False])

    oracle_result = run_oracle_sequence(
        wp, DEVICE, vertices, faces, poses, depth_frames, cam_rays, vignette_mask, taus
    )

    def pose_provider(frame_idx):
        M = poses[frame_idx]
        return M[:3, :3].T, M[3, :3]

    def depth_provider(frame_idx):
        return depth_to_mm(depth_frames[frame_idx])

    fusion_results = run_pose_variant_sequence(
        wp, DEVICE, vertices, faces, 3, pose_provider, {"gt": depth_provider}, cam_rays, vignette_mask, taus
    )
    fusion_result = fusion_results["gt"]

    assert fusion_result.ray_miss_count == oracle_result.ray_miss_count
    assert fusion_result.evaluable_count == oracle_result.evaluable_count
    assert fusion_result.d_pred_unavailable_count == oracle_result.d_pred_unavailable_count
    for tau in taus:
        assert fusion_result.tau_reject_count[tau] == oracle_result.tau_reject_count[tau]
        np.testing.assert_array_equal(
            fusion_result.predicted_observed[tau], oracle_result.predicted_observed[tau]
        )


def test_fusion_miss_reports_no_hit_and_no_observed(warp_ready):
    verts, faces = _single_triangle_plane(z=10.0, half_extent=1.0)

    def pose_provider(frame_idx):
        return np.eye(3), np.zeros(3)

    def depth_provider(frame_idx):
        raw = np.array([12345], dtype=np.uint16)
        return depth_to_mm(raw)

    cam_rays = np.array([[0.0, 0.0, -1.0]], dtype=np.float64)  # pointing away
    vignette_mask = np.array([False])

    results = run_pose_variant_sequence(
        wp, DEVICE, verts, faces, 1, pose_provider, {"depth": depth_provider}, cam_rays, vignette_mask, [0.25]
    )
    result = results["depth"]
    assert result.ray_miss_count == 1
    assert result.evaluable_count == 0
    assert result.predicted_observed[0.25][0] == False  # noqa: E712


def test_fusion_two_depth_sources_share_one_raycast(warp_ready):
    """Two depth providers at the same pose: ray_miss_count/evaluable_count
    must be identical across both results (shared ray-cast), while
    tau_reject_count/predicted_observed can differ (different depth
    values feeding the same cached geometry)."""
    cam_rays = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)
    vertices, faces = _single_triangle_plane(z=50.0)

    def pose_provider(frame_idx):
        return np.eye(3), np.zeros(3)

    # depth source A: matches d_hit exactly (50mm) -> passes any tau.
    def depth_a(frame_idx):
        return depth_to_mm(np.array([int(round(50.0 / 100.0 * 65535))], dtype=np.uint16))

    # depth source B: way off (10mm) -> fails a tight tau.
    def depth_b(frame_idx):
        return depth_to_mm(np.array([int(round(10.0 / 100.0 * 65535))], dtype=np.uint16))

    vignette_mask = np.array([False])
    results = run_pose_variant_sequence(
        wp, DEVICE, vertices, faces, 1, pose_provider, {"a": depth_a, "b": depth_b},
        cam_rays, vignette_mask, [0.25],
    )

    assert results["a"].ray_miss_count == results["b"].ray_miss_count == 0
    assert results["a"].evaluable_count == results["b"].evaluable_count == 1
    assert results["a"].predicted_observed[0.25][0] == True  # noqa: E712
    assert results["b"].predicted_observed[0.25][0] == False  # noqa: E712
