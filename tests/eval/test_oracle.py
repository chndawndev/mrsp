"""End-to-end synthetic test for src/eval/oracle.py's run_oracle_sequence:
tau boundary (strict '<'), ray-miss counting, d_pred_unavailable counting,
evaluable/ignore-set bookkeeping, and OR-accumulation into
predicted_observed -- all on a tiny synthetic 2-ray, 3-frame, 1-face
scenario (CPU Warp device, no GPU/real dataset needed). The real-data
validation (docs/eval_protocol.md section 6) lives in
scripts/eval_oracle.py; this file only checks the wiring/logic.
"""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from eval.ignore_set import compute_ignore_set  # noqa: E402
from eval.oracle import run_oracle_sequence  # noqa: E402
from gt.depth import depth_to_mm  # noqa: E402
from gt.rasterizer import build_mesh  # noqa: E402

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


def test_oracle_sequence_boundary_and_bookkeeping(warp_ready):
    # 2 rays: ray 0 hits the plane straight on (axis-aligned, identity pose
    # -> d_hit exactly 50.0); ray 1 points away and always misses.
    cam_rays = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]], dtype=np.float64)
    vertices, faces = _single_triangle_plane(z=50.0)

    identity_pose = np.eye(4)  # raw "pose.txt" convention: M[:3,:3]=I, M[3,:3]=0
    poses = np.stack([identity_pose, identity_pose, identity_pose])

    # Frame 0: ray0's GT depth reading ~= 50.0 (near-exact match -> passes any tau).
    raw_near_50 = int(round(50.0 / 100.0 * 65535))
    # Frame 1: ray0's GT depth reading picked to exercise the strict '<' boundary.
    raw_frame1 = int(round(57.0 / 100.0 * 65535))
    # Frame 2: ray0's GT depth reading is raw==0 (no reading) -> d_pred_unavailable.
    raw_unavailable = 0

    depth_frames = [
        np.array([raw_near_50, 12345], dtype=np.uint16),
        np.array([raw_frame1, 12345], dtype=np.uint16),
        np.array([raw_unavailable, 12345], dtype=np.uint16),
    ]

    # Compute frame 1's exact relative error the same way oracle.py does,
    # so the boundary tau is bit-consistent with the implementation instead
    # of a hand-picked round number.
    d_hit_ray0 = 50.0
    d_pred_frame1, valid = depth_to_mm(np.array([raw_frame1], dtype=np.uint16))
    assert valid[0]
    boundary_tau = float(abs(d_pred_frame1[0] - d_hit_ray0) / d_hit_ray0)
    taus = [boundary_tau, boundary_tau + 1e-6]

    vignette_mask = np.array([False, False])

    result = run_oracle_sequence(
        wp, DEVICE, vertices, faces, poses, depth_frames, cam_rays, vignette_mask, taus
    )

    assert result.n_faces == 1
    assert result.n_frames == 3
    assert result.ray_miss_count == 3  # ray1 misses every frame
    assert result.evaluable_count == 3  # ray0 evaluable every frame (hit/miss independent of d_pred)
    assert result.d_pred_unavailable_count == 1  # frame 2's ray0 (raw==0)

    # At the exact boundary tau, frame 1's pixel is rejected (strict '<');
    # frame 0's near-exact match still passes -> face 0 still ends up
    # observed (OR-accumulated), but via frame 0, not frame 1.
    assert result.tau_reject_count[boundary_tau] == 1
    assert result.predicted_observed[boundary_tau][0] == True  # noqa: E712

    # Just above the boundary, frame 1 now passes too -> zero rejects.
    assert result.tau_reject_count[boundary_tau + 1e-6] == 0
    assert result.predicted_observed[boundary_tau + 1e-6][0] == True  # noqa: E712

    # ever_evaluable_hit / ignore_set: single face, reached by an evaluable
    # ray (ray0, every frame) -> in ever_evaluable_hit, not in ignore_set
    # (a GT-observed, evaluable-reached face is exactly the non-gap case --
    # src/eval/ignore_set.py's narrow, gt_observed-restricted definition).
    assert result.ever_evaluable_hit[0] == True  # noqa: E712
    gt_observed = np.array([True])
    ignore_set = compute_ignore_set(gt_observed, result.ever_evaluable_hit)
    assert ignore_set[0] == False  # noqa: E712


def test_oracle_sequence_face_never_evaluable_goes_to_ignore_set(warp_ready):
    """A ray that only ever hits the mesh through a vignetted pixel never
    marks its face in ever_evaluable_hit. If the released GT nonetheless
    credits that face as observed (the classic "gap" scenario -- GT-observed
    but never reachable via an evaluable ray), it lands in the ignore set
    (src/eval/ignore_set.py's narrow, gt_observed-restricted definition)."""
    cam_rays = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)
    vertices, faces = _single_triangle_plane(z=50.0)
    identity_pose = np.eye(4)
    poses = np.stack([identity_pose])
    depth_frames = [np.array([int(round(50.0 / 100.0 * 65535))], dtype=np.uint16)]
    vignette_mask = np.array([True])  # the only ray is vignetted

    result = run_oracle_sequence(
        wp, DEVICE, vertices, faces, poses, depth_frames, cam_rays, vignette_mask, [0.25]
    )

    assert result.evaluable_count == 0
    assert result.ever_evaluable_hit[0] == False  # noqa: E712
    gt_observed = np.array([True])
    ignore_set = compute_ignore_set(gt_observed, result.ever_evaluable_hit)
    assert ignore_set[0] == True  # noqa: E712
    assert result.predicted_observed[0.25][0] == False  # noqa: E712
