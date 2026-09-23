"""Unit tests for src/eval/scale_recovery.py: depth scale (median-of-
per-frame-medians), Umeyama similarity fit, and the world-frame pose
anchoring formula.
"""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from eval.scale_recovery import (  # noqa: E402
    aligned_predicted_pose,
    compute_depth_scale,
    compute_pose_alignment,
    umeyama,
)


def test_compute_depth_scale_constant_ratio():
    rng = np.random.default_rng(0)
    true_scale = 177.0
    gt_frames, valid_frames, pred_frames = [], [], []
    for _ in range(5):
        pred = rng.uniform(0.1, 0.3, size=100)
        gt = pred * true_scale
        valid = np.ones(100, dtype=bool)
        gt_frames.append(gt)
        valid_frames.append(valid)
        pred_frames.append(pred)
    result = compute_depth_scale(gt_frames, valid_frames, pred_frames)
    assert result.median == pytest.approx(true_scale, rel=1e-6)
    assert len(result.per_frame_ratios) == 5


def test_compute_depth_scale_ignores_invalid_pixels():
    pred = np.array([0.1, 0.2, 999.0])  # last pixel is garbage
    gt = np.array([10.0, 20.0, -1.0])
    valid = np.array([True, True, False])
    result = compute_depth_scale([gt], [valid], [pred])
    # median(gt[valid])=15.0, median(pred[valid])=0.15000000000000002 -> ratio=100
    assert result.median == pytest.approx(100.0, rel=1e-6)


def test_compute_depth_scale_median_of_medians_not_pooled():
    """Frames with different pixel counts -- median-of-per-frame-ratios
    must differ from a single ratio pooled across all pixels of all
    frames (the explicitly rejected alternative aggregation,
    docs/eval_protocol.md section 1)."""
    gt_frames = [np.array([10.0, 10.0, 10.0]), np.array([100.0]), np.array([100.0])]
    pred_frames = [np.array([1.0, 1.0, 1.0]), np.array([1.0]), np.array([1.0])]
    valid_frames = [np.array([True, True, True]), np.array([True]), np.array([True])]

    result = compute_depth_scale(gt_frames, valid_frames, pred_frames)
    # per-frame ratios: 10.0, 100.0, 100.0 -> median = 100.0
    assert result.median == pytest.approx(100.0)

    # the (rejected) pooled alternative: median(all gt)/median(all pred) = 10.0/1.0 = 10.0
    pooled_gt = np.concatenate(gt_frames)
    pooled_pred = np.concatenate(pred_frames)
    pooled_ratio = np.median(pooled_gt) / np.median(pooled_pred)
    assert pooled_ratio == pytest.approx(10.0)
    assert result.median != pytest.approx(pooled_ratio)


def test_compute_depth_scale_zero_pred_median_raises():
    gt = np.array([10.0, 10.0])
    pred = np.array([0.0, 0.0])
    valid = np.array([True, True])
    with pytest.raises(ValueError):
        compute_depth_scale([gt], [valid], [pred])


def test_compute_depth_scale_no_valid_pixels_raises():
    gt = np.array([10.0, 10.0])
    pred = np.array([1.0, 1.0])
    valid = np.array([False, False])
    with pytest.raises(ValueError):
        compute_depth_scale([gt], [valid], [pred])


def test_umeyama_recovers_known_similarity_transform():
    rng = np.random.default_rng(1)
    src = rng.uniform(-10, 10, size=(50, 3))

    theta = 0.4
    R_true = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    t_true = np.array([5.0, -3.0, 2.0])
    c_true = 177.0

    dst = (c_true * (R_true @ src.T).T) + t_true

    R, t, c = umeyama(src, dst)
    np.testing.assert_allclose(R, R_true, atol=1e-8)
    np.testing.assert_allclose(t, t_true, atol=1e-6)
    assert c == pytest.approx(c_true, rel=1e-8)


def test_compute_pose_alignment_matches_umeyama():
    rng = np.random.default_rng(2)
    src = rng.uniform(-5, 5, size=(30, 3))
    R_true = np.eye(3)
    t_true = np.array([1.0, 1.0, 1.0])
    c_true = 2.0
    dst = c_true * (R_true @ src.T).T + t_true

    alignment = compute_pose_alignment(src, dst)
    assert alignment.s_pose == pytest.approx(c_true, rel=1e-8)
    np.testing.assert_allclose(alignment.t_u, t_true, atol=1e-6)


def test_aligned_predicted_pose_formula():
    R_u = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # 90deg about Z
    t_u = np.array([10.0, 0.0, 0.0])
    s_pose = 3.0
    from eval.scale_recovery import PoseAlignment

    alignment = PoseAlignment(R_u=R_u, t_u=t_u, s_pose=s_pose)

    R_i = np.eye(3)
    t_i = np.array([1.0, 0.0, 0.0])

    R_world, T_world = aligned_predicted_pose(alignment, R_i, t_i)
    np.testing.assert_allclose(R_world, R_u @ R_i)
    expected_T = s_pose * (R_u @ t_i) + t_u
    np.testing.assert_allclose(T_world, expected_T)
    # explicit expected value: R_u @ [1,0,0] = [0,1,0]; s_pose*[0,1,0]=[0,3,0]; +t_u=[10,3,0]
    np.testing.assert_allclose(T_world, [10.0, 3.0, 0.0])
