"""Unit tests for the small pure functions in src/eval/ (Stage 2):
evaluable_pixel_mask's boolean gate, face_set_iou/face_set_disagreement's
edge cases, and compute_ignore_set's (narrow, gt_observed-restricted)
definition.
"""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from eval.evaluable import MAX_HIT_DISTANCE_MM, evaluable_pixel_mask  # noqa: E402
from eval.ignore_set import compute_ignore_set  # noqa: E402
from eval.metrics_face import face_set_disagreement, face_set_iou  # noqa: E402


def test_evaluable_pixel_mask_requires_hit():
    hit = np.array([True, False, True])
    d_hit = np.array([50.0, 50.0, 50.0])
    vignette = np.array([False, False, False])
    result = evaluable_pixel_mask(hit, d_hit, vignette)
    np.testing.assert_array_equal(result, [True, False, True])


def test_evaluable_pixel_mask_excludes_vignette():
    hit = np.array([True, True])
    d_hit = np.array([50.0, 50.0])
    vignette = np.array([True, False])
    result = evaluable_pixel_mask(hit, d_hit, vignette)
    np.testing.assert_array_equal(result, [False, True])


def test_evaluable_pixel_mask_100mm_boundary():
    """Exactly 100mm passes (<=), just over fails -- section 2a's "<=100mm"."""
    hit = np.array([True, True, True])
    d_hit = np.array([99.999, 100.0, 100.001])
    vignette = np.array([False, False, False])
    result = evaluable_pixel_mask(hit, d_hit, vignette)
    np.testing.assert_array_equal(result, [True, True, False])
    assert MAX_HIT_DISTANCE_MM == 100.0


def test_face_set_iou_identical():
    a = np.array([True, True, False, False])
    assert face_set_iou(a, a) == pytest.approx(1.0)


def test_face_set_iou_disjoint():
    a = np.array([True, False])
    b = np.array([False, True])
    assert face_set_iou(a, b) == pytest.approx(0.0)


def test_face_set_iou_empty_union_is_nan():
    a = np.array([False, False])
    assert np.isnan(face_set_iou(a, a))


def test_face_set_iou_partial_overlap():
    a = np.array([True, True, False, False])
    b = np.array([True, False, True, False])
    # intersection={0}, union={0,1,2} -> 1/3
    assert face_set_iou(a, b) == pytest.approx(1 / 3)


def test_face_set_disagreement_counts():
    a = np.array([True, True, False, False])
    b = np.array([True, False, True, False])
    a_not_b, b_not_a = face_set_disagreement(a, b)
    assert a_not_b == 1  # index 1
    assert b_not_a == 1  # index 2


def test_ignore_set_is_gt_observed_and_unreachable():
    """Narrow definition (resolved 2026-09-22, approved by Chen after Stage
    3 found the unrestricted reading degenerate): ignore_set =
    gt_observed & ~ever_evaluable_hit -- the literal Part 1 "gap", not the
    full complement of ever_evaluable_hit.

    index0: observed, reachable -> not ignored (normal case).
    index1: observed, NOT reachable -> ignored (the classic gap).
    index2: NOT observed, reachable -> not ignored (the rare aliasing case,
        Stage 1's silhouette noise -- narrow definition doesn't touch it).
    index3: NOT observed, NOT reachable -> not ignored (a genuinely-missed
        region correctly staying out of the ignore set is the whole point
        of the narrow definition; the broad one would have wrongly
        swallowed this too).
    """
    gt_observed = np.array([True, True, False, False])
    ever_evaluable_hit = np.array([True, False, True, False])
    ignore_set = compute_ignore_set(gt_observed, ever_evaluable_hit)
    np.testing.assert_array_equal(ignore_set, [False, True, False, False])
