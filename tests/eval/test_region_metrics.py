"""Unit tests for src/eval/region_metrics.py: detection thresholds, region
recall by size class, false reassurance/alarm rate (incl. ignore-set
exclusion), area fraction/calibration ratio, and localization error /
component matching.
"""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from eval.region_metrics import (  # noqa: E402
    area_fraction_and_calibration,
    build_face_to_component_id,
    detection_at_threshold,
    detection_sweep,
    false_alarm_rate,
    false_reassurance_rate,
    localization_error,
    matching_predicted_component,
    region_coverage_fraction,
    region_recall_by_size_class,
    segment_intersects_mesh,
)

trimesh = pytest.importorskip("trimesh")
from eval.regions import Region  # noqa: E402


def _region(faces, area, size_class="medium"):
    from eval.regions import classify_size

    d = 2 * np.sqrt(area / np.pi)
    return Region(faces=np.array(faces), area=area, diameter=d, size_class=classify_size(d))


def test_region_coverage_fraction_full_and_partial():
    region = _region([0, 1, 2, 3], area=4.0)  # 4 faces, area 1 each
    predicted_unobserved = np.array([True, True, False, False, True])  # faces 0,1,4
    areas = np.ones(5)
    frac = region_coverage_fraction(region, predicted_unobserved, areas)
    assert frac == pytest.approx(0.5)  # 2 of 4 region faces covered


def test_detection_at_threshold():
    r1 = _region([0, 1, 2, 3], area=4.0)  # 50% covered
    r2 = _region([4, 5], area=2.0)  # fully covered
    predicted_unobserved = np.array([True, True, False, False, True, True])
    areas = np.ones(6)

    result_50 = detection_at_threshold([r1, r2], predicted_unobserved, areas, 0.50)
    assert result_50.detected == [True, True]  # 50% clears >=50%

    result_75 = detection_at_threshold([r1, r2], predicted_unobserved, areas, 0.75)
    assert result_75.detected == [False, True]


def test_region_recall_by_size_class():
    # one small (d~5-10), one large (d>=20) region, both fully detected.
    small = _region([0], area=np.pi * (6.0 / 2) ** 2)  # diameter 6mm -> small
    large = _region([1], area=np.pi * (25.0 / 2) ** 2)  # diameter 25mm -> large
    assert small.size_class == "small"
    assert large.size_class == "large"
    predicted_unobserved = np.array([True, True])
    areas = np.array([small.area, large.area])

    recall = region_recall_by_size_class([small, large], predicted_unobserved, areas, threshold=0.5)
    assert recall["small"]["n_regions"] == 1
    assert recall["small"]["n_detected"] == 1
    assert recall["small"]["recall"] == pytest.approx(1.0)
    assert recall["large"]["recall"] == pytest.approx(1.0)
    assert recall["medium"]["n_regions"] == 0
    assert np.isnan(recall["medium"]["recall"])


def test_detection_sweep_covers_all_three_thresholds():
    # area chosen so diameter=15mm ("medium"), not just an arbitrary area.
    medium_area = np.pi * (15.0 / 2) ** 2
    r = _region([0, 1, 2, 3], area=medium_area)  # 50% covered
    predicted_unobserved = np.array([True, True, False, False])
    areas = np.full(4, medium_area / 4)
    sweep = detection_sweep([r], predicted_unobserved, areas)
    assert set(sweep.keys()) == {0.25, 0.50, 0.75}
    assert sweep[0.25]["medium"]["n_detected"] == 1  # 50% >= 25%
    assert sweep[0.75]["medium"]["n_detected"] == 0  # 50% < 75%


def test_false_reassurance_rate():
    gt_unobserved = np.array([True, True, True, False])
    predicted_observed = np.array([True, False, False, True])  # 1 of 3 GT-unobserved faces marked observed
    areas = np.ones(4)
    rate = false_reassurance_rate(gt_unobserved, predicted_observed, areas)
    assert rate == pytest.approx(1 / 3)


def test_false_alarm_rate_excludes_ignore_set():
    # 4 faces: 0,1 are GT-unobserved; 2,3 are GT-observed.
    # predicted_unobserved = {0,1,2,3} (marks everything unobserved).
    # face 3 (GT-observed) is in the ignore set -> excluded from both numerator and denominator.
    predicted_unobserved = np.array([True, True, True, True])
    gt_observed = np.array([False, False, True, True])
    ignore_set = np.array([False, False, False, True])
    areas = np.ones(4)

    rate = false_alarm_rate(predicted_unobserved, gt_observed, ignore_set, areas)
    # scored = {0,1,2} (face 3 excluded); numerator = scored & gt_observed = {2}; denom = {0,1,2}
    assert rate == pytest.approx(1 / 3)


def test_false_alarm_rate_zero_when_ignore_set_covers_all_gt_observed_overlap():
    predicted_unobserved = np.array([True, True, True])
    gt_observed = np.array([False, False, True])
    ignore_set = np.array([False, False, True])  # the one GT-observed face is fully ignored
    areas = np.ones(3)
    rate = false_alarm_rate(predicted_unobserved, gt_observed, ignore_set, areas)
    assert rate == pytest.approx(0.0)


def test_area_fraction_and_calibration():
    predicted_unobserved = np.array([True, True, False, False])
    gt_unobserved = np.array([True, False, False, False])
    areas = np.array([2.0, 3.0, 1.0, 4.0])  # total = 10
    total_area = areas.sum()

    frac, ratio = area_fraction_and_calibration(predicted_unobserved, gt_unobserved, areas, total_area)
    assert frac == pytest.approx(5.0 / 10.0)  # faces 0,1 -> area 5
    assert ratio == pytest.approx(5.0 / 2.0)  # gt_unobserved area is just face 0 -> 2.0


def test_matching_predicted_component_largest_overlap():
    gt_region = _region([0, 1, 2, 3, 4], area=5.0)
    comp_small_overlap = _region([4, 10], area=2.0)
    comp_large_overlap = _region([0, 1, 2], area=3.0)
    components = [comp_small_overlap, comp_large_overlap]
    face_to_component_id = build_face_to_component_id(11, components)
    match = matching_predicted_component(gt_region, components, face_to_component_id)
    assert match is comp_large_overlap


def test_matching_predicted_component_none_when_no_overlap():
    gt_region = _region([0, 1], area=2.0)
    comp = _region([5, 6], area=2.0)
    components = [comp]
    face_to_component_id = build_face_to_component_id(7, components)
    assert matching_predicted_component(gt_region, components, face_to_component_id) is None


def test_localization_error_known_distance():
    # 3 unit right-triangles in a row, each shifted +1 in x from the last
    # (face i's centroid is at x = i + 1/3, y=1/3, z=0). GT region = faces
    # {0,1} (overlaps predicted component on face 1, so matching succeeds);
    # predicted component = faces {1,2}. Centroids differ only in x, by
    # exactly 1.0 (area-weighted average of two unit-shifted centroids,
    # shifted again by one more unit).
    vertices = np.array(
        [
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],  # face 0
            [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 1.0, 0.0],  # face 1
            [2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [2.0, 1.0, 0.0],  # face 2
        ]
    )
    faces = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    areas = np.array([0.5, 0.5, 0.5])

    gt_region = _region([0, 1], area=1.0)
    pred_comp = _region([1, 2], area=1.0)
    components = [pred_comp]
    face_to_component_id = build_face_to_component_id(3, components)

    err = localization_error(gt_region, components, face_to_component_id, vertices, faces, areas)
    assert err == pytest.approx(1.0, abs=1e-9)


def test_localization_error_none_when_undetected():
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    faces = np.array([[0, 1, 2]])
    areas = np.array([0.5])
    gt_region = _region([0], area=0.5)
    face_to_component_id = build_face_to_component_id(1, [])
    assert localization_error(gt_region, [], face_to_component_id, vertices, faces, areas) is None


def _wall_mesh(half_extent: float = 1000.0) -> "trimesh.Trimesh":
    """A large flat wall in the plane z=0."""
    verts = np.array(
        [
            [-half_extent, -half_extent, 0.0],
            [half_extent, -half_extent, 0.0],
            [half_extent, half_extent, 0.0],
            [-half_extent, half_extent, 0.0],
        ]
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def test_segment_intersects_mesh_crossing_wall():
    mesh = _wall_mesh()
    point_a = np.array([0.0, 0.0, -5.0])
    point_b = np.array([0.0, 0.0, 5.0])
    assert segment_intersects_mesh(mesh, point_a, point_b) is True


def test_segment_intersects_mesh_same_side_no_crossing():
    mesh = _wall_mesh()
    point_a = np.array([0.0, 0.0, -5.0])
    point_b = np.array([0.0, 0.0, -1.0])
    assert segment_intersects_mesh(mesh, point_a, point_b) is False


def test_segment_intersects_mesh_endpoints_on_surface_no_false_positive():
    """Both endpoints essentially on the mesh surface, segment lying flat
    within the plane (not crossing through it) -- must not register a
    spurious self-intersection from the margin-excluded endpoints."""
    mesh = _wall_mesh()
    point_a = np.array([-10.0, 0.0, 0.0])
    point_b = np.array([10.0, 0.0, 0.0])
    assert segment_intersects_mesh(mesh, point_a, point_b) is False


def test_segment_intersects_mesh_zero_length_segment():
    mesh = _wall_mesh()
    point_a = np.array([5.0, 5.0, -3.0])
    assert segment_intersects_mesh(mesh, point_a, point_a) is False
