"""Unit tests for src/eval_ext/region_iou.py, on small synthetic face
sets with hand-computed answers. Faces are abstract indices with unit
area (area=1.0 each) unless noted -- no real mesh geometry is needed
since `region_iou` only consumes face-index sets and an areas array.
"""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from eval.regions import Region, classify_size  # noqa: E402
from eval.region_metrics import build_face_to_component_id  # noqa: E402
from eval_ext.region_iou import region_iou, region_iou_many  # noqa: E402


def _region(faces, area=None):
    faces = np.array(faces)
    if area is None:
        area = float(len(faces))
    d = 2 * np.sqrt(area / np.pi)
    return Region(faces=faces, area=area, diameter=d, size_class=classify_size(d))


def test_identical_component_gives_iou_1():
    n_faces = 10
    areas = np.ones(n_faces)
    region = _region([0, 1, 2, 3])
    components = [_region([0, 1, 2, 3])]
    f2c = build_face_to_component_id(n_faces, components)
    assert region_iou(region, components, f2c, areas) == 1.0


def test_disjoint_component_gives_iou_0():
    n_faces = 10
    areas = np.ones(n_faces)
    region = _region([0, 1, 2, 3])
    components = [_region([5, 6, 7])]
    f2c = build_face_to_component_id(n_faces, components)
    assert region_iou(region, components, f2c, areas) == 0.0


def test_no_overlapping_component_at_all_gives_iou_0():
    n_faces = 10
    areas = np.ones(n_faces)
    region = _region([0, 1, 2, 3])
    components: list[Region] = []
    f2c = build_face_to_component_id(n_faces, components)
    assert region_iou(region, components, f2c, areas) == 0.0


def test_component_strict_superset_of_region():
    # region = {0,1,2,3} (area 4), component = {0,1,2,3,4,5,6,7} (area 8)
    # intersection = region (area 4), union = component (area 8) -> IoU = 0.5
    n_faces = 10
    areas = np.ones(n_faces)
    region = _region([0, 1, 2, 3])
    components = [_region([0, 1, 2, 3, 4, 5, 6, 7])]
    f2c = build_face_to_component_id(n_faces, components)
    assert region_iou(region, components, f2c, areas) == 0.5


def test_several_fragments_inside_region():
    # region = {0..7} (area 8). Two disjoint predicted fragments inside it:
    # {0,1} and {4,5} (area 2 each, total 4). Since both are subsets of
    # the region, union = region (area 8), intersection = fragments (area 4)
    # -> IoU = 4/8 = 0.5
    n_faces = 10
    areas = np.ones(n_faces)
    region = _region(list(range(8)))
    components = [_region([0, 1]), _region([4, 5])]
    f2c = build_face_to_component_id(n_faces, components)
    assert region_iou(region, components, f2c, areas) == 0.5


def test_fragment_straddling_region_boundary():
    # region = {0,1,2,3} (area 4). One predicted component = {2,3,4,5}
    # (area 4), half inside the region (faces 2,3) and half outside
    # (faces 4,5). intersection = {2,3} (area 2), union = {0,1,2,3,4,5}
    # (area 6) -> IoU = 2/6 = 1/3.
    n_faces = 10
    areas = np.ones(n_faces)
    region = _region([0, 1, 2, 3])
    components = [_region([2, 3, 4, 5])]
    f2c = build_face_to_component_id(n_faces, components)
    assert abs(region_iou(region, components, f2c, areas) - (2.0 / 6.0)) < 1e-12


def test_two_regions_touched_by_one_merged_component():
    # One predicted component spans both region A's territory and region
    # B's territory (a merge across a GT gap that predicted-unobserved
    # bridged). component = {0,1,2,3,4,5,6}. region A = {0,1,2} (area 3),
    # region B = {4,5,6} (area 3). Each region's IoU is computed
    # independently against the SAME merged component:
    #   A: intersection={0,1,2} (3), union=component (7) -> IoU = 3/7
    #   B: intersection={4,5,6} (3), union=component (7) -> IoU = 3/7
    # The merge penalizes both regions symmetrically via the inflated
    # union, not just one of them.
    n_faces = 10
    areas = np.ones(n_faces)
    region_a = _region([0, 1, 2])
    region_b = _region([4, 5, 6])
    components = [_region([0, 1, 2, 3, 4, 5, 6])]
    f2c = build_face_to_component_id(n_faces, components)
    iou_a, iou_b = region_iou_many([region_a, region_b], components, f2c, areas)
    assert abs(iou_a - 3.0 / 7.0) < 1e-12
    assert abs(iou_b - 3.0 / 7.0) < 1e-12


def test_region_iou_many_matches_individual_calls():
    n_faces = 12
    areas = np.array([1.0, 2.0, 1.5, 0.5, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    region1 = _region([0, 1])
    region2 = _region([6, 7, 8])
    components = [_region([0, 1, 2]), _region([7, 8, 9])]
    f2c = build_face_to_component_id(n_faces, components)
    individual = [region_iou(r, components, f2c, areas) for r in [region1, region2]]
    batched = region_iou_many([region1, region2], components, f2c, areas)
    assert individual == batched


def test_nonuniform_areas_superset_ratio():
    # region = face 0 (area 5.0). component = faces {0,1} with face 1
    # area 15.0. intersection area = 5.0, union area = 20.0 -> IoU = 0.25.
    n_faces = 5
    areas = np.array([5.0, 15.0, 1.0, 1.0, 1.0])
    region = _region([0], area=5.0)
    components = [_region([0, 1], area=20.0)]
    f2c = build_face_to_component_id(n_faces, components)
    assert abs(region_iou(region, components, f2c, areas) - 0.25) < 1e-12
