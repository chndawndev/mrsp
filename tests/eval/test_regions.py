"""Unit tests for src/eval/regions.py: size classification boundaries and
connected-component extraction, on fabricated adjacency (no real mesh
geometry needed -- connected_components_of_subset only consumes face-index
adjacency pairs and a boolean mask).
"""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from eval.regions import classify_size, compute_regions  # noqa: E402


def test_classify_size_boundaries():
    assert classify_size(4.999) == "below_headline"
    assert classify_size(5.0) == "small"
    assert classify_size(9.999) == "small"
    assert classify_size(10.0) == "medium"
    assert classify_size(19.999) == "medium"
    assert classify_size(20.0) == "large"
    assert classify_size(100.0) == "large"


def test_compute_regions_two_disjoint_components():
    # 5 faces: {0,1,2} adjacent in a chain, {3,4} adjacent separately, all in mask.
    n_faces = 5
    adjacency = np.array([[0, 1], [1, 2], [3, 4]])
    mask = np.array([True, True, True, True, True])
    areas = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

    regions = compute_regions(n_faces, adjacency, mask, areas)
    assert len(regions) == 2
    sizes = sorted(len(r.faces) for r in regions)
    assert sizes == [2, 3]
    areas_found = sorted(r.area for r in regions)
    assert areas_found == pytest.approx([2.0, 3.0])


def test_compute_regions_excludes_masked_out_faces():
    # face 2 is excluded from the mask -> splits the {0,1,2} chain into {0,1} and nothing at 2.
    n_faces = 3
    adjacency = np.array([[0, 1], [1, 2]])
    mask = np.array([True, True, False])
    areas = np.array([1.0, 1.0, 1.0])

    regions = compute_regions(n_faces, adjacency, mask, areas)
    assert len(regions) == 1
    assert set(regions[0].faces.tolist()) == {0, 1}


def test_compute_regions_size_class_from_area():
    # A single triangle-region with area pi*25 -> diameter 2*sqrt(25)=10mm -> "medium".
    n_faces = 1
    adjacency = np.empty((0, 2), dtype=int)
    mask = np.array([True])
    areas = np.array([np.pi * 25.0])

    regions = compute_regions(n_faces, adjacency, mask, areas)
    assert len(regions) == 1
    assert regions[0].diameter == pytest.approx(10.0)
    assert regions[0].size_class == "medium"
