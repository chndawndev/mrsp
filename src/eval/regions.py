"""Region extraction, `docs/success_criteria.md` section 1 ("GT region",
"Region size").

**GT region**: a connected component of GT-unobserved faces on
`coverage_mesh.obj`. Components computed by face adjacency (shared edge),
reusing `src/geometry/mesh_stats.py`'s `connected_components_of_subset`
directly (per `docs/eval_protocol.md` section 5's explicit instruction),
not new component-extraction logic.

**Region size**: equivalent diameter `d = 2*sqrt(area/pi)`. Size classes:
small (5mm<=d<10mm), medium (10mm<=d<20mm), large (d>=20mm); d<5mm is
"below_headline" -- excluded from headline metrics, appendix only
(`docs/success_criteria.md` section 1, lines 40-46).

Predicted-unobserved components (used for localization error and the
ignore-set validation, `docs/eval_protocol.md` section 6) are built with
the exact same `compute_regions` function, just on a different face mask
(`predicted_unobserved & ~ignore_set`, ignore-set faces excluded from
construction so they cannot bridge two otherwise-separate components --
section 6's "ignore set" effects list, item 2).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from geometry.mesh_stats import area_to_diameter, connected_components_of_subset

SMALL_MIN_MM = 5.0
MEDIUM_MIN_MM = 10.0
LARGE_MIN_MM = 20.0


@dataclass
class Region:
    faces: np.ndarray  # face indices belonging to this region
    area: float  # mm^2, sum of face areas
    diameter: float  # mm, equivalent-circle diameter
    size_class: str  # "below_headline" | "small" | "medium" | "large"


def classify_size(diameter_mm: float) -> str:
    if diameter_mm < SMALL_MIN_MM:
        return "below_headline"
    if diameter_mm < MEDIUM_MIN_MM:
        return "small"
    if diameter_mm < LARGE_MIN_MM:
        return "medium"
    return "large"


def compute_regions(n_faces: int, adjacency: np.ndarray, mask: np.ndarray, areas: np.ndarray) -> list[Region]:
    """mask: (n_faces,) bool, the faces to partition into connected
    components (GT-unobserved for GT regions; predicted_unobserved &
    ~ignore_set for predicted components).

    Returns one Region per connected component, unsorted.
    """
    components = connected_components_of_subset(n_faces, adjacency, mask)
    regions = []
    for comp in components:
        area = float(areas[comp].sum())
        diameter = area_to_diameter(area)
        regions.append(Region(faces=comp, area=area, diameter=diameter, size_class=classify_size(diameter)))
    return regions
