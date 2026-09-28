"""Region IoU, `docs/success_criteria.md` section 6 (2026-09-27 deviation,
item 1) -- a new primary localization metric, implemented OUTSIDE the
locked `src/eval/`/`src/gt/` paths per that entry's own instruction
("Implemented outside the locked paths").

Definition, quoted verbatim from `docs/success_criteria.md` section 6,
item 1:

    For each GT unobserved region R: let C be the union of all predicted-
    unobserved connected components (built exactly as in the locked
    component construction, ignore set excluded) that intersect R.
    IoU(R) = area(R intersect C) / area(R union C); IoU(R) = 0 if no
    component intersects R.

"The locked component construction" is `eval.regions.compute_regions`
(imported below, not reimplemented), the same function `src/eval/
region_metrics.py`'s `matching_predicted_component` uses for its
single-best-overlap component. Region IoU differs from that: C here is
the UNION of every predicted component touching R, not just the
largest-overlap one -- a region partially detected by several small,
disjoint predicted fragments still gets full credit for however much of
R those fragments jointly cover, and is penalized for however much of R
they jointly still miss, plus however far any of them spill outside R.
"""
from __future__ import annotations

import numpy as np

from eval.regions import Region


def region_iou(
    gt_region: Region,
    predicted_components: list[Region],
    face_to_component_id: np.ndarray,
    areas: np.ndarray,
) -> float:
    """IoU(R) as defined above. `predicted_components` and
    `face_to_component_id` are built by the caller exactly as
    `src/eval/region_metrics.py` already does for localization error
    (`eval.regions.compute_regions` on `predicted_unobserved & ~ignore_set`,
    then `eval.region_metrics.build_face_to_component_id`) -- not
    rebuilt here, so a caller scoring both localization error and region
    IoU on the same mask only pays the component-construction cost once.
    """
    ids = face_to_component_id[gt_region.faces]
    ids = np.unique(ids[ids >= 0])
    if len(ids) == 0:
        return 0.0

    n_faces = len(areas)
    region_mask = np.zeros(n_faces, dtype=bool)
    region_mask[gt_region.faces] = True

    c_mask = np.zeros(n_faces, dtype=bool)
    for i in ids:
        c_mask[predicted_components[i].faces] = True

    intersection = region_mask & c_mask
    union = region_mask | c_mask
    area_union = float(areas[union].sum())
    if area_union == 0.0:
        return 0.0
    area_intersection = float(areas[intersection].sum())
    return area_intersection / area_union


def region_iou_many(
    gt_regions: list[Region],
    predicted_components: list[Region],
    face_to_component_id: np.ndarray,
    areas: np.ndarray,
) -> list[float]:
    """`region_iou` for every region in `gt_regions`, same component
    construction reused across all of them."""
    return [region_iou(r, predicted_components, face_to_component_id, areas) for r in gt_regions]
