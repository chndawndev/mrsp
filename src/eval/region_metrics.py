"""Region-level metrics, `docs/success_criteria.md` section 1 (frozen
definitions) and `docs/eval_protocol.md` section 5 (area fraction /
calibration ratio). Consumes the face-level outputs of
`src/eval/oracle.py` (Stage 2) / the predicted-path equivalent (Stage 4)
plus `src/eval/regions.py`'s region extraction.

**Ignore-set scope, exactly per `docs/eval_protocol.md` section 6's
"Effect, applied everywhere a method's output is scored" list** (the only
place the frozen definitions below are modified by it):
  - Region recall / detection: uses the RAW predicted-unobserved set,
    unfiltered. Section 6 does not list this as an ignore-set use, and
    detection's definition (coverage fraction of a real GT-unobserved
    region) is not distorted by the structural gap the ignore set exists
    to handle -- a region being *over*-covered by structurally-unreachable
    faces only helps recall, never hurts it.
  - False alarm rate: BOTH numerator and denominator restricted to
    `~ignore_set` (section 6, item 1, "neither the numerator nor
    denominator counts an ignore-set face").
  - Predicted-unobserved connected components (used here only for
    localization error's region-to-component matching): built from
    `predicted_unobserved & ~ignore_set` (section 6, item 2) --
    `src/eval/regions.py::compute_regions` already takes this
    ignore-set-excluded mask as its `mask` argument; this module does not
    re-derive it.
  - False reassurance rate and the area-fraction/calibration-ratio pair:
    not listed in section 6's ignore-set effects at all -- computed raw,
    unfiltered, exactly as `docs/success_criteria.md` / section 5 define
    them.

**Localization error: Euclidean, decided (2026-09-23, approved by Chen)**.
"Surface distance" (`docs/success_criteria.md` line 66-67) is ambiguous
between geodesic (along-mesh) and Euclidean (straight-line through 3D
space); resolved in favor of Euclidean, between area-weighted face
centroids -- parameter-free and reproducible, vs. geodesic distance on a
~700k-face mesh being implementation-dependent (Dijkstra-along-edges vs.
heat method vs. exact geodesic give different numbers). Full reasoning in
`docs/eval_protocol.md` section 5. Paired with a diagnostic, not a change
to the metric: `segment_intersects_mesh` below tests whether the straight
line between a matched pair's centroids tunnels through the mesh, which
would mean Euclidean understates the true surface separation for that
pair -- reported as a per-sequence fraction alongside localization error.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .regions import Region

DETECTION_THRESHOLDS = [0.25, 0.50, 0.75]
DETECTION_PRIMARY = 0.50


def region_coverage_fraction(region: Region, predicted_unobserved: np.ndarray, areas: np.ndarray) -> float:
    """Fraction of `region`'s area covered by `predicted_unobserved`
    (raw, not ignore-set-filtered -- see module docstring)."""
    covered_area = float(areas[region.faces][predicted_unobserved[region.faces]].sum())
    return covered_area / region.area if region.area > 0 else float("nan")


@dataclass
class DetectionResult:
    threshold: float
    detected: list[bool]  # one per region, in the same order as the input region list
    coverage: list[float]  # one per region, coverage fraction (independent of threshold)


def detection_at_threshold(
    regions: list[Region], predicted_unobserved: np.ndarray, areas: np.ndarray, threshold: float
) -> DetectionResult:
    coverage = [region_coverage_fraction(r, predicted_unobserved, areas) for r in regions]
    detected = [c >= threshold for c in coverage]
    return DetectionResult(threshold=threshold, detected=detected, coverage=coverage)


def region_recall_by_size_class(
    regions: list[Region], predicted_unobserved: np.ndarray, areas: np.ndarray, threshold: float = DETECTION_PRIMARY
) -> dict[str, dict]:
    """Region recall (fraction of GT regions detected), grouped by
    `Region.size_class`. `regions` should be GT regions (headline classes
    small/medium/large use d>=5mm; "below_headline" is reported too, per
    `docs/success_criteria.md`'s "still reported in an appendix table")."""
    result = detection_at_threshold(regions, predicted_unobserved, areas, threshold)
    by_class: dict[str, dict] = {}
    for size_class in ["below_headline", "small", "medium", "large"]:
        idx = [i for i, r in enumerate(regions) if r.size_class == size_class]
        n = len(idx)
        n_detected = sum(result.detected[i] for i in idx)
        by_class[size_class] = {
            "n_regions": n,
            "n_detected": n_detected,
            "recall": n_detected / n if n else float("nan"),
        }
    return by_class


def detection_sweep(
    regions: list[Region], predicted_unobserved: np.ndarray, areas: np.ndarray, thresholds: list[float] = DETECTION_THRESHOLDS
) -> dict[float, dict[str, dict]]:
    """The 25/50/75% robustness sweep (`docs/success_criteria.md` line
    54), region recall by size class at each threshold."""
    return {t: region_recall_by_size_class(regions, predicted_unobserved, areas, t) for t in thresholds}


def false_reassurance_rate(gt_unobserved: np.ndarray, predicted_observed: np.ndarray, areas: np.ndarray) -> float:
    """Fraction of GT-unobserved area the pipeline marks as observed
    (`docs/success_criteria.md` line 59-61). Raw, not ignore-set-filtered."""
    gt_unobserved_area = float(areas[gt_unobserved].sum())
    marked_observed_area = float(areas[gt_unobserved & predicted_observed].sum())
    return marked_observed_area / gt_unobserved_area if gt_unobserved_area > 0 else float("nan")


def false_alarm_rate(
    predicted_unobserved: np.ndarray, gt_observed: np.ndarray, ignore_set: np.ndarray, areas: np.ndarray
) -> float:
    """Fraction of predicted-unobserved area GT marks as observed
    (`docs/success_criteria.md` line 63-64), with BOTH numerator and
    denominator restricted to non-ignore-set faces
    (`docs/eval_protocol.md` section 6, "Excluded from false alarm rate's
    computation")."""
    scored = predicted_unobserved & ~ignore_set
    denom_area = float(areas[scored].sum())
    numer_area = float(areas[scored & gt_observed].sum())
    return numer_area / denom_area if denom_area > 0 else float("nan")


def area_fraction_and_calibration(
    predicted_unobserved: np.ndarray, gt_unobserved: np.ndarray, areas: np.ndarray, total_area: float
) -> tuple[float, float]:
    """`docs/eval_protocol.md` section 5: (predicted-unobserved area as a
    fraction of total mesh area, ratio of predicted-unobserved area to
    GT-unobserved area). Raw, not ignore-set-filtered (section 6's
    ignore-set effects list does not mention this pair)."""
    pred_area = float(areas[predicted_unobserved].sum())
    gt_unobserved_area = float(areas[gt_unobserved].sum())
    area_fraction = pred_area / total_area if total_area > 0 else float("nan")
    calibration_ratio = pred_area / gt_unobserved_area if gt_unobserved_area > 0 else float("nan")
    return area_fraction, calibration_ratio


def region_centroid(region: Region, vertices: np.ndarray, faces: np.ndarray, areas: np.ndarray) -> np.ndarray:
    """Area-weighted centroid of a region's faces (mean of each face's own
    centroid, weighted by that face's area)."""
    face_centroids = vertices[faces[region.faces]].mean(axis=1)  # (n_region_faces, 3)
    w = areas[region.faces]
    return (face_centroids * w[:, None]).sum(axis=0) / w.sum()


def build_face_to_component_id(n_faces: int, components: list[Region]) -> np.ndarray:
    """`(n_faces,)` int array: index into `components` for the component
    each face belongs to, or -1 if the face isn't in any component.
    Precomputed once per component list so `matching_predicted_component`
    is O(|gt_region.faces|) per call instead of O(n_components) --
    with potentially thousands of small predicted components (single/few-
    face aliasing specks, `docs/visibility_limitations.md`), a naive
    per-region-per-component set-intersection scan is O(regions *
    components) and does not finish in reasonable time on a full sequence."""
    face_to_id = np.full(n_faces, -1, dtype=np.int64)
    for i, comp in enumerate(components):
        face_to_id[comp.faces] = i
    return face_to_id


def matching_predicted_component(
    gt_region: Region, predicted_components: list[Region], face_to_component_id: np.ndarray
) -> Region | None:
    """The predicted component overlapping `gt_region`, by largest shared
    face count -- `docs/success_criteria.md`'s "the overlapping predicted
    component" (singular) doesn't specify a tie-break when more than one
    predicted component overlaps a GT region (possible if the method only
    partially detects a region in a way that splits it); largest-overlap
    is the natural reading, flagged here as an INTERPRETATION, not stated
    in the frozen text. Returns None if no predicted component overlaps
    at all (an undetected region has no localization error).

    `face_to_component_id`: `build_face_to_component_id(n_faces,
    predicted_components)`, built once and reused across every GT region
    (not rebuilt per call)."""
    ids = face_to_component_id[gt_region.faces]
    ids = ids[ids >= 0]
    if len(ids) == 0:
        return None
    counts = np.bincount(ids)
    best_id = int(np.argmax(counts))
    return predicted_components[best_id]


def localization_error(
    gt_region: Region,
    predicted_components: list[Region],
    face_to_component_id: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    areas: np.ndarray,
) -> float | None:
    """Euclidean surface distance between `gt_region`'s centroid and the
    overlapping predicted component's centroid -- see module docstring,
    "Localization error: Euclidean, decided". None if no predicted
    component overlaps `gt_region` (undetected -- no error to report)."""
    comp = matching_predicted_component(gt_region, predicted_components, face_to_component_id)
    if comp is None:
        return None
    c_gt = region_centroid(gt_region, vertices, faces, areas)
    c_pred = region_centroid(comp, vertices, faces, areas)
    return float(np.linalg.norm(c_gt - c_pred))


def segment_intersects_mesh(
    mesh, point_a: np.ndarray, point_b: np.ndarray, margin_frac: float = 1e-3
) -> bool:
    """The localization-error diagnostic (`docs/eval_protocol.md` section 5,
    "Localization error: Euclidean, decided") -- NOT part of the metric
    itself, never used to compute or adjust `localization_error`.

    True if the straight segment from `point_a` to `point_b` crosses the
    mesh surface anywhere strictly between its two endpoints. `mesh`: a
    `trimesh.Trimesh` (built once per sequence and reused across pairs --
    not rebuilt here). `margin_frac`: fraction of the segment length
    excluded at each end, since a region centroid sits on/near the mesh
    surface itself and a ray cast from exactly there would otherwise
    register a spurious self-intersection at t~0.

    An intersecting segment means the straight line cuts through the
    lumen or the wall -- Euclidean distance understates the true surface
    separation for that pair.
    """
    vec = np.asarray(point_b, dtype=np.float64) - np.asarray(point_a, dtype=np.float64)
    length = float(np.linalg.norm(vec))
    if length == 0.0:
        return False
    direction = vec / length
    margin = margin_frac * length
    origin = np.asarray(point_a, dtype=np.float64) + margin * direction

    locations, index_ray, index_tri = mesh.ray.intersects_location(
        origin.reshape(1, 3), direction.reshape(1, 3), multiple_hits=True
    )
    if len(locations) == 0:
        return False

    t = (locations - origin) @ direction  # distance along the ray from `origin`
    valid_range = length - 2 * margin
    return bool(np.any((t > 0) & (t < valid_range)))
