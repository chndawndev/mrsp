# Cost of undistorting to a pinhole model

Question: before choosing the pipeline's input camera format, what does undistorting the omnidirectional (Scaramuzza) model to a pinhole model actually cost, in FOV, coverage, and resampling quality? Script: `scripts/pinhole_tradeoff.py`. Diagnostic only — **no pipeline changed**; the pinhole model is local to this script and does not touch `src/geometry/` or `scripts/visibility_full.py` (frozen). 3 sequences spanning segments: `c2_cecum_t4_v1` (cecum), `c1_descending_t1_v1` (descending), `c1_sigmoid2_t1_v1` (sigmoid2) — chosen for segment spread and small frame counts (123-191 frames) to keep this a quick diagnostic. Full numbers: `results/pinhole_tradeoff/pinhole_tradeoff.json`.

## The pinhole model chosen

**f = 541.29px, image 1350x1080 (same resolution as the source), principal point at image center.**

This was **not** found analytically (matching corner angle to the vignette boundary gives f=362.66px, but that overshoots the actual captured frame — its own edge midpoints project up to 91px outside the real 1350x1080 image, i.e. it would need source pixels that don't exist). Found instead by an empirical search over the model's own projection: the smallest f (widest FOV) whose entire image perimeter maps, through the real omnidirectional model, to pixel positions that are both inside the captured frame and inside the vignette (real-data) circle (`find_max_fov_f()` in the script). The omnidirectional model's stretch matrix makes it mildly anisotropic, so a purely angular/isotropic argument is insufficient — this cost real iteration to get right and is worth remembering for any future FOV-related derivation from this camera model.

| | Original (omnidirectional, real-data extent) | Pinhole (this choice) |
|---|---|---|
| Horizontal FOV | ~122.4° (edge-to-edge at y=cy, confirmed within the vignette) | **102.55°** |
| Vertical FOV | ~90.5° (edge-to-edge at x=cx) | **89.86°** |

**Vertical FOV is preserved almost exactly (loses 0.6°); horizontal FOV loses ~20°** — the 1350x1080 aspect ratio's binding constraint under a single shared focal length is horizontal, not vertical, once corners are properly excluded.

## Coverage cost (GT-criterion visibility, full omni vs. pinhole-restricted)

| Sequence | Segment | omni observed frac | pinhole observed frac | drop | newly-lost >5mm components (area) |
|---|---|---|---|---|---|
| c2_cecum_t4_v1 | cecum | 84.33% | 73.10% | **11.22pp** (13.3% relative) | 3: 1848.3, 179.5, 62.3 mm² |
| c1_descending_t1_v1 | descending | 75.49% | 70.55% | **4.94pp** (6.5% relative) | 1: 1067.7 mm² |
| c1_sigmoid2_t1_v1 | sigmoid2 | 75.46% | 68.95% | **6.51pp** (8.6% relative) | 4: 612.1, 237.5, 74.5, 24.9 mm² |

"Newly-lost" = faces observed under the full omnidirectional model but not under the pinhole-restricted model, connected-componented and filtered to the same >5mm equivalent-diameter threshold used as the primary-analysis size class elsewhere in this project (`docs/coverage_stats.md`, `results/regions.csv`).

**Every one of the 3 sequences produces at least one new >5mm "lost" region under pinhole restriction that does not exist under the full omnidirectional model** — 1-4 per sequence, total area 949-2090mm² per sequence. These are not sub-pixel artifacts (contrast `docs/visibility_outliers.md`'s sigmoid1 diagnosis, largest disagreement there was 0.25mm²) — the largest single newly-lost region here (1848mm², cecum) is nearly 4mm² per average mesh triangle's worth of contiguous surface, comfortably a "real" region by this project's own size conventions.

GT-unobserved >5mm regions (sanity check, not a cost measure): 2-5 per sequence, touched by at least one omni-observed face in most cases (4/5, 4/4, 2/2) — consistent with the small, already-characterized false-observed noise rate (`docs/gpu_validation.md`, `docs/visibility_outliers.md`). Pinhole touches fewer of these (2/5, 3/4, 1/2) simply because it sees less overall, not because it's more accurate — expected, not informative about which model is "better."

## Resampling cost: corner pixel stretch

**3.38x.** A one-pixel step in the source omnidirectional image, at the corner, spans 3.38 pixels in the destination pinhole image after undistortion. This is the classic wide-angle-correction artifact: the periphery of the source image is locally magnified when rectified to a rectilinear projection, degrading effective resolution exactly where a real pipeline needs corner data most (near the edge of usable FOV). This number is purely geometric (computed from the two camera models directly) and does not depend on which sequence or frame is used.

## Bottom line

Restricting to the widest practical pinhole FOV loses 5-11 percentage points of observed surface per sequence and introduces new sub-headline-threshold-sized coverage gaps (1-4 per sequence, order 25-1850mm² each) that the omnidirectional model does not have, on top of a 3.4x local resampling penalty at the image corners. None of this is free, and the cost is not confined to the horizontal periphery in an easily-discardable way — the newly-lost regions are large enough to matter at this project's own >5mm size threshold. This is reported for the pipeline-format decision to use; no recommendation is made here on which format to choose, and no pipeline was changed.
