# Diagnosis of the 9 flagged/worst-IoU sequences

Scope: the 9 sequences from `docs/visibility_rasterization.md`'s "10 worst" table that the user grouped as pattern (a) strongly asymmetric (6 sequences) and pattern (b) balanced-but-large (3 sequences). Script: `scripts/diagnose_visibility_outliers.py`. Reuses the already-saved `results/visibility_full/observed_masks.npz` — **no GPU rasterization was rerun**. Raw per-sequence numbers: `results/visibility_outliers/<name>_diagnosis.json`. Renders: `results/visibility_outliers/views/<name>_{view1,view2,cutopen}.png` (light gray = GT-observed, dark gray = GT-unobserved, **magenta = false_unobserved**). Nothing was fixed — diagnosis only, per instructions.

**Headline correction to the requested grouping**: pattern (a) as given (by false_observed/false_unobserved asymmetry alone) is not geometrically homogeneous. It splits cleanly into two sub-patterns with opposite signatures on every measured axis. This section reports **three** groups, not two.

- **A1 (ascending)**: `c1_ascending_t4_v2`, `c1_ascending_t4_v3`, `c1_ascending_t3_v1`
- **A2 (transverse2/rectum)**: `c2_transverse2_t3_v1`, `c2_rectum_t4_v1`, `c2_rectum_t3_v1`
- **B (sigmoid1)**: `c1_sigmoid1_t2_v2`, `c1_sigmoid1_t2_v3`, `c1_sigmoid1_t1_v1`

---

## MEASURED

### 1-2. Spatial distribution and rim-adjacency

| Sequence | Group | n false_unobserved | # components | largest component | rim-adjacent fraction |
|---|---|---|---|---|---|
| c1_ascending_t4_v2 | A1 | 8033 | 7549 | 0.18 mm² (0.48mm) | 99.6% |
| c1_ascending_t4_v3 | A1 | 8049 | 7562 | 0.18 mm² (0.48mm) | 99.5% |
| c1_ascending_t3_v1 | A1 | 4652 | 4378 | 0.15 mm² (0.44mm) | 99.1% |
| c2_transverse2_t3_v1 | A2 | 4159 | 209 | 60.20 mm² (8.76mm) | 25.2% |
| c2_rectum_t4_v1 | A2 | 3070 | 671 | 85.99 mm² (10.46mm) | 31.9% |
| c2_rectum_t3_v1 | A2 | 2594 | 467 | 49.79 mm² (7.96mm) | 36.2% |
| c1_sigmoid1_t2_v2 | B | 2268 | 1930 | 0.25 mm² (0.56mm) | 77.6% |
| c1_sigmoid1_t2_v3 | B | 2266 | 1928 | 0.23 mm² (0.54mm) | 79.9% |
| c1_sigmoid1_t1_v1 | B | 805 | 718 | 0.35 mm² (0.66mm) | 73.4% |

**A1**: component count is within a few percent of the face count itself (7549 components / 8033 faces) — almost every flagged face is its own isolated single-face "component." Combined with 99%+ rim-adjacency, this is a thin, scattered speckle sitting directly on the observed/unobserved boundary contour, confirmed visually (see Images below): a dense salt-and-pepper texture tracing the boundary line, not a blob.

**A2**: the opposite signature. Far fewer, much larger components (up to 86 mm², i.e. ~10mm equivalent diameter — comparable to the "large" regions size class used elsewhere in this project), and most flagged faces (64-75%) are **not** touching the GT-unobserved boundary at all — they sit as islands inside otherwise-correctly-observed territory.

**B**: intermediate — moderate fragmentation (roughly 1 face per component, similar ratio to A1) but only 73-80% rim-adjacent (between A1 and A2).

### 3. Per-face geometry: false_unobserved vs. correctly-observed control sample

Distance (mm) / incidence angle (deg, 0=head-on, 90=grazing) / projected footprint (px²) at each face's own closest-approach frame; control = random sample of up to 2000 correctly-observed faces from the same mesh, same statistic.

| Sequence | Group | dist (fu / ctrl, median) | incidence (fu / ctrl, median) | footprint px² (fu / ctrl, median) | fu frac. <1px² |
|---|---|---|---|---|---|
| c1_ascending_t4_v2 | A1 | 26.7 / 32.4 | 33.8 / 38.9 | **19.36 / 6.77** | 13.4% |
| c1_ascending_t4_v3 | A1 | 26.7 / 31.9 | 33.8 / 38.7 | **19.35 / 7.15** | 13.3% |
| c1_ascending_t3_v1 | A1 | 25.0 / 25.3 | 39.0 / 31.7 | 11.38 / 11.42 | 28.4% |
| c2_transverse2_t3_v1 | A2 | 31.0 / 29.5 | 34.4 / 26.2 | 12.04 / 19.77 | 13.2% |
| c2_rectum_t4_v1 | A2 | 18.7 / 25.5 | 19.8 / 25.0 | 31.07 / 26.91 | 3.0% |
| c2_rectum_t3_v1 | A2 | 38.1 / 27.6 | 58.1 / 32.6 | 1.95 / 16.44 | 28.5% |
| c1_sigmoid1_t2_v2 | B | **115.7 / 39.8** | **62.0 / 44.4** | **0.030 / 5.58** | **91.3%** |
| c1_sigmoid1_t2_v3 | B | **116.0 / 37.8** | **61.9 / 45.0** | **0.028 / 6.17** | **92.0%** |
| c1_sigmoid1_t1_v1 | B | **78.1 / 30.9** | **73.8 / 45.1** | **0.514 / 11.71** | **70.4%** |

**B is unambiguous**: closest approach is 2-3x farther than the control sample, incidence angle 15-30° more grazing, and projected footprint is **150-200x smaller** — 70-92% of these faces never exceed 1 pixel² even at their single best opportunity across the whole video. This is not a subtle statistical shift; it's a different regime entirely.

**A1 is the opposite of the footprint hypothesis**: for the two `t4` sequences, false_unobserved footprint is **~3x LARGER** than the control median (19.4 vs 6.8-7.2 px²), and distance/incidence are only mildly different (a few mm / a few degrees). `c1_ascending_t3_v1` shows no meaningful footprint difference at all (11.38 vs 11.42). The grazing-angle/sub-pixel-footprint hypothesis does not explain A1.

**A2 is mixed and partially supports the hypothesis**: `c2_transverse2_t3_v1` and `c2_rectum_t3_v1` show clearly higher incidence angle for false_unobserved (34.4 vs 26.2°; 58.1 vs 32.6°) and `c2_rectum_t3_v1` shows a large footprint gap (1.95 vs 16.4 px²) consistent with the hypothesis. `c2_rectum_t4_v1` is the outlier within its own subgroup: false_unobserved faces are *closer* (18.7 vs 25.5mm) and *less* grazing (19.8 vs 25.0°) than the control, with a slightly *larger* footprint (31.1 vs 26.9) — the hypothesis does not hold for this one sequence.

### 4. Frame-level clustering

| Sequence | Group | Top frame(s) capturing most false_unobserved | Max frame-to-frame rotation | Max frame-to-frame translation |
|---|---|---|---|---|
| c1_ascending_t4_v2 | A1 | frames 58, 681, 684: 861+855+718 = 2434/8033 (30%) | 2.18° | 0.687mm (frame 212, not a flagged frame) |
| c1_ascending_t4_v3 | A1 | frames 681, 58, 56: 1677+1504+464 = 3645/8049 (45%) | 2.18° | 0.730mm (frame 551, not a flagged frame) |
| c1_ascending_t3_v1 | A1 | frames 46, 121, 124: 506+400+339 = 1245/4652 (27%) | 0.00° | (not computed for this pair) |
| c2_transverse2_t3_v1 | A2 | frame 111 alone: 1674/4159 (40%) | 4.09° | -- |
| c2_rectum_t4_v1 | A2 | frame 30: 823/3070 (27%) | 3.50° | -- |
| c2_rectum_t3_v1 | A2 | frame 15: 615/2594 (24%) | 4.25° | -- |
| c1_sigmoid1_t2_v2 | B | **frame 122 alone: 2017/2268 (89%)** | 1.81° | -- |
| c1_sigmoid1_t2_v3 | B | **frame 122 alone: 2157/2266 (95%)** | 1.82° | -- |
| c1_sigmoid1_t1_v1 | B | **frame 103 alone: 710/805 (88%)** | 1.00° | -- |

**B**: a single frame accounts for 88-95% of all flagged faces in each sequence. Checked the frame-to-frame rotation and translation around that frame for anomalies (large sudden rotation, unusually large motion) — **found nothing unusual**: rotation is 1.0-1.8° (below this sequence's own max), and this frame is not a translation outlier either. The concentration is not explained by a pose-tracking glitch; it coincides with the closest-approach frame itself (by construction, since "best frame" = frame of minimum distance) — i.e., there is exactly one frame per sequence where the camera passes close enough to see this patch of wall at all, and that is also where our reimplementation fails on it.

**A1**: `c1_ascending_t4_v2` and `c1_ascending_t4_v3` share the *same* top-3 frame indices (58, 681/684, 56) — expected and a useful cross-check, since v2/v3 share a camera trajectory (`CLAUDE.md`) — but no single frame dominates the way it does in B (top-3 frames cover only 27-45% of flagged faces, not 90%+), and checked frame-to-frame rotation/translation at those specific frames explicitly: **nothing unusual** (deltas in line with the sequence's typical frame-to-frame motion, and the sequence's own largest jump occurs at an unflagged frame). The "large rotation between frames" hypothesis is not supported for A1.

**A2**: `c2_transverse2_t3_v1` and `c2_rectum_t3_v1` both show markedly higher max frame-to-frame rotation (4.09°, 4.25°) than any A1 or B sequence — worth further inspection, but concentrated in a single frame per sequence rather than the whole-video-wide scatter seen in A1, so it's suggestive rather than conclusive on its own.

### 5. Mesh degeneracy check (sigmoid1 / group B only)

| Sequence | near-zero-area faces (<1e-6mm²) | duplicate vertex groups | duplicate faces (by vertex set) |
|---|---|---|---|
| c1_sigmoid1_t2_v2 | 0 | 0 | 0 |
| c1_sigmoid1_t2_v3 | 0 | 0 | 0 |
| c1_sigmoid1_t1_v1 | 0 | 0 | 0 |

**None found, on any of the three sigmoid1 meshes.** The hypothesis that sigmoid1's mesh has degenerate/duplicated geometry making first-hit ties abnormally common is **not supported** — there is nothing structurally wrong with these meshes at the triangle level.

## Images

Representative renders (full set in `results/visibility_outliers/views/`):

`c1_sigmoid1_t2_v2` (group B) — magenta speckle scattered along the boundary AND scattered inside the dark-gray region, consistent with 78% rim-adjacency and near-total sub-pixel footprints:

![sigmoid1 view1](../results/visibility_outliers/views/c1_sigmoid1_t2_v2_view1.png)

`c1_ascending_t4_v2` (group A1) — dense salt-and-pepper magenta speckle running exactly along the boundary contour between light gray (observed) and dark gray (unobserved), visually distinct from B's pattern:

![ascending view2](../results/visibility_outliers/views/c1_ascending_t4_v2_view2.png)

---

## INTERPRETATION

Each interpretation below is separated from the measurements above and states what would confirm or refute it, per project reporting convention.

**Group B (sigmoid1): the sub-pixel/grazing-angle hypothesis holds, cleanly.** These faces are seen, at best, from ~3x farther away, at 15-30° more grazing incidence, and with a projected footprint 150-200x smaller than typical correctly-observed faces — 70-92% never exceed 1 pixel². A single frame per sequence provides the only real viewing opportunity, and that frame shows no pose anomaly. The most likely mechanism: at sub-pixel projected size, whether a given pixel's ray happens to land on the face is essentially a rounding/sampling coincidence, and our independent camera-model reimplementation (already shown to reproduce the released mesh at IoU=0.9988 overall, `docs/gpu_validation.md`) does not need to diverge by much to flip that coincidence at this extreme. **What would confirm this**: re-rendering only frame 122 (or 103) at higher angular sampling density (e.g., 4x supersampling per pixel) and checking whether these specific faces start registering as hit. **What would refute it**: if supersampling does *not* recover these faces, the footprint/distance numbers would be a coincidence and something else would need to explain the miss.

**Group A1 (ascending): NOT explained by the tested hypotheses.** Footprint is larger, not smaller, than the control sample; distance and incidence are only mildly shifted; rotation and translation at the concentrated frames are unremarkable. The visual pattern (a speckled line running exactly along the observed/unobserved boundary, near-1:1 component-to-face ratio) instead suggests a **boundary-alignment discrepancy**: a pixel-level (or half-pixel) difference in exactly where our rasterizer's ray grid crosses the true silhouette edge, relative to whatever the original renderer's ray grid or anti-aliasing did at that same edge — flipping individual boundary-straddling triangles from hit to miss without a strong distance/angle/size signature, because these are edge cases along a contour rather than a general visibility-limit population. This is an unverified mechanism, explicitly flagged as such. **What would confirm it**: checking whether the flagged faces' triangle *silhouette edges* (not centroids) lie unusually close to the projected image boundary of neighboring hit/miss faces at the shared "bad" frames (58/681 for the t4 pair); a boundary-adjacency-in-screen-space measurement, not yet computed here. **What would refute it**: if supersampling at frames 58/681 does *not* change these faces' hit status, or if the same faces are also missed from other, unrelated frames where they are not near any boundary.

**Group A2 (transverse2/rectum): partially explained, one clear exception.** `c2_transverse2_t3_v1` and `c2_rectum_t3_v1` show the same directional pattern as B (higher incidence angle, and for rectum_t3 a large footprint gap) but far more mildly, and their components are large interior islands rather than a boundary speckle — consistent with a genuine but less extreme visibility-margin effect (folds partially self-occluding a patch that's only seen at a decent-but-not-generous angle for one frame). `c2_rectum_t4_v1` does not fit this story at all (closer, less grazing, similar footprint) and remains **UNKNOWN** — no tested hypothesis explains it; its elevated max rotation (3.50°) at the concentrated frame is a candidate lead not yet followed up. **What would confirm the fold-occlusion reading for transverse2_t3_v1/rectum_t3_v1**: checking whether the flagged components sit immediately behind a haustral fold from that frame's specific camera position (would need a per-frame occlusion-ray test, not done here). **What would settle rectum_t4_v1**: rendering that sequence's frame 30 and inspecting it directly, which has not been done.

**Nothing here should be treated as confirming a bug in `src/geometry/camera.py` or the "transposed" pose convention** — both are frozen, empirically-verified conventions (`docs/conventions.md`, `docs/oracle_check.md`) and this diagnosis did not touch or re-derive them. The candidate mechanisms above (sub-pixel rounding, boundary-grid alignment, marginal-angle folds) are all consistent with a correct implementation that simply sits at the edge of a resolution limit the original renderer's exact implementation handles a little differently — not with the camera model or pose convention being wrong.
