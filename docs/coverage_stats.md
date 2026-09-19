# Coverage stats — Task B Part 1 (GT, as released)

Script: `scripts/coverage_stats.py`. **169** `c1`/`c2` `registered_videos` sequences, discovered ONLY from `*.zip` archives (never from extracted directories — see correction below), streamed via `unzip -p` — no extraction. All 22 `c0_*` archives excluded. Full log: `logs/coverage_stats.log` (~12 min runtime, boundary-distance pass included; that run predates the row-count correction below, hence 171 raw output rows — see correction). Outputs: `results/coverage_stats.csv` (one row/sequence, joined with `docs/release_v1.csv` on `Video Name`, 169/169 matched), `results/coverage_components.json` (full unobserved-component area lists per sequence, 169 keys).

**Area-to-diameter assumption** (for the 5mm/10mm thresholds): each unobserved connected component's total triangle area is treated as the area of an equivalent circle, `diameter = 2*sqrt(area/pi)`. This is a size proxy for irregular blobs of triangles, not a measured physical diameter — a long thin sliver and a compact patch of the same area get the same "diameter."

**Data issue found and fixed while building this**: `c1_cecum_t1_v3.zip` packages its contents inside a `c1_cecum_t1_v3/` subfolder, unlike every other archive checked (which store `coverage_mesh.obj` at the zip root). Confirmed as an isolated one-off (checked all 55 `*_v3.zip` archives specifically). `src/geometry/coverage_mesh.py`'s zip streamer now resolves the actual internal path via `unzip -Z1` first rather than assuming a fixed layout, so this didn't require excluding the sequence.

**Correction (2026-09-18)**: the original run also discovered sequences from 2 extracted directories present at the time (`c1_transverse1_t1_v1/`, `c2_transverse1_t1_v1/`) alongside their matching `.zip` archives — those directories have since disappeared from the (nominally read-only) dataset root, confirming they were an incidental/temporary extraction, not part of the dataset proper. This produced 2 duplicate rows (identical content either way, since both paths pointed at the same `coverage_mesh.obj`) in a 171-row CSV. `discover_sequences()` now reads only `*.zip` archives, deduplicates by name, and asserts exactly 169 sequences. `results/coverage_stats.csv` was corrected by dropping the 2 duplicate rows and fixing a trailing-space column name (`"Video Number "` → `"Video Number"`); every table below is recomputed from the corrected 169-row CSV (no mesh re-analysis needed — the duplicates carried identical values). All numbers below shifted negligibly from the original 171-row version. Separately, while recomputing, an unrelated pre-existing error was caught and fixed: the old "By Qualitative Score" table claimed "23 rows have no score," which was simply wrong (87+40+44 already summed to the full row count) — there are in fact zero missing Qualitative Scores in this file.

## Overall distribution

| Metric | min | 25% | median | 75% | max | mean |
|---|---|---|---|---|---|---|
| unobserved fraction (by area) | 0.043 | 0.169 | 0.241 | 0.289 | 0.534 | 0.233 |
| # unobserved components | 16 | 54 | 86 | 135 | 1550 | 143.2 |
| # components > 5mm equiv. diam. | 1 | 3 | 5 | 7 | 13 | 5.7 |
| # components > 10mm equiv. diam. | 1 | 3 | 4 | 6 | 9 | 4.3 |
| largest component equiv. diam. (mm) | 21.0 | 49.2 | 56.5 | 65.9 | 106.0 | 57.9 |

Unobserved-by-count and unobserved-by-area fractions track each other closely per sequence (both computed, see CSV) — unobserved faces aren't systematically smaller/larger than observed ones on average.

Every sequence has at least one unobserved region exceeding 10mm equivalent diameter — full 100% coverage of any segment never actually happens in this dataset, consistent with the phantoms' fixed camera trajectories not fully circling every surface.

## By Segment

| Segment | median unobs. frac (area) | mean | n |
|---|---|---|---|
| cecum | 0.097 | 0.110 | 22 |
| rectum | 0.182 | 0.193 | 24 |
| sigmoid2 (c1 only) | 0.198 | 0.187 | 12 |
| transverse2 | 0.246 | 0.272 | 23 |
| descending | 0.256 | 0.250 | 17 |
| ascending | 0.262 | 0.241 | 23 |
| sigmoid (c2 only) | 0.273 | 0.275 | 12 |
| transverse1 | 0.285 | 0.287 | 24 |
| sigmoid1 (c1 only) | 0.312 | 0.315 | 12 |

Cecum is by far the best-covered segment (median 0.097) — it's a closed pouch the camera can circle fully. Segments with tubular/traversal geometry (transverse, sigmoid) consistently leave 25-31% of surface area unobserved. Note `sigmoid`/`sigmoid1`/`sigmoid2` are different anatomical-segment labels used by `c1` vs. `c2` phantoms respectively (colon-shape-specific segmentation, not a typo — confirmed via the Segment x Colon crosstab: `sigmoid` only appears for `c2`, `sigmoid1`/`sigmoid2` only for `c1`).

## By Debris (v3 vs. v1/v2)

| Debris | median | mean | n |
|---|---|---|---|
| no | 0.240 | 0.232 | 114 |
| yes | 0.244 | 0.234 | 55 |

Essentially no difference, as expected: the README states v3's pixel-wise ground truth (including `coverage_mesh.obj`) represents the "clean" colon and shares v2's camera trajectory — debris changes the RGB video, not the coverage geometry.

## By Open End Visible

(Normalized case/whitespace — see `docs/data_inventory.md` for the `"No"`/`"no "` inconsistency in the raw index.)

| Open End Visible | median | mean | n |
|---|---|---|---|
| no | 0.137 | 0.169 | 45 |
| yes | 0.264 | 0.256 | 124 |

Sequences where the far/uninserted end is visible have nearly double the unobserved fraction — plausible: seeing down an open tube usually means that distal region is glimpsed from a distance rather than closely traversed and covered.

## By Qualitative Score

| Qualitative Score | median | mean | n |
|---|---|---|---|
| 1 (best alignment) | 0.235 | 0.221 | 85 |
| 2 (good alignment) | 0.230 | 0.222 | 40 |
| 3 (misalignment, phantom defects) | 0.262 | 0.264 | 44 |

(85+40+44 = 169: every sequence has a Qualitative Score, none missing.)

Weak trend: score-3 (misaligned) sequences skew slightly higher unobserved fraction, but the effect is small relative to the segment-level differences above — coverage fraction alone isn't a strong proxy for alignment quality.

## Boundary-distance breakdown (full-dataset generalization of `docs/openend_artifact_check.md`)

`docs/openend_artifact_check.md` qualitatively checked 6 sequences by hand (rendering + boundary-loop analysis) and concluded that large unobserved regions typically touch an open-end boundary but mostly consist of genuine lumen wall well inside the tube. This section reruns that boundary-distance idea as a cheap per-face metric over **all 169 sequences**: for every unobserved face, Euclidean distance (KD-tree) to the nearest boundary (open-end) vertex, then area-weighted aggregation. Same area-to-diameter assumption as above for the ">10mm" component filter. New columns in `results/coverage_stats.csv`: `unobs_area_within_{5,10}mm_of_boundary_{mm2,frac}`, `large_unobs_area_within_{5,10}mm_of_boundary_frac` (restricted to components with equivalent diameter > 10mm), `n_fully_interior_components`, `fully_interior_area_mm2` (components with 0% of faces within 5mm of any boundary — fully interior gaps).

### Overall

| Metric | median | mean |
|---|---|---|
| unobserved area within 5mm of a boundary | 19.8% | 24.0% |
| unobserved area within 10mm of a boundary | 37.7% | 41.6% |
| — restricted to components >10mm diam., within 5mm | 20.0% | 23.9% |
| — restricted to components >10mm diam., within 10mm | 37.5% | 41.9% |
| # fully-interior components (0% within 5mm) | 58 | 97.6 |
| fully-interior components' area, as % of total unobserved area | 11.2% | 16.8% |

**This confirms the 6-sequence finding at full dataset scale: most unobserved area is not an open-end artifact.** Median 80.2% of unobserved area is farther than 5mm from any boundary, 62.3% farther than 10mm — and restricting to only the large (>10mm diameter) components barely changes these numbers (20.0% vs 19.8% within 5mm), so large components are not disproportionately edge-hugging. Only 11.2% of unobserved area (median) comes from components that are *fully* interior (zero near-boundary faces) — meaning most of the "far from boundary" area belongs to large components that also touch a boundary somewhere (exactly the pattern seen in the rendered images: one big region anchored at an open end but extending deep inward), not to fully isolated patches.

### By Segment (median)

| Segment | % unobs. area <5mm of boundary | % unobs. area <10mm | # fully-interior components | fully-interior area, % of unobs. |
|---|---|---|---|---|
| descending | 13.4% | 25.3% | 33 | 28.8% |
| transverse1 | 16.7% | 28.1% | 56 | 31.5% |
| sigmoid1 | 17.2% | 29.9% | 256.5 | 17.2% |
| ascending | 19.1% | 35.6% | 91 | 36.6% |
| transverse2 | 19.1% | 33.2% | 36 | 10.3% |
| rectum | 25.8% | 46.1% | 99 | 0.6% |
| sigmoid | 26.2% | 47.7% | 16.5 | 0.05% |
| cecum | 29.9% | 47.7% | 89.5 | 9.6% |
| sigmoid2 | 43.4% | 70.6% | 21 | 1.1% |

No clean monotonic pattern with overall coverage quality (cecum is best-covered overall but has one of the *highest* near-boundary fractions — its unobserved area, small as it is, sits close to its single opening). `sigmoid1` stands out with a huge fully-interior component count (median 256.5) — consistent with `docs/coverage_stats.md`'s earlier note that `sigmoid1` sequences had unusually high component counts (up to 1550) — lots of small isolated fold-occlusion artifacts, not boundary effects.

### By Open End Visible (median)

| Open End Visible | % unobs. area <5mm of boundary | % unobs. area <10mm | # fully-interior components | fully-interior area, % of unobs. |
|---|---|---|---|---|
| no | 31.0% | 52.8% | 95 | 7.0% |
| yes | 18.2% | 33.6% | 48 | 17.2% |

**Counter-intuitive, and consistent with the 6-sequence qualitative check**: sequences where the far end is **not** visible have a *higher* near-boundary unobserved fraction, not lower. A naive story ("visible far end → glimpsed at a distance → more edge artifact") predicts the opposite. The more likely explanation: when the far end isn't visible, the camera never reaches/opens up that end at all, so the unobserved region there is a small cap immediately at the opening (high near-boundary fraction, few fully-interior components); when the far end *is* visible, the camera reaches further down the tube, converting what would otherwise be an edge-cap gap into a longer stretch of unobserved lumen wall reaching inward (lower near-boundary fraction, more fully-interior components, nearly 2.5x the fully-interior area share).

## Extremes

Most-covered: `c2_cecum_t4_v2`/`_v3`, `c2_cecum_t3_v1` (all cecum, ~4-7% unobserved). Least-covered: `c2_transverse2_t3_v2`/`_v3` (~52-53% unobserved), `c1_sigmoid1_t1_v2`/`_v3` (~43%).
