# Coverage stats — Task B Part 1 (GT, as released)

Script: `scripts/coverage_stats.py`. 171 `c1`/`c2` `registered_videos` sequences (169 zip archives + 2 pre-extracted dirs; all 22 `c0_*` archives excluded), streamed via `unzip -p` — no extraction. Full log: `logs/coverage_stats.log`. Outputs: `results/coverage_stats.csv` (one row/sequence, joined with `docs/release_v1.csv` on `Video Name`, 171/171 matched), `results/coverage_components.json` (full unobserved-component area lists per sequence).

**Area-to-diameter assumption** (for the 5mm/10mm thresholds): each unobserved connected component's total triangle area is treated as the area of an equivalent circle, `diameter = 2*sqrt(area/pi)`. This is a size proxy for irregular blobs of triangles, not a measured physical diameter — a long thin sliver and a compact patch of the same area get the same "diameter."

**Data issue found and fixed while building this**: `c1_cecum_t1_v3.zip` packages its contents inside a `c1_cecum_t1_v3/` subfolder, unlike every other archive checked (which store `coverage_mesh.obj` at the zip root). Confirmed as an isolated one-off (checked all 55 `*_v3.zip` archives specifically). `src/geometry/coverage_mesh.py`'s zip streamer now resolves the actual internal path via `unzip -Z1` first rather than assuming a fixed layout, so this didn't require excluding the sequence.

## Overall distribution

| Metric | min | 25% | median | 75% | max | mean |
|---|---|---|---|---|---|---|
| unobserved fraction (by area) | 0.043 | 0.172 | 0.242 | 0.287 | 0.534 | 0.233 |
| # unobserved components | 16 | 54.5 | 87 | 136 | 1550 | 143.6 |
| # components > 5mm equiv. diam. | 1 | 3.5 | 5 | 7 | 13 | 5.7 |
| # components > 10mm equiv. diam. | 1 | 3 | 4 | 6 | 9 | 4.3 |
| largest component equiv. diam. (mm) | 21.0 | 49.2 | 56.5 | 65.9 | 106.0 | 58.0 |

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
| transverse1 | 0.285 | 0.286 | 26 |
| sigmoid1 (c1 only) | 0.312 | 0.315 | 12 |

Cecum is by far the best-covered segment (median 0.097) — it's a closed pouch the camera can circle fully. Segments with tubular/traversal geometry (transverse, sigmoid) consistently leave 25-31% of surface area unobserved. Note `sigmoid`/`sigmoid1`/`sigmoid2` are different anatomical-segment labels used by `c1` vs. `c2` phantoms respectively (colon-shape-specific segmentation, not a typo — confirmed via the Segment x Colon crosstab: `sigmoid` only appears for `c2`, `sigmoid1`/`sigmoid2` only for `c1`).

## By Debris (v3 vs. v1/v2)

| Debris | median | mean | n |
|---|---|---|---|
| no | 0.242 | 0.233 | 116 |
| yes | 0.244 | 0.234 | 55 |

Essentially no difference, as expected: the README states v3's pixel-wise ground truth (including `coverage_mesh.obj`) represents the "clean" colon and shares v2's camera trajectory — debris changes the RGB video, not the coverage geometry.

## By Open End Visible

(Normalized case/whitespace — see `docs/data_inventory.md` for the `"No"`/`"no "` inconsistency in the raw index.)

| Open End Visible | median | mean | n |
|---|---|---|---|
| no | 0.137 | 0.169 | 45 |
| yes | 0.266 | 0.256 | 126 |

Sequences where the far/uninserted end is visible have nearly double the unobserved fraction — plausible: seeing down an open tube usually means that distal region is glimpsed from a distance rather than closely traversed and covered.

## By Qualitative Score

| Qualitative Score | median | mean | n |
|---|---|---|---|
| 1 (best alignment) | 0.238 | 0.223 | 87 |
| 2 (good alignment) | 0.230 | 0.222 | 40 |
| 3 (misalignment, phantom defects) | 0.262 | 0.264 | 44 |
| (23 rows have no score in the index) | | | |

Weak trend: score-3 (misaligned) sequences skew slightly higher unobserved fraction, but the effect is small relative to the segment-level differences above — coverage fraction alone isn't a strong proxy for alignment quality.

## Extremes

Most-covered: `c2_cecum_t4_v2`/`_v3`, `c2_cecum_t3_v1` (all cecum, ~4-7% unobserved). Least-covered: `c2_transverse2_t3_v2`/`_v3` (~52-53% unobserved), `c1_sigmoid1_t1_v2`/`_v3` (~43%).
