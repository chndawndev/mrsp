# Unobserved regions (component-level analysis)

Shifts the unit of analysis from sequences to individual unobserved connected components ("regions"). Script: `scripts/regions.py`, streamed from all 169 zip archives (reuses `scripts/coverage_stats.py`'s `discover_sequences()`, so the same 169-sequence guarantee applies). Full log: `logs/regions.log` (~12 min). Outputs: `results/regions.csv` (24,208 rows, one per unobserved component) and `results/mesh_identity.csv` (169 rows, one per sequence, with an exact content hash of its mesh's vertex array — used for the independent-mesh analysis below).

Also added to `results/coverage_stats.csv`: `fragmentation_score_lt10mm` = fraction of a sequence's unobserved *area* held by components with equivalent diameter < 10mm (continuous companion to the categorical `difficulty_subset`, which is kept as-is, descriptive only).

## `results/regions.csv` columns

`Video Name, Colon, Segment, Phantom Number, Video Number, Debris` (identifying columns, joined from `docs/release_v1.csv`), `area_mm2, equiv_diam_mm, n_faces, min_boundary_dist_mm, median_boundary_dist_mm, fully_interior, compactness, centroid_axis_position`.

- **compactness** = triangle area / (2D bounding-box area of the component in its own local best-fit plane, found via SVD of the component's vertices). Near 1 = compact patch filling its bounding box; near 0 = long thin sliver. **Values can exceed 1** (max observed 1.71) when a component has real out-of-plane surface curvature (e.g. crosses a fold) — its true 3D surface area then exceeds the area of its flattened 2D footprint. This is expected, not a bug.
- **centroid_axis_position**: the component's centroid, projected onto the mesh's PCA long axis (same method as `scripts/render_coverage_views.py`), min-max normalized to [0,1] using that mesh's own vertex extent. 0/1 = the two ends of the tube.

## Counts

| Filter | count |
|---|---|
| all unobserved regions | 24,208 |
| equivalent diameter > 5mm | 957 |
| equivalent diameter > 10mm | 719 |

The vast majority of "regions" (23,251 of 24,208, 96%) are under 5mm equivalent diameter — tiny fragments. This matches the earlier finding that `fragmentation_score_lt10mm` (fraction of *area*, not count, from <10mm components) has a median of only 1.5%: there are enormous numbers of small components, but they carry almost none of the total unobserved area. Region **count** and region **area** tell very different stories and shouldn't be conflated.

## Distributions

| Metric | population | min | 25% | median | 75% | max |
|---|---|---|---|---|---|---|
| area (mm2) | all regions | 0.000 | 0.014 | 0.038 | 0.062 | 8828.0 |
| area (mm2) | >5mm only | 19.7 | 79.2 | 247.1 | 1166.3 | 8828.0 |
| compactness | all regions | 0.000 | 0.467 | 0.479 | 0.496 | 1.710 |
| compactness | >5mm only | 0.280 | 0.596 | 0.677 | 0.763 | 1.658 |

- **fully_interior fraction**: 68.1% of ALL regions are fully interior (0% of faces within 5mm of a boundary) — but restricted to >5mm regions, only 46.5% are. Small regions are overwhelmingly interior fold-occlusion artifacts; among the larger, more consequential regions, it's closer to a coin flip between "interior" and "touches an open end somewhere."
- **compactness** is markedly lower (less compact / more elongated) for the mass of tiny regions (median 0.479) than for >5mm regions (median 0.677) — small fragments tend to be thin slivers (e.g. along a fold edge), while larger regions are more compact blobs, consistent with the qualitative renders in `docs/openend_artifact_check.md`.
- **centroid_axis_position** (>5mm regions): median 0.479 (near the middle), but 23.6% sit near either end (<0.1 or >0.9) — a modest, not overwhelming, concentration at the tube extremities. Fully-interior >5mm regions skew slightly more central (median axis position 0.490) than non-fully-interior ones (0.418, i.e. closer to an end) — small directional signal, consistent with (not a repeat of) the `docs/openend_artifact_check.md`/`docs/coverage_stats.md` boundary-distance findings, since axis position and boundary distance measure related but distinct things (position along the tube vs. literal distance to the nearest boundary edge).

## Regions per sequence, and how many distinct meshes contribute

Regions per sequence: min 16, median 86, mean 143.2, max 1550 (same distribution as `n_unobserved_components` in `docs/coverage_stats.md`, since each region *is* one component).

**169 sequences are backed by only 103 distinct meshes** (exact vertex-array hash match) — a third of all "sequences" are geometric duplicates of another sequence's phantom scan, just viewed under a different camera path.

## Independent meshes and trajectories: the key caveat for any per-sequence statistics

This matters directly for how much independent statistical information 169 "sequences" actually carry, and the mesh-sharing pattern is **more nuanced than the README's wording suggests**:

- **v2 and v3 share the identical mesh in 48 of 58 combos where both exist (82.8%) — but NOT all**: 10 combos (17.2%) have genuinely different v2/v3 meshes despite the README stating v3 "uses the same camera trajectory and imaging settings as v2." (Confirmed independently: v2/v3 `pose.txt` files are numerically close but not byte-identical either — same frame count, values agreeing to ~3 decimal places, consistent with a repeated-but-not-identical robot trajectory, not a shared trajectory file.)
- **v1 usually has its own distinct mesh from v2/v3**: only 18 of 58 combos (31%) have v1 sharing v2's mesh; in 40 of 58 (69%), v1 is a completely different mesh (different vertex/face count, not just different coordinates) — i.e., the "same phantom segment" was independently re-scanned/re-registered for v1's recording session in most cases.
- Per (Colon, Segment, Phantom Number) combo: **20 combos (34%) share exactly 1 mesh** across all their videos, **31 combos (53%) have 2 distinct meshes**, **7 combos (12%) have 3 fully distinct meshes** (v1, v2, v3 all different).
- Overall: **169 sequences → 103 distinct meshes → 113 distinct trajectories** (using the README's definition that v3 shares v2's trajectory; trajectories exceed meshes because a mesh shared by v1=v2 is still imaged by 2 different camera paths).

### By segment: distinct meshes, distinct trajectories, and single-mesh dominance

| Segment | n sequences | n distinct meshes | n distinct trajectories | n regions | top-mesh region-count share | top-mesh region-area share |
|---|---|---|---|---|---|---|
| ascending | 23 | 14 | 16 | 3293 | 16.8% | 15.7% |
| cecum | 22 | 14 | 14 | 2907 | 14.3% | 15.5% |
| descending | 17 | 12 | 11 | 2297 | 16.7% | 13.1% |
| rectum | 24 | 14 | 16 | 3418 | 14.4% | 12.0% |
| **sigmoid** | 12 | 6 | 8 | 433 | **34.4%** | **27.1%** |
| **sigmoid1** | 12 | 7 | 8 | 7837 | **38.5%** | **24.1%** |
| **sigmoid2** | 12 | 7 | 8 | 408 | **28.4%** | **27.1%** |
| transverse1 | 24 | 13 | 16 | 2420 | 27.4% | 15.5% |
| transverse2 | 23 | 16 | 16 | 1195 | 13.8% | 10.2% |

**No segment has a single mesh contributing a literal majority of its regions or region-area** (all top-mesh shares are under 50%), so none is "dominated" in the strictest sense. **But `sigmoid`, `sigmoid1`, and `sigmoid2` stand out clearly** as the segments to treat with the most caution: each has only 12 sequences reducing to just 6-7 distinct meshes (roughly half the nominal sample size), and their single most-represented mesh accounts for 24-34% of that segment's total region area — 1.5-2x the share seen in any other segment. These three are also exactly the segments exclusive to one colon shape (`sigmoid`=c2 only, `sigmoid1`/`sigmoid2`=c1 only, per `docs/coverage_stats.md`), so they additionally get no cross-colon diversity that the other 6 segments have. **Any conclusion drawn specifically about `sigmoid`/`sigmoid1`/`sigmoid2` in isolation should be read as resting on effectively 6-8 independent geometries, not 12 independent sequences.**

## Practical implication

Treating each of the 169 registered sequences (or worse, each of the 24,208 regions) as an independent sample overstates the dataset's effective diversity by roughly 1.6x at the mesh level (169/103) dataset-wide, and by up to ~2x within the three sigmoid-family segments specifically. Any downstream statistical test (e.g. comparing coverage or fragmentation across conditions) should account for this — at minimum by being aware that many "different" sequences are the same physical surface geometry evaluated under a different camera path, not an independent draw of anatomy.
