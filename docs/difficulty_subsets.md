# Coverage difficulty subsets

Classifies the 169 registered sequences by the *shape* of their unobserved-area distribution, not just how much is unobserved. Built entirely from already-computed `results/coverage_stats.csv` + `results/coverage_components.json` — no mesh re-analysis. Script: `scripts/difficulty_subsets.py`. Adds a `difficulty_subset` column to `results/coverage_stats.csv` (plus a `largest_component_frac` helper column = `comp_area_top1_mm2 / unobserved_area_mm2`, cross-checked against a direct recomputation from `coverage_components.json`'s raw area lists — max absolute difference 3.3e-16, i.e. the two sources agree exactly).

## Definitions

- **monolithic**: largest unobserved component holds **>70%** of total unobserved area — coverage is bad in essentially one place.
- **fragmented**: **>100** unobserved components **AND** largest component **<40%** of area — coverage gaps are scattered across many small spots, no single dominant hole.
- **mixed**: everything else.

## Counts

| Subset | n | % |
|---|---|---|
| mixed | 98 | 58.0% |
| monolithic | 56 | 33.1% |
| fragmented | 15 | 8.9% |

## Per-subset medians

| Subset | median unobs. frac (area) | median #components >5mm | median #components >10mm | median #fully-interior components |
|---|---|---|---|---|
| fragmented | 0.262 | 8 | 6 | 212 |
| mixed | 0.224 | 6 | 5 | 56 |
| monolithic | 0.244 | 4 | 3 | 41.5 |

Fragmented sequences have both more large (>5/10mm) components *and* dramatically more fully-interior (fold-occlusion-style) components — consistent with their definition: many separate gaps, none dominant. Monolithic sequences have the fewest of everything except overall unobserved fraction, which sits in the middle — a monolithic sequence isn't necessarily better- or worse-covered overall, its badness is just concentrated into one region rather than spread out.

## Segment distribution

| Segment | fragmented | mixed | monolithic |
|---|---|---|---|
| ascending | 10 | 8 | 5 |
| cecum | 0 | 12 | 10 |
| descending | 0 | 10 | 7 |
| rectum | 0 | 14 | 10 |
| sigmoid | 0 | 7 | 5 |
| sigmoid1 | 2 | 7 | 3 |
| sigmoid2 | 0 | 7 | 5 |
| transverse1 | 3 | 17 | 4 |
| transverse2 | 0 | 16 | 7 |

**Fragmentation is heavily concentrated in `ascending`** (10 of 15 fragmented sequences overall, i.e. two-thirds, out of only 23 ascending sequences total) plus a handful of `sigmoid1`/`transverse1`. Five segments (`cecum`, `descending`, `rectum`, `sigmoid`, `sigmoid2`) have **zero** fragmented sequences — their unobserved area, whatever its size, is always concentrated into few-enough components to land in mixed or monolithic. This lines up with `sigmoid1`'s previously-noted extreme component counts (up to 1550, `docs/coverage_stats.md`) and suggests `ascending`'s camera trajectories produce more scattered, less contiguous coverage gaps than other segments.

## Threshold sanity check: 5 sequences closest to each boundary

### Monolithic boundary (`largest_component_frac` vs. 0.70)

| Video Name | Segment | largest_component_frac | n_unobserved_components | assigned |
|---|---|---|---|---|
| c2_sigmoid_t3_v3 | sigmoid | 0.6933 | 47 | mixed |
| c2_sigmoid_t3_v2 | sigmoid | 0.6932 | 62 | mixed |
| c2_transverse1_t2_v1 | transverse1 | 0.7077 | 52 | monolithic |
| c1_ascending_t2_v2 | ascending | 0.7085 | 71 | monolithic |
| c1_ascending_t2_v3 | ascending | 0.7085 | 64 | monolithic |

Tight clustering right at the line (0.693 to 0.709, i.e. within 1.5 percentage points of the 70% cutoff on both sides) — this is a real, populated boundary, not an arbitrary cut through empty space. A sequence like `c2_sigmoid_t3_v3` (69.3%) is one bad frame/trajectory-tweak away from flipping to monolithic; treat the monolithic/mixed distinction as a soft one near this edge, not a hard categorical fact.

### Fragmented count boundary (`n_unobserved_components` vs. 100)

| Video Name | Segment | n_unobserved_components | largest_component_frac | assigned |
|---|---|---|---|---|
| c1_rectum_t4_v2 | rectum | 101 | 0.6252 | mixed |
| c1_ascending_t4_v2 | ascending | 102 | 0.8669 | monolithic |
| c1_descending_t4_v1 | descending | 102 | 0.7634 | monolithic |
| c1_descending_t1_v1 | descending | 97 | 0.7107 | monolithic |
| c1_cecum_t4_v3 | cecum | 97 | 0.7609 | monolithic |

This confirms the **AND** condition is doing real work, not just the count: `c1_rectum_t4_v2` clears the >100 component-count bar but its largest component still holds 62.5% of the area, so it correctly lands in `mixed`, not `fragmented` — component *count* alone doesn't imply fragmentation if one component still dominates. None of the other 4 near this count boundary are anywhere near the 40% frac cutoff either (they're all monolithic-range, 71-87%), so in practice the count threshold rarely binds without the frac condition already having decided the outcome first.

### Fragmented fraction boundary (`largest_component_frac` vs. 0.40)

| Video Name | Segment | largest_component_frac | n_unobserved_components | assigned |
|---|---|---|---|---|
| c2_cecum_t4_v3 | cecum | 0.4048 | 97 | mixed |
| c2_transverse1_t3_v3 | transverse1 | 0.4064 | 71 | mixed |
| c2_transverse1_t3_v2 | transverse1 | 0.4066 | 76 | mixed |
| c2_cecum_t4_v2 | cecum | 0.4068 | 80 | mixed |
| c1_sigmoid2_t1_v3 | sigmoid2 | 0.4091 | 30 | mixed |

None of these 5 have `n_unobserved_components > 100` (max is 97), so this boundary isn't actually "live" for any of them — even if their fraction dropped further, they'd still fail the count condition and stay `mixed`. In this dataset the 40%-fraction cutoff and the 100-component cutoff rarely co-occur near their respective edges; `fragmented` sequences tend to clear both thresholds by a comfortable margin rather than being marginal on one axis.
