# Region IoU: implementation and validity gate

Implements `docs/success_criteria.md` section 6 (2026-09-27 deviation),
items 1 and 2. Evaluated on **oracle and the two location-blind baselines
of the D1.3 diagnosis only** (all-unobserved; area-matched random, 20
seeds) — no real pipeline (EndoDAC, MASt3R-SLAM, or any other) is scored
with this metric in this task, per instructions. `src/eval/`, `src/gt/`,
`docs/success_criteria.md`, and `docs/eval_protocol.md` are unedited;
region IoU is implemented in `src/eval_ext/region_iou.py`, outside the
locked paths, importing (not reimplementing) the locked component
construction (`eval.regions.compute_regions`).

MEASURED sections report only what was directly observed (tests run,
files read, numbers computed). INTERPRETATION is marked explicitly.

---

## Definition as implemented

Quoted verbatim from `docs/success_criteria.md` section 6, item 1
(2026-09-27 deviation):

> For each GT unobserved region R: let C be the union of all predicted-
> unobserved connected components (built exactly as in the locked
> component construction, ignore set excluded) that intersect R.
> IoU(R) = area(R intersect C) / area(R union C); IoU(R) = 0 if no
> component intersects R. Corpus value: mean over regions, pooled per
> docs/eval_protocol.md, mesh-level cluster bootstrap, with the
> (Colon, Segment) and (Colon, Segment, Phantom Number) intervals
> reported alongside. Implemented outside the locked paths.

Implementation: `src/eval_ext/region_iou.py::region_iou`. For a GT
region `R`, finds every predicted component (from
`eval.regions.compute_regions` on `predicted_unobserved & ~ignore_set`,
the same construction `src/eval/region_metrics.py`'s localization error
already uses) that shares at least one face with `R`, takes the face-set
union of all of them as `C`, and returns
`area(R ∩ C) / area(R ∪ C)` (0 if no component intersects `R`). This
differs from `matching_predicted_component` (single largest-overlap
component, used for localization error) by using the union of *every*
overlapping component, not just the best one — a region split across
several small predicted fragments is credited for all of them jointly,
and penalized once for however far any of them spill outside `R`.

Quoted, item 2 (validity gate):

> Validity gate for region IoU, evaluated only on oracle and the two
> location-blind baselines of the D1.3 diagnosis (all-unobserved;
> area-matched random, 20 seeds), at tau = 0.25, medium + large regions.
> No real pipeline's output is used for the gate.
> Pass: oracle median IoU >= 0.80, and oracle median IoU exceeds each
> baseline's median by >= 0.30.
> If the gate fails, region IoU is not used as a primary endpoint; stop
> and diagnose before any pipeline is scored with it.

---

## Unit tests (MEASURED)

`tests/eval_ext/test_region_iou.py`, 9 tests, all passing (synthetic
face sets, unit face area unless noted, hand-computed expected values):

| Case | Setup | Expected | Result |
|---|---|---|---|
| Identical | component = region | IoU = 1.0 | PASS |
| Disjoint | no shared faces | IoU = 0.0 | PASS |
| No overlapping component at all | empty component list | IoU = 0.0 | PASS |
| Component strict superset | region area 4, component area 8, region ⊂ component | IoU = 4/8 = 0.5 | PASS |
| Several fragments inside region | region area 8, two disjoint fragments (area 2 each) both ⊂ region | IoU = 4/8 = 0.5 | PASS |
| Fragment straddling boundary | region {0,1,2,3}, component {2,3,4,5} | IoU = 2/6 = 1/3 | PASS |
| Two regions touched by one merged component | one component spans both regions' territory | both regions get IoU = 3/7 independently, symmetric penalty | PASS |
| `region_iou_many` batch consistency | same inputs via loop vs. batch call | identical results | PASS |
| Non-uniform face areas | area-weighted intersection/union, not face-count | IoU = 5/20 = 0.25 | PASS |

Full existing suite (`tests/`, 60 tests including these 9) re-run after
adding this module: **60/60 pass**, no regressions.

---

## Corpus computation (MEASURED)

`scripts/region_iou_gate.py`, all 169 registered sequences (confirmed via
`discover_sequences_with_ignore_set()`, reused from `scripts/
diagnose_d1_3.py` — same set already used for D1 Stage B and the D1.3
diagnosis), tau = 0.25. CPU-only, no GPU, no new ray casting: oracle's
`predicted_unobserved` is unpacked directly from Stage B's
already-computed `predicted_observed_packed.bin`; all-unobserved and
area-matched random are constructed the same way `scripts/
diagnose_d1_3.py` already does (same target area — `fully_predicted`'s
predicted-unobserved area at tau=0.25 — same seed formula
`1000*20260925 + seed`, `seed` in 0..19). Wall time: 1130.7 s (≈18.8
min), 21,054 raw (region × baseline × seed) rows written to
`results/region_iou_gate/region_iou_raw.csv`.

Aggregation: pooled mean and median over regions (equal weight per
region), mesh-level cluster bootstrap (10,000 replicates, seed
20260925), (Colon, Segment) and (Colon, Segment, Phantom Number)
sensitivity intervals alongside — all via
`scripts/region_iou_gate.py::bootstrap_mean_and_median`, structurally
the same replicate loop as `scripts/eval_d1_aggregate.py::bootstrap_median`
extended to track the mean too (region IoU's corpus value is a pooled
mean/median over dimensionless per-region ratios, not a ratio of summed
areas, so `bootstrap_ratio` doesn't apply here). For the area-matched
random baseline, both reporting modes the task asked for are computed:
(a) the 20 seeds' own per-seed point estimates, reported as
mean-of-seeds plus min–max range over seeds, and (b) a seed-averaged
per-region value (each region's IoU averaged across its 20 seeds first),
then pooled and mesh-bootstrapped the same way as oracle/all-unobserved
— matching the dual-reporting pattern `scripts/diagnose_d1_3.py` already
established for exactly this baseline.

### Results, tau = 0.25, by size class

n_regions: small=238 (75 meshes), medium=277 (82 meshes), large=442 (103
meshes), medium+large=719 (103 meshes).

**Oracle**

| size class | mean | median | median 95% CI (mesh) |
|---|---|---|---|
| small | 0.9905 | 0.9932 | [0.9916, 0.9950] |
| medium | 0.9967 | 0.9983 | [0.9976, 0.9992] |
| large | 0.9987 | 0.9992 | [0.9991, 0.9994] |
| **medium+large** | **0.9979** | **0.9991** | **[0.9989, 0.9993]** |

**All-unobserved baseline**

| size class | mean | median | median 95% CI (mesh) |
|---|---|---|---|
| small | 0.0733 | 0.0020 | [0.0018, 0.0023] |
| medium | 0.0367 | 0.0074 | [0.0070, 0.0083] |
| large | 0.0855 | 0.0618 | [0.0575, 0.0725] |
| **medium+large** | **0.0667** | **0.0296** | **[0.0229, 0.0395]** |

**Area-matched random baseline (20 seeds)**

| size class | mean-of-seeds (range) | median-of-seeds (range) | seed-avg pooled median (95% CI) |
|---|---|---|---|
| small | 0.3586 [0.3491, 0.3680] | 0.3977 [0.3888, 0.4091] | 0.4013 [0.3728, 0.4205] |
| medium | 0.3993 [0.3952, 0.4069] | 0.4393 [0.4259, 0.4503] | 0.4368 [0.3950, 0.4730] |
| large | 0.4520 [0.4491, 0.4549] | 0.4983 [0.4943, 0.5014] | 0.4998 [0.4699, 0.5150] |
| **medium+large** | **0.4317 [0.4290, 0.4358]** | **0.4764 [0.4735, 0.4788]** | **0.4758 [0.4386, 0.5003]** |

Full per-size-class results (including both sensitivity clustering
levels, (Colon, Segment) and (Colon, Segment, Phantom Number)) in
`results/region_iou_gate/region_iou_summary.json`. Raw per-region rows
in `results/region_iou_gate/region_iou_raw.csv`.

---

## Gate (MEASURED, mechanical)

Medium+large regions, tau = 0.25, medians (`results/region_iou_gate/gate_result.json`):

| | value |
|---|---|
| Oracle median IoU | 0.9991 |
| All-unobserved median IoU | 0.0296 |
| Area-matched random median IoU (seed-averaged) | 0.4764 |
| Margin, oracle − all-unobserved | 0.9695 |
| Margin, oracle − area-matched random | 0.5227 |
| Threshold: oracle median ≥ | 0.80 |
| Threshold: margin ≥ | 0.30 |

- Oracle median ≥ 0.80: **PASS** (0.9991).
- Margin over all-unobserved ≥ 0.30: **PASS** (0.9695).
- Margin over area-matched random ≥ 0.30: **PASS** (0.5227).

**GATE RESULT: PASS.** Region IoU is validated as usable for a primary
endpoint under item 2's rule.

---

## INTERPRETATION (not part of the mechanical gate)

- Oracle's median IoU is not exactly 1.0 (0.9991, not 1.0) despite oracle
  using GT depth and GT pose. This is consistent with the same
  silhouette-aliasing noise already characterized in Stage 3 of the
  `src/eval/` implementation (a small number of extra/missing single-face
  specks per sequence) — plausible, not independently re-confirmed in
  this task.
- The area-matched random baseline's IoU (≈0.48 median at medium+large)
  is much higher than the all-unobserved baseline's (≈0.03) and not
  close to 0. This is expected, not a sign of a mis-specified baseline:
  the random mask is sized to match `fully_predicted`'s over-flagged area
  (roughly double GT's true unobserved area, per the D1.3 diagnosis), and
  a randomly scattered mask of that size still has substantial expected
  overlap with a large GT region purely from the region's own share of
  total mesh area — this is exactly the kind of degenerate-but-plausible
  score the gate's margin requirement (not just a floor on oracle) is
  designed to catch, and here it still leaves oracle 0.52 above it.
- The gate's oracle margin over the random baseline (0.52) is
  comfortably larger than the required 0.30, and the margin over
  all-unobserved (0.97) is close to the maximum possible (1.0) — the
  gate is not passing narrowly.

---

## Open issues

1. No real pipeline is scored with region IoU yet — this task's scope
   was the gate only, per instructions. A subsequent stage will need to
   compute region IoU for EndoDAC (and any other evaluated pipeline)
   under `tau ∈ {0.15, 0.25, 0.35, 0.50}` before D2b.2 (section 6, item
   3) can be evaluated.
2. Oracle's sub-1.0 median IoU (0.9991) is attributed to the
   already-characterized aliasing noise but not independently
   re-verified in this task (see INTERPRETATION above) — would need a
   face-level diff against Stage 3's already-published extra/missing
   face counts to confirm the same mechanism, not a new one.
