# D1.3 failure diagnosis (post-hoc, descriptive)

D1.3 failed on the full corpus (Stage B, `docs/d1_stage_b.md`): oracle
minus fully_predicted recall on medium+large regions = **1.67
percentage points**, threshold 10pp. `docs/success_criteria.md` §2's
pre-registered fail branch requires a diagnosis before adding methods,
naming two candidate causes without deciding between them: **(A) the
detection definition is insensitive**, or **(B) the regions are too
easy**. Everything below is a **post-hoc diagnostic, designed after
seeing the D1 results** -- none of it changes D1's verdict, produces a
new pass/fail judgment, or proposes a replacement metric. I am not
writing the `docs/success_criteria.md` §6 entry.

Code: `scripts/diagnose_d1_3_ignore_set.py` (recomputes the per-face
ignore set Stage B didn't save, GPU, ~24 min actual for all 169
sequences), `scripts/diagnose_d1_3.py` (the four diagnostics, CPU-only,
~24 min actual). Bootstrap seed `20260925`, 10,000 replicates throughout,
same as Stage B. Outputs in `results/d1/diagnosis/` (gitignored).

---

## MEASURED

### Diagnostic 1: location-blind baselines (tau=0.25)

Two baselines that see no image data at all, evaluated through the exact
same locked `src/eval/region_metrics.py` functions as the real
configurations:

| baseline | recall (medium+large) @25% | @50% | @75% | false_reassurance | false_alarm | loc. err. median (mm) |
|---|---|---|---|---|---|---|
| **all-unobserved** (every face predicted unobserved) | 1.000 | 1.000 | 1.000 | 0.000 | 0.759 [0.734, 0.770] | 48.14 [46.67, 49.58] |
| **area-matched random**, area = fully_predicted's tau=0.25 area, 20 seeds | 0.973 (seed range 0.969-0.976) | 0.624 [0.527, 0.718] | 0.032 (seed range 0.028-0.033) | 0.479 [0.438, 0.494] | 0.759 [0.734, 0.770] | 8.04 [6.85, 10.40] |
| (for reference) **oracle** | 1.000 | 1.000 | 1.000 | 0.0015 | 0.0006 | 0.008 |
| (for reference) **fully_predicted** | 0.993 | 0.983 | 0.958 | 0.031 | 0.543 | 17.67 |

`[lo, hi]` = mesh-level bootstrap 95% CI; the random baseline's range is
the min-max across its 20 seeds (a different quantity from the CI of the
seed-averaged value, both reported per the task spec).

**Sensitivity CIs** (`docs/eval_protocol.md`'s two coarser clusterings,
reported alongside every mesh-level CI above, not decision-driving),
`results/d1/diagnosis/sensitivity_cis_followup.json`:

| baseline | quantity | (Colon, Segment), 15 clusters | (Colon, Segment, Phantom Number), 58 clusters |
|---|---|---|---|
| all-unobserved | recall@50% (medium+large) | [1.000, 1.000] | [1.000, 1.000] |
| all-unobserved | false_reassurance | [0.000, 0.000] | [0.000, 0.000] |
| all-unobserved | false_alarm | [0.724, 0.795] | [0.738, 0.782] |
| all-unobserved | loc. err. median | [44.32, 51.04] | [46.33, 49.85] |
| area-matched random | recall@50% (medium+large) | [0.445, 0.772] | [0.511, 0.728] |
| area-matched random | false_reassurance | [0.433, 0.534] | [0.451, 0.515] |
| area-matched random | false_alarm | [0.725, 0.796] | [0.738, 0.782] |
| area-matched random | loc. err. median | [6.82, 10.51] | [6.85, 10.10] |

**False reassurance, oracle vs. pred_depth_only, tau=0.25** (corpus-wide
pooled, `docs/eval_protocol.md`'s aggregation rule): oracle **0.0015**,
pred_depth_only **0.0004**. Face count predicted-observed but
GT-unobserved, pred_depth_only, tau=0.25, corpus-wide: **7,669 faces**
(`results/d1/diagnosis/sensitivity_cis_followup.json` and a one-off
recount from `results/d1/per_sequence/*/metrics.json` +
`predicted_observed_packed.bin`, both already-saved Stage B/diagnosis
outputs).

### Diagnostic 2: recall sensitivity table (all configs x taus x thresholds x size classes)

Full table: `results/d1/diagnosis/recall_sensitivity.csv` (192 rows).
Oracle-minus-config gap, medium+large, across the whole sweep:

| config | tau | thresh=25% | thresh=50% | thresh=75% |
|---|---|---|---|---|
| pred_depth_only | any | 0.00pp | 0.00pp | 0.00pp |
| pred_pose_only | 0.25 | 1.67pp | 3.48pp | 9.18pp |
| fully_predicted | 0.15 | 0.28pp | 1.25pp | 3.34pp |
| fully_predicted | 0.25 | 0.70pp | 1.67pp | 4.17pp |
| fully_predicted | 0.35 | 0.97pp | 1.81pp | 4.73pp |
| fully_predicted | 0.50 | 1.39pp | 2.92pp | 5.84pp |

**The gap never reaches 10pp anywhere in the sweep** -- not at the
stricter 75% detection threshold, not at any tau. `pred_depth_only`'s
gap is exactly 0.00pp at every cell (it detects every medium+large
region regardless of threshold or tau).

### Diagnostic 3: region difficulty (coverage fraction distribution, tau=0.25)

Full table: `results/d1/diagnosis/region_difficulty.csv`.

| config | size class | min | q25 | median | q75 | max | share in [0.40, 0.60] |
|---|---|---|---|---|---|---|---|
| oracle | small/medium/large | 0.89-0.99 | 0.986-0.999 | 0.993-0.9996 | 0.999-1.0 | 1.0 | 0.0% |
| pred_depth_only | small/medium/large | 0.89-0.99 | 0.998-1.0 | 1.0 | 1.0 | 1.0 | 0.0% |
| pred_pose_only | small/medium/large | 0.0-0.23 | 0.85-0.93 | 0.989-0.994 | 1.0 | 1.0 | 1.1-5.1% |
| fully_predicted | small/medium/large | 0.0-0.21 | 0.98-0.995 | 1.0 | 1.0 | 1.0 | 0.5-3.8% |

Median coverage is above 0.98 for every configuration and size class
(oracle and pred_depth_only above 0.999). The share of regions sitting
near the 50% detection boundary ([0.40, 0.60]) is 0% for the two
depth-accurate configs and only 1-5% for the two pose-error configs.

### Diagnostic 4: region-merging check (tau=0.25, every configuration)

For every matched headline region: ratio of the matched predicted
component's area to the GT region's own area, correlated (Spearman,
mesh-level bootstrap CI) against localization error:

p-values are not reported: they assume independent regions, which these
are not (957 regions come from only 103 meshes, 15 (Colon, Segment)
groups) -- the mesh-level and sensitivity bootstrap CIs are the
uncertainty measure used throughout instead.

| config | n regions | Spearman rho | mesh-level CI (103 meshes) | (Colon, Segment) CI (15 clusters) | (Colon, Segment, Phantom) CI (58 clusters) |
|---|---|---|---|---|---|
| oracle | 957 | **-0.649** | [-0.710, -0.584] | [-0.755, -0.532] | [-0.720, -0.570] |
| pred_depth_only | 957 | **+0.839** | [0.801, 0.869] | [0.759, 0.892] | [0.796, 0.873] |
| pred_pose_only | 951 | **+0.816** | [0.788, 0.836] | [0.768, 0.843] | [0.787, 0.837] |
| fully_predicted | 956 | **+0.782** | [0.740, 0.817] | [0.706, 0.833] | [0.736, 0.819] |

Localization error by area-ratio quartile (`results/d1/diagnosis/region_merging_summary.json`
has the full table):

| config | Q1 median loc.err (area ratio range) | Q4 median loc.err (area ratio range) |
|---|---|---|
| oracle | 0.024mm (ratio 0.63-1.00) | ~0mm (ratio 1.0000-1.0057) |
| pred_depth_only | 0.75mm (ratio 0.63-1.57) | 26.20mm (ratio 17.7-574) |
| pred_pose_only | 1.15mm (ratio 0.0-1.10) | 26.74mm (ratio 7.8-570) |
| fully_predicted | 2.57mm (ratio 0.002-1.80) | 28.24mm (ratio 23.9-606) |

---

## INTERPRETATION

For each measurement below: which candidate cause it bears on, in which
direction, and what would confirm/refute it further.

- **Diagnostic 1 bears on cause A (detection definition insensitive), in
  the "insensitive" direction, at the 50%+ thresholds.** A baseline that
  sees no image data (all-unobserved) scores perfect recall at every
  threshold tested (25/50/75%) -- identical to oracle's own recall on
  this axis. This means the region-recall metric alone cannot
  distinguish a maximally naive predictor from a well-behaved one, at
  least on regions of this size and threshold range. *Would be
  strengthened by*: showing the same holds on an independently-collected
  corpus (not available here). *Would be weakened by*: a stricter
  threshold or size-class cut where all-unobserved's recall drops below
  oracle's -- diagnostic 2 already tests this at 75% and it does not
  happen (`pred_depth_only` still detects everything at every
  threshold).
- **Diagnostic 1's random baseline is mixed evidence, cutting against a
  simple "any baseline passes" version of cause A.** At the 50%
  threshold the random baseline scores only 0.624 -- far below oracle's
  1.0 and fully_predicted's 0.983 -- so recall@50% *does* separate a
  genuinely structured prediction from unstructured noise of the same
  area. But at 25% it scores 0.973, nearly indistinguishable from the
  real configurations, showing the metric's discriminating power is
  threshold-dependent, not uniformly absent. *Would be confirmed further
  by*: testing intermediate thresholds (30%, 40%) to locate exactly where
  random stops competing with real predictions.
- **Diagnostic 1's localization-error result is separate evidence for
  cause A, specific to that metric, not to recall.** The random
  baseline's localization error (median 8.04mm) is *six times smaller*
  than the deliberately-uninformative all-unobserved baseline's
  (48.14mm), even though random is, definitionally, uninformative about
  location. This is a real property of "matched largest-overlap
  component" localization error under a highly fragmented predicted set:
  a random selection covering ~50% of the mesh produces mostly tiny,
  isolated matched components that happen to sit inside or near the true
  region purely by chance, driving the reported error down. This does
  not affect D1.3's own recall-based criterion, but is a direct
  demonstration that localization error, taken alone, understates
  fragmentation-driven unreliability. *Would be confirmed further by*:
  reporting the *number* of components sharing the "matched" designation
  or a component-count/fragmentation statistic alongside localization
  error in future work (not computed here).
- **Diagnostic 3 bears on cause B (regions too easy), in the "too easy"
  direction, and is the strongest single piece of evidence for either
  cause.** Median coverage fraction is above 0.98 for every
  configuration and every size class, including the two configurations
  with real, substantial prediction error (pose and combined). Only
  1-5% of regions land near the 50% decision boundary for those
  configurations, and 0% for the depth-accurate ones. A region-level
  detection criterion set at 50% coverage is, empirically, almost never
  the deciding factor on this corpus -- most regions are covered with
  enormous margin or (rarely) missed outright, not decided at the
  margin. *Would be confirmed further by*: checking whether this pattern
  holds specifically for the smallest regions (near the 5mm headline
  cutoff) as opposed to being driven by the large class; not separately
  broken out here beyond the existing small/medium/large split, which
  already shows the same pattern in each class individually.
- **Diagnostic 4 is evidence for cause A, specific to how recall
  interacts with predicted-region size, not visible in the recall number
  itself.** For every non-oracle configuration, a larger matched
  predicted component relative to the true region (evidence of several
  GT regions or a large area being "merged" into one predicted blob)
  strongly predicts worse localization error (rho 0.78-0.84, p practically
  0). Area ratios in the worst quartile reach 17-600x the true region's
  area. **Region recall itself does not penalize this**: a region is
  "detected" once >=50% of its own area is covered, regardless of how
  much larger the covering predicted component is than the region --
  so a method that merges many true regions into one enormous
  over-flagged blob can still score high recall while being
  substantively wrong about *where* and *how large* the unobserved area
  actually is. Oracle's own regime (rho -0.649, area ratios clustered
  at 0.6-1.0) is a different, near-perfect-match dynamic and is not
  comparable to the other three configurations' merging pattern. *Would
  be confirmed further by*: a size-class-stratified version of this
  correlation (not run here) to check whether merging is concentrated in
  small regions specifically, which the small-class coverage numbers in
  diagnostic 3 (largest spread, most near-threshold regions) are already
  suggestive of.

**Combined picture, stated carefully**: the two candidate causes are not
mutually exclusive and the evidence here implicates both, acting
together. Regions are covered with enough margin (diagnostic 3) that the
recall metric rarely reaches a genuinely contested decision; when
predicted-unobserved area is generously over-large relative to the true
region (diagnostic 4, driven by the same predicted-depth over-flagging
`docs/d1_stage_b.md` already measured), it still clears the coverage bar
easily, so recall stays high even though the *shape and extent* of the
prediction is badly wrong. Diagnostic 1 shows this is not merely a
property of this particular set of predictions -- a content-blind
baseline clears the same bar. None of this is a claim about what should
be done differently; it is the measured basis for the pre-registered
diagnosis requirement, reported for the record.
