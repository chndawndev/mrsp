# Success criteria (pre-registered)

Status: **FROZEN** as of 2026-09-18, before any depth/pose pipeline has been run.

Purpose: fix the decision thresholds *before* seeing results, so that "is this
project working?" is answered by the numbers and not by whichever reading makes
the project look alive.

---

## 0. Change control

This file is pre-registration, not a working note.

- **Claude Code must never edit this file.** Not to reword, not to "clarify",
  not to add a criterion. If an agent believes a criterion is wrong or
  unmeasurable, it must say so in its report and stop, leaving this file
  untouched.
- **Only the author edits this file**, and only by appending to §6 (Deviations).
  Existing text is never rewritten or deleted.
- A deviation entry must be written **before** the affected result is
  interpreted, and must state what was changed, why, and what the original
  criterion was.
- `src/eval/` and `src/gt/` are locked in the same sense: no agent changes a
  metric definition, a threshold, or a filtering rule in those paths without
  explicit approval in the conversation. "The metric was wrong, so I fixed it
  and reran" is a protocol violation, even when the fix is correct.

---

## 1. Frozen definitions

These are the definitions all criteria below refer to. They are fixed now so
that a disappointing result cannot be rescued by redefining the target.

**GT region.** A connected component of GT-unobserved faces on
`coverage_mesh.obj` (per-face `vt` flag as released). Components are computed
by face adjacency (shared edge).

**Region size.** Equivalent diameter `d = 2*sqrt(area/pi)` from total triangle
area. Size classes:
- small: 5 mm <= d < 10 mm
- medium: 10 mm <= d < 20 mm
- large: d >= 20 mm
- Regions with d < 5 mm are excluded from headline metrics (they cannot hide a
  clinically relevant lesion) but are still reported in an appendix table.

**Predicted-unobserved set.** The set of mesh faces a pipeline's output marks as
never adequately observed, after registration of the prediction to the GT mesh.
The registration procedure is part of `src/eval/` and is frozen with it.

**Detection (primary).** A GT region counts as detected if at least **50% of its
area** is covered by the predicted-unobserved set. This single threshold is the
primary definition; a sweep over 25/50/75% is reported as a robustness curve,
but the headline number is always 50%.

**Region recall.** Fraction of GT regions detected, computed per size class.

**False reassurance rate.** Fraction of GT-unobserved area that the pipeline
marks as observed. This is the safety-critical metric and is always reported
alongside recall.

**False alarm rate.** Fraction of predicted-unobserved area that GT marks as
observed.

**Localization error.** Surface distance between the centroid of a detected GT
region and the centroid of the overlapping predicted component.

**Clustering.** Because v2/v3 pairs share geometry and trajectory, and all
sequences within a (colon, segment) share one mesh, every aggregate statistic
and every significance test is clustered at the **mesh level**. The number of
independent meshes behind each reported number is stated in every table.

---

## 2. Decision point D1: first pipeline, four-way decomposition

Trigger: one end-to-end pipeline has been run on all 169 registered sequences in
the four configurations (GT depth + GT pose, predicted depth + GT pose, GT depth
+ predicted pose, both predicted).

D1 asks one question: **does this benchmark have measurement power?** A
benchmark where every configuration scores the same cannot rank methods.

| # | Criterion | Pass threshold |
|---|---|---|
| D1.1 | Pipeline completes (produces a usable trajectory and depth for the full sequence, no crash, no permanent track loss) | on **>= 70%** of the 169 sequences |
| D1.2 | Oracle sanity: region recall (medium + large) under GT depth + GT pose | **>= 0.80** |
| D1.3 | Sensitivity: absolute gap in region recall (medium + large) between the oracle configuration and the fully-predicted configuration | **>= 10 percentage points** |
| D1.4 | Size discrimination: region recall on small regions is lower than on large regions | gap **>= 15 percentage points**, or an explicit finding that it is not |
| D1.5 | False reassurance rate under the fully-predicted configuration is measurable and non-degenerate | strictly between **0.02 and 0.98** |

**D1 passes** if D1.1, D1.2, D1.3 all pass. D1.4 and D1.5 are recorded but do
not by themselves block.

**If D1.2 fails** the evaluation code is wrong, not the method. Fix the
evaluation, rerun, and log it in §6. This is the only case where changing
`src/eval/` is expected.

**If D1.1 fails** (pipeline completes on < 70%): this is itself a reportable
finding, but the benchmark cannot be run as designed on this method. Try one
alternative pipeline before concluding anything.

**If D1.3 fails** (oracle and prediction score nearly the same): stop and
diagnose before adding methods. Either the detection definition is insensitive,
or the regions are too easy. Do **not** fix this by switching to a metric that
happens to separate them; write down the diagnosis first.

---

## 3. Decision point D2: multiple pipelines

Trigger: 4 to 6 pipelines evaluated.

| # | Criterion | Pass threshold |
|---|---|---|
| D2.1 | Spread between best and worst pipeline on region recall (medium + large) | **>= 10 percentage points** |
| D2.2 | The ranking is stable under the detection-threshold sweep (25/50/75%) | top and bottom method do not swap |
| D2.3 | Per-mesh clustered bootstrap CI for the best-vs-worst difference excludes zero | 95% CI |

**D2 passes** if D2.1 and D2.3 pass. D2.2 failing is itself worth reporting.

**If D2 fails**, the honest paper is a negative/limits result: current pipelines
are indistinguishable on this task. That is publishable but must be framed as
such from the start, not retrofitted.

---

## 4. Decision point D3: is there a finding?

Trigger: D2 passed; criteria sensitivity analysis (distance / incidence angle /
pixel footprint) complete.

At least **one** of the following pre-stated hypotheses must hold, with the
effect size given:

- **H1.** Depth accuracy does not predict localization quality: Spearman
  correlation between per-sequence AbsRel and per-sequence region recall is
  weaker than |rho| = 0.4, across at least 4 pipelines.
- **H2.** Tightening the observation criterion changes the method ranking:
  Spearman correlation between rankings under the released GT and under the
  strictest clinical criterion is below 0.7.
- **H3.** Fully-interior regions (behind folds) are detected far less often than
  boundary-touching regions of the same size class: gap >= 20 percentage points.
- **H4.** Debris degrades localization on the v2/v3 paired subset: region recall
  drops by >= 10 percentage points, with geometry and trajectory held identical.

If none holds, the paper has a benchmark and no message. Options at that point:
report the negative result honestly, or extend to the real full-length videos
for a robustness study. Do **not** go hunting for a fourth hypothesis in the
data and present it as if it had been predicted; any post-hoc hypothesis is
labeled exploratory in the paper.

---

## 5. Prohibited moves

These are the specific ways this project could fool itself. None is permitted
without a §6 deviation entry written first.

1. Changing a threshold in §1 to §4 after seeing the result it applies to.
2. Switching the primary metric, or promoting a secondary metric to primary.
3. Dropping sequences from an aggregate because a pipeline failed on them.
   Failures are reported as failures, with the count.
4. Introducing a per-sequence or per-region filter that was not pre-stated,
   unless it is applied identically to every method and logged.
5. Tuning a pipeline's hyperparameters on the same sequences used for the
   headline numbers, unless every pipeline gets the same tuning budget and this
   is stated.
6. Reporting an unverified mechanism as a finding. Every causal claim names the
   experiment that would refute it.
7. Reporting counts of sequences or regions without the number of independent
   meshes behind them.
8. Silently correcting an evaluation bug and rerunning. Bug fixes are logged in
   §6 with before/after numbers.

---

## 6. Deviations log

Append-only. Each entry: date, what changed, original text, reason, and whether
the affected result had already been seen.

(no entries yet)

---

## 7. Sign-off

Pre-registered by: Chen
Date: 2026-09-18
Commit at time of freezing: (fcb2f25)