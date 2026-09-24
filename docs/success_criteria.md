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

### 2026-09-19 — Deviation: D1 mixed configurations redefined as post-hoc substitution

**Results seen at the time of this entry:** none. No pipeline has been run. This
constraint was discovered during the candidate survey
(`docs/pipeline_candidates.md`, 2026-09-19), not in response to any observed
number.

**Original criterion (§2, unchanged in place):** D1 requires the four
configurations GT depth + GT pose, predicted depth + GT pose, GT depth +
predicted pose, and both predicted.

**What changed:** Of the ten pipelines surveyed, none exposes a documented
mechanism for conditioning inference on an injected GT pose or GT depth. The
foundation-model group predicts depth, pose and intrinsics jointly with no
conditioning input; the SLAM group couples depth and pose inside a bundle
adjustment. The two mixed configurations are therefore produced by **post-hoc
substitution at the fusion stage**: the method's own predicted depth is fused
using GT poses, and GT depth is fused using the method's own predicted poses.
The method itself runs unmodified, on its own estimated inputs, in every
configuration. Only the fusion stage substitutes one input stream after the
fact. This term, "post-hoc substitution", is the single name used for this
design in all documents and in the paper.

**What this still answers:** D1's actual question is whether the evaluation has
measurement power. Post-hoc substitution answers it: if fusing with GT poses
changes region-level results substantially, pose error matters at the fusion
stage, whatever happens inside the method.

**Caveat carried into the paper:** post-hoc substitution does not simulate "the
method running with correct poses", nor "with correct depth". The predicted
depth used in the pose-substituted configuration was still produced under the
method's own erroneous pose estimates; the method never saw the GT pose. This
isolates how each error source propagates through fusion. It does not estimate
how the method would behave if one of its inputs were actually correct.

**Unaffected:** D1.3's 10-percentage-point threshold applies to the oracle
versus fully-predicted gap, which this redefinition does not touch. D1.1, D1.2,
D1.4 and D1.5 are unchanged.

**Future-proofing:** if a pipeline is later found to support genuine GT-pose or
GT-depth conditioning, run both variants on that method and report the
difference. Do not silently switch definitions for one method while others use
post-hoc substitution.

### 2026-09-19 — Clarification: the factual basis of the clustering rule in §1

**Results seen at the time of this entry:** none from any pipeline. The numbers
below come from GT-side analysis only (`docs/regions.md`,
`results/mesh_identity.csv`).

**Text being clarified (§1, "Clustering", left unchanged in place):** "Because
v2/v3 pairs share geometry and trajectory, and all sequences within a (colon,
segment) share one mesh..."

**Correction:** both premises are looser than stated. Verified by exact
vertex-array hashing: v2 and v3 share an identical mesh in 48 of 58 combos, not
all; v1 shares v2's mesh in only 18 of 58; and sequences within a (colon,
segment) do not all share a single mesh. Dataset-wide, 169 sequences are backed
by 103 distinct meshes and 113 distinct trajectories.

**Effect on the criteria:** none. The rule itself, that every aggregate and
every test is clustered at the mesh level and states the number of independent
meshes behind it, stands unchanged and is if anything better justified. Mesh
identity is determined by hash from `results/mesh_identity.csv`, not by assuming
that sequences in the same segment share geometry.

**Related constraint recorded elsewhere:** the debris ablation referenced by
hypothesis H4 (§4) is restricted to the 48 mesh-identical v2/v3 combos. H4's
requirement that geometry and trajectory be held identical is therefore met by
construction on that subset; the remaining 10 combos are excluded and the
exclusion is reported.

### 2026-09-23: Pre-run note: anticipated risk to D1.3 (no change to criteria)

Written before the full-corpus D1 run. Only one sequence (c1_cecum_t1_v1)
has been evaluated so far.

Observation (single sequence, not a result): under fully_predicted with
EndoDAC, the predicted-unobserved area fraction is 2.0x to 3.8x the GT
fraction across tau in {0.15, 0.25, 0.35, 0.50}, and every headline region
is detected in every configuration. Region recall is non-decreasing in the
size of the predicted-unobserved set, so a pipeline that over-flags can
reach high recall regardless of localization quality.

Prediction: D1.3 (oracle minus fully_predicted recall on medium + large
regions >= 10 pp) may fail on the full corpus because predicted-depth
over-flagging inflates fully_predicted recall.

Commitments:
- D1.3's definition and threshold are unchanged.
- If D1.3 fails, the pre-registered failure branch for D1.3 applies as
  written.
- Predicted-unobserved area fraction and false reassurance will be
  reported next to recall for every configuration. These are descriptive
  diagnostics only and do not substitute for D1.3 in any pass/fail
  decision.

### 2026-09-23: Deviation: localization error operationalized as Euclidean

§1 defines localization error as "surface distance" between centroids.
This term admits a geodesic reading. The protocol (docs/eval_protocol.md,
decision of 2026-09-23) uses the Euclidean distance between the
area-weighted centroid of the GT region and that of the matched predicted
component, and reports as a diagnostic whether the connecting segment
intersects the mesh (segment_intersects_mesh).

Recorded as a deviation rather than a clarification because the original
wording does not rule out the geodesic reading.

Timing: decided before the full-corpus run; a single-sequence pilot
(c1_cecum_t1_v1) had been run, in which all matched segments stayed
inside the lumen.

Trigger fixed in advance: if more than 20% of matched GT-predicted pairs
on the full corpus have a segment that intersects the mesh, geodesic
localization error will be computed and reported alongside Euclidean
for all configurations. Euclidean remains the primary metric either way.

### 2026-09-23: Clarification: operational definition of D1.1 completion

§2 D1.1 requires "a usable trajectory and depth for the full sequence, no
crash, no permanent track loss". Operationalized before the full-corpus
run as: every frame of the sequence has finite predicted depth and a
finite predicted pose, and the Sim(3) trajectory alignment succeeds.

Frame-to-frame pose pipelines such as EndoDAC cannot lose track by
construction; for them D1.1 measures crash-free completion only. This is
stated wherever D1.1 is reported.

Trajectory quality (ATE after Sim(3) alignment, endpoint drift as a
fraction of GT path length) is reported per sequence as a descriptive
quantity. No threshold is attached to it, because one sequence
(c1_cecum_t1_v1, endpoint drift 2.78%) had already been seen.

---

## 7. Sign-off

Pre-registered by: Chen
Date: 2026-09-18
Commit at time of freezing: (b3fa3fb7095fe3ded13428c81a1bfaa5922e3923)