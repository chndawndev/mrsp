# EndoDAC: depth/pose scale consistency, `c1_cecum_t1_v1`

Measured before writing any evaluation code, per instructions. Uses the
outputs already produced by the full-sequence run
(`docs/pipelines/endodac.md` §8, baseline input, all 218 frames) — no new
inference. Script: `scratch/pipelines/endodac_scale_analysis.py`, log:
`logs/endodac_scale_analysis.log`, raw numbers:
`results/pipelines/endodac_scale_analysis.json`. **All MEASURED.**

---

## 1. Depth scale: per-frame median(GT)/median(pred), 218 frames

For each frame, `ratio = median(GT depth) / median(pred depth)` over
GT-valid pixels (`0 < GT_mm < 100`, this project's frozen convention).

| Statistic | Value |
|---|---|
| n frames | 218 |
| median | 177.01 |
| IQR (Q1, Q3) | [167.77, 191.50], width 23.73 |
| min | 135.79 |
| max | 226.33 |
| std | 19.26 |
| relative IQR (IQR / median) | **13.4%** |
| relative range ((max−min) / median) | **51.2%** |

**A single global depth scale is not a tight fit.** The middle 50% of
frames alone spans 13.4% of the median value, and the full range spans
51.2% — over half the median value from the most-underestimated to the
most-overestimated frame. This is a wide distribution, not a narrow one.
Per §1's framing: a real system cannot do per-frame GT scaling, so if a
single global scale is applied (the only option without GT access at
inference time), individual frames will carry depth error on the order of
10-50%+ from scale mismatch alone, on top of whatever shape/structural
error the depth prediction itself has (Check 5, `docs/pipelines/endodac.md`
§6 — baseline Spearman ρ ≈ -0.87, i.e., real correlation but not perfect).

---

## 2. Pose scale: Umeyama-with-scale, full 218-frame trajectory

Umeyama (rotation + translation + scale) alignment of the predicted
trajectory (`poses_pred.npy`, §8's convention: each relative transform
inverted before chaining) to GT (`pose.txt`), over all 218 frames — not
a short window, since this measurement needs the full-sequence scale
factor.

| Quantity | Value |
|---|---|
| Umeyama scale `s_pose` | **557.87** |
| Post-alignment ATE (RMSE) | **9.23 mm** |

---

## 3. Depth scale vs. pose scale: a geometric consistency test

**Correction (2026-09-20): this section originally compared `s_pose`
(557.87) directly against the median depth scale (177.01) as a ratio
(3.15x, "+215%") and concluded depth and pose were substantially
inconsistent. That comparison was invalid and has been removed, not just
reworded.** `s_pose` converts network-native *translation* units to mm;
the depth scale converts network-native *depth* units to mm. These are
two independently-initialized output heads with unrelated internal unit
systems (`docs/pipelines/endodac.md` §3) — there is no reason the two
raw numeric factors should be comparable at all, any more than a
currency-exchange rate and a unit-conversion factor for length would be.
A large numeric ratio between them measures nothing. Replaced with an
actual geometric test, below.

### The test

Four configurations, 20 frames spread evenly across `c1_cecum_t1_v1`
(indices `0, 11, 22, ..., 217`), backprojected via the project's own
validated camera model (`src/geometry/camera.py`) and compared against
`coverage_mesh.obj` using the exact same methodology as
`scripts/oracle_check.py` (pixel stride 8, `trimesh` nearest-surface
query, capped at 20,000 query points per configuration — same cap
`oracle_check.py` uses). Script:
`scratch/pipelines/endodac_fusion_check.py`, log:
`logs/endodac_fusion_check.log`, raw numbers:
`results/pipelines/endodac_fusion_check/metrics.json`.

| Config | Depth | Pose | Median dist. | p95 dist. |
|---|---|---|---|---|
| **A. oracle** | GT (mm) | GT (world frame) | **0.028 mm** | 0.099 mm |
| **B. fully predicted** | pred × 177.01 | pred, world-anchored | **10.12 mm** | 33.91 mm |
| **C. pred depth only** | pred × 177.01 | GT | **10.17 mm** | 33.32 mm |
| **D. pred pose only** | GT (mm) | pred, world-anchored | **3.70 mm** | 10.29 mm |

Config A reproduces `docs/oracle_check.md`'s established 0.028mm almost
exactly (0.0282mm here, on a different 20-frame/stride-8 sample vs. the
original all-218-frame run) — confirms this script's camera-model/pose
plumbing matches the validated pipeline, not a new implementation with
its own bugs.

**Predicted poses need a world-frame anchor, not just a scalar.**
`poses_pred.npy`'s chain starts at an arbitrary reference frame (frame 0
= identity) — a bare multiply by `s_pose` fixes the *scale* of that
arbitrary frame but not its *position/orientation* relative to the mesh's
real-world coordinates, which is a separate, necessary piece of
bookkeeping. Configs B and D apply the **full** Umeyama similarity
transform (rotation `R_u`, translation `t_u`, scale `s_pose` — the same
`s_pose` from section 2, unchanged) fitted over the whole 218-frame
trajectory to re-express each predicted camera pose in the mesh's world
frame: for predicted rotation `R_i` and translation `t_i` at frame `i`,
`R_world = R_u @ R_i`, `T_world = s_pose * (R_u @ t_i) + t_u`. This is
required to ask the geometric question at all — without it, *any*
depth/pose combination would land arbitrarily far from the mesh
regardless of scale correctness, which would itself be a units error of
the same kind being corrected here.

### Result: the two scales are geometrically consistent, order of magnitude

**Config B (fully predicted) lands at a median 10.12mm from the true
mesh surface — not hundreds of mm.** For comparison, the *wrong* pose
interpretation tested during setup (`docs/pipelines/endodac.md` §7's
precursor to that convention check, and the original pose.txt transpose
ambiguity in `docs/oracle_check.md`) produced ~400mm errors — two orders
of magnitude worse. 10mm is small relative to the mesh's own geometric
scale (this sequence's unobserved regions alone range 5mm to several cm,
`docs/success_criteria.md` §1's size classes). **This confirms the
user's stated hypothesis: the fully-predicted configuration lands near
the mesh, so the two independently-fit global scales are geometrically
consistent when combined, and the earlier "3.15x" reading was exactly
the incommensurable-units artifact it's now described as** — not
evidence of a real scale mismatch.

### Isolating the source: depth error dominates, not a depth/pose interaction

Comparing the two controls against the fully-predicted result:

- **C (pred depth + GT pose) ≈ B (pred depth + pred pose)**: 10.17mm vs.
  10.12mm — nearly identical. Swapping in perfect pose barely changes the
  outcome.
- **D (GT depth + pred pose)**: 3.70mm — noticeably smaller than either
  depth-involving configuration.

**Depth error dominates the misplacement; pose error is a smaller,
mostly-independent contributor.** This is consistent with section 1's
finding that a single global depth scale is a poor fit (13.4-51.2%
relative spread) — that per-frame scale mismatch, plus depth's own
structural/shape error (`docs/pipelines/endodac.md` §6, baseline Spearman
ρ ≈ -0.87, real but imperfect correlation), plausibly explains most of
configs B and C's ~10mm gap. **Illustrative check, not a proven
decomposition law**: treating the two error sources as roughly
independent, `sqrt(C² + D²) = sqrt(10.17² + 3.70²) ≈ 10.82mm`, close to
B's measured 10.12mm — consistent with roughly-independent, not strongly
super-additive, error sources, though this quadrature check is offered as
a sanity-check on the qualitative "depth dominates" reading, not as a
validated error-propagation model.

**Consequence for fusion, stated as instructed:** the earlier
"depth and pose must be scaled separately, one factor cannot be reused
for the other" advice was correct in effect (they clearly ARE two
separate fits with different numeric values, 177.01 and 557.87, in
unrelated units) but for the wrong stated reason (a meaningless ratio,
not a demonstrated geometric mismatch). The corrected reasoning: fit each
scale independently (as already done) and expect the fused reconstruction
to land within roughly the 10mm order of magnitude shown here — dominated
by depth's own accuracy, not by any depth/pose scale disagreement.

---

## 4. Trajectory drift

Predicted path length computed in the network's raw units, then scaled by
`s_pose` (Umeyama's fitted scale) for direct comparison to GT — equivalent
to, and cross-checked against, computing path length directly on the
full Umeyama-aligned trajectory (both give 320.38mm, confirming rotation
and translation alone don't change path length, only the scale factor
does, as expected).

| Quantity | Value |
|---|---|
| GT path length | 348.35 mm |
| Predicted path length (scaled by `s_pose`) | 320.38 mm |
| Path length ratio (pred / GT) | **0.920** |
| Endpoint error (aligned pred vs. GT, frame 217) | 9.68 mm |
| **Endpoint error as fraction of GT path length** | **2.78%** |

The predicted trajectory, once given its own best-fit global scale, tracks
GT's overall path length reasonably well (8% short over the full
218-frame, ~348mm traverse) and ends up only 2.78% of total distance
traveled away from the true endpoint. **This is the candidate degradation
metric for D1.1** the instructions asked to identify: endpoint error as a
fraction of distance traveled is a single, real-number-per-sequence,
scale-normalized quantity that doesn't require per-frame GT access to
define (only a single sequence-level scale fit), and 2.78% is a
concrete, reportable number for this sequence.

---

## Summary

| Check | Result |
|---|---|
| Depth scale tightness | Wide: 13.4% relative IQR, 51.2% relative range across 218 frames — not a tight single global scale |
| Pose scale | `s_pose` = 557.87, ATE 9.23mm over the full trajectory |
| Depth vs. pose scale consistency | **Corrected via geometric test (§3)**: fully-predicted reconstruction lands 10.12mm median from the mesh (vs. oracle's 0.028mm, and vs. ~400mm for a genuinely wrong transform) — the two independently-fit scales are geometrically consistent; the earlier "3.15x inconsistent" reading compared incommensurable units and was removed. Depth error dominates the ~10mm gap (pred-depth-only ≈ fully-predicted; pred-pose-only is 3.70mm), not a depth/pose scale mismatch. |
| Trajectory drift (candidate D1.1 metric) | Path length ratio 0.920 (8% short); **endpoint error 2.78% of distance traveled** |

No evaluation code written. This is measurement only, ahead of designing
the fusion/registration step referenced in `docs/success_criteria.md`
§6's post-hoc-substitution deviation.
