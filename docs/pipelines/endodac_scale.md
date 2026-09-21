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

## 3. Depth scale vs. pose scale: substantially inconsistent

| Quantity | Value |
|---|---|
| `s_pose` | 557.87 |
| median depth scale | 177.01 |
| ratio (`s_pose` / median depth scale) | **3.15** |
| percent difference | **+215%** |
| differ by more than 20%? | **YES, by over an order of magnitude past the threshold** |

**Depth and pose carry inconsistent scale.** The pose-derived scale factor
is more than 3x the depth-derived scale factor — nowhere near the 20%
tolerance. This is not surprising in retrospect: depth and pose come from
two entirely separate network heads (`depth_model.pth` vs.
`pose_encoder.pth`/`pose.pth`) with independent, arbitrarily-initialized
output unit systems (`docs/pipelines/endodac.md` §3's scale-ambiguity
note applies to both, independently) — there is no architectural reason
their native units would agree, and they measurably don't.

**Consequence for fusion, stated as instructed:** depth and pose must be
scaled **separately** — a single shared scale factor recovered from one
(e.g. from pose via Umeyama-to-GT, or from depth via median-matching)
cannot be reused for the other. Any fusion step (post-hoc substitution,
`docs/success_criteria.md` §6) that projects predicted depth using
predicted or GT poses must resolve depth's own scale independently from
whatever scale resolution pose uses, or the fused reconstruction will be
off by roughly the 3.15x ratio measured here.

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
| Depth vs. pose scale consistency | **Inconsistent**: ratio 3.15x, +215% — well past the 20% threshold. Must scale depth and pose separately in fusion. |
| Trajectory drift (candidate D1.1 metric) | Path length ratio 0.920 (8% short); **endpoint error 2.78% of distance traveled** |

No evaluation code written. This is measurement only, ahead of designing
the fusion/registration step referenced in `docs/success_criteria.md`
§6's post-hoc-substitution deviation.
