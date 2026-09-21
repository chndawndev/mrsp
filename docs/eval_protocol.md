# Evaluation protocol design for `src/eval/`

Design document — **no code written yet**, per instructions. Once
implemented, this becomes the "registration procedure" and metric
definitions that `docs/success_criteria.md` §0 locks
("`src/eval/` ... no agent changes a metric definition, a threshold, or a
filtering rule ... without explicit approval"). Review this design before
implementation, since after that point changing it is a §5-prohibited-move
unless logged as a deviation.

Justified throughout by the measurements in
`docs/pipelines/endodac_scale.md` and `docs/pipelines/endodac.md`. Every
number cited from those documents is MEASURED; every threshold proposed
here is a reasoned design choice (INTERPRETATION), flagged as such,
subject to your review before it's implemented and frozen.

---

## 0. The one thing to read before anything else

**Scale recovery (§1) uses GT, for both the oracle and every predicted
configuration. This makes every number this protocol produces an upper
bound on what a deployed system (no GT available at inference time) could
achieve — not an estimate of deployed performance.** This is intentional:
D1's actual question is whether depth/pose error *propagates* into
region-level miss detection, not whether a real product could self-scale
without ground truth. Stated once, prominently, here, because it colors
the interpretation of every metric below. Restated at the point of use in
§1 as instructed, and it belongs in the eventual paper's methods section
alongside every headline number, not just this file.

---

## 1. Scale recovery

### Depth: single global factor per sequence, median matching

For each sequence, compute the per-frame ratio `median(GT depth) /
median(pred depth)` over GT-valid pixels (`docs/pipelines/endodac_scale.md`
§1's exact method), for every frame with GT depth available, then take
the **median of those per-frame ratios** (median-of-medians, each frame
weighted equally) as the sequence's single global depth scale factor.
This exact aggregation — not a single ratio pooled across all pixels of
all frames, a different and not-yet-computed number — is what produced
the c1_cecum_t1_v1 example value of **177.01** cited throughout this
document; future implementation must use the same aggregation to keep
numbers comparable to what's already been measured.

**Justification for why this must be global, not per-frame:** a real
deployed system has no per-frame GT to scale against; a global,
per-sequence factor is the most GT access a fusion step could plausibly
use and still resemble a real pipeline's single calibration step. Per-frame
GT-scaling was explicitly rejected in `docs/pipelines/endodac_scale.md`
§1 for exactly this reason.

**Known limitation, carried forward, not fixed here:** this factor is a
poor per-frame fit — 13.4% relative IQR, 51.2% relative range across
c1_cecum_t1_v1's 218 frames (`docs/pipelines/endodac_scale.md` §1). Every
downstream metric in this protocol inherits that residual per-frame scale
noise. §2's tau derivation accounts for it explicitly.

### Pose: Umeyama similarity to the GT trajectory

Umeyama (rotation + translation + **scale**) alignment of the predicted
camera trajectory to GT, fit over the **entire** sequence trajectory (not
a window — `docs/pipelines/endodac_scale.md` §2 found short-window fits
of this kind can be uninformative, see `docs/pipelines/endodac.md` §7's
account of why a short-window chain+Umeyama+ATE test was inconclusive for
a different reason; using the full trajectory avoids that specific
failure mode by construction, though it's a different fit than that §7
test, not a reuse of it). Produces `(R_u, t_u, s_pose)`.

**Method-specific prerequisite, not part of this step**: before Umeyama
fitting, the pipeline's raw relative-pose output must already be chained
into a single candidate trajectory using that pipeline's own validated
chaining convention (for EndoDAC: invert each `transformation_from_
parameters` output before chaining, per `docs/pipelines/endodac.md` §7 —
empirically validated per-method, not assumed). **A new pipeline needs
its own §7-style empirical convention check before this protocol applies
to it** — this is a gap this protocol doesn't close, flagged for whoever
onboards the next method.

**World-frame anchoring for ray-casting (§2)**: the fitted `(R_u, t_u,
s_pose)` is applied to *every* predicted camera pose, not just used to
scale a scalar. For predicted rotation `R_i` and translation `t_i` at
frame `i` (in the pipeline's own arbitrary reference frame):

```
R_world_i = R_u @ R_i
T_world_i = s_pose * (R_u @ t_i) + t_u
```

This is required, not optional bookkeeping: a predicted trajectory's
chain starts at an arbitrary reference frame (frame 0 = identity), so a
bare scalar multiply by `s_pose` fixes only the *scale* of that arbitrary
frame, not its position/orientation relative to the mesh's real-world
coordinates. Exact formula and justification:
`docs/pipelines/endodac_scale.md` §3, validated there by reproducing the
oracle configuration's already-established 0.028mm median point-to-mesh
distance almost exactly using this construction.

### Stated plainly, as instructed

**Both scale-recovery steps use GT.** The depth factor is fit against GT
depth; the pose alignment is fit against the GT trajectory. Every
configuration in §4 that includes predicted depth or predicted pose
benefits from this GT-based correction — including the "fully predicted"
configuration, which is not what a deployed system without GT access
could do. **This is intentional and by design**: D1 asks whether this
benchmark has *measurement power* — whether depth/pose error propagates
into detectable region-level differences — not whether EndoDAC (or any
method) is deployment-ready. Any report or paper section that cites D1-D3
numbers must carry this caveat explicitly, in the same paragraph as the
numbers, not as a footnote.

---

## 2. Predicted-observed set: ray-cast + tau-gated depth agreement

### Procedure

For each frame, for **every pixel** (full density — see the flag on
sampling density below), cast one ray through the project's own camera
model (`src/geometry/camera.py`'s `unproject()`, the Scaramuzza
omnidirectional model — **never the pipeline's own predicted intrinsics**,
matching `CLAUDE.md`'s frozen decision that "GT visibility evaluation
always uses the omnidirectional model, regardless of what a method
consumed"; the same rule is extended here to the *predicted*-observed
side, for the same reason — a fixed, shared camera model is the only way
different pipelines' outputs are comparable at all) from the aligned
camera pose (§1: `R_world_i`, `T_world_i` for predicted pose, or the
direct GT `pose.txt` transform for GT pose).

Find the nearest intersection of that ray with `coverage_mesh.obj`
(world-space ray-mesh query, e.g. via `trimesh`'s embree-backed nearest-
hit query, already available in the project venv — confirmed present when
`scratch/pipelines/endodac_fusion_check.py` ran). Let `hit_world` be the
intersection point. Convert it back into the ray's own camera frame,
`hit_cam = R_world_i.T @ (hit_world - T_world_i)`, and take
**`d_hit = hit_cam[2]`, the camera-frame Z-depth of the hit** — not the
Euclidean ray length. This distinction matters and is exactly the kind
of bug `src/geometry/camera.py`'s own docstring warns about for
back-projection ("NOT the same as `unproject(px) * depth`, which would
use depth as a ray length, i.e. radial distance") — the same care applies
in reverse here, converting a hit point back to a Z-depth.

Let `d_pred` be the scaled predicted depth (§1) at that pixel. Mark the
hit face **observed** for this pixel/frame if:

```
|d_pred - d_hit| / d_hit < tau
```

**OR-accumulate across all frames**: a face is in the predicted-observed
set if *any* pixel, in *any* frame, passes this test for it — exactly
mirroring the GT criterion's own "any primary ray hit, any pixel, any
frame, no distance ... threshold" language (`CLAUDE.md`'s frozen
conventions), with the one necessary addition (the tau gate) that the GT
side doesn't need, because GT depth is exact geometry, not a noisy
estimate.

**Predicted-unobserved set = all mesh faces minus the accumulated
predicted-observed set.** This is definitionally the same quantity
`docs/success_criteria.md` §1 calls "the set of mesh faces a pipeline's
output marks as never adequately observed" — restated here to make the
equivalence explicit, not introducing a new definition.

### Why a tau gate is necessary (not just permitted)

Without it, "predicted-observed" would degenerate into "the camera's
frustum geometrically reached this face" — a function of pose and FOV
alone, blind to whether the pipeline's own depth estimate at that pixel
is trustworthy. That would make D1's actual question (does depth
accuracy matter for missed-region detection?) unanswerable by
construction. Requiring depth agreement is also the conservative,
safety-appropriate choice for this project's stated goal: a pixel whose
predicted depth doesn't match the true surface is exactly the situation
where the pipeline *shouldn't* be trusted to have reliably imaged that
location, even if the ray geometrically got there.

### Deriving tau from measurement, not guessing

MEASURED inputs, all from `docs/pipelines/endodac_scale.md` and a fresh
computation on the same 20 frames used there:

| Quantity | Value | Source |
|---|---|---|
| Fully-predicted point-to-mesh distance, median | 10.12 mm | `endodac_scale.md` §3, config B |
| Fully-predicted point-to-mesh distance, p95 | 33.91 mm | same |
| Median GT depth (same 20 frames) | 41.33 mm | computed for this document |
| GT depth p95 | 86.26 mm | computed for this document |
| Depth scale relative IQR (scale-noise floor) | 13.4% | `endodac_scale.md` §1 |

Two independent estimates of a reasonable tau:

- **Position-error-to-typical-depth ratio**: 10.12mm / 41.33mm ≈ **24.5%**
  (median-to-median), up to 33.91mm / 86.26mm ≈ **39.3%** (p95-to-p95).
  This measures roughly what fraction of typical depth a typical
  reconstruction error represents — a natural scale for "how loose does
  tau need to be to admit most of the fully-predicted configuration's own
  typical error."
- **Scale-noise floor**: tau must clear 13.4% by a comfortable margin, or
  the tau test would be dominated by the *already-known, already-accepted*
  per-frame residual of a single global scale factor (§1), not by genuine
  depth-shape error — testing something already measured and priced in,
  not new information.

**Proposed primary value: `tau = 0.25`** — close to the measured
median-to-median ratio (24.5%, rounded), and comfortably (~1.9x) above
the 13.4% scale-noise floor.

**Proposed sweep: `tau ∈ {0.15, 0.25, 0.35, 0.50}`** — 0.15 as the
stringent end (just above the scale-noise floor, testing whether results
are sensitive to being close to that floor), 0.25 primary, 0.35 and 0.50
as looser anchors bracketing the p95-based estimate. Mirrors the existing
precedent in `docs/success_criteria.md` §1's detection-threshold sweep
(25/50/75%, headline at 50%) and D1.4's robustness-curve framing — same
sweep-plus-headline pattern, not a new one.

**Explicit limitation**: this derivation is grounded in **one sequence's**
20-frame sample (`c1_cecum_t1_v1`). It is a reasoned starting point, not
a calibrated constant — flagged for revisit once more sequences and
methods are evaluated, per the same spirit as `docs/success_criteria.md`
§5 prohibited move #1 (don't change a threshold after seeing the result
it applies to) — meaning: settle tau **before** running the corpus, not
after, and if a revision looks warranted later, it goes through a
`docs/success_criteria.md` §6-style deviation entry, not a silent change.

**Sampling density flag**: `docs/oracle_check.md` already diagnosed a
false-negative-inflating artifact from pixel subsampling (IoU 0.869 at
stride 8 rising to 0.944 at stride 4, trending toward the ~0.999 achieved
at full density in `docs/gpu_validation.md`). This protocol specifies
**full pixel density** (every pixel, every frame) for exactly this
reason — matching the frozen GT criterion's own "any pixel" language, and
avoiding re-discovering the same already-diagnosed artifact under a new
name. This has real compute cost at corpus scale, not estimated here.

---

## 3. Rays that miss the mesh entirely

Discard from the tau test (no face to attribute the pixel to), but
**count and report the fraction per sequence, per configuration** — not
merged across configurations, since a badly-misaligned predicted pose
(more likely in the pred-pose configurations, D1's "predicted pose"
column) could plausibly send many more rays off into empty space than a
GT-anchored pose would. A high miss fraction in a predicted-pose
configuration is itself a finding worth surfacing, not just a discarded
statistic — flagged for the eventual results write-up, not scored against
any threshold here since none is pre-stated for it.

---

## 4. Configurations

Exactly `docs/success_criteria.md` §2 D1's four: GT depth + GT pose
(oracle), predicted depth + GT pose, GT depth + predicted pose, both
predicted — produced by **post-hoc substitution**
(`docs/success_criteria.md` §6's deviation entry): the method runs
unmodified in every configuration; only §1's scale-recovery and §2's
ray-casting/fusion stage substitutes GT for one input stream.

**Scale factors are fit once per sequence per input type, reused across
configurations** — not refit per configuration. The same depth scale
factor is used in both "predicted depth" configurations (pred+GT-pose and
both-predicted); the same pose alignment `(R_u, t_u, s_pose)` is used in
both "predicted pose" configurations. This is a deliberate consistency
choice: depth scale and pose alignment are properties of the *method's
output relative to GT for that sequence*, not of which configuration is
currently being evaluated, and refitting per configuration would let
subtle differences in the fit itself (not the actual depth/pose being
substituted) leak into cross-configuration comparisons.

---

## 5. Outputs per sequence

- **Predicted-observed face set** (boolean, one per mesh face) — the
  primary artifact, per configuration.
- **Predicted-unobserved face set** — its complement (§2).
- **Ray-miss fraction** (§3), per configuration.
- **Region-level metrics**, exactly as frozen in `docs/success_criteria.md`
  §1, computed per configuration:
  - Region recall (fraction of GT regions detected, per size class:
    small/medium/large, `d<5mm` excluded from headline, appendix only).
  - Detection at the primary 50% threshold, plus the 25/50/75% robustness
    sweep.
  - False reassurance rate (fraction of GT-unobserved area marked
    observed).
  - False alarm rate (fraction of predicted-unobserved area GT marks
    observed).
  - Localization error (surface distance, detected-GT-region centroid to
    overlapping predicted-component centroid) — this needs connected
    components of the *predicted*-unobserved face set too (not just GT
    regions), reusing `src/geometry/mesh_stats.py`'s
    `connected_components_of_subset` directly rather than writing new
    component-extraction logic.
  - All aggregates clustered at the mesh level, with the number of
    independent meshes stated, per `docs/success_criteria.md` §1's
    Clustering rule.
- **tau-sweep table**: every metric above, at each of the 4 proposed tau
  values, not just the primary — needed for D1.4-style robustness
  reporting and for validating tau itself (§6).

---

## 6. Validation plan for the eval code

**What the oracle configuration must reproduce**: GT depth + GT pose,
run through this protocol's own full ray-cast + tau-gate pipeline (not
a shortcut re-derivation of `coverage_mesh.obj`'s `vt` flags), should
recover a predicted-observed set matching the mesh's own GT-observed
labeling almost exactly — because GT depth + GT pose + the same camera
model is nearly the same information the original renderer used to
produce those labels in the first place.

**Target and precedent**: `docs/gpu_validation.md` already validated a
related but not identical check (a direct visibility ray-caster, not a
depth-image-backprojection-and-tau-gate pipeline) at **full pixel
density, achieving IoU 0.9988** against the released mesh. This protocol
adopts the same **IoU ≥ 0.99** bar for its own oracle validation, at full
density — a reasonable target given the closely related methodology, but
not a guarantee this exact number reproduces, since the pipelines aren't
identical (flagged so nobody mistakes 0.9988 as already having validated
*this* code).

**Tau's effect on the oracle should be negligible, and that's itself a
check**: since `d_pred` and `d_hit` are both derived from the same GT
mesh geometry in the oracle configuration, they should agree almost
exactly regardless of tau (modulo ray-casting numerical precision).
**Validation requirement**: run the oracle configuration across the full
tau sweep (§2), and require IoU ≥ 0.99 at **every** sweep value,
including the most stringent (0.15). If the oracle fails at the
stringent end, that tau value is numerically too tight even for
near-perfect input and should be dropped from the sweep — a useful
diagnostic on tau's own lower bound, not just on the eval code's
correctness.

**If the oracle doesn't immediately clear 0.99**: diagnose before
concluding a bug, following the exact precedent `docs/oracle_check.md`
already set for this same mesh (there: an apparent IoU shortfall at
stride 8 was traced to pixel-sampling density, confirmed by rerunning at
higher density and watching `false_unobserved` fall while
`false_observed` stayed negligible — the same diagnostic move applies
here if the oracle underperforms: check whether it's a
`false_observed`-vs-`false_unobserved` asymmetry consistent with a known,
already-understood cause before touching any metric code, per
`docs/success_criteria.md` §0's rule that only a genuine D1.2 (oracle
recall) failure justifies changing `src/eval/`, and even then only after
logging it).

**Passing tolerance, stated plainly**: IoU ≥ 0.99 at full density, across
the full tau sweep, is the pass bar for the eval code itself, before any
predicted configuration's numbers are trusted. This is a code-correctness
check, distinct from `docs/success_criteria.md` §2's D1.2 ("oracle
sanity: region recall (medium + large) under GT depth + GT pose ≥ 0.80"),
which is a *result* threshold on the oracle configuration's region-level
recall, not a face-set IoU against the raw mesh labels — both should be
checked, in that order (code correctness first, then the pre-registered
result threshold), since a D1.2 failure with broken eval code would be
mis-diagnosed as a method problem.

---

## Flags: where this design touches frozen definitions

None of the following are violations found in the frozen text — nothing
here contradicts an explicit rule in `docs/success_criteria.md`. They are
places where this design fills in something the frozen document left
open, or creates a real interpretive risk worth surfacing before
implementation, exactly as instructed. `docs/success_criteria.md` is not
modified by writing this list.

1. **GT-based scale recovery makes every "predicted" configuration an
   upper bound, not a deployability estimate** (§0, §1). Not a rule
   violation — `docs/success_criteria.md` doesn't prohibit this, and
   leaves "the registration procedure" undefined for `src/eval/` to
   specify — but it is the single most consequential interpretive fact
   about every number this protocol will produce, and needs to travel
   with those numbers wherever they're reported, not just live in this
   file.

2. **tau is a new filter, not previously specified anywhere in
   `docs/success_criteria.md`.** §5 prohibited move #4 permits an
   unstated filter only "if it is applied identically to every method and
   logged." This design satisfies that by construction (one fixed tau
   sweep + primary value, applied identically across every pipeline and
   sequence, logged here) — flagged so this requirement is visible before
   implementation, not discovered as a violation after the fact.

3. **Risk, not a conflict**: D1.5 requires false reassurance rate
   "strictly between 0.02 and 0.98" under the fully-predicted
   configuration. Depth error dominates reconstruction error
   (`docs/pipelines/endodac_scale.md` §3), and the depth-scale-noise
   floor (13.4%) sits close to the proposed tau range (15-50%) — a real
   possibility is that the tau gate rejects so many otherwise-correct
   hits that the predicted-unobserved set balloons, pushing false
   reassurance rate toward the *low* end (very little gets marked
   "observed" that shouldn't be) rather than the usual worry (too much
   false reassurance). D1.5 doesn't block D1 (`docs/success_criteria.md`
   §2: "D1.4 and D1.5 are recorded but do not by themselves block"), so
   this isn't a blocking conflict — but it's a specific, plausible
   failure mode worth watching for in the first real run, not a generic
   "results may vary" caveat.

4. **"Registration procedure" is interpreted here as registering the
   predicted camera trajectory to GT** (via §1's Umeyama fit), not
   registering a reconstructed point cloud or mesh directly. This is a
   specific reading of an intentionally open term in
   `docs/success_criteria.md` §1 ("Predicted-unobserved set ... after
   registration of the prediction to the GT mesh. The registration
   procedure is part of `src/eval/` and is frozen with it.") — stated
   explicitly so it's confirmed or corrected before it becomes frozen by
   implementation, rather than assumed silently.

5. **Predicted intrinsics are not used anywhere in this protocol** — all
   ray-casting uses the project's own fixed Scaramuzza camera model,
   never a pipeline's own predicted (pinhole) intrinsics output, matching
   `CLAUDE.md`'s frozen decision for the GT side and extending it to the
   predicted side for the same reason (comparability across methods). Not
   a conflict, just making an implicit consequence of an existing frozen
   decision explicit before someone assumes predicted intrinsics get used
   somewhere in fusion.

---

No code written. This document is the thing to review/approve before
`src/eval/` gets implemented against it.
