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

### 2a. Evaluable pixel — the actual gate on the tau test

Settled by `docs/eval_protocol_oracle_gap.md`'s investigation (vignette
mask, open-end/mold escape diagnosis). A pixel is **evaluable**, applied
identically to every configuration and every method, if and only if:

- it is **inside the fixed camera valid-pixel mask** — the 135-sequence
  majority mask, 102,049 px (6.9992%), `docs/eval_protocol.md` §8 — AND
- **the hit distance along that pixel's ray is ≤100mm** (camera-frame
  Z-depth, the same `d_hit` defined above, not Euclidean ray length).

**Only evaluable pixels enter the tau test.** A pixel outside this gate
never contributes to the predicted-observed set, regardless of what `tau`
or the predicted/GT depth values say — there's nothing meaningful to test
there (§ below). **Predicted depth is masked with the same valid-pixel
mask before use**: a method's own output inside the vignette carries no
information (the corresponding GT pixel is always forced to `raw==0` by
the same hardware mask, `docs/gpu_validation.md`), so a method's
predicted value there must not be trusted or scored either, symmetrically.

This single gate is what separates the two distinct non-evaluable
populations found during the investigation — a fixed hardware mask, and a
depth-clamp — from the two *evaluable*-but-still-miss populations that
need no special casing at all: open-end escapes and mold/extra-scene-
geometry escapes are simply rays that find **no hit** against
`coverage_mesh.obj`, already covered by §3 (discard, count), not folded
into "evaluable" one way or the other.

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
name. This has real compute cost at corpus scale — measured in §7, not
just flagged: ~35 hours on CPU/embree for the full corpus, which is why
§7 exists and why §7's decision (extend the GPU rasterizer) applies here.

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
- **Predicted-unobserved area fraction and calibration ratio** (per
  configuration, per tau — see below, own paragraph since it's the check
  that keeps every recall number honest).
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

### 2026-09-23: Aggregation rule for corpus-level metrics (decided before the full-corpus run)

Point estimate: pooled over regions. For region recall in a size class,
the number of detected GT regions divided by the total number of GT
regions in that class, across all evaluated sequences. Every region has
equal weight. Area-based metrics (false reassurance rate, false alarm
rate, area fraction, calibration ratio) are pooled the same way: summed
numerator area over summed denominator area across sequences.

Uncertainty: cluster bootstrap at the mesh level (resample mesh hash ids
from results/mesh_identity.csv with replacement, carrying all sequences
and regions of each resampled mesh), 10,000 replicates, 95% percentile
interval. The number of regions and the number of independent meshes
behind every estimate are reported with it.

Sensitivity (reported, not used for any decision): the same bootstrap
clustered at two coarser levels: (Colon, Segment), and (Colon, Segment,
Phantom Number), both taken from the dataset summary sheet, since
distinct mesh hashes can share one physical phantom or one mold.

All D1 to D2 pass/fail decisions use the point estimate, as the
thresholds in docs/success_criteria.md are stated on point values.

### Predicted-unobserved area fraction — the check that keeps recall honest

Two numbers, per configuration, per tau:

- **Predicted-unobserved area as a fraction of total mesh area.**
- **Ratio of predicted-unobserved area to GT-unobserved area** — 1.0 for
  a perfectly calibrated method (marks the same total area unobserved as
  GT actually is, whether or not it's the *same* area); above 1.0 means
  over-flagging, below 1.0 means under-flagging.

**Why this is required, not optional**: region recall alone can't
distinguish a method that genuinely detects missed regions from one that
just marks almost everything unobserved. A method returning the entire
mesh as unobserved scores perfect region recall (every GT region is
trivially ≥50% covered) while being useless — and a low tau (§2) could
push a genuinely mediocre depth predictor toward exactly this degenerate
behavior (flagged already in the "Flags" section, risk #3, as a specific
concern for D1.5; this area-fraction check is the direct instrument for
catching it, not just a related worry). **Report this area fraction
directly next to every recall number in every results table** — not in a
separate appendix — so a reviewer (or a future version of the person
running this pipeline) can't read a strong recall number without also
seeing whether it was bought by over-flagging. A recall improvement that
comes with the unobserved-area fraction climbing toward 1.0 (i.e., toward
"everything is unobserved") is not the same finding as a recall
improvement at a stable, GT-comparable area fraction, and the two must
never be presented identically.

### Localization error: Euclidean, not geodesic — decided (2026-09-23, approved by Chen)

`docs/success_criteria.md` §1 defines localization error as a "surface
distance" between two centroids, without saying whether that means
geodesic (along the mesh surface) or Euclidean (straight-line through 3D
space) — an ambiguity neither document resolved before `src/eval/` had to
implement something. **Decision: Euclidean**, between the two regions'
area-weighted face centroids.

**Reasoning**: geodesic distance on a ~700k-face mesh is not a single
well-defined number — Dijkstra-along-edges, the heat method, and an exact
(e.g. MMP-family) geodesic solver give genuinely different results, and
picking one adds an implementation-dependent methodological choice
without adding clarity to what's being measured. Euclidean is
parameter-free, deterministic, and trivially reproducible by anyone
re-running this code, at the cost of not being a "distance along the
surface" in the literal sense the phrase suggests.

**Diagnostic, not a change to the metric**: for each matched GT-region /
predicted-component pair, test whether the straight segment between the
two centroids intersects the mesh anywhere strictly between its two
endpoints (`src/eval/region_metrics.py::segment_intersects_mesh`, a small
margin excluded at each end so a centroid sitting on/near the surface
doesn't register a spurious self-hit). An intersecting segment means the
straight line cuts through the lumen or the wall, so Euclidean
understates the true surface separation for that pair. **Report the
fraction of matched pairs where this happens, per sequence, alongside
localization error** — not folded into the metric itself. If that
fraction is large on the corpus, geodesic distance should be revisited;
if small, Euclidean stands on measured evidence rather than an assumption
about mesh geometry.

---

## 6. Validation plan for the eval code

**Step 0, prerequisite to everything below: validate the GPU rasterizer's
new distance-output capability itself**, before trusting any oracle check
that depends on it. §7's decision approved exactly two extensions to the
frozen rasterizer — hit-distance output and arbitrary-pose input — and
the existing IoU 0.9988 validation (`docs/gpu_validation.md`) covers only
the rasterizer's original observed/unobserved logic, not this new
capability. **Required check**: for a sample of frames, compare the
rasterizer's reported hit distance against an independent computation of
the same quantity (e.g. `src/geometry/camera.py`'s `backproject_depth`
applied to the corresponding GT depth pixel, which should yield the same
camera-frame Z-depth up to floating-point/ray-BVH precision, since both
are describing the same GT geometry along the same ray). Agreement to
within a small, stated numerical tolerance (not yet chosen — proposed
alongside whoever implements the extension) is required before the hit
distance is trusted for the tau test anywhere else in this protocol.

**What the oracle configuration must reproduce — rewritten (2026-09-21)
after `docs/eval_protocol_oracle_gap.md`'s Part 1 measurement**: GT depth
+ GT pose, run through this protocol's own full ray-cast + tau-gate
pipeline, is compared against **our own GT rasterization restricted to
evaluable pixels** (§2a) — *not* directly against the released mesh's
`vt` flags. This is a deliberate change from the original plan: Part 1
measured that even a perfect, error-free oracle **cannot** reach IoU 1.0
against the released `vt` flags — vignette and the 100mm depth clamp
create a structural ceiling of **0.948-0.968** (3 sequences measured,
`docs/eval_protocol_oracle_gap.md`), for reasons that have nothing to do
with the eval code's correctness. Comparing the code's oracle output
directly against the released mesh would conflate a known, measured,
method-independent gap with an actual code bug — exactly the ambiguity
this rewrite avoids: the **evaluable-pixel-restricted** GT rasterization
is a fair target that a correctly-implemented oracle configuration really
can hit at IoU ≈ 1.0, since restricting to evaluable pixels removes
precisely the two structural gap causes before comparison.

**The released-GT comparison is kept too, as a separate, always-reported
number, with Part 1's ceiling stated next to it every time** — so a
reader sees "oracle vs. released GT: 0.95-ish (expected, see the
measured ceiling)" and "oracle vs. evaluable-pixel GT: should be ≈1.0
(actual code-correctness signal)" side by side, and nobody reads the
first number as a bug.

**Target and precedent**: `docs/gpu_validation.md` already validated a
related but not identical check (a direct visibility ray-caster, not a
depth-image-backprojection-and-tau-gate pipeline) at **full pixel
density, achieving IoU 0.9988** against the released mesh — before the
evaluable-pixel restriction existed as a concept. This protocol adopts
the same **IoU ≥ 0.99** bar, now against the evaluable-pixel-restricted
target, at full density — a reasonable target given the closely related
methodology, but not a guarantee this exact number reproduces, since the
pipelines aren't identical (flagged so nobody mistakes 0.9988 as already
having validated *this* code).

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

### The oracle gap is a known, method-independent property — not a per-method finding

Non-evaluable faces (vignette + 100mm-clamp, `docs/eval_protocol_oracle_gap.md`
Part 1) are marked **unobserved** for every method equally, including a
perfect oracle — no method, however accurate, can ever mark them observed,
because the pixels that would need to are gated out before the tau test
even runs (§2a). This has one specific, predictable consequence worth
stating before any method's numbers are reported: it **raises false alarm
rate uniformly** (`docs/success_criteria.md` §1: "fraction of
predicted-unobserved area that GT marks as observed") — every non-evaluable
face that the released GT calls observed contributes to false alarm rate
for *every* method, oracle included, by construction, not because any
method actually failed to detect something reachable.

**Measured size of the effect, corrected (2026-09-21)**: an earlier
version of this paragraph reported "3.16-5.13%," computed as
`gap / GT_observed` — the wrong denominator. `docs/success_criteria.md`
§1 defines false alarm rate as "fraction of **predicted-unobserved
area** that GT marks as observed," which for the oracle configuration is
`gap / (GT_unobserved + gap)` (predicted-unobserved = GT_unobserved ∪
gap, and only the gap portion is GT-observed). **Correct measured
values: 10.06%-23.85%** across the 3 tested sequences
(`docs/eval_protocol_oracle_gap.md`) — far larger than first reported,
and large enough that it isn't just "read false alarm rate against a
small floor" (the original framing) but a real problem needing the
ignore-set fix below, not just a caveat.

### The ignore set — ADOPTED (2026-09-21, approved by Chen)

**Adopted**, in-conversation, explicitly and by name — not inferred.
Originally proposed as a draft with measured before/after numbers (kept
below as the evidence base for the decision); now part of the protocol.

**Definition**: the **ignore set** for a sequence is the set of faces
never reachable through an evaluable pixel (§2a) under **GT pose** —
exactly the "gap" set already measured in
`docs/eval_protocol_oracle_gap.md` Part 1 (vignette + 100mm-clamp
causes). Computed **once per sequence**, from GT pose alone — **method-
independent**, the same set used for every configuration and every
pipeline evaluated on that sequence, never recomputed per method (so it
can't be tuned, satisfying the same §5-rule-4 "applied identically"
requirement as §2a and Flag 6).

**Effect, applied everywhere a method's output is scored**:
- **Excluded from false alarm rate's computation** — neither the
  numerator nor denominator counts an ignore-set face, for any
  configuration, any method.
- **Excluded from predicted-unobserved connected-component
  construction** — an ignore-set face can never bridge two otherwise-
  separate predicted-unobserved components together, removing the
  spurious-merge/spurious-new-component distortion measured above.
- **GT regions are unchanged** — the ignore set only ever touches the
  *predicted* side's bookkeeping, never `coverage_mesh.obj`'s own `vt`-
  flagged region definition (same as Flag 6 already states for §2a).

**Effect on the oracle numbers, MEASURED** (ignore-set faces = exactly
the already-measured gap faces, so "with the ignore set" numbers are
recovered directly from data already collected, not a new computation):

| Sequence | false alarm rate, no ignore set | false alarm rate, **with ignore set** | predicted-unobserved components, no ignore set | **with ignore set** |
|---|---|---|---|---|
| `c1_cecum_t1_v1` | 23.85% | **0.00%** | 3 | **3** |
| `c1_ascending_t3_v1` | 18.31% | **0.00%** | 5 (1 spurious) | **4** |
| `c2_rectum_t1_v1` | 10.06% | **0.00%** | 2 (2 merged away) | **4** |

With the ignore set, the oracle's false alarm rate drops to exactly 0
(predicted-unobserved becomes exactly GT-unobserved, which by definition
shares no faces with GT-observed) and the component counts return to
matching the true GT-unobserved regions exactly — the distortion is
fully explained by, and fully reversed by excluding, the same gap set
already measured. This was the evidence the adoption decision above was
based on.

**Mold-index-collision faces fall into the ignore set by construction,
not as a special case.** `docs/eval_protocol_oracle_gap.md` Item 1's
false_unobserved set (faces our own ray-cast never hits at all, from any
frame) is a strict subset of the "gap"/ignore set (faces never reached
by an *evaluable* ray specifically — a weaker, more inclusive condition
than "never hit at all"). Any face the released GT mislabels observed
because of the suspected `primID` collision with a mold triangle is,
by definition, never hit by our own lumen-only ray-cast — so it is
already excluded via the ignore set, with no additional logic needed to
special-case it.

---

## 7. Compute cost estimate — MEASURED, before writing code

Ray-casting (§2) is the dominant cost; backprojection, tau comparison,
face-set accumulation, and region-level metrics are cheap elementwise
`numpy` operations by comparison. Benchmarked directly rather than
guessed: `scratch/pipelines/eval_protocol_cost_benchmark.py`, log:
`logs/eval_protocol_cost_benchmark.log`, using the project's own camera
model, `coverage_mesh.obj`, and `trimesh`'s embree-backed ray intersector
(`trimesh.ray.ray_pyembree`, confirmed active by default in
`scratch/.venv`).

**Design point validated first**: ray-casting depends only on
`(frame, pose-variant)`, not on `(frame, configuration)` or
`(frame, configuration, tau)`. Two pose variants exist (GT pose, shared by
the oracle and pred-depth-only configurations; aligned predicted pose,
shared by the fully-predicted and pred-pose-only configurations) — not
four. `tau` is applied post-hoc to cached hit distances, exactly as
instructed, at zero extra ray-casting cost.

**MEASURED, full pixel density (1350x1080 = 1,458,000 rays/frame), one
pose variant, mean over 3 frames spread across `c1_cecum_t1_v1`**:

| Quantity | Value |
|---|---|
| Mean ray-cast time / frame | 1.71 s |
| Throughput | ~0.9M rays/sec (0.57-1.18M across the 3 sampled frames) |
| Per-sequence (218 frames × 2 pose variants) | **743.7 s ≈ 12.4 min** |
| **Corpus extrapolation (169 sequences)** | **125,680 s ≈ 2,095 min ≈ 34.9 hours** |

**This exceeds "a few hours" by roughly an order of magnitude.** Per
instructions: not silently reducing density. Options below, none chosen
here.

### Option 1 (recommended): reuse the project's existing GPU rasterizer

`scripts/visibility_full.py` / the GPU (Warp) rasterizer already built
for this project's own GT-side visibility work solved a **structurally
identical problem** — one ray per pixel per frame, nearest hit on this
same `coverage_mesh.obj` — at full density, for the entire 169-sequence
corpus, in **19.3 minutes total**
(`docs/visibility_rasterization.md`/`EXPERIMENTS.md`, 2026-09-19),
validated to IoU 0.9988 against the released mesh
(`docs/gpu_validation.md`). That's roughly **100x faster** than this
benchmark's CPU/embree throughput for a comparable single-pose-variant,
full-corpus task. Extended to 2 pose variants, the precedent throughput
would plausibly land the entire corpus well under an hour.

**Two extensions needed, neither touching the core algorithm**: (a) also
return hit **distance** (`d_hit`), not just hit/miss and face index — the
rasterizer must already compute this internally to determine the nearest
hit, so exposing it should not add meaningful cost, but it's currently
unused/undocumented, per what's visible in this design process; (b)
accept an arbitrary camera pose (GT or aligned-predicted), not only the
GT trajectory it's been run against so far.

**Historical record**: `docs/visibility_limitations.md` states plainly
"Visibility rasterization tooling (`scripts/visibility_full.py`,
`scripts/render_coverage_views.py`, `src/geometry/`) is now frozen," so
extending it needed explicit sign-off before this option could be
pursued. That sign-off was requested here, as a flag alongside three
other options, and **was subsequently obtained — see the "Decision
(2026-09-21) — approved by Chen" note below**, scoped to exactly these
two extensions. The existing IoU 0.9988 validation covers the
observed/unobserved logic, which this extension doesn't change; the new
distance-output capability needs its own, separate validation pass
(§6 Step 0) before being trusted.

### Option 2: reduce pixel density on CPU/embree

Quantified, not assumed: extrapolating this benchmark's throughput
linearly with pixel count, stride 4 → ~8.7 hours, stride 8 → ~4.4 hours
(both still ≥ "a few hours" under most readings of that phrase). Stride 8
is exactly the density `docs/oracle_check.md` already found produces a
real, diagnosed IoU shortfall (0.869 vs. 0.9988 at full density) — reusing
it here would put that already-known artifact into the actual eval
numbers, not a diagnostic script, and would need its own fresh
re-validation against this protocol's IoU ≥ 0.99 oracle bar (§6) before
being trusted at any stride below 1. Not recommended given Option 1
exists.

### Option 3: reduce sequence scope — ruled out, not proposed as viable

D1's own trigger condition (`docs/success_criteria.md` §2) is "one
end-to-end pipeline has been run on **all 169** registered sequences."
Running fewer isn't a compute optimization available within the frozen
protocol without a `docs/success_criteria.md` §6-style deviation entry
changing D1's trigger itself — out of scope for an eval-code design
decision, noted only to rule it out explicitly.

### Option 4: CPU/embree, parallelized across sequences

No methodology change, no touching frozen tooling: run independent
sequences as separate OS processes (embree/trimesh ray intersectors are
not meant to be shared across threads on one mesh object, but separate
processes each with their own mesh copy are straightforward and safe).
The shared box has 48 CPUs (`nproc`, checked earlier this session).
Naive per-sequence parallelism across, say, 8-16 concurrent processes
could plausibly cut ~35 hours to roughly 2-4 hours wall-clock without any
change to density or accuracy — but shares the same "don't hog a shared
resource" courtesy this project already practices for GPUs
(`CLAUDE.md`'s workflow rules); worth stating as a real constraint, not
assuming unlimited concurrency is free to take on a shared box, and not
benchmarked here (single-process cost only was measured).

### Decision (2026-09-20) — approved by Chen

**Option 1 approved by Chen**, in-conversation, via an explicit choice
between the four options presented above (not inferred from an absence of
objection): extend the existing GPU (Warp) visibility rasterizer — add
hit-distance output and accept an arbitrary camera pose (GT or
aligned-predicted), not only the GT trajectory it currently runs against.
This is the sign-off `docs/visibility_limitations.md`'s "frozen" note
requires, obtained here, for **exactly these two extensions (hit-distance
output + arbitrary-pose input) and nothing else** — not a blanket
reopening of that tooling for unrelated changes. Options 2-4 are
documented above for the record but not being pursued. **Not yet
implemented** — this section records the decision; the extension itself
and `src/eval/` are separate, still-unwritten next steps. The new
distance-output capability needs its own validation before use, added as
an explicit step in §6.

---

## 8. Vignette mask — decided (2026-09-21)

Resolved the `docs/eval_protocol_oracle_gap.md` investigation's Part A
question: **the fixed camera valid-pixel mask is the 135-sequence
majority mask, 102,049 px (6.9992%)**, computed as the intersection of GT
depth `raw==0` across every frame, checked identical across 135 of the
169 registered sequences (streamed from the archives, no extraction;
`scratch/pipelines/oracle_gap_vignette_169.py`). The other 34 sequences'
excess pixels (each sequence's own unique amount, +7px to +34,787px above
this baseline) are **trajectory-dependent persistent escapes, not
vignette** — a pixel that happens to face an opening for a whole
sequence's duration, not a hardware property. These are handled by §3
(rays that miss the mesh: discarded, counted), the same as the transient
open-end bulges found earlier, not folded into the vignette mask. This
value feeds §2a's "evaluable pixel" definition once Part 2 is written.

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

   **Rejected alternative, stated explicitly**: registering the
   *predicted point cloud* (backprojected predicted depth, at predicted
   pose) directly onto `coverage_mesh.obj` — e.g. via ICP — was
   considered and rejected. That approach lets **depth error drive the
   alignment**: ICP would warp the predicted trajectory to minimize
   point-to-mesh distance, which depends on predicted depth's own
   accuracy. This would silently contaminate the "GT depth + predicted
   pose" configuration (§4) — the whole point of that configuration is to
   isolate pose error *alone*, with depth held exactly correct, but a
   point-cloud-ICP registration computed from a run that includes
   predicted depth (or fit once and reused, which just moves the
   contamination) breaks that isolation and makes the four configurations
   in §4 no longer independently interpretable. **Trajectory alignment
   depends on pose only** (§1's Umeyama fit uses camera *positions*,
   never depth) — it keeps depth error and pose error separable across
   all four configurations, which is the entire reason §4's post-hoc-
   substitution design exists in the first place
   (`docs/success_criteria.md` §6's deviation entry).

5. **Predicted intrinsics are not used anywhere in this protocol** — all
   ray-casting uses the project's own fixed Scaramuzza camera model,
   never a pipeline's own predicted (pinhole) intrinsics output, matching
   `CLAUDE.md`'s frozen decision for the GT side and extending it to the
   predicted side for the same reason (comparability across methods). Not
   a conflict, just making an implicit consequence of an existing frozen
   decision explicit before someone assumes predicted intrinsics get used
   somewhere in fusion.

6. **The evaluable-pixel restriction (§2a) is a filter under
   `docs/success_criteria.md` §5 prohibited move #4** ("Introducing a
   per-sequence or per-region filter that was not pre-stated, unless it
   is applied identically to every method and logged"). It qualifies:
   the mask (135-sequence majority, 102,049px) and the 100mm cutoff are
   fixed, sequence-independent, method-independent, and applied
   identically to the oracle and every predicted configuration alike —
   never tuned per sequence or per method — and it's logged here, in
   `docs/eval_protocol_oracle_gap.md`, and in §6's rewritten
   code-correctness target. **The GT region definition itself is
   unchanged**: regions are still exactly the released `vt`-flagged
   components (`docs/success_criteria.md` §1's "GT region"), not
   recomputed against the evaluable-pixel mask — only the *predicted*
   side (what counts as a valid comparison pixel) is restricted, never
   the ground truth's own definition of what a region is.

7. **The ignore set (§6, adopted 2026-09-21) is also a filter under
   `docs/success_criteria.md` §5 prohibited move #4**, on the same terms
   as Flag 6: computed once per sequence from GT pose alone, before any
   method's output is scored, identical across every configuration and
   every pipeline evaluated on that sequence, never tuned, and logged
   here and in `docs/eval_protocol_oracle_gap.md`. It excludes ignore-set
   faces from false alarm rate and from predicted-unobserved
   connected-component construction — a narrower effect than §2a's
   evaluable-pixel restriction (which gates the tau test itself), applied
   downstream of it. **GT regions are unchanged by this filter too** —
   same principle as Flag 6, restated because it's a separate adopted
   filter, not a consequence of the first.

---

No code written. This document is the thing to review/approve before
`src/eval/` gets implemented against it.
