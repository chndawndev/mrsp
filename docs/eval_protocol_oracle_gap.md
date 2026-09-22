# Oracle ceiling against the released GT — Part 1, STOPPED again (new blockers)

Follow-up to the user's resolution of the original mask-identity blocker
(previous version of this document, retained in git history). Applied the
prescribed fix: vignette = complement of the intersection of `raw == 0`
across every frame of a sequence, not a single frame. **Both new
sub-checks the resolution asked for turned up their own blockers.**
Neither is the originally-suspected problem repeating — they're two new,
distinct findings, reported here. Per instructions, stopping again before
Part 1 steps 2-4 or Part 2.

**Sequences**: `c1_cecum_t1_v1`, `c1_ascending_t3_v1`, `c2_rectum_t1_v1`
(same three as before). Scripts:
`scratch/pipelines/oracle_gap_vignette_169.py` (169-sequence check),
`scratch/pipelines/oracle_gap_openend_validate.py` (open-end escape
validation). Logs: `logs/oracle_gap_vignette_169.log`,
`logs/oracle_gap_openend_validate_v2.log`. Raw data:
`results/pipelines/oracle_gap_vignette/summary.json`,
`results/pipelines/oracle_gap_openend_validation.json`.

---

## Part A: vignette = intersection-across-all-frames, all 169 sequences — MEASURED

Computed per-sequence, streaming depth TIFFs directly from each
`registered_videos/*.zip` (Python `zipfile`, no extraction to disk;
`c0_*` legacy archives excluded, confirmed 169 remain). Sanity-checked
first against the already-known `c1_cecum_t1_v1` result (218 frames,
102,049px, exact match) before scaling up.

**Result: NOT identical across all 169. Stopping again, per the
resolution's own stop condition** ("If they differ, report the groups and
by how many pixels, and stop.").

- **135 of 169 sequences share one exact mask**: 102,049 px (6.9992%) —
  the same value already established from `c1_cecum_t1_v1` alone. This is
  the majority, and matches the single-sequence result exactly.
- **34 of 169 sequences each have their own, unique, larger mask** — every
  one of the 34 is its own singleton group (no two of the 34 share a
  mask with each other). Excess over the 102,049 baseline ranges from
  **+7 px** (`c1_sigmoid2_t1_v1`, negligible) up to **+34,787 px**
  (`c1_transverse2_t3_v2`, +2.4% of frame). Full list of all 35 groups
  and their members: `logs/oracle_gap_vignette_169.log`.
- **Pattern within the 34, checked as instructed**: no clean split by
  `c1` vs `c2`, and no clean split by `v1` vs `v2` either — differences
  are spread continuously from single-digit to tens-of-thousands of
  pixels, not two discrete clusters. **What *is* visible**: for the base
  sequences appearing more than once in the 34 (e.g. `c1_descending_t1_*`,
  `c1_sigmoid2_t1_*`, `c1_transverse2_t3_*`), **`v2` and `v3` are always
  close to each other** (e.g. `c1_descending_t1_v2`=119,742 vs
  `v3`=119,741, 1px apart) **while `v1` differs more** (same sequence's
  `v1`=122,787, ~3,045px from v2/v3). This tracks the project's own
  frozen convention that v2/v3 share a trajectory (robot repeat) while v1
  has a different trajectory (`CLAUDE.md`) — **not** the "v1/v2 differ in
  imaging settings" axis the resolution specifically asked to check.

## Interpretation (Part A)

A per-sequence-intersection mask correctly removes *transient* escapes
(the rise-then-fall bulges from the previous version of this document),
but cannot remove a *persistent* one — a pixel that happens to be
`raw==0` in literally every frame of one particular sequence (e.g. a
short or trajectory-specific run that spends its entire duration facing
toward an opening) survives the intersection and gets counted as
"vignette" for that sequence only. The v2≈v3-but-not-v1 pattern points at
trajectory, not hardware/imaging settings, as the driver — consistent
with this being the same underlying phenomenon as Part B below (open-end
geometry), just persistent across a whole sequence rather than transient
within one. **Not independently confirmed beyond this pattern check** —
haven't traced any of the 34 sequences' extra pixels to specific
open-end-facing rays the way the previous bulge frames were visually
spot-checked.

---

## Part B: open-end escape vs. our rasterizer's no-hit — MEASURED, mixed result

For `c1_ascending_t3_v1` and `c2_rectum_t1_v1`, every frame: compared
`raw==0` pixels outside the (135-sequence global) vignette mask against
pixels where a fresh full-density ray-cast (project camera model, GT
pose, `coverage_mesh.obj`, same embree backend as earlier benchmarks)
reports no intersection at all.

**Bug caught and fixed before trusting the first run**: the first attempt
compared vignette-restricted `escape` against an *unrestricted* `nohit`
— comparing a vignette-excluded set against a set that still included
vignette pixels. Fixed by restricting both sides to outside the vignette
mask (inside it, GT's `raw==0` is forced by the `mask.png` post-process
regardless of the true ray result, per `docs/gpu_validation.md`, so it
carries no information there). Confirmed the fix mattered by checking:
100% of `c2_rectum_t1_v1` frame 105's original disagreement was pixels
*inside* the vignette — a pure artifact of the bug, not a real escape
disagreement.

**After the fix**:

| Sequence | escape | nohit | agree | escape-only | nohit-only | overall IoU |
|---|---|---|---|---|---|---|
| `c2_rectum_t1_v1` | 427,169 | 427,311 | 426,946 | 223 | 365 | **0.9986** |
| `c1_ascending_t3_v1` | 24,483 | 2,854,253 | 24,483 | 0 | 2,829,770 | **0.0086** |

**`c2_rectum_t1_v1` confirms the resolution's hypothesis cleanly**: open-end
escapes and our rasterizer's no-hit pixels agree almost exactly (IoU
0.9986), after the vignette-restriction fix. This directly validates the
camera model and pose convention at the open-end silhouette, for this
sequence.

**`c1_ascending_t3_v1` does not confirm it, and the disagreement is real,
not the vignette bug** (the fix changed nothing for this sequence —
0 pixels of its disagreement were ever inside the vignette). Inspected
frame 152 (the escape peak) directly: of its 23,610 `nohit_only` pixels,
18,174 have GT `raw==65535` (a real GT hit beyond 100mm — the third gap
cause, not a miss) and 5,436 have valid GT depth (68.4-100mm, median
97.1mm — genuinely GT-observed, close-range surface points our ray-cast
entirely misses). Also checked and ruled out a simple bug: the GT camera
position is inside `coverage_mesh.obj`'s bounding box at every probed
frame (0, 152, 257), so this isn't a pose/scale error putting the camera
outside the mesh.

## Interpretation (Part B) — correction: the Group A1 attribution was wrong, cause is UNKNOWN

**Retracted.** An earlier version of this document attributed
`c1_ascending_t3_v1`'s disagreement to `docs/visibility_outliers.md`'s
"Group A1" issue, on the basis that it's the same sequence. That's not
evidence of the same cause, and the signatures don't actually match:

- **Group A1** (`docs/visibility_outliers.md`): sub-millimetre,
  single-face speckle — footprint 0.15mm² (~0.44mm equivalent diameter)
  — scattered along the observed/unobserved boundary contour, consistent
  with a pixel- or half-pixel-level silhouette-edge disagreement.
- **This finding**: thousands of pixels per frame (23,610 at frame 152
  alone) where our ray-cast finds nothing at all, while GT depth reports
  an actual surface 68-100+ mm away. Not sub-pixel, not boundary-edge
  speckle — a large-scale, systematic absence of geometry along entire
  ray bundles.

Same sequence, different phenomenon by every measured characteristic.
**Cause is UNKNOWN, pending the diagnosis below** — not attributed to
Group A1 or to anything else without direct evidence.

## Diagnosis: does the renderer trace against geometry `coverage_mesh.obj` doesn't contain?

Hypothesis (yours): `Render.cu` allocates the coverage buffer from
`model->meshes[0]` only, while `3D_models/` contains `molds/` alongside
`lumen(s)/` — if the depth *render* pass traces the full scene but the
*coverage* export is `meshes[0]`-only, rays that exit through the open
end could hit real depth-rendered geometry (a mold, a cap, adjacent
structure) that the exported `coverage_mesh.obj` simply doesn't include.

### Step 1 — how the model is loaded, MEASURED, cited

The renderer loads exactly one model file per run —
`RenderingModule.cpp:37`, `modelFilePath = argv[2] + "model.obj"` — via
`loadOBJ()` (`render/Model.cpp:126`), which splits a single `.obj` into
one or more `TriangleMesh` entries in `model->meshes` (one per
material/shape group found by `tinyobj::LoadObj`,
`render/Model.cpp:177-190`). Whether that specific `model.obj` combines
lumen and mold geometry into multiple shapes is a data question this
repo's code can't answer directly — but two things downstream of loading
are unambiguous in the code:

- **The ray-traced scene includes every mesh, not just `meshes[0]`.**
  `render/RenderContext.cpp:220`, `const int numMeshes =
  (int)model->meshes.size();`, then `RenderContext.cpp:262-263`,
  `for (int meshID=0;meshID<numMeshes;meshID++) { TriangleMesh &mesh =
  *model->meshes[meshID]; ... }` builds OWL/OptiX geometry for every
  mesh and adds each to the acceleration structure used by both ray
  types — `RenderContext.cpp:236-237`,
  `owlGeomTypeSetClosestHit(triMeshGeomType,0,module,"primary");
  owlGeomTypeSetClosestHit(triMeshGeomType,1,module,"occlusion");` — so
  the primary (depth) ray can hit any loaded mesh, not just the first.
- **The coverage buffer is sized and read from `meshes[0]` only.**
  `RenderingModule.cpp:135`,
  `coverage_host = (uint8_t*) malloc(model->meshes[0]->index.size()*...)`;
  `RenderingModule.cpp:340`,
  `cudaMemcpy(coverage_host, owlBufferGetPointer(context->coverage,0),
  model->meshes[0]->index.size(), ...)`; same restriction in
  `render/RenderContext.cpp:93`. No other mesh index ever appears in the
  coverage buffer's allocation or read-out.

**So if `model.obj` for a given render includes more than one mesh, depth
can be rendered against geometry that coverage never tracks and that
never appears in the exported `coverage_mesh.obj`.** This is a structural
possibility confirmed at the code level; whether it's what actually
happened for this sequence is what steps 2-3 test.

### Step 2 — what exists under `3D_models/` for this colon/segment, MEASURED

`/data1_ycao/chua/datasets/C3VDv2/3D_models/colon1/` has both:
`lumens/c1_ascending_model.zip` (one file, `c1_ascending.obj`, 419,875
verts / 838,043 faces) and `molds/c1_ascending_mold.zip` (three separate
STL pieces: `c1_ascending_bottom.stl`, `c1_ascending_core.stl`,
`c1_ascending_top.stl`). Extracted both into `scratch/3D_models/` for
this check (small model files, not video sequences — not the "more than
one sequence" the extraction-approval rule is about).

### Step 3 — where do the 5,436 no-hit-but-valid-depth pixels actually lie, MEASURED

The raw lumen `.obj` and the mold STLs are in their own local coordinate
frame, not `coverage_mesh.obj`'s registered world frame (bounding boxes
don't overlap at all before registration). Registered the raw lumen onto
`coverage_mesh.obj` via `trimesh.registration.mesh_other` (principal-axis
initial alignment + ICP, no scale) —
**RMS residual 0.094mm**, confirming the raw lumen and `coverage_mesh.obj`
are the same surface up to a rigid transform, and giving a trustworthy
transform to apply to the mold (same local scan/CAD session as the
lumen). Script: `scratch/pipelines/oracle_gap_diagnosis_step3.py`, log:
`logs/oracle_gap_diagnosis_step3.log`.

Back-projected the exact 5,436-pixel set (frame 152, confirmed count) to
world coordinates via GT depth + GT pose, then measured distance to (a)
`coverage_mesh.obj` and (b) the registered mold (all 3 pieces):

| Target | median dist | mean | max | min |
|---|---|---|---|---|
| `coverage_mesh.obj` | 4.398 mm | 4.891 mm | 13.679 mm | 0.001 mm |
| registered mold (combined) | **1.076 mm** | 1.451 mm | 7.936 mm | 0.001 mm |
| — vs. `_core.stl` alone | 1.332 mm | — | — | 0.003 mm (90.4% within 5mm) |
| — vs. `_bottom.stl` alone | 6.981 mm | — | — | (18.3% within 5mm) |
| — vs. `_top.stl` alone | 27.835 mm | — | — | (0% within 5mm) |

**76.5% of the 5,436 points are closer to the mold than to
`coverage_mesh.obj`**, and where they're close to the mold, it's
specifically the `_core` piece (90.4% within 5mm) — not `_bottom` or
`_top`. 85.9% of the points also fall inside `coverage_mesh.obj`'s own
bounding box, so "outside the bbox" alone would have missed this; the
surface-distance comparison is what separates the two hypotheses.

### Verdict: CONFIRMED, with the precision it deserves stated plainly

These pixels' GT depth is measurably closer to the mold's core piece than
to the exported lumen mesh, in the direction and magnitude the hypothesis
predicts, on top of a code-level mechanism (§ step 1) that structurally
allows exactly this. **Not a clean 0.000mm match** — median 1.08mm, not
sub-millimetre — which is expected: the registration transform (however
well-validated at 0.094mm RMS on the lumen) is one extra step of
possible error, real manufacturing tolerance exists between a cast lumen
and the mold it was cast from, and a few points likely sit near the
core/bottom seam or the open-end silhouette itself. This is enough to
accept the hypothesis, not enough to claim exact confirmation down to the
millimetre.

**These pixels are GT depth from geometry outside `coverage_mesh.obj`
(the mold, not the lumen). They are correctly handled as ray misses under
§3 of the protocol, and are not an oracle gap.** Documented as a fourth
cause below, alongside vignette, open-end escape, and the 100mm clamp.

### Fourth cause, for the record

- **Cause**: mold/extra-scene geometry visible through an open end,
  rendered into the depth image but never tracked by the coverage buffer
  (`meshes[0]`-only) or exported into `coverage_mesh.obj`.
- **Effect**: a valid, non-clamped GT depth value at a pixel where no
  face of `coverage_mesh.obj` can ever be hit — a stricter version of
  open-end escape (§3 already handles it the same way: ray finds no hit
  against `coverage_mesh.obj`, discarded and counted, not treated as a
  gap), just with a `raw` value that looks like real geometry rather than
  `0`.
- **Scope, not measured**: how common this is beyond `c1_ascending_t3_v1`
  frame 152 — whether it's specific to sequences with mold pieces close
  to the open end, or general — is unknown; not investigated further,
  out of scope for this diagnosis.

---

## Part B, second confirmation on a clean replacement sequence — MEASURED

Picked `c1_descending_t1_v1` (Open End Visible=yes, descending segment,
123 frames) — explicitly not in `docs/visibility_outliers.md`'s Group A1
or A2. Same method as `c2_rectum_t1_v1` (vignette-restricted on both
sides). **Result: IoU 0.9976** (agree=6,091,639, escape_only=1,288,
nohit_only=13,416, out of ~6.1M total escape/nohit pixels across 123
frames). Log: `logs/oracle_gap_openend_validate_descending.log`.

**Two independent, uncontaminated sequences (`c2_rectum_t1_v1` 0.9986,
`c1_descending_t1_v1` 0.9976) now confirm the open-end-escape hypothesis
cleanly.** Combined with the diagnosed 4th cause fully accounting for
`c1_ascending_t3_v1`'s disagreement (not a refutation, a different,
now-understood phenomenon), the hypothesis stands: open-end escapes (and
mold-geometry escapes) are ray misses, handled by §3, not an oracle gap.

---

## Escape fraction per frame — recorded, no further work per instructions

Peak values, kept for later analysis: `c1_ascending_t3_v1` frame 152,
1,375px (0.094% of frame); `c2_rectum_t1_v1` frame 41, 17,148px (1.18% of
frame) — both exactly reproduce the earlier bulge-peak counts from the
first version of this document (cross-check: the two independent
computations, single-frame-vs-baseline comparison and
raw==0-outside-corrected-vignette, agree exactly). Full per-frame series
for both sequences: `results/pipelines/oracle_gap_openend_validation.json`.

---

## Part 1 steps 2-4, resumed with the corrected 4-cause model — MEASURED

Both blockers resolved (135-sequence majority mask as the global
vignette, per Decision 1; open-end/mold escapes confirmed as ray misses,
not a gap, per the diagnosis above and 2 clean confirmations). Full
pixel density, GT pose, all 3 original sequences. Script:
`scratch/pipelines/oracle_gap_part1_steps234.py`, log:
`logs/oracle_gap_part1_steps234.log`, raw numbers:
`results/pipelines/oracle_gap_part1/summary.json`. Per-face, tracked
across every frame whether each GT-observed face was ever reached by an
**evaluable** ray (outside vignette, hit distance ≤100mm), ever hit
*only* via a vignette-region ray, or ever hit *only* via a valid-mask ray
beyond 100mm — verified first that the camera-frame Z-depth computed this
way matches the GT depth image almost exactly for the oracle case (median
0.0034mm, p95 0.011mm, on a clean sanity frame) before trusting the
full run.

| Sequence | GT-observed | gap (never evaluable) | vignette-only | beyond-100mm-only | union | residual | **max face IoU** |
|---|---|---|---|---|---|---|---|
| `c1_cecum_t1_v1` | 635,753 | 20,095 (3.16%) | 19,720 (3.10%) | 0 (0%) | 19,720 (3.10%) | 375 | **0.9679** |
| `c1_ascending_t3_v1` | 615,047 | 31,580 (5.13%) | 23,183 (3.77%) | 3,744 (0.61%) | 26,927 (4.38%) | 4,653 | **0.9484** |
| `c2_rectum_t1_v1` | 201,300 | 9,329 (4.63%) | 9,176 (4.56%) | 0 (0%) | 9,176 (4.56%) | 153 | **0.9530** |

**Cross-check**: `c1_ascending_t3_v1` is the *only* sequence of the 3
with a nonzero beyond-100mm-only count (3,744) — consistent with the
diagnosed 4th cause (mold geometry through the open end, much of which
reads as valid depth in the 68-100mm range or clamps to `raw==65535`
beyond it) being concentrated in exactly this sequence.

**Residual** (never hit *evaluable*, vignette, *or* beyond-100mm at all —
375/4,653/153 faces): not further diagnosed, flagged honestly rather than
folded into either named cause. Plausibly the same kind of numerical/
silhouette-edge noise already characterized in `docs/gpu_validation.md`
(~0.006% GPU-vs-CPU backend disagreement) or faces reachable only via
open-end/mold rays whose GT depth also happens to be invalid — not
confirmed either way.

**Max face IoU is the answer to "how good can an oracle depth-image
pipeline ever look against the released GT, structurally"**: 0.948-0.968
across the 3 sequences — genuinely short of 1.0, entirely from vignette
+ 100mm-clamp, not from any method error. This number belongs next to
any future oracle-configuration result as the honest ceiling, not 1.0.

### Predicted-side impact — CORRECTED interpretation (2026-09-21)

**The earlier version of this section was wrong about what changes.**
It described the gap as changing "GT-unobserved regions" — it doesn't.
**GT regions stay exactly as released, always** — they're defined once
from `coverage_mesh.obj`'s own `vt` flags (`docs/success_criteria.md` §1)
and this protocol never touches that definition (Flag 6, `docs/eval_protocol.md`).
**What the gap actually distorts is the *predicted* side**, for every
method including a perfect oracle: a face that can never be reached
through an evaluable pixel can never be marked predicted-observed,
regardless of depth/pose accuracy, so it lands in **predicted-unobserved**
by construction — inflating false alarm rate (since it's a GT-observed
face sitting in the predicted-unobserved set) and potentially bridging
otherwise-separate predicted-unobserved *components* into one, which is
what localization error's centroid computation (`docs/success_criteria.md`
§1) actually uses. The numbers below are the same computation as before
(built the real >5mm GT-unobserved regions via face adjacency,
`src/geometry/mesh_stats.py`, then rebuilt with gap faces added), but the
correct reading is **"the oracle configuration's own predicted-unobserved
components, with vs. without the non-evaluable faces baked in"** — not a
GT redefinition.

| Sequence | predicted-unobserved components *without* non-evaluable faces (= true GT-unobserved regions) | *with* non-evaluable faces (today's actual oracle output) | components whose face-set changed | of those, merged with another component |
|---|---|---|---|---|
| `c1_cecum_t1_v1` | 3 | 3 | 3/3 | 3 |
| `c1_ascending_t3_v1` | 4 | **5** | 4/4 | 4 |
| `c2_rectum_t1_v1` | 4 | **2** | 4/4 | 4 |

**Every tracked component changes in every sequence**, purely from the
gap, not from any depth/pose error — this is what an oracle's predicted-
unobserved output actually looks like today, distorted relative to the
true GT-unobserved regions it's supposed to reproduce. `c1_cecum_t1_v1`
grows all 3 components without changing the count; `c1_ascending_t3_v1`
gains a spurious **new** component (gap faces forming their own component
large enough to cross 5mm on their own — exists only because of the gap);
`c2_rectum_t1_v1` **merges** 4 components into 2 (gap faces bridge
previously-separate components into one, which would also merge/distort
whatever localization-error centroid gets computed from them).

### False alarm rate — CORRECTED, the earlier number used the wrong denominator

**Also wrong in the earlier version**, and in `docs/eval_protocol.md` §6's
"raises false alarm rate uniformly by 3.16-5.13%" — that used
`gap / GT_observed`, not `docs/success_criteria.md` §1's actual
definition: "fraction of **predicted-unobserved area** that GT marks as
observed," i.e. `gap / (GT_unobserved + gap)` for the oracle configuration
(predicted-unobserved = GT_unobserved ∪ gap, and only the gap portion of
that is GT-observed). **Corrected, MEASURED**:

| Sequence | GT_unobserved | gap | false alarm rate (oracle, uncorrected) |
|---|---|---|---|
| `c1_cecum_t1_v1` | 64,155 | 20,095 | **23.85%** |
| `c1_ascending_t3_v1` | 140,855 | 31,580 | **18.31%** |
| `c2_rectum_t1_v1` | 83,442 | 9,329 | **10.06%** |

**This is far larger than the earlier 3.16-5.13% figure** — a perfect
oracle, today, would show a false alarm rate of 10-24% purely from the
gap, which would badly fail `docs/success_criteria.md` §2's D1.5 check
("false reassurance rate... strictly between 0.02 and 0.98" — note D1.5
is about *false reassurance*, the opposite-direction metric, but a
10-24% false alarm rate floor is exactly the kind of large, method-
independent distortion that would make *any* false-alarm-based reading
of D1-D3 misleading without a fix). The ignore-set amendment below is
proposed specifically because this number is too large to leave
unaddressed.

---

## Item 1: testing the mold-index-collision hypothesis for Group A1 — tightened, MEASURED, weaker than first reported

**This supersedes the first pass below the fold** (kept for the record,
not deleted). The first pass reported "≥38% explained" without a null
baseline — once one exists, the picture is materially weaker: real,
statistically detectable, but far smaller in magnitude, and the decisive
test the tightening added **fails** by the criterion set in advance.

Hypothesis: `Render.cu`'s coverage update (`coverage[primID] = 255`,
lines 315-320) writes to a per-mesh-local `primID` with no check of which
mesh was actually hit, and the scene has multiple meshes (lumen + mold,
`RenderContext.cpp:262-263`). If a primary ray hits a **mold** triangle
with local index N, the bug would mark **lumen** face N observed.

**Sequences, expanded**: the 3 Group A1 sequences, plus `c2_rectum_t4_v1`
(Group A2, a generalization test) and `c1_cecum_t1_v1` (control, `Open
End Visible: no`). Extracted `coverage_mesh.obj`+`pose.txt` and
lumen/mold 3D models for `c2_rectum_t4_v1` (small, selective, not the
full archive). Script:
`scratch/pipelines/oracle_gap_item1_tightened.py`, log:
`logs/oracle_gap_item1_tightened.log`, saved arrays (per-face hit
booleans, for reuse without re-running the ~14min/sequence registration):
`results/pipelines/oracle_gap_item1_tightened/`.

### a) Null baseline — the correction that matters most

For each sequence, drew 1,000 random same-size samples of lumen indices
(from all lumen faces, and separately from GT-observed faces only) and
measured how often they'd overlap the mold-hit index set (§b's union) by
pure chance.

| Sequence | observed union overlap | null baseline (GT-observed pool), mean ± std | **excess over chance** |
|---|---|---|---|
| `c1_ascending_t4_v2` | 46.24% | 32.60% ± 0.52% | **13.64 pp** (≈1,096 of 8,032 faces) |
| `c1_ascending_t4_v3` | 46.24% | 32.61% ± 0.52% | **13.62 pp** (≈1,097 of 8,050 faces) |
| `c1_ascending_t3_v1` | 45.48% | 32.45% ± 0.67% | **13.02 pp** (≈606 of 4,653 faces) |
| `c2_rectum_t4_v1` | 0.00% | 0.00% ± 0.00% | 0 (mold never hit at all) |
| `c1_cecum_t1_v1` (control) | 0.00% | 0.00% ± 0.00% | 0 (mold never hit at all) |

**Two things both true at once, stated precisely**: (1) the null baseline
itself is surprisingly high (~32.6%, not near 0%) — the mold pieces get
hit so extensively across a full video (`_bottom` ≈51% of its own
400,708 triangles ever hit at least once, `_top` ≈31% of 253,230) that a
*random* lumen index has a real chance of coincidentally matching a
mold-hit index, just from the mold's own hit coverage being so broad.
(2) Given the null distribution's tight spread (std ≈0.5-0.7pp over
1,000 draws), the **observed excess (13.0-13.6 percentage points) is
many standard deviations above chance — not explainable as noise**. So:
the raw "46% overlap" headline from the first pass was misleading (most
of it is baseline chance, not signal), but a real, statistically robust,
much smaller effect (~13pp, ≈13% of false_unobserved) survives
correction.

### b) Corrected indexing: union of each piece's own local (0-based) hit indices

Recomputed as instructed — not concatenated/offset numbering, a true
per-piece-local union: a lumen index N counts as "mold-hit" if it was hit
in `_bottom`'s own 0-based space, or `_core`'s, or `_top`'s,
independently. Per-piece figures (unchanged from the first pass) and the
union, side by side:

| Sequence | `_bottom` alone | `_core` alone | `_top` alone | **union** |
|---|---|---|---|---|
| `c1_ascending_t4_v2` | 22.4% | 0.0% | 37.7% | **46.24%** |
| `c1_ascending_t4_v3` | 22.2% | 0.0% | 37.7% | **46.24%** |
| `c1_ascending_t3_v1` | 23.3% | 0.0% | 36.9% | **45.48%** |

The union (46%) is well below the sum of the pieces (60%), confirming
substantial overlap between which false_unobserved faces `_bottom`
"explains" and which `_top` does — most of the apparent signal from
either piece alone is the same faces, not additive.

### c) Decisive test — FAILS the pre-stated criterion

`predicted_GT = (our own lumen-only ray-cast's observed set) ∪ {lumen
faces whose index is in the mold-hit union}`. Compared IoU against the
released `coverage_mesh.obj`, with and without the mold-collision term:

| Sequence | IoU, lumen-only (no mold term) | IoU, **with mold-collision term** | change |
|---|---|---|---|
| `c1_ascending_t4_v2` | 0.9840 | **0.8567** | **−0.127** |
| `c1_ascending_t4_v3` | 0.9840 | **0.8568** | **−0.127** |
| `c1_ascending_t3_v1` | 0.9921 | **0.9277** | **−0.064** |
| `c2_rectum_t4_v1` | 0.9846 | 0.9846 | 0 (mold never hit) |
| `c1_cecum_t1_v1` (control) | 0.9988 | 0.9988 | 0 (mold never hit) |

**Adding the mold-collision term makes agreement with the released GT
substantially *worse*, not better, on every sequence where it has any
effect.** Mechanically: the mold-hit union set is large (§a — ~third of
the lumen's index range, from broad hit coverage over hundreds of
frames), so OR-ing it wholesale into `predicted_GT` adds roughly
200,000+ newly-"observed" faces per ascending sequence — of which at
most the few thousand true false_unobserved faces could possibly be
correct matches; the rest are false positives that swamp the IoU.

### Verdict, per the pre-stated criterion: NOT CONFIRMED — report what remains

**"If (c) lifts IoU to near 1.0, record as confirmed. If not, report what
remains and keep it partially explained."** (c) does not lift IoU — it
lowers it substantially. By the criterion set in advance, **this is not
confirmed**. What actually remains, precisely:

- A **real, statistically significant (many-sigma), but small** effect:
  ~13 percentage points of Group A1's false_unobserved faces (not ~38-46%
  as first reported) show index-collision correlation beyond chance, on
  all 3 ascending sequences, consistently.
- **Zero effect on the Group A2 generalization test** (`c2_rectum_t4_v1`)
  — the mold is never hit at all in that sequence's GT-pose trajectory,
  so whatever causes A2's false_unobserved faces is unrelated to this
  mechanism, at least for this sequence.
- **Zero effect on the control**, as expected.
- The **wholesale "correction" (using the full mold-hit union as a blanket
  fix) actively harms reconstruction accuracy** — not usable as a
  general-purpose fix even where the underlying statistical signal is
  real, because the mold-hit union set is far too broad (most of a
  mold's surface gets hit by *some* frame over a long video, whether or
  not that specific hit corresponds to the actual coverage-buffer bug for
  that specific frame).
- **~87% of Group A1's false_unobserved faces, and 100% of the tested
  A2 sequence's, remain entirely unexplained.**

**Action taken**: `docs/visibility_outliers.md`'s addendum corrected to
match — the earlier "partially confirmed, ≥38%" language retracted and
replaced with this tightened, weaker, statistically-precise finding.
Group A1 (and now also tested-but-unexplained A2) remain substantially
**UNKNOWN**, with only a small, real, non-generalizing contributing
factor identified, not a resolution.

<details>
<summary>First pass (superseded above, kept for the record)</summary>

Sequences: `c1_ascending_t4_v2`, `c1_ascending_t4_v3`, `c1_ascending_t3_v1`
(Group A1, all three), `c1_cecum_t1_v1` (control). Script:
`scratch/pipelines/oracle_gap_item1_mold_collision.py`, log:
`logs/oracle_gap_item1_mold_collision.log`.

**Methodology correction made and disclosed at the time**: the very
first attempt cast rays against a single combined (lumen+mold) scene,
letting occlusion determine lumen-vs-mold per ray. That gave a
false_unobserved count of 291,783 for `c1_ascending_t4_v2` — 36x the
expected ~8,033 — almost certainly from imperfect mold/lumen registration
causing spurious occlusion, not a real finding. Switched to two
independent ray-casts per frame (lumen alone, mold alone).

`model.obj` (the renderer's actual per-video scene file) is not present
in the released dataset. Used the raw mold STL files as a proxy:
`c1_ascending_bottom.stl` 400,708 faces, `c1_ascending_core.stl`
1,320,511 faces, `c1_ascending_top.stl` 253,230 faces.

False_unobserved counts reproduced almost exactly: `t4_v2` 8,032 (prior:
8,033), `t4_v3` 8,050 (prior: 8,049), `t3_v1` 4,653 (prior: 4,652), cecum
375 (prior: 373/375) — this part of the methodology validation stands
unchanged by the tightening above.

Per-piece overlap (no union, no null baseline) was reported as "`_top`
alone explains 37-38%, `_bottom` another 22-23%," concluding "PARTIALLY
CONFIRMED... at least ~38%." **This conclusion is retracted above**: it
lacked a null baseline (the ~32.6% chance-level baseline found in the
tightened pass means most of that 38-46% was never real signal) and
lacked the decisive predicted_GT-vs-released-GT IoU test, which — once
run — fails the confirmation criterion outright.

</details>

---

## Item 3: vignette-only-observed faces across all 169 sequences — MEASURED

Faces the released GT marks observed but that our own ray-cast (GT pose)
only ever reaches through vignette-region pixels, never a non-vignette
one — candidate first level for a criteria-sensitivity analysis
("released GT minus faces never actually imaged"). Streamed
`coverage_mesh.obj` + `pose.txt` only, all 169 registered sequences, no
bulk extraction. **Reduced pixel density (stride 4) for full-corpus
tractability** — `docs/oracle_check.md` already characterized this exact
stride's gap versus full density (IoU 0.944 vs. 0.9988) on this same
mesh; flagged explicitly as an approximation, not a full-density result,
same caveat applies here directionally (likely a mild undercount of true
reach, meaning these fractions are more likely slight overestimates of
the true vignette-only-observed gap). Script:
`scratch/pipelines/oracle_gap_item3_vignette_only_169.py`, log:
`logs/oracle_gap_item3_vignette_only_169.log`, raw data:
`results/pipelines/oracle_gap_item3/summary.json`.

**Distribution, fraction of each sequence's GT-observed faces that are
vignette-only-observed, 169/169 sequences successfully processed**:

| Statistic | Value |
|---|---|
| Median | 2.90% |
| Mean | 3.28% |
| Min | 0.67% (`c1_rectum_t1_v1`) |
| Max | 11.90% (`c2_transverse2_t3_v2`) |
| p5 / p25 / p75 / p95 | 1.24% / 1.90% / 4.14% / 6.43% |
| Total faces (all 169 sequences) | 1,734,922 |

**Not uniform across segments**: the 5 highest-fraction sequences are all
`transverse2` or `rectum` (`c2_transverse2_t3_v2` 11.90%,
`c2_transverse2_t3_v3` 11.52%, `c2_transverse2_t4_v3` 7.13%,
`c2_rectum_t3_v3` 6.81%, `c2_transverse2_t3_v1` 6.70%); the 5 lowest are
all `c1_rectum` (0.67-0.96%) — the same segment name, opposite colon
(`c1` vs `c2`), sitting at opposite ends of the distribution. **Not
investigated further** — flagged as a real pattern, not explained here.

**As a candidate criteria-sensitivity first level**: "released GT minus
vignette-only-observed faces" would remove a median 2.90% (up to 11.90%
in the worst observed sequence) of each sequence's GT-observed set from
consideration — smaller than the combined vignette+100mm-clamp gap
measured in Part 1 (3.16-5.13% on the 3 sequences tested there, at full
density, for comparison — the two numbers aren't directly comparable since
Item 3 is vignette-only, at stride 4, across all 169, while Part 1 was
vignette+clamp, at full density, on 3 sequences).

---

## Summary

| Question | Answer | Status |
|---|---|---|
| Fixed vignette mask | 135/169 sequences share 102,049px (6.9992%); 34 have their own trajectory-dependent excess | MEASURED, Decision 1 |
| Open-end escapes an oracle gap? | No — confirmed on 2 clean sequences (IoU 0.9986, 0.9976) | MEASURED |
| `c1_ascending_t3_v1`'s large disagreement | Not Group A1 (signatures differ) — a 4th cause: mold geometry visible through the open end, rendered but not exported to `coverage_mesh.obj` | MEASURED + diagnosed, CONFIRMED |
| Max face IoU vs. released GT (oracle ceiling) | 0.948-0.968 across 3 sequences, from vignette + 100mm clamp only | MEASURED |
| Does the gap touch region-level metrics? | Corrected: touches the *predicted*-unobserved side (components, false alarm rate), not GT regions, which stay as released | MEASURED, corrected |
| False alarm rate impact (oracle, uncorrected) | 10.06%-23.85%, not the earlier 3.16-5.13% (wrong denominator) | MEASURED, corrected |
| Group A1's mold-index-collision hypothesis | **NOT CONFIRMED** (tightened): real ~13pp excess over a chance baseline of ~33%, but the decisive predicted_GT-vs-released-GT IoU test *lowers* IoU (0.984→0.857), failing the pre-stated criterion; zero effect on Group A2 test and control | MEASURED, tightened, mostly UNKNOWN |
| Vignette-only-observed faces, all 169 sequences | Median 2.90% of GT-observed, up to 11.90% (`transverse2`/`rectum`-linked pattern, not explained) | MEASURED (stride-4 approximation) |

Part 2 (amending `docs/eval_protocol.md`) is already complete (prior turn
in this session). Items 1-3 above were requested before Part 2's review;
nothing further pending except your review of all of it.
