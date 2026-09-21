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

### Region-level impact — MEASURED, touches the primary analysis unit, unevenly

For each sequence: built the real >5mm GT-unobserved regions (face
adjacency, `src/geometry/mesh_stats.py`, matching `docs/success_criteria.md`
§1's region definition exactly), then rebuilt regions after reclassifying
every gap face (GT-observed but non-evaluable) as unobserved too, and
compared.

| Sequence | original >5mm regions | modified >5mm regions | regions whose face-set changed | of those, merged with another region's faces |
|---|---|---|---|---|
| `c1_cecum_t1_v1` | 3 | 3 | 3/3 | 3 |
| `c1_ascending_t3_v1` | 4 | **5** | 4/4 | 4 |
| `c2_rectum_t1_v1` | 4 | **2** | 4/4 | 4 |

**Every single tracked region changed in every sequence** — the gap is
not negligible at the region level. The *kind* of change varies:
`c1_cecum_t1_v1` grows all 3 regions without changing the count;
`c1_ascending_t3_v1` gains a **new** >5mm region (gap faces forming their
own component large enough to cross the threshold on their own — a
region that exists only because of the gap, not because of any real
missed area); `c2_rectum_t1_v1` **merges** 4 regions down to 2 (gap faces
bridge previously-separate regions into fewer, larger ones). "Merged"
here means a tracked region's face-set gained faces beyond the gap set
itself (i.e. absorbed area from what was a separate unobserved
component, tracked or not) — a real structural change, not just growth
from newly-reclassified gap faces alone.

**This directly answers the original question**: yes, the gap touches
`docs/success_criteria.md`'s primary analysis unit (regions), and it does
so differently per sequence — sometimes only growing existing regions,
sometimes creating a new one, sometimes merging several into fewer. Not
uniform, and not ignorable.

---

## Summary

| Question | Answer | Status |
|---|---|---|
| Fixed vignette mask | 135/169 sequences share 102,049px (6.9992%); 34 have their own trajectory-dependent excess | MEASURED, Decision 1 |
| Open-end escapes an oracle gap? | No — confirmed on 2 clean sequences (IoU 0.9986, 0.9976) | MEASURED |
| `c1_ascending_t3_v1`'s large disagreement | Not Group A1 (signatures differ) — a 4th cause: mold geometry visible through the open end, rendered but not exported to `coverage_mesh.obj` | MEASURED + diagnosed, CONFIRMED |
| Max face IoU vs. released GT (oracle ceiling) | 0.948-0.968 across 3 sequences, from vignette + 100mm clamp only | MEASURED |
| Does the gap touch region-level metrics? | Yes, on every tracked region, unevenly (growth / new spurious region / merges) | MEASURED |

Part 2 (amending `docs/eval_protocol.md`) follows in this same session.
