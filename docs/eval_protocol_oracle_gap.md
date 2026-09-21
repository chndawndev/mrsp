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

## Interpretation (Part B) — connects to a known, already-unresolved issue

**`c1_ascending_t3_v1` is one of the three sequences in `docs/
visibility_outliers.md`'s "Group A1"** — a discrepancy discovered and
investigated at length during this project's original GPU-rasterizer
validation work, months before this task, and explicitly **logged there
as UNKNOWN**: "dense salt-and-pepper... speckle running exactly along the
boundary contour," footprint *larger* than a control sample (opposite of
the grazing-angle/sub-pixel hypothesis that explained a different flagged
group), distance/incidence/rotation all unremarkable, supersampling
doesn't move it. The leading hypothesis there — "a pixel-level (or
half-pixel) difference in exactly where our rasterizer's ray grid crosses
the true silhouette edge" — was never confirmed or refuted.

This session's finding (a large, silhouette-adjacent ray-cast disagreement
specifically on `c1_ascending_t3_v1`, absent on `c2_rectum_t1_v1`) is
very likely **the same phenomenon**, not a new bug introduced by this
task's own camera/ray-casting code — that code just reproduced
`c2_rectum_t1_v1` almost perfectly, using the identical method. **Not
independently re-confirmed against A1's specific proposed mechanism** —
this is a strong circumstantial match (same sequence, same general
silhouette-boundary character), not a re-derivation of the original
boundary-alignment hypothesis.

**Practical consequence**: `c1_ascending_t3_v1` is a poor, contaminated
choice for validating the open-end-escape hypothesis specifically — its
disagreement is dominated by an unrelated, already-known, already-
unresolved issue, not by open-end behavior. `c2_rectum_t1_v1` alone
provides a clean, direct confirmation.

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

## Where this leaves Part 1 steps 2-4 and Part 2 — not resumed

Two open questions, your call on both:

1. **Part A (169-sequence vignette)**: use the 135-sequence majority mask
   (102,049px) as the single global vignette, treating each of the 34
   outlier sequences' excess as a separate, per-sequence phenomenon (same
   spirit as resolution option 1 from the original blocker) — or
   something else?
2. **Part B (open-end validation)**: is `c2_rectum_t1_v1`'s clean 0.9986
   IoU sufficient confirmation on its own (with `c1_ascending_t3_v1`
   excluded as contaminated by the known A1 issue), or do you want a
   second, uncontaminated open-end-visible sequence tested before trusting
   "open-end escape = no oracle gap, handled by section 3" as a general
   protocol statement?

Not proceeding to Part 1 steps 2-4, Part 2, or Part 2's evaluable-pixel
definition until you weigh in on both. Part 3 (approval-record fix) was
completed independently in a prior turn and stands.
