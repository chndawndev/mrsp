# Oracle ceiling against the released GT — Part 1, STOPPED at step 1

Requested as Part 1 of a 3-part task amending `docs/eval_protocol.md`. Do
them in order; **step 1 failed its own stated stop condition**, so steps
2-4 were not run. This document reports exactly what was measured, no
further. Sequences, script, and log below.

**Sequences used**: `c1_cecum_t1_v1` (already extracted, used throughout
this project's EndoDAC work) plus 2 sequences with `Open End Visible =
yes` from different segments, per `docs/release_v1.csv`:
`c1_ascending_t3_v1` (ascending, colon 1) and `c2_rectum_t1_v1` (rectum,
colon 2) — extracted into `scratch/` with your approval this session.
Script: `scratch/pipelines/oracle_gap_part1.py`, log:
`logs/oracle_gap_part1_mask_check.log`.

---

## Step 1: fixed camera valid-pixel mask — MEASURED

Built from `c1_cecum_t1_v1` frame `0000`'s GT depth `raw == 0` region:
**102,049 pixels (6.9992% of 1,458,000)**, used as the reference mask.

Checked pixel-for-pixel identity of this mask against every one of the
`raw == 0` regions in all 631 frames across the 3 sequences (218 + 259 +
154).

**Result: NOT identical. Per instructions, stopping here — steps 2-4 not
run.**

- **`c1_cecum_t1_v1`'s own 218 frames: all match the reference exactly**,
  zero mismatches. The mask *is* fixed within this sequence.
- **`c1_ascending_t3_v1`: mismatches in frames 140-170** (31 of 259
  frames). Extra `raw==0` pixel count rises from the 102,049 baseline to
  a peak of 103,424 at frame 152 (+1,375 pixels, +1.35% relative to
  baseline), then falls back to exactly the baseline by frame 171. Zero
  mismatches outside this range.
- **`c2_rectum_t1_v1`: mismatches in frames 14-78** (65 of 154 frames).
  Extra pixel count rises from baseline to a peak of 119,197 at frame 41
  (+17,148 pixels, +16.80% relative to baseline), then falls back to
  exactly the baseline by frame 78. Zero mismatches outside this range.

**The pattern in both cases is a clean rise-then-fall bulge over a
contiguous frame range, returning to exactly the reference count at both
ends** — not a constant offset, not noise, not scattered mismatches
throughout the sequence. Full per-frame counts in
`logs/oracle_gap_part1_mask_check.log`.

## Interpretation (not measured, flagged as such)

Visually inspected the RGB frame at each bulge's peak
(`scratch/c1_ascending_t3_v1/rgb/0152.png`,
`scratch/c2_rectum_t1_v1/rgb/0041.png`): both show a dark, recessed
opening in the field of view, consistent with looking toward or through
an open lumen end. Both sequences were deliberately selected for `Open
End Visible = yes`.

**Leading interpretation**: `raw == 0` in the depth channel conflates two
physically different things — (1) the fixed hardware vignette (constant
across every frame, the ~7.0% baseline, confirmed stable in
`c1_cecum_t1_v1`) and (2) genuine, scene-dependent ray-misses where a ray
exits through an open colon end into space the renderer's mesh doesn't
model, which only occurs while an open end is actually in view. The
bulge's shape — rising as the open end comes more centrally into frame,
peaking, then falling as the camera moves past it — matches this
explanation. **Not independently confirmed** beyond the two visual spot
checks; I have not, for example, traced individual extra-zero pixels back
to specific rays known to exit through the modeled mesh boundary.

## Why this blocks the "fixed camera valid-pixel mask" premise

Part 1's own framing assumes a single mask, constant across all
sequences, built once from `raw == 0`. That premise holds for
`c1_cecum_t1_v1` but measurably does not hold for either open-end-visible
sequence — meaning `raw == 0` is not a safe proxy for "hardware vignette"
in general, and using it naively would silently fold scene-dependent
open-end misses into what's supposed to be a fixed camera property,
inflating the apparent vignette specifically on open-end-visible
sequences and nowhere else.

## Proposed resolutions — none chosen, your call

1. **Use `c1_cecum_t1_v1` (or any confirmed-clean sequence/frame range) as
   the sole reference for the fixed mask**, and apply that single mask
   uniformly to all sequences including the two open-end-visible ones —
   treating the bulge pixels in those sequences as a separate,
   *additional*, scene-dependent phenomenon, not part of the fixed mask.
   This seems most consistent with what "fixed camera valid-pixel mask"
   was originally asking for, and lets steps 2-4 proceed using a mask
   that's actually constant.
2. **Extend Part 1's 2-cause model (vignette, 100mm clamp) to 3 causes**,
   adding open-end ray-misses as its own category, measured per-sequence
   rather than assumed fixed — more work, but doesn't silently discard
   what the bulge sequences are actually showing, which is itself germane
   to the "open end" regions this project already cares about
   (`docs/openend_artifact_check.md`).
3. **Drop the two open-end-visible sequences and use 3 non-open-end
   sequences instead**, avoiding the bulge phenomenon entirely — but this
   deviates from the original sequence-selection instruction (open-end
   sequences specifically, to stress-test the depth-clamp/vignette gap
   near an open end) and may just relocate the same issue to whichever
   sequences get picked next, unverified.

Not proceeding to Part 1 steps 2-4, Part 2, or re-checking Part 3 against
these numbers until you weigh in. Part 3 (the approval-record fix) was
independent of this blocker and has already been completed in
`docs/eval_protocol.md` §6-7.
