# Viewer data export, Stage 1: `c1_cecum_t1_v1`

Exports everything a 3D results viewer needs for one sequence to
`results/viewer/c1_cecum_t1_v1/`, and proves the export is faithful by
recomputing every metric in `results/eval_stage4/summary.json` from the
exported files alone, in a fresh process. No viewer code written this
stage. `src/eval/`, `src/gt/`, `docs/success_criteria.md` untouched.

Script: `scripts/export_viewer_data.py --export` (GPU, then)
`scripts/export_viewer_data.py --verify` (CPU, fresh process). Log:
`logs/export_viewer_data.log`. GPU selection followed
`scripts/eval_stage4.py`'s convention exactly: index 7 (RTX 6000 Ada, 32.1
GB free, 0% utilized), single `CUDA_VISIBLE_DEVICES`. Total export runtime
4m48s (streamed mesh+poses 3.2s, GT depth load 6.0s, pose variant GT
52.9s, pose variant pred 48.9s, oracle.py cross-check 44.0s, metrics+JSON
writing ~4s).

---

## MEASURED

### What was exported

All under `results/viewer/c1_cecum_t1_v1/`:

| file | dtype | shape | size |
|---|---|---|---|
| `vertices_f32.bin` | float32 | (350595, 3) | 4.0M |
| `vertices_f64.bin` | float64 | (350595, 3) | 8.0M |
| `faces_i32.bin` | int32 | (699908, 3) | 8.0M |
| `gt_observed_u8.bin` | uint8 (0/1) | (699908,) | 684K |
| `ignore_set_u8.bin` | uint8 (0/1) | (699908,) | 684K |
| `face_area_mm2_f32.bin` | float32 | (699908,) | 2.7M |
| `face_area_mm2_f64.bin` | float64 | (699908,) | 5.4M |
| `gt_region_id_i32.bin` | int32 | (699908,) | 2.7M |
| `predicted_observed_packed.bin` | uint8, packed bits | (4 config, 4 tau, 87489 bytes) | 1.4M |
| `regions.json` | -- | 416 regions | 136K |
| `metrics.json` | -- | 4 config × 4 tau | 52K |
| `trajectory.json` | -- | 218 frames × 2 cameras | 96K |
| `manifest.json` | -- | file index + hashes | 4.0K |
| `verification.json` | -- | check A/B/C results | 32K |

**Format**: raw little-endian `.bin` arrays (a browser viewer can load
these directly as typed arrays) plus JSON for structured/variable-size
data, all indexed by `manifest.json` (path, dtype, shape, byte size,
sha256 per file; git commit `485d52a`, clean tree at export time; tau
list; config name list; `predicted_observed_packed`'s axis order
`[config, tau, byte]` and bit order `little`).

**Dual precision (flagged in planning, not in the original spec)**:
vertices and face areas are exported as both float32 and float64. The
mesh's own vertex coordinates are not exactly representable in float32 --
recomputing face areas from the float32-cast vertices differs from the
float64 areas by up to **1.055e-05 mm²** (measured, `check_A.area_recompute_max_diff_f32`
in `verification.json`), which would have exceeded any strict recomputation
tolerance. The float64 copies exist specifically so check A can hold
metrics to 1e-9; the float32 copies are for the viewer's rendering path,
where this precision loss doesn't matter.

**Region-id mapping (flagged in planning)**: the task spec's check C names
"GT region 0." In `compute_regions`' actual output order, id 0 is the
416-region list's large region (58,304 faces) -- not the 1,478-face region
the small-region diagnosis (`results/stage4_small_region_diagnosis/summary.json`)
calls "small region 0." Verified by exact face-set match: the diagnosis's
1,478 face ids are identical to `compute_regions`' region **id 45**. Check
C below uses id 45, and `verification.json`'s `check_C.region_id_mapping_ok`
records the match.

### Verification A -- recompute every metric from the export, fresh process

`--verify` loads only the exported files (plus the reference
`summary.json` and diagnosis `summary.json`, read for the check-C
face-set match) and recomputes every `metrics.json` field via the same
locked `src/eval` functions (`region_metrics.py`, `regions.py`), on the
float64 arrays.

**Result: worst diff across every metric, every configuration, every tau
= 0.0 (bit-identical). PASS** (`verification.json`: `check_A.worst_diff =
0.0`, `check_A.pass = true`). This covers: `area_fraction`,
`calibration_ratio`, `false_reassurance_rate`, `false_alarm_rate`,
`n_predicted_unobserved_faces`, `localization_error_median_mm`,
`localization_error_mean_mm`, `segment_intersect_fraction`,
`n_headline_regions_undetected`, region recall at 50% and the full
25/50/75% detection sweep, all by size class -- 4 configs × 4 taus each.
`ignore_set_frac_faces` diff = 0.0. Recomputed GT region ids exactly match
the exported `gt_region_id` array and order.

Informational only, not part of the pass/fail: recomputing face areas
from the float32 vertex copy gives a max diff of 1.055e-05 mm² against the
float64 areas (see "Dual precision" above) -- not run through the full
metric pipeline, since check A's tolerance (1e-9) was defined against the
float64 export.

**Not independently recomputable from the exported per-face arrays**
(flagged, not silently treated as verified): `ray_miss_frac`,
`evaluable_frac`, `d_pred_unavailable_count`, `depth_scale_median`,
`pose_alignment_s_pose`. These are ray-level counters and scale factors
that require rerunning the GPU ray-cast to rederive; the exported
`metrics.json` values are copied from the same in-memory result the
`--export` process also wrote `summary.json`'s sibling from, so this is a
copy-identity check (diff 0.0 for all), not an independent verification.
The per-face metrics recomputed above (area fraction, false alarm rate,
etc.) *do* independently confirm the underlying `predicted_observed`
arrays are correct, since those are derived from the same ray-cast output
these counters come from.

### Verification B -- sanity counts

| check | expected | measured | pass |
|---|---|---|---|
| `n_faces` | 699908 | 699908 | yes |
| `ignore_set` count | 20094 | 20094 | yes |
| headline region count | 3 (2 small, 0 medium, 1 large) | 3 (2 small, 0 medium, 1 large) | yes |

### Verification C -- diagnostic count, region 45, tau=0.25

Region 45's 1,478 faces, `fully_predicted` vs. `pred_pose_only`, tau=0.25,
raw (unfiltered) predicted-unobserved sets:

- Faces predicted-unobserved under `fully_predicted` but not under
  `pred_pose_only`: **73** (matches the expected count; this is the same
  73 already reported in `docs/eval_stage4.md`'s appendix for this region
  and tau).
- Of those 73, also predicted-unobserved under `pred_depth_only`: **72**.
- Not predicted-unobserved under `pred_depth_only` (i.e. `pred_depth_only`
  marks it observed): **1**.

---

## INTERPRETATION

- **72/73 overlap with `pred_depth_only` is consistent with the
  already-reported mechanism** (`docs/eval_stage4.md` appendix item (a)):
  the extra faces `fully_predicted` marks unobserved beyond
  `pred_pose_only`, in this region, are overwhelmingly the same faces
  `pred_depth_only`'s predicted-depth-driven over-flagging behavior
  independently marks unobserved. This is consistent with, not
  independent proof of, that mechanism -- it would be confirmed by
  checking whether the same near-total overlap (72/73 here) holds for
  other regions and other tau values where `fully_predicted` similarly
  exceeds `pred_pose_only`, which this export does not do (region 45 /
  tau=0.25 only, as specified). It would be weakened if other
  region/tau combinations showed a much lower overlap fraction.
- **The single non-overlapping face** is not traced further here (out of
  scope for this export's acceptance test); it would be confirmed or
  explained by inspecting that one face's per-frame ray-cast history
  under both depth sources.
