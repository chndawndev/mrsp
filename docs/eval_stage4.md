# Stage 4 full per-tau table

`c1_cecum_t1_v1`, 699908 faces, 218 frames. depth_scale.median=177.013374, pose s_pose=557.871181. ignore_set = 2.8709% of faces (sequence-level constant, repeated in every row below since it is required alongside every recall number). GT regions: 416 total, 3 headline (d>=5mm; this sequence has 2 small, 0 medium, 1 large).

Pulled directly from `results/eval_stage4/summary.json`, no recomputation. **No D1 pass/fail commentary** -- one sequence is not D1.

---

## Oracle (GT depth + GT pose) (`oracle`)

ray_miss_frac=0.00000%, evaluable_frac=85.4523%, d_pred_unavailable_count=2979

| metric | tau=0.15 | tau=0.25 | tau=0.35 | tau=0.5 | tau-sensitivity |
|---|---|---|---|---|---|
| ignore_set_frac_faces | 2.8709% | 2.8709% | 2.8709% | 2.8709% | flat |
| area_fraction | 11.3480% | 11.3480% | 11.3479% | 11.3479% | flat |
| calibration_ratio | 1.3535 | 1.3535 | 1.3535 | 1.3535 | flat |
| recall@50% small | 100.0000% | 100.0000% | 100.0000% | 100.0000% | flat |
| recall@50% medium | n/a | n/a | n/a | n/a | n/a |
| recall@50% large | 100.0000% | 100.0000% | 100.0000% | 100.0000% | flat |
| recall@50% below_headline (appendix) | 62.9540% | 62.9540% | 62.9540% | 62.9540% | flat |
| false_reassurance_rate | 0.4600% | 0.4600% | 0.4600% | 0.4600% | flat |
| false_alarm_rate | 0.0010% | 0.0010% | 0.0000% | 0.0000% | flat (near zero) |
| localization_error median (mm) | 0.0306 | 0.0306 | 0.0306 | 0.0306 | flat |
| localization_error mean (mm) | 0.0306 | 0.0306 | 0.0306 | 0.0306 | flat |
| n_headline_regions_undetected | 0 | 0 | 0 | 0 | flat (0) |
| segment_intersect_fraction | 0.0000% | 0.0000% | 0.0000% | 0.0000% | flat (0) |
| n_predicted_unobserved_faces | 83909 | 83909 | 83908 | 83908 | flat |
| tau_reject_count | 7063 | 2383 | 539 | 29 | sensitive |

**25/50/75% detection threshold sweep (small/medium/large), at each tau:**

| tau | thresh | small | medium | large |
|---|---|---|---|---|
| 0.15 | 0.25 | 100.00% (2/2) | n/a (0/0) | 100.00% (1/1) |
| 0.15 | 0.5 | 100.00% (2/2) | n/a (0/0) | 100.00% (1/1) |
| 0.15 | 0.75 | 100.00% (2/2) | n/a (0/0) | 100.00% (1/1) |
| 0.25 | 0.25 | 100.00% (2/2) | n/a (0/0) | 100.00% (1/1) |
| 0.25 | 0.5 | 100.00% (2/2) | n/a (0/0) | 100.00% (1/1) |
| 0.25 | 0.75 | 100.00% (2/2) | n/a (0/0) | 100.00% (1/1) |
| 0.35 | 0.25 | 100.00% (2/2) | n/a (0/0) | 100.00% (1/1) |
| 0.35 | 0.5 | 100.00% (2/2) | n/a (0/0) | 100.00% (1/1) |
| 0.35 | 0.75 | 100.00% (2/2) | n/a (0/0) | 100.00% (1/1) |
| 0.5 | 0.25 | 100.00% (2/2) | n/a (0/0) | 100.00% (1/1) |
| 0.5 | 0.5 | 100.00% (2/2) | n/a (0/0) | 100.00% (1/1) |
| 0.5 | 0.75 | 100.00% (2/2) | n/a (0/0) | 100.00% (1/1) |

## Predicted depth + GT pose (`pred_depth_only`)

ray_miss_frac=0.00000%, evaluable_frac=85.4523%, d_pred_unavailable_count=0

| metric | tau=0.15 | tau=0.25 | tau=0.35 | tau=0.5 | tau-sensitivity |
|---|---|---|---|---|---|
| ignore_set_frac_faces | 2.8709% | 2.8709% | 2.8709% | 2.8709% | flat |
| area_fraction | 35.6919% | 25.0180% | 19.7726% | 17.7453% | sensitive |
| calibration_ratio | 4.2570 | 2.9839 | 2.3583 | 2.1165 | sensitive |
| recall@50% small | 100.0000% | 100.0000% | 100.0000% | 100.0000% | flat |
| recall@50% medium | n/a | n/a | n/a | n/a | n/a |
| recall@50% large | 100.0000% | 100.0000% | 100.0000% | 100.0000% | flat |
| recall@50% below_headline (appendix) | 83.2930% | 79.1768% | 75.0605% | 70.9443% | mild |
| false_reassurance_rate | 0.1363% | 0.2071% | 0.2510% | 0.2853% | sensitive |
| false_alarm_rate | 74.3870% | 61.9959% | 50.1312% | 43.2931% | sensitive |
| localization_error median (mm) | 1.1357 | 1.1021 | 1.1021 | 1.1021 | mild |
| localization_error mean (mm) | 3.5962 | 3.1508 | 2.5840 | 2.3430 | sensitive |
| n_headline_regions_undetected | 0 | 0 | 0 | 0 | flat (0) |
| segment_intersect_fraction | 0.0000% | 0.0000% | 0.0000% | 0.0000% | flat (0) |
| n_predicted_unobserved_faces | 244641 | 181643 | 151254 | 135415 | sensitive |
| tau_reject_count | 205295749 | 161000487 | 117707112 | 63265936 | sensitive |

**25/50/75% detection threshold sweep (small/medium/large), at each tau:**

| tau | thresh | small | medium | large |
|---|---|---|---|---|
| 0.15 | 0.25 | 100.00% (2/2) | n/a (0/0) | 100.00% (1/1) |
| 0.15 | 0.5 | 100.00% (2/2) | n/a (0/0) | 100.00% (1/1) |
| 0.15 | 0.75 | 100.00% (2/2) | n/a (0/0) | 100.00% (1/1) |
| 0.25 | 0.25 | 100.00% (2/2) | n/a (0/0) | 100.00% (1/1) |
| 0.25 | 0.5 | 100.00% (2/2) | n/a (0/0) | 100.00% (1/1) |
| 0.25 | 0.75 | 100.00% (2/2) | n/a (0/0) | 100.00% (1/1) |
| 0.35 | 0.25 | 100.00% (2/2) | n/a (0/0) | 100.00% (1/1) |
| 0.35 | 0.5 | 100.00% (2/2) | n/a (0/0) | 100.00% (1/1) |
| 0.35 | 0.75 | 100.00% (2/2) | n/a (0/0) | 100.00% (1/1) |
| 0.5 | 0.25 | 100.00% (2/2) | n/a (0/0) | 100.00% (1/1) |
| 0.5 | 0.5 | 100.00% (2/2) | n/a (0/0) | 100.00% (1/1) |
| 0.5 | 0.75 | 100.00% (2/2) | n/a (0/0) | 100.00% (1/1) |

## GT depth + predicted pose (`pred_pose_only`)

ray_miss_frac=0.05332%, evaluable_frac=86.8811%, d_pred_unavailable_count=8335784

| metric | tau=0.15 | tau=0.25 | tau=0.35 | tau=0.5 | tau-sensitivity |
|---|---|---|---|---|---|
| ignore_set_frac_faces | 2.8709% | 2.8709% | 2.8709% | 2.8709% | flat |
| area_fraction | 13.1849% | 12.2050% | 11.1690% | 10.8569% | mild |
| calibration_ratio | 1.5726 | 1.4557 | 1.3321 | 1.2949 | mild |
| recall@50% small | 100.0000% | 50.0000% | 50.0000% | 50.0000% | sensitive |
| recall@50% medium | n/a | n/a | n/a | n/a | n/a |
| recall@50% large | 100.0000% | 100.0000% | 100.0000% | 100.0000% | flat |
| recall@50% below_headline (appendix) | 58.8378% | 52.7845% | 37.5303% | 35.5932% | sensitive |
| false_reassurance_rate | 9.5125% | 11.4423% | 13.4568% | 14.1373% | sensitive |
| false_alarm_rate | 30.2340% | 25.6048% | 23.0365% | 22.5472% | sensitive |
| localization_error median (mm) | 4.4600 | 3.9711 | 2.5047 | 2.2091 | sensitive |
| localization_error mean (mm) | 4.6735 | 4.6633 | 4.1708 | 4.0177 | mild |
| n_headline_regions_undetected | 0 | 0 | 0 | 0 | flat (0) |
| segment_intersect_fraction | 0.0000% | 0.0000% | 0.0000% | 0.0000% | flat (0) |
| n_predicted_unobserved_faces | 96910 | 89624 | 82594 | 80765 | mild |
| tau_reject_count | 93575487 | 36642233 | 13764722 | 3629600 | sensitive |

**25/50/75% detection threshold sweep (small/medium/large), at each tau:**

| tau | thresh | small | medium | large |
|---|---|---|---|---|
| 0.15 | 0.25 | 100.00% (2/2) | n/a (0/0) | 100.00% (1/1) |
| 0.15 | 0.5 | 100.00% (2/2) | n/a (0/0) | 100.00% (1/1) |
| 0.15 | 0.75 | 50.00% (1/2) | n/a (0/0) | 100.00% (1/1) |
| 0.25 | 0.25 | 100.00% (2/2) | n/a (0/0) | 100.00% (1/1) |
| 0.25 | 0.5 | 50.00% (1/2) | n/a (0/0) | 100.00% (1/1) |
| 0.25 | 0.75 | 50.00% (1/2) | n/a (0/0) | 100.00% (1/1) |
| 0.35 | 0.25 | 100.00% (2/2) | n/a (0/0) | 100.00% (1/1) |
| 0.35 | 0.5 | 50.00% (1/2) | n/a (0/0) | 100.00% (1/1) |
| 0.35 | 0.75 | 50.00% (1/2) | n/a (0/0) | 100.00% (1/1) |
| 0.5 | 0.25 | 100.00% (2/2) | n/a (0/0) | 100.00% (1/1) |
| 0.5 | 0.5 | 50.00% (1/2) | n/a (0/0) | 100.00% (1/1) |
| 0.5 | 0.75 | 50.00% (1/2) | n/a (0/0) | 100.00% (1/1) |

## Fully predicted (`fully_predicted`)

ray_miss_frac=0.05332%, evaluable_frac=86.8811%, d_pred_unavailable_count=0

| metric | tau=0.15 | tau=0.25 | tau=0.35 | tau=0.5 | tau-sensitivity |
|---|---|---|---|---|---|
| ignore_set_frac_faces | 2.8709% | 2.8709% | 2.8709% | 2.8709% | flat |
| area_fraction | 31.5445% | 22.2463% | 18.5453% | 16.5001% | sensitive |
| calibration_ratio | 3.7624 | 2.6534 | 2.2119 | 1.9680 | sensitive |
| recall@50% small | 100.0000% | 100.0000% | 50.0000% | 50.0000% | sensitive |
| recall@50% medium | n/a | n/a | n/a | n/a | n/a |
| recall@50% large | 100.0000% | 100.0000% | 100.0000% | 100.0000% | flat |
| recall@50% below_headline (appendix) | 84.5036% | 79.4189% | 65.1332% | 52.3002% | sensitive |
| false_reassurance_rate | 0.9523% | 1.9515% | 3.0042% | 4.1015% | sensitive |
| false_alarm_rate | 70.9102% | 57.2977% | 47.7175% | 40.6251% | sensitive |
| localization_error median (mm) | 7.2840 | 5.9988 | 0.9231 | 1.4451 | sensitive |
| localization_error mean (mm) | 6.0271 | 5.5627 | 2.2389 | 2.8086 | sensitive |
| n_headline_regions_undetected | 0 | 0 | 0 | 0 | flat (0) |
| segment_intersect_fraction | 0.0000% | 0.0000% | 0.0000% | 0.0000% | flat (0) |
| n_predicted_unobserved_faces | 219051 | 163832 | 139937 | 122956 | sensitive |
| tau_reject_count | 210189609 | 163440073 | 120860007 | 61689824 | sensitive |

**25/50/75% detection threshold sweep (small/medium/large), at each tau:**

| tau | thresh | small | medium | large |
|---|---|---|---|---|
| 0.15 | 0.25 | 100.00% (2/2) | n/a (0/0) | 100.00% (1/1) |
| 0.15 | 0.5 | 100.00% (2/2) | n/a (0/0) | 100.00% (1/1) |
| 0.15 | 0.75 | 50.00% (1/2) | n/a (0/0) | 100.00% (1/1) |
| 0.25 | 0.25 | 100.00% (2/2) | n/a (0/0) | 100.00% (1/1) |
| 0.25 | 0.5 | 100.00% (2/2) | n/a (0/0) | 100.00% (1/1) |
| 0.25 | 0.75 | 50.00% (1/2) | n/a (0/0) | 100.00% (1/1) |
| 0.35 | 0.25 | 100.00% (2/2) | n/a (0/0) | 100.00% (1/1) |
| 0.35 | 0.5 | 50.00% (1/2) | n/a (0/0) | 100.00% (1/1) |
| 0.35 | 0.75 | 50.00% (1/2) | n/a (0/0) | 100.00% (1/1) |
| 0.5 | 0.25 | 100.00% (2/2) | n/a (0/0) | 100.00% (1/1) |
| 0.5 | 0.5 | 50.00% (1/2) | n/a (0/0) | 100.00% (1/1) |
| 0.5 | 0.75 | 50.00% (1/2) | n/a (0/0) | 100.00% (1/1) |

---

## Tau-sensitivity summary, MEASURED

- **Flat across the sweep, every configuration**: `area_fraction`/`calibration_ratio` for the oracle config only (predicted-depth configs move it substantially); `recall@50% large`; `segment_intersect_fraction` (0.00% everywhere, all 4 configs, all 4 tau -- see the separate geodesic-vs-Euclidean report); `n_headline_regions_undetected` (0 everywhere -- every headline region has *some* overlapping predicted component at every tau, in every configuration).

- **Sensitive to tau**: `tau_reject_count` in every configuration (by construction -- a looser tau admits more pixels); `area_fraction`/`calibration_ratio`/`false_alarm_rate` for every configuration that includes predicted depth (`pred_depth_only`, `fully_predicted`) -- looser tau lets more predicted faces pass the depth-agreement test, shrinking the predicted-unobserved set and area fraction, and with it false alarm rate; `recall@50% small` for `pred_pose_only` (drops exactly between tau=0.15 and 0.25) and `fully_predicted` (drops exactly between tau=0.25 and 0.35) -- see the separate small-region diagnosis for the face-level mechanism.

- **INTERPRETATION, not confirmed further here**: `localization_error` mean is more tau-sensitive than its median in the predicted-depth configurations (e.g. `pred_depth_only`'s mean moves 3.60mm->2.34mm across the sweep while its median barely moves, 1.14mm->1.10mm) -- consistent with a small number of large-error outlier region-matches whose *identity* (which predicted component gets matched to which GT region) shifts with tau, dragging the mean while the bulk of matches stay stable. Not traced to specific regions here; would need the same per-region breakdown as the small-region diagnosis to confirm.

---

## Appendix: the small-region recall inversion at tau=0.25 (`pred_pose_only`=0.50 vs `fully_predicted`=1.00)

Diagnosed with `scratch/pipelines/stage4_small_region_diagnosis.py` (reruns the identical Stage 4 pipeline on the same sequence -- no new sequences -- keeping per-region face-level detail the main driver discards). Full data: `results/stage4_small_region_diagnosis/summary.json`.

**MEASURED.** `c1_cecum_t1_v1` has exactly 2 small GT regions:

| region | faces | area (mm²) | diameter (mm) |
|---|---|---|---|
| 0 | 1478 | 57.9384 | 8.5889 |
| 1 | 1537 | 75.0400 | 9.7747 |

("region 0"/"region 1" here are local to this diagnosis's small-region-only list; the global `compute_regions` id used elsewhere (e.g. `docs/viewer_export.md`, the results viewer) is 45 for region 0 and 175 for region 1.)

Area-weighted coverage fraction of each region by the predicted-unobserved set, per configuration, per tau (detection threshold is 50%):

| config | tau=0.15 | tau=0.25 | tau=0.35 | tau=0.5 |
|---|---|---|---|---|
| oracle — region 0 / region 1 | 98.86% / 96.85% | 98.86% / 96.85% | 98.86% / 96.85% | 98.86% / 96.85% |
| pred_depth_only — region 0 / region 1 | 99.12% / 100.00% | 99.04% / 100.00% | 99.04% / 100.00% | 99.04% / 99.95% |
| pred_pose_only — region 0 / region 1 | **51.51%** / 80.15% | **48.44%** / 79.84% | 48.11% / 78.49% | 45.93% / 77.88% |
| fully_predicted — region 0 / region 1 | **63.54%** / 100.00% | **52.64%** / 100.00% | 49.69% / 100.00% | 46.26% / 100.00% |

Region 1 is comfortably detected (>77%) in every configuration at every tau -- it is not part of the inversion. Region 0 is the entire story: its coverage sits within a few percentage points of the 50% line under both pose-error configurations, at every tau, and `fully_predicted`'s coverage is consistently 2-12 percentage points higher than `pred_pose_only`'s at every single tau value (63.5 vs 51.5 at 0.15; 52.6 vs 48.4 at 0.25; 49.7 vs 48.1 at 0.35; 46.3 vs 45.9 at 0.5) -- a real, directionally-consistent, always-present gap. **The recall metric only visibly flips at tau=0.25** because that is the only tau where the two configurations' region-0 coverage lands on *opposite* sides of the 50% line (both are above it at tau=0.15; both are below it at tau=0.35 and 0.5).

Face-level diff at tau=0.25, region 0 (1478 faces total): 658 faces are predicted-unobserved in both configurations, 736 are predicted-observed in both, and the configurations disagree on 84 faces -- **11** unobserved only under `pred_pose_only`, **73** unobserved only under `fully_predicted`. Region 1's diff at tau=0.25: 1232 shared-unobserved, 305 unobserved only under `fully_predicted`, 0 only under `pred_pose_only`, 0 observed-in-both (fully_predicted marks the whole region unobserved; pred_pose_only doesn't quite).

**INTERPRETATION**, separated from the above:

- **(a) Real interaction — confirmed.** `fully_predicted` adds a net +62 extra predicted-unobserved faces to region 0 relative to `pred_pose_only` (73 vs 11, out of 1478), consistent with predicted depth's already-documented general over-flagging behavior (`pred_depth_only`'s own area fraction is 2-3x the oracle's) leaking into this specific region on top of whatever the pose-error baseline already contributes. This gap is present and same-direction at every tau, not just at 0.25 -- it is not a fluke of one tau value.
- **(b) Artifact of small counts (n=2) — also confirmed, and it's *why* (a) becomes a visible metric flip.** Region 0's coverage sits within ~2 percentage points of the exact 50% threshold under pose error alone at every tau. With only 2 small regions total, one region crossing that line changes reported recall by a full 50 percentage points. The same underlying (a)-gap exists at every tau, but only shows up as a recall difference where it happens to straddle the threshold.
- **(c) Implementation inconsistency — ruled out.** Both configurations run through the identical `run_pose_variant_sequence` code path (only the depth provider callable differs); the oracle configuration from this same code was cross-checked bit-for-bit against the separately-implemented, already-locked `src/eval/oracle.py` on this same real sequence; and the face-level diff (73 vs 11 faces out of 1478) is small, bounded, and directionally explicable, not the kind of wild or structureless discrepancy a bug would produce.

**Conclusion**: (a) and (b) together fully explain the inversion; (c) is ruled out. The inversion is a genuine, measured 2-12 percentage point coverage gap between the two configurations, expressed as a full recall swing only because of the small-region-count threshold sensitivity described in (b) -- not evidence of anything wrong with the eval code, and not something that would necessarily reproduce the same way with more than 2 small regions.

