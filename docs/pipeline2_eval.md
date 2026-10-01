# Pipeline 2 evaluation: MASt3R-SLAM corpus, region IoU for both pipelines, H5/H6

MASt3R-SLAM evaluated on all 169 registered sequences, 4 configurations x
4 taus, exactly as EndoDAC in D1 Stage B (`docs/d1_stage_b.md`), with the
missing-prediction rules of `docs/eval_protocol.md` lines 196-216
("2026-09-29: Missing predictions, internal crops, confidence maps").
Region IoU for EndoDAC and MASt3R-SLAM, each next to its own area-matched
random baseline. H5 and H6 tested mechanically on MASt3R-SLAM.

**Every number below inherits the GT-based scale recovery of
`docs/eval_protocol.md` section 1 (depth scale fit to GT depth, Umeyama
alignment to the GT trajectory): it is an upper bound on what a deployed
system without GT could achieve, not an estimate of deployed performance.**

**D2b not triggered (2 of 3 pipelines).** No D2b or D2 pass/fail is
stated anywhere in this document. EndoDAC vs MASt3R-SLAM differences are
descriptive pairwise numbers only.

Code (committed): `scripts/pipeline_adapters.py` (EndoDAC and MASt3R-SLAM
adapters), `scripts/eval_pipeline_sequence.py` (per-sequence driver,
generalizes `scripts/eval_d1_sequence.py`, which is untouched),
`scripts/pipeline2_preflight.py`, `scripts/pipeline2_region_iou.py`,
`scripts/pipeline2_aggregate.py`, `scripts/pipeline2_report_tables.py`,
`tests/conventions/test_mast3r_grid_mapping.py`. Results (not committed):
`results/pipeline2_eval/`. No file under `src/eval/`, `src/gt/`,
`src/eval_ext/region_iou.py`, `docs/success_criteria.md` or
`docs/eval_protocol.md` was modified.

---

# MEASURED

## 1. Pre-flight

### 1.1 Reproduction gate (EndoDAC, `c1_cecum_t1_v1`): PASS

`scripts/eval_pipeline_sequence.py --pipeline endodac --sequence
c1_cecum_t1_v1` (GPU 4), exact NaN-aware diff against
`results/eval_stage4/summary.json`
(`results/pipeline2_eval/preflight/gate1_reproduction_diff.json`):
**980/980 reference leaves, 0 missing, 0 mismatches, bit-identical.** In
addition, `predicted_observed_packed.bin` and `regions.csv` are
byte-identical to D1 Stage B's `results/d1/per_sequence/c1_cecum_t1_v1/`.
The EndoDAC adapter loads exactly the files and dtype conversions of
`scripts/eval_d1_sequence.py`; with every frame and pixel predicted, every
missing-prediction branch of the new driver reduces to D1 Stage B's
computation (confirmed by this gate, not assumed).

### 1.2 MASt3R-SLAM adapter check (`c1_cecum_t1_v1`, no metric): PASS

`scripts/pipeline2_preflight.py adapter-check`
(`results/pipeline2_eval/preflight/gate2_adapter_check.json`):

| item | value |
|---|---|
| GT frames | 218 |
| frames with pose / with depth / with both | 218 / 218 / 218 |
| missing frames (pose / depth / either) | 0 / 0 / 0 |
| valid (non-vignette) pixels | 1,355,951 |
| valid pixels with no depth | 28,898 (**2.131%**): 13,380 above the grid (top crop), 14,358 below it, 1,160 right of the last grid column |
| round trip: full-res depth vs independent scalar bilinear of the 512x400 grid at Stage 2's mapped coordinates, 10 random valid pixels (seed 20260930) x 2 frames (0, 129) | max abs diff **2.2e-16** (native units) |
| unavailable pixels all NaN / available pixels all finite | yes / yes |
| mapping formula vs Stage 2's `original_to_model` (1,000 random points) | max abs diff 0.0 |

The 2.131% equals the figure Stage 3 reported (`docs/pipelines/
mast3r_slam.md`, Stage 3 pre-flight 2). It is identical for every
sequence (fixed 1350x1080 input; measured min = max over 169 sequences).

Bilinear rule: a 1350x1080 pixel has depth iff its mapped grid coordinate
lies in [0, 511] x [0, 399]; then its four bilinear neighbours are grid
cells holding a prediction. Outside that box the pixel is unavailable
(never extrapolated, never clamped). Unit tests:
`tests/conventions/test_mast3r_grid_mapping.py` (4 tests; full suite
64/64 pass).

Pose convention check (descriptive, not a metric): after position-only
Umeyama alignment, the rotation residual between aligned predicted and GT
camera rotations is median 10.2 deg, p95 22.7 deg (EndoDAC, same sequence:
median 7.4 deg). Reading the quaternion as (w,x,y,z) instead of (x,y,z,w)
gives median 175.6 deg, ruling that order out. Transposing the rotation
alone gives median 11.4 deg, p95 28.6 deg; relative rotations between
frames 30 apart do not separate the two readings for MASt3R (predicted
relative rotation median 13.9 deg vs GT 4.9 deg), while they do for
EndoDAC (error 3.6 deg vs 7.2 deg transposed). The c2w convention rests on
Stage 1's translation-direction test (`docs/pipelines/mast3r_slam.md` Q3);
rotation and translation come from the same lietorch SE3, so a mixed
convention cannot arise.

### 1.3 Timing (MASt3R-SLAM) and projection

Uncached driver, single process, GPU 4:

| sequence | frames | seconds |
|---|---|---|
| c1_transverse1_t1_v3 (shortest) | 117 | 172.1 |
| c2_rectum_t2_v3 (median) | 383 | 496.7 |
| c1_cecum_t4_v3 (longest) | 959 | 1205.0 |

Linear fit 1.227 s/frame + 27.8 s/sequence: **projected 24.5 h** for
67,886 frames, single process. Then two changes, neither touching any
computed value: (a) the adapter caches each frame's upsampled depth
instead of decoding it four times; rerunning c1_transverse1_t1_v3 with
the cache gave byte-identical `predicted_observed_packed.bin`,
`regions.csv` and `metrics.json` (timings excluded), 172 s -> 121 s
(`logs/pipeline2_cache_equivalence.log`); (b) three processes (shards)
on the same single GPU (index 4), since the GPU was mostly idle and the
work CPU-bound. The three timing sequences' outputs (uncached code) were
kept; the cache equivalence was verified on one of them only.

## 2. Main run

`scripts/eval_pipeline_sequence.py --pipeline mast3r_slam --all
--shard-index {0,1,2} --shard-total 3`, `CUDA_VISIBLE_DEVICES=4` for all
three (`logs/pipeline2_main_gpu_status.log`: GPU 4, 48.5 GB free, 0%
utilized at launch). **169/169 sequences status ok, 0 failures.** Wall
time 8 h 55 min (2026-10-01T01:18:47Z to 10:13:27Z); summed per-sequence
compute 26.5 h. Logs: `logs/pipeline2_main_mast3r_shard{0,1,2}.log`.

Per-sequence checks, all 169 sequences:
- oracle configuration via `src/eval/fusion.py` vs `src/eval/oracle.py`:
  bit-identical at every tau (the run raises otherwise);
- oracle predicted-observed bits vs D1 Stage B's saved bits: 0 faces
  differ, every tau;
- ignore set vs `results/d1/diagnosis/ignore_set/`: 0 faces differ.

Scale recovery: depth scale from every frame with predicted depth (169/169
sequences: all GT frames), over pixels that are GT-valid AND have a
prediction; Umeyama over frames with a predicted pose. Same factors reused
in every configuration (`docs/eval_protocol.md` section 4).

## 3. Missing frames and crop

| quantity | value |
|---|---|
| GT frames, 169 sequences | 67,886 |
| frames with no predicted pose | **6,188 (9.1%)** |
| frames with no predicted depth | 0 |
| sequences with any missing frame | **26** (19 distinct meshes, 18 distinct trajectories) |
| sequences with every frame predicted (pose and depth) | 143 |
| valid pixels with no depth, frames with depth | 2.131% in every sequence |
| valid pixels with no depth, all GT frames, pooled | 2.131% |

Handling (`docs/eval_protocol.md` lines 201-216): a missing-pose frame is
not ray-cast in pred_pose_only or fully_predicted; it is used in
pred_depth_only (its depth exists) and in oracle. No frame is filled or
interpolated. No sequence is excluded.

**Discrepancy with the Stage 3 record.** `docs/pipelines/mast3r_slam.md`
Stage 3 reported these 26 sequences as each "missing exactly one frame"
and "not a permanent track loss". Against the GT frame count they miss 61
to 441 frames each. After the skipped frame, MASt3R-SLAM enters
relocalization; frames that fail to relocalize never reach `track()` and
have no row in `poses_per_frame.csv`, which is what Stage 3 counted. In
21 of the 26 every frame from the skipped frame to the end has no pose;
in 5 (c1_cecum_t1_v3, c2_ascending_t4_v1, c2_rectum_t1_v2,
c2_rectum_t3_v2, c2_transverse1_t3_v1) tracking resumed later. A
correction block was appended to `docs/pipelines/mast3r_slam.md` (original
text left in place). The strict-D1.1 count (143/169) was right.

Per sequence (`results/pipeline2_eval/mast3r_missing_frames.csv`):

| sequence | GT frames | frames with pose | missing pose | first missing-pose frame |
|---|---|---|---|---|
| c1_cecum_t1_v2 | 423 | 170 | 253 | 170 |
| c1_cecum_t1_v3 | 423 | 212 | 211 | 171 |
| c1_sigmoid1_t1_v3 | 307 | 56 | 251 | 56 |
| c2_ascending_t4_v1 | 292 | 231 | 61 | 151 |
| c2_ascending_t4_v3 | 455 | 324 | 131 | 324 |
| c2_cecum_t1_v1 | 304 | 173 | 131 | 173 |
| c2_cecum_t1_v2 | 431 | 48 | 383 | 48 |
| c2_cecum_t1_v3 | 431 | 45 | 386 | 45 |
| c2_cecum_t3_v1 | 228 | 74 | 154 | 74 |
| c2_cecum_t4_v1 | 191 | 68 | 123 | 68 |
| c2_cecum_t4_v2 | 520 | 336 | 184 | 336 |
| c2_cecum_t4_v3 | 520 | 341 | 179 | 341 |
| c2_rectum_t1_v2 | 712 | 448 | 264 | 389 |
| c2_rectum_t1_v3 | 712 | 400 | 312 | 400 |
| c2_rectum_t3_v2 | 333 | 250 | 83 | 125 |
| c2_rectum_t3_v3 | 333 | 133 | 200 | 133 |
| c2_rectum_t4_v1 | 198 | 36 | 162 | 36 |
| c2_sigmoid_t1_v3 | 558 | 196 | 362 | 196 |
| c2_transverse1_t1_v2 | 627 | 186 | 441 | 186 |
| c2_transverse1_t1_v3 | 627 | 227 | 400 | 227 |
| c2_transverse1_t3_v1 | 853 | 578 | 275 | 498 |
| c2_transverse1_t3_v2 | 298 | 105 | 193 | 105 |
| c2_transverse1_t4_v2 | 382 | 84 | 298 | 84 |
| c2_transverse1_t4_v3 | 382 | 226 | 156 | 226 |
| c2_transverse2_t1_v2 | 552 | 252 | 300 | 252 |
| c2_transverse2_t1_v3 | 552 | 257 | 295 | 257 |

## 4. MASt3R-SLAM corpus tables (layout of `docs/d1_stage_b.md`)

Pooled point estimates; mesh-level cluster bootstrap 10,000 replicates,
seed 20260925; 169 sequences, 103 meshes, 113 trajectories
(`docs/regions.md:50`), 58 (Colon, Segment, Phantom Number) groups, 15
(Colon, Segment) groups behind every row. Full tables (every config x tau
x metric x detection threshold, all three cluster levels):
`results/pipeline2_eval/mast3r_corpus_tables.json`. Aggregation code is
`scripts/eval_d1_aggregate.py::build_corpus_tables`, imported; on
EndoDAC's Stage B outputs it reproduces `results/d1/corpus_tables.json`
exactly (3,408/3,408 leaves).

Recall by size class, tau = 0.25, 50%: oracle 1.000 / 1.000 / 1.000
(small 238 / medium 277 / large 442 regions); pred_depth_only 1.000 /
1.000 / 1.000; pred_pose_only 0.7605 / 0.8267 / 0.9548; fully_predicted
0.9034 / 0.9278 / 0.9774. Below-headline (appendix, 23,251 regions):
0.5997 / 0.9439 / 0.7024 / 0.8760.

Corpus-wide segment_intersects_mesh fraction (all configs and taus
pooled): 3.90%.

### Corpus table, tau = 0.25 (point estimate, mesh-level 95% CI)

| config | recall m+l @50% | area_fraction | calibration (raw / ignore-excl.) | false_reassurance | false_alarm | loc_err median mm (n) | segment_intersect |
|---|---|---|---|---|---|---|---|
| oracle | 1.0000 [1.0000, 1.0000] (n=719, 103 meshes) | 0.2628 [0.2427, 0.2829] | 1.154 / 0.999 | 0.00150 [0.00125, 0.00176] | 0.0006 [0.0003, 0.0009] | 0.008 (n=957) | 0.0021 |
| pred_depth_only | 1.0000 [1.0000, 1.0000] (n=719, 103 meshes) | 0.4910 [0.4611, 0.5200] | 2.156 / 2.001 | 0.00020 [0.00014, 0.00027] | 0.5004 [0.4737, 0.5263] | 15.716 (n=957) | 0.0334 |
| pred_pose_only | 0.9054 [0.8738, 0.9340] (n=719, 103 meshes) | 0.3346 [0.3023, 0.3697] | 1.470 / 1.357 | 0.11357 [0.09425, 0.13531] | 0.3466 [0.2986, 0.3943] | 5.964 (n=946) | 0.0772 |
| fully_predicted | 0.9583 [0.9385, 0.9763] (n=719, 103 meshes) | 0.5704 [0.5442, 0.5971] | 2.505 / 2.362 | 0.05169 [0.04011, 0.06501] | 0.5986 [0.5733, 0.6250] | 19.143 (n=944) | 0.0477 |

### All taus: recall (m+l, 50%), area fraction, false reassurance, false alarm (point estimates)

| config | tau | recall m+l | recall small | recall large | area_fraction | false_reassurance | false_alarm |
|---|---|---|---|---|---|---|---|
| oracle | 0.15 | 1.0000 | 1.0000 | 1.0000 | 0.2629 | 0.00142 | 0.0008 |
| oracle | 0.25 | 1.0000 | 1.0000 | 1.0000 | 0.2628 | 0.00150 | 0.0006 |
| oracle | 0.35 | 1.0000 | 1.0000 | 1.0000 | 0.2627 | 0.00152 | 0.0004 |
| oracle | 0.5 | 1.0000 | 1.0000 | 1.0000 | 0.2627 | 0.00154 | 0.0003 |
| pred_depth_only | 0.15 | 1.0000 | 1.0000 | 1.0000 | 0.5411 | 0.00012 | 0.5499 |
| pred_depth_only | 0.25 | 1.0000 | 1.0000 | 1.0000 | 0.4910 | 0.00020 | 0.5004 |
| pred_depth_only | 0.35 | 1.0000 | 1.0000 | 1.0000 | 0.4430 | 0.00029 | 0.4417 |
| pred_depth_only | 0.5 | 1.0000 | 1.0000 | 1.0000 | 0.3688 | 0.00051 | 0.3176 |
| pred_pose_only | 0.15 | 0.9235 | 0.8319 | 0.9706 | 0.3743 | 0.09414 | 0.4053 |
| pred_pose_only | 0.25 | 0.9054 | 0.7605 | 0.9548 | 0.3346 | 0.11357 | 0.3466 |
| pred_pose_only | 0.35 | 0.8915 | 0.7395 | 0.9434 | 0.3125 | 0.12637 | 0.3085 |
| pred_pose_only | 0.5 | 0.8818 | 0.7059 | 0.9344 | 0.2904 | 0.13859 | 0.2644 |
| fully_predicted | 0.15 | 0.9708 | 0.9076 | 0.9864 | 0.6285 | 0.03932 | 0.6325 |
| fully_predicted | 0.25 | 0.9583 | 0.9034 | 0.9774 | 0.5704 | 0.05169 | 0.5986 |
| fully_predicted | 0.35 | 0.9513 | 0.8908 | 0.9706 | 0.5064 | 0.06547 | 0.5517 |
| fully_predicted | 0.5 | 0.9235 | 0.8571 | 0.9434 | 0.4053 | 0.09048 | 0.4472 |

### Sensitivity intervals, tau = 0.25: (Colon, Segment) and (Colon, Segment, Phantom Number)

| config | metric | mesh CI | (Colon, Segment) CI | (Colon, Segment, Phantom) CI |
|---|---|---|---|---|
| oracle | recall m+l | [1.0000, 1.0000] | [1.0000, 1.0000] | [1.0000, 1.0000] |
| oracle | false_reassurance_rate | [0.00125, 0.00176] | [0.00106, 0.00206] | [0.00121, 0.00180] |
| oracle | false_alarm_rate | [0.00031, 0.00091] | [0.00029, 0.00094] | [0.00031, 0.00089] |
| oracle | area_fraction | [0.24267, 0.28289] | [0.22032, 0.30508] | [0.23882, 0.28698] |
| pred_depth_only | recall m+l | [1.0000, 1.0000] | [1.0000, 1.0000] | [1.0000, 1.0000] |
| pred_depth_only | false_reassurance_rate | [0.00014, 0.00027] | [0.00011, 0.00031] | [0.00013, 0.00029] |
| pred_depth_only | false_alarm_rate | [0.47366, 0.52630] | [0.44085, 0.55312] | [0.46934, 0.53108] |
| pred_depth_only | area_fraction | [0.46110, 0.52000] | [0.42367, 0.55143] | [0.45501, 0.52589] |
| pred_pose_only | recall m+l | [0.8738, 0.9340] | [0.8743, 0.9353] | [0.8750, 0.9342] |
| pred_pose_only | false_reassurance_rate | [0.09425, 0.13531] | [0.08534, 0.15335] | [0.09169, 0.14016] |
| pred_pose_only | false_alarm_rate | [0.29856, 0.39434] | [0.28058, 0.40345] | [0.29608, 0.39712] |
| pred_pose_only | area_fraction | [0.30235, 0.36967] | [0.27618, 0.39745] | [0.29729, 0.37280] |
| fully_predicted | recall m+l | [0.9385, 0.9763] | [0.9362, 0.9797] | [0.9373, 0.9775] |
| fully_predicted | false_reassurance_rate | [0.04011, 0.06501] | [0.03256, 0.07607] | [0.03871, 0.06691] |
| fully_predicted | false_alarm_rate | [0.57330, 0.62505] | [0.55420, 0.64722] | [0.56928, 0.62761] |
| fully_predicted | area_fraction | [0.54423, 0.59705] | [0.52452, 0.61578] | [0.53914, 0.59993] |

### Diagnostic b (over 169 sequences: mean / median / max)

| config | ray_miss_frac | d_pred_unavailable_frac_of_evaluable |
|---|---|---|
| oracle | 0.0164 / 0.0106 / 0.0943 (c2_transverse2_t2_v1) | 0.0000 / 0.0000 / 0.0010 |
| pred_depth_only | 0.0164 / 0.0106 / 0.0943 (c2_transverse2_t2_v1) | 0.0218 / 0.0221 / 0.0234 |
| pred_pose_only | 0.0332 / 0.0194 / 0.8998 (c2_rectum_t4_v1) | 0.0205 / 0.0205 / 0.0733 |
| fully_predicted | 0.0332 / 0.0194 / 0.8998 (c2_rectum_t4_v1) | 0.0220 / 0.0220 / 0.0495 |

### Diagnostic c (pooled faces)

| tau | only-unobserved under fully_predicted, not pred_pose_only | also unobserved under pred_depth_only | fraction |
|---|---|---|---|
| 0.15 | 19,807,607 | 13,264,897 | 0.670 |
| 0.25 | 17,909,849 | 11,696,308 | 0.653 |
| 0.35 | 14,808,344 | 9,580,018 | 0.647 |
| 0.5 | 9,277,331 | 5,677,846 | 0.612 |

Notes on diagnostic b: in pred_pose_only, "d_pred" is GT depth, so its
unavailable fraction counts GT-invalid pixels hit under the predicted pose
(as in D1 Stage B). In pred_depth_only and fully_predicted it is
dominated by the 2.13% crop. ray_miss_frac in the predicted-pose
configurations is over ray-cast (posed) frames only; its maximum, 0.8998,
is c2_rectum_t4_v1, which has 36 posed frames out of 198.

## 5. Region IoU, both pipelines, with matched baselines

`src/eval_ext/region_iou.py` unchanged; components from the locked
`eval.regions.compute_regions` on predicted_unobserved & ~ignore_set.
For every (pipeline, sequence, cell), 20 area-matched random sets at that
cell's own predicted-unobserved area on that sequence, with the seeds and
construction of `scripts/diagnose_d1_3.py` (default_rng(1000 * 20260925 +
seed), seeds 0-19). Check: EndoDAC fully_predicted tau = 0.25 random rows
reproduce `results/region_iou_gate/region_iou_raw.csv` exactly (19,140
rows, max diff 0.0). Rows: `results/pipeline2_eval/region_iou/`;
summary: `results/pipeline2_eval/region_iou_summary.json`.

Columns: pipeline IoU (pooled mean over regions, mesh CI) and median;
random baseline seed-averaged per region, then pooled (range of the 20
per-seed pooled means in brackets); paired difference = pooled mean of
per-region (pipeline IoU - seed-averaged random IoU), mesh CI and both
sensitivity CIs; false alarm of the pipeline (and of the random sets,
mean over seeds); predicted-unobserved area fraction (identical for the
random sets by construction, to within one face). 719 medium + large
regions, 103 meshes, 113 trajectories in every row.

### endodac: region IoU (medium + large) vs area-matched random baseline

| config | tau | IoU mean [mesh CI] | IoU median | random mean, seed-avg (range over 20 seeds) | random median | paired diff (pipeline - random) [mesh CI] | (C,S) CI | (C,S,P) CI | false_alarm (random) | area_fraction | n regions / meshes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| oracle | 0.25 | 0.9979 [0.9975, 0.9983] | 0.9991 | 0.2711 (0.2704-0.2719) | 0.2784 | +0.7269 [0.7086, 0.7455] | [0.6951, 0.7657] | [0.7062, 0.7493] | 0.0006 (0.7311) | 0.2628 | 719 / 103 |
| pred_depth_only | 0.25 | 0.3518 [0.3221, 0.3818] | 0.2427 | 0.4369 (0.4351-0.4386) | 0.4617 | -0.0851 [-0.1251, -0.0413] | [-0.1528, -0.0050] | [-0.1312, -0.0336] | 0.4982 (0.7453) | 0.4888 | 719 / 103 |
| pred_pose_only | 0.25 | 0.4622 [0.4235, 0.5015] | 0.4642 | 0.3565 (0.3548-0.3580) | 0.3693 | +0.1057 [0.0562, 0.1561] | [0.0353, 0.1852] | [0.0541, 0.1622] | 0.3648 (0.7354) | 0.3687 | 719 / 103 |
| fully_predicted | 0.25 | 0.2908 [0.2629, 0.3203] | 0.1924 | 0.4317 (0.4290-0.4358) | 0.4758 | -0.1409 [-0.1811, -0.0980] | [-0.1972, -0.0746] | [-0.1842, -0.0934] | 0.5428 (0.7447) | 0.5167 | 719 / 103 |
| fully_predicted | 0.15 | 0.2459 [0.2224, 0.2701] | 0.1548 | 0.4121 (0.4084-0.4155) | 0.4696 | -0.1661 [-0.2047, -0.1269] | [-0.2160, -0.1120] | [-0.2076, -0.1205] | 0.5882 (0.7487) | 0.5742 | 719 / 103 |
| fully_predicted | 0.35 | 0.3306 [0.2995, 0.3629] | 0.2347 | 0.4310 (0.4283-0.4330) | 0.4581 | -0.1004 [-0.1431, -0.0550] | [-0.1640, -0.0252] | [-0.1477, -0.0481] | 0.4908 (0.7418) | 0.4639 | 719 / 103 |
| fully_predicted | 0.5 | 0.4416 [0.4047, 0.4806] | 0.3930 | 0.3840 (0.3828-0.3849) | 0.3921 | +0.0576 [0.0086, 0.1065] | [-0.0130, 0.1427] | [0.0057, 0.1143] | 0.3858 (0.7381) | 0.3856 | 719 / 103 |

#### endodac: by size class, fully_predicted tau = 0.25

| size class | IoU mean | random mean | paired diff [mesh CI] | n |
|---|---|---|---|---|
| small | 0.2845 | 0.3586 | -0.0740 [-0.1342, -0.0094] | 238 |
| medium | 0.2106 | 0.3993 | -0.1886 [-0.2558, -0.1183] | 277 |
| large | 0.3410 | 0.4520 | -0.1109 [-0.1521, -0.0685] | 442 |
| medium_plus_large | 0.2908 | 0.4317 | -0.1409 [-0.1811, -0.0980] | 719 |

### mast3r_slam: region IoU (medium + large) vs area-matched random baseline

| config | tau | IoU mean [mesh CI] | IoU median | random mean, seed-avg (range over 20 seeds) | random median | paired diff (pipeline - random) [mesh CI] | (C,S) CI | (C,S,P) CI | false_alarm (random) | area_fraction | n regions / meshes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| oracle | 0.25 | 0.9979 [0.9975, 0.9983] | 0.9991 | 0.2711 (0.2704-0.2719) | 0.2784 | +0.7269 [0.7085, 0.7461] | [0.6946, 0.7648] | [0.7060, 0.7492] | 0.0006 (0.7311) | 0.2628 | 719 / 103 |
| pred_depth_only | 0.25 | 0.3578 [0.3247, 0.3912] | 0.2421 | 0.4349 (0.4321-0.4379) | 0.4703 | -0.0770 [-0.1201, -0.0318] | [-0.1467, -0.0027] | [-0.1264, -0.0248] | 0.5004 (0.7450) | 0.4910 | 719 / 103 |
| pred_pose_only | 0.25 | 0.4829 [0.4448, 0.5226] | 0.5271 | 0.3142 (0.3132-0.3151) | 0.3123 | +0.1687 [0.1217, 0.2186] | [0.0857, 0.2642] | [0.1165, 0.2278] | 0.3466 (0.7403) | 0.3346 | 719 / 103 |
| fully_predicted | 0.25 | 0.2630 [0.2444, 0.2827] | 0.1698 | 0.4139 (0.4102-0.4185) | 0.4794 | -0.1509 [-0.1882, -0.1107] | [-0.2029, -0.0884] | [-0.1903, -0.1092] | 0.5986 (0.7527) | 0.5704 | 719 / 103 |
| fully_predicted | 0.15 | 0.2274 [0.2088, 0.2471] | 0.1397 | 0.3627 (0.3570-0.3673) | 0.4556 | -0.1353 [-0.1783, -0.0915] | [-0.2000, -0.0662] | [-0.1826, -0.0883] | 0.6325 (0.7549) | 0.6285 | 719 / 103 |
| fully_predicted | 0.35 | 0.3047 [0.2825, 0.3277] | 0.2192 | 0.4415 (0.4397-0.4431) | 0.4768 | -0.1367 [-0.1704, -0.1010] | [-0.1842, -0.0837] | [-0.1733, -0.0981] | 0.5517 (0.7494) | 0.5064 | 719 / 103 |
| fully_predicted | 0.5 | 0.3774 [0.3499, 0.4048] | 0.3158 | 0.3875 (0.3867-0.3884) | 0.4153 | -0.0102 [-0.0488, 0.0302] | [-0.0637, 0.0564] | [-0.0519, 0.0351] | 0.4472 (0.7452) | 0.4053 | 719 / 103 |

#### mast3r_slam: by size class, fully_predicted tau = 0.25

| size class | IoU mean | random mean | paired diff [mesh CI] | n |
|---|---|---|---|---|
| small | 0.2312 | 0.3502 | -0.1190 [-0.1745, -0.0592] | 238 |
| medium | 0.1797 | 0.4024 | -0.2227 [-0.2710, -0.1712] | 277 |
| large | 0.3152 | 0.4211 | -0.1058 [-0.1471, -0.0633] | 442 |
| medium_plus_large | 0.2630 | 0.4139 | -0.1509 [-0.1882, -0.1107] | 719 |

Cells where the pipeline's IoU is below its matched random baseline with
the mesh CI of the paired difference below zero, reported as such per the
reporting rule (`docs/eval_protocol.md` lines 448-463):
- EndoDAC: pred_depth_only 0.25; fully_predicted 0.15, 0.25, 0.35.
- MASt3R-SLAM: pred_depth_only 0.25; fully_predicted 0.15, 0.25, 0.35.
- Not distinguishable from the baseline (CI includes zero): MASt3R-SLAM
  fully_predicted 0.50.
- Above the baseline: both pipelines pred_pose_only 0.25 and oracle;
  EndoDAC fully_predicted 0.50 (mesh CI [0.009, 0.107]; the (Colon,
  Segment) CI [-0.013, 0.143] includes zero).

## 6. H5 and H6 (MASt3R-SLAM only)

Criterion text, `docs/success_criteria.md` lines 424-431, quoted:

> 4. New pre-stated hypotheses, tested only on pipelines evaluated after
>    this entry (EndoDAC excluded, since these were derived from its data).
>    H5 (masking): for each new pipeline, false reassurance under
>        fully_predicted is lower than under pred_pose_only, with the
>        mesh-level CI of the paired difference excluding zero.
>    H6 (depth drives false alarm): for each new pipeline, false alarm
>        under pred_depth_only exceeds that under pred_pose_only, with the
>        mesh-level CI of the paired difference excluding zero.

MASt3R-SLAM is the only pipeline evaluated after that entry. Paired by
sequence (same 169 sequences in both configurations), pooled ratio of
sums per configuration, difference recomputed on every mesh-cluster
resample (10,000, seed 20260925; `results/pipeline2_eval/h5_h6.json`).
169 sequences, 103 meshes, 113 trajectories.

**H5** (false reassurance, A - B < 0; A = fully_predicted, B = pred_pose_only)

| tau | A | B | A - B | mesh CI | (C,S) CI | (C,S,P) CI | criterion met (mesh CI) |
|---|---|---|---|---|---|---|---|
| 0.15 | 0.03932 | 0.09414 | -0.05482 | [-0.06832, -0.04305] | [-0.07458, -0.03892] | [-0.07039, -0.04186] | yes |
| 0.25 | 0.05169 | 0.11357 | -0.06188 | [-0.07534, -0.04982] | [-0.08084, -0.04658] | [-0.07824, -0.04815] | yes |
| 0.35 | 0.06547 | 0.12637 | -0.06090 | [-0.07345, -0.04922] | [-0.07851, -0.04572] | [-0.07663, -0.04775] | yes |
| 0.5 | 0.09048 | 0.13859 | -0.04811 | [-0.05994, -0.03642] | [-0.06628, -0.02930] | [-0.06089, -0.03566] | yes |

**H6** (false alarm, A - B > 0; A = pred_depth_only, B = pred_pose_only)

| tau | A | B | A - B | mesh CI | (C,S) CI | (C,S,P) CI | criterion met (mesh CI) |
|---|---|---|---|---|---|---|---|
| 0.15 | 0.54991 | 0.40533 | +0.14458 | [0.09932, 0.18994] | [0.07852, 0.21082] | [0.09904, 0.19075] | yes |
| 0.25 | 0.50043 | 0.34662 | +0.15381 | [0.10123, 0.20533] | [0.07374, 0.23084] | [0.09935, 0.20752] | yes |
| 0.35 | 0.44168 | 0.30846 | +0.13322 | [0.07734, 0.18754] | [0.04942, 0.21452] | [0.07518, 0.18947] | yes |
| 0.5 | 0.31763 | 0.26436 | +0.05327 | [-0.00328, 0.10642] | [-0.02977, 0.13156] | [-0.00627, 0.10991] | no |

**Mechanical result.** The criterion text names no tau.
- H5: lower false reassurance under fully_predicted, mesh CI excluding
  zero, **at all four taus**. Holds as written, whichever tau is meant.
- H6: higher false alarm under pred_depth_only, mesh CI excluding zero,
  **at tau = 0.15, 0.25, 0.35; not at tau = 0.50** (point +0.053, mesh CI
  [-0.003, 0.106]). Whether H6 holds as written depends on which tau the
  criterion refers to: **UNKNOWN**. The only tau fixed in the same
  deviation entry is D2b's tau = 0.25 (item 3), which item 4 does not
  reference. Settling it needs the author's reading; this document does
  not choose one.

## 7. Descriptive pairwise differences, MASt3R-SLAM minus EndoDAC

Paired by sequence (169), pooled ratios recomputed per mesh-cluster
resample; region IoU and recall paired per region (same 719 medium +
large regions in both pipelines). 103 meshes, 113 trajectories. No
pass/fail attached. Oracle rows are identical by construction (both use
GT depth and GT pose) and serve as a check. File:
`results/pipeline2_eval/pairwise_mast3r_minus_endodac.json`.

| config | tau | metric | MASt3R | EndoDAC | MASt3R - EndoDAC | mesh CI | (C,S) CI | (C,S,P) CI |
|---|---|---|---|---|---|---|---|---|
| oracle | 0.25 | false_reassurance_rate | 0.00150 | 0.00150 | +0.00000 | [0.00000, 0.00000] | [0.00000, 0.00000] | [0.00000, 0.00000] |
| oracle | 0.25 | false_alarm_rate | 0.00056 | 0.00056 | +0.00000 | [0.00000, 0.00000] | [0.00000, 0.00000] | [0.00000, 0.00000] |
| oracle | 0.25 | area_fraction | 0.26279 | 0.26279 | +0.00000 | [0.00000, 0.00000] | [0.00000, 0.00000] | [0.00000, 0.00000] |
| oracle | 0.25 | recall_medium_plus_large_at_50pct | 1.00000 | 1.00000 | +0.00000 | [0.00000, 0.00000] | [0.00000, 0.00000] | [0.00000, 0.00000] |
| oracle | 0.25 | region_iou_mean_medium_plus_large | 0.99793 | 0.99793 | +0.00000 | [0.00000, 0.00000] | [0.00000, 0.00000] | [0.00000, 0.00000] |
| pred_depth_only | 0.25 | false_reassurance_rate | 0.00020 | 0.00043 | -0.00023 | [-0.00033, -0.00015] | [-0.00039, -0.00011] | [-0.00035, -0.00014] |
| pred_depth_only | 0.25 | false_alarm_rate | 0.50043 | 0.49817 | +0.00226 | [-0.01061, 0.01511] | [-0.01550, 0.02115] | [-0.01222, 0.01672] |
| pred_depth_only | 0.25 | area_fraction | 0.49101 | 0.48885 | +0.00216 | [-0.00962, 0.01406] | [-0.01405, 0.01949] | [-0.01104, 0.01513] |
| pred_depth_only | 0.25 | recall_medium_plus_large_at_50pct | 1.00000 | 1.00000 | +0.00000 | [0.00000, 0.00000] | [0.00000, 0.00000] | [0.00000, 0.00000] |
| pred_depth_only | 0.25 | region_iou_mean_medium_plus_large | 0.35784 | 0.35178 | +0.00606 | [-0.01147, 0.02352] | [-0.00855, 0.02106] | [-0.01228, 0.02392] |
| pred_pose_only | 0.25 | false_reassurance_rate | 0.11357 | 0.05517 | +0.05840 | [0.04227, 0.07557] | [0.03787, 0.08344] | [0.04122, 0.07806] |
| pred_pose_only | 0.25 | false_alarm_rate | 0.34662 | 0.36475 | -0.01813 | [-0.07065, 0.03371] | [-0.09282, 0.04772] | [-0.07167, 0.03422] |
| pred_pose_only | 0.25 | area_fraction | 0.33464 | 0.36870 | -0.03406 | [-0.06164, -0.00469] | [-0.07653, 0.00716] | [-0.06391, -0.00222] |
| pred_pose_only | 0.25 | recall_medium_plus_large_at_50pct | 0.90542 | 0.96523 | -0.05981 | [-0.08976, -0.03220] | [-0.08716, -0.03549] | [-0.08832, -0.03319] |
| pred_pose_only | 0.25 | region_iou_mean_medium_plus_large | 0.48289 | 0.46223 | +0.02066 | [-0.02269, 0.06596] | [-0.03578, 0.08271] | [-0.02067, 0.06262] |
| fully_predicted | 0.25 | false_reassurance_rate | 0.05169 | 0.03074 | +0.02095 | [0.01036, 0.03252] | [0.00927, 0.03479] | [0.01007, 0.03315] |
| fully_predicted | 0.25 | false_alarm_rate | 0.59857 | 0.54276 | +0.05581 | [0.03788, 0.07373] | [0.02701, 0.08631] | [0.03697, 0.07631] |
| fully_predicted | 0.25 | area_fraction | 0.57039 | 0.51673 | +0.05366 | [0.03253, 0.07491] | [0.02200, 0.08595] | [0.03109, 0.07666] |
| fully_predicted | 0.25 | recall_medium_plus_large_at_50pct | 0.95828 | 0.98331 | -0.02503 | [-0.04447, -0.00769] | [-0.04209, -0.00703] | [-0.04386, -0.00705] |
| fully_predicted | 0.25 | region_iou_mean_medium_plus_large | 0.26303 | 0.29079 | -0.02776 | [-0.05249, -0.00386] | [-0.06094, 0.00137] | [-0.05637, -0.00213] |
| fully_predicted | 0.15 | false_reassurance_rate | 0.03932 | 0.02347 | +0.01585 | [0.00665, 0.02578] | [0.00662, 0.02602] | [0.00611, 0.02630] |
| fully_predicted | 0.15 | false_alarm_rate | 0.63253 | 0.58817 | +0.04436 | [0.03075, 0.05829] | [0.02201, 0.06743] | [0.02931, 0.06072] |
| fully_predicted | 0.15 | area_fraction | 0.62847 | 0.57421 | +0.05426 | [0.03531, 0.07439] | [0.02386, 0.08420] | [0.03240, 0.07651] |
| fully_predicted | 0.15 | recall_medium_plus_large_at_50pct | 0.97079 | 0.98748 | -0.01669 | [-0.03253, -0.00150] | [-0.03050, -0.00257] | [-0.03276, -0.00144] |
| fully_predicted | 0.15 | region_iou_mean_medium_plus_large | 0.22742 | 0.24594 | -0.01851 | [-0.04000, 0.00224] | [-0.05101, 0.00856] | [-0.04382, 0.00473] |
| fully_predicted | 0.35 | false_reassurance_rate | 0.06547 | 0.03816 | +0.02731 | [0.01455, 0.04105] | [0.01116, 0.04645] | [0.01421, 0.04228] |
| fully_predicted | 0.35 | false_alarm_rate | 0.55172 | 0.49075 | +0.06097 | [0.03985, 0.08300] | [0.03014, 0.09385] | [0.03785, 0.08451] |
| fully_predicted | 0.35 | area_fraction | 0.50644 | 0.46388 | +0.04256 | [0.02186, 0.06424] | [0.01153, 0.07240] | [0.01935, 0.06559] |
| fully_predicted | 0.35 | recall_medium_plus_large_at_50pct | 0.95132 | 0.98192 | -0.03060 | [-0.05270, -0.01049] | [-0.05122, -0.00990] | [-0.05147, -0.01169] |
| fully_predicted | 0.35 | region_iou_mean_medium_plus_large | 0.30472 | 0.33062 | -0.02590 | [-0.05081, -0.00102] | [-0.05325, 0.00596] | [-0.05373, 0.00150] |
| fully_predicted | 0.5 | false_reassurance_rate | 0.09048 | 0.04953 | +0.04095 | [0.02386, 0.05990] | [0.01598, 0.07361] | [0.02324, 0.06164] |
| fully_predicted | 0.5 | false_alarm_rate | 0.44725 | 0.38584 | +0.06141 | [0.03367, 0.08989] | [0.02054, 0.10162] | [0.03181, 0.09176] |
| fully_predicted | 0.5 | area_fraction | 0.40526 | 0.38557 | +0.01969 | [-0.00042, 0.04099] | [-0.01331, 0.05138] | [-0.00304, 0.04331] |
| fully_predicted | 0.5 | recall_medium_plus_large_at_50pct | 0.92350 | 0.97079 | -0.04729 | [-0.07224, -0.02406] | [-0.07278, -0.02394] | [-0.07143, -0.02489] |
| fully_predicted | 0.5 | region_iou_mean_medium_plus_large | 0.37737 | 0.44160 | -0.06424 | [-0.10082, -0.02802] | [-0.12824, -0.00748] | [-0.10807, -0.02145] |

## 8. D2b

**D2b not triggered (2 of 3 pipelines).** `docs/success_criteria.md`
line 408: "Trigger: 3 pipelines evaluated, including EndoDAC". D2 (lines
111-125) needs 4 to 6 pipelines and is not triggered either.

---

# INTERPRETATION

Each item is a reading of the numbers above, not a finding.

1. **Predicted depth drives both pipelines' region IoU below the
   location-blind baseline.** For both pipelines, configurations with
   predicted depth (pred_depth_only, fully_predicted at tau <= 0.35)
   score below their area-matched random baseline, while pred_pose_only
   scores above it, and false alarm is about 0.5 to 0.6 wherever
   predicted depth is used. Consistent with predicted-depth over-flagging
   producing large predicted components that merge GT regions with
   GT-observed surface, inflating the union in IoU. Would be confirmed by
   decomposing each region's union into area inside R, area of touching
   components outside R, and showing the outside share drives the deficit
   for predicted-depth configurations; refuted if the deficit comes mainly
   from low intersection (regions partly marked observed) instead.

2. **MASt3R-SLAM's higher false reassurance under predicted pose reflects
   pose error, not missing frames.** Missing-pose frames can only remove
   observations, which lowers false reassurance, yet MASt3R-SLAM's
   pred_pose_only false reassurance is 0.114 vs EndoDAC's 0.055.
   Consistent with its larger rotation residual (Section 1.2, one
   sequence; ATE median 6.97 vs 5.16 mm, Stage 3). Would be confirmed by
   a per-sequence association between false reassurance and rotation
   residual / ATE across the 169 sequences; refuted if the excess is
   concentrated in a few sequences unrelated to trajectory error.

3. **The 6,188 missing-pose frames inflate MASt3R-SLAM's predicted-pose
   area fraction and false alarm on 26 sequences.** Frames never ray-cast
   observe nothing. Would be quantified (descriptively only, not
   replacing the full-corpus numbers) by the pairwise differences
   restricted to the 143 complete sequences; refuted if those differences
   are essentially unchanged.

4. **H5's direction is the masking mechanism the hypothesis names.** Under
   the GT pose, predicted depth can only remove observations
   (`docs/success_criteria.md` lines 350-356). Under a predicted pose it
   can also add them (a pixel whose GT depth fails the tau test against the
   misplaced surface may pass with predicted depth), so H5's outcome is
   not guaranteed by construction. Diagnostic c counts faces unobserved
   only under fully_predicted (17.9 M at tau = 0.25, pooled); the reverse
   count (observed under fully_predicted but not pred_pose_only) was not
   computed. That count, restricted to GT-unobserved faces, would show how
   much of the H5 difference is removal versus addition.

---

# Open questions

1. H6 at tau = 0.50: which tau does item 4 of the 2026-09-27 deviation
   refer to? (Section 6.)
2. The Stage 2 grid mapping (`ox = (px + half_crop_w) * scale_w`) has no
   half-pixel offset. Whether it matches the pixel-center convention of
   MASt3R-SLAM's own resize is UNKNOWN (not verified here); the largest
   possible shift is about (scale - 1) / 2 = 0.8 original pixels. Used
   exactly as documented, per instruction.
3. MASt3R-SLAM's relative rotations over 30 frames are about three times
   GT's on c1_cecum_t1_v1 (13.9 vs 4.9 deg median). Not examined on
   other sequences.
4. Interpretations 1-3 each name an analysis that was not run.
