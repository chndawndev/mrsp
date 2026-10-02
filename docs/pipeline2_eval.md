# Pipeline 2 evaluation: MASt3R-SLAM corpus, region IoU for both pipelines, H5/H6

MASt3R-SLAM evaluated on all 169 registered sequences, 4 configurations x
4 taus, exactly as EndoDAC in D1 Stage B (`docs/d1_stage_b.md`), with the
missing-prediction rules of `docs/eval_protocol.md` lines 196-216
("2026-09-29: Missing predictions, internal crops, confidence maps").
Region IoU for EndoDAC and MASt3R-SLAM, each next to its own area-matched
random baseline. H5 and H6 tested mechanically on MASt3R-SLAM.

**Revision 2026-10-02 (grid mapping).** The first version of this document
(commit 64e61e1) mapped MASt3R-SLAM's 512x400 depth grid to 1350x1080 with
a corner-aligned formula. MASt3R-SLAM's resize is pixel-center aligned
(section 1.2). The mapping was corrected, the full MASt3R-SLAM evaluation
was rerun, and every MASt3R-SLAM number below is from the rerun. Old
values and the size of each change are in section 8. EndoDAC numbers are
unaffected. Section 9 adds a post-hoc descriptive sensitivity analysis on
the 143 sequences MASt3R-SLAM predicted completely.

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
`scripts/mast3r_slam_grid_alignment_test.py`,
`scripts/pipeline2_complete_sequence_sensitivity.py`,
`tests/conventions/test_mast3r_grid_mapping.py`. Results (not committed):
`results/pipeline2_eval/`; the superseded corner-aligned outputs are kept
in `results/pipeline2_eval_corner_aligned/`. No file under `src/eval/`,
`src/gt/`, `src/eval_ext/region_iou.py`, `docs/success_criteria.md` or
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

### 1.2 Grid mapping and MASt3R-SLAM adapter check (`c1_cecum_t1_v1`, no metric): PASS

**Grid mapping: pixel-center aligned.** The code MASt3R-SLAM executes for
an RGB folder (repository `scratch/pipelines/MASt3R-SLAM`, commit
e6f4e3d; Pillow 11.3.0):

- `mast3r_slam/dataloader.py:43-44, 50`: `img = cv2.imread(self.rgb_files[idx])`,
  `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`, `return img.astype(self.dtype) / 255.0`.
  No resize here.
- `main.py:253, 263` (and this project's
  `scripts/mast3r_slam_run_perframe.py:242, 249`):
  `timestamp, img = dataset[i]`,
  `frame = create_frame(i, img, T_WC, img_size=dataset.img_size, device=device)`.
- `mast3r_slam/frame.py:112`: `img = resize_img(img, img_size)`.
- `mast3r_slam/mast3r_utils.py:247, 254, 261, 264`:
  `img = PIL.Image.fromarray(np.uint8(img * 255))`,
  `img = _resize_pil_image(img, size)`,
  `halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8`,
  `img = img.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))`.
- `thirdparty/mast3r/dust3r/dust3r/utils/image.py:66, 69-70` (DUSt3R):
  `interp = PIL.Image.LANCZOS`,
  `new_size = tuple(int(round(x*long_edge_size/S)) for x in img.size)`,
  `return img.resize(new_size, interp)`.

The only resampling step is `PIL.Image.resize` with no `box` argument
(1350x1080 -> 512x410), followed by an integer crop (rows 5..404). PIL's
resize maps output pixel center `r + 0.5` to input coordinate
`(r + 0.5) * scale`, which is the pixel-center convention
`source = (dst + 0.5) * scale - 0.5`.

Direct test (`scripts/mast3r_slam_grid_alignment_test.py`, CPU, seed
20261001, `results/pipelines/mast3r_slam_grid_alignment/grid_alignment_test.json`):
a synthetic 1350x1080 PNG with 63 Gaussian markers (sigma 8 px) at known
sub-pixel positions, loaded through MASt3R-SLAM's own `RGBFiles` dataset
and `create_frame`; markers located on the 512x400 `frame.uimg` by
intensity-weighted centroid. Residual = located - predicted, grid pixels:

| formula | mean dx | mean dy | std dx / dy | max abs dx / dy |
|---|---|---|---|---|
| center aligned, `px = (ox + 0.5) / scale_w - 0.5 - half_crop_w` | +0.0000 | -0.0004 | 0.0031 / 0.0024 | 0.0098 / 0.0068 |
| corner aligned, `px = ox / scale_w - half_crop_w` (first version) | -0.3104 | -0.3106 | 0.0031 / 0.0024 | 0.3181 / 0.3156 |

The two formulas differ by `0.5 / scale - 0.5` = -0.3104 (x), -0.3102 (y)
grid pixels, 0.82 original pixels; the measured corner-aligned residual
equals that offset. The evidence is not ambiguous. The adapter's
`original_to_model` now uses the center-aligned formula (the test script
asserts the adapter function against the located markers, tolerance 0.02
grid px). `scripts/mast3r_slam_resolution_mapping.py` holds the same
formula.

**Stage 2 round-trip test, rerun** (`scripts/mast3r_slam_resolution_mapping.py
--round-trip-only`, `logs/mast3r_slam_round_trip_center_aligned.log`):
2,000 random points + 5 named points, max error 1.6e-13 px. This test
applies the forward formula and then its own inverse; it passed with the
corner-aligned pair too and cannot detect an alignment error. The marker
test above is the check that can.

**Adapter check, rerun** (`scripts/pipeline2_preflight.py adapter-check`,
`results/pipeline2_eval/preflight/gate2_adapter_check.json`):

| item | value |
|---|---|
| GT frames | 218 |
| frames with pose / with depth / with both | 218 / 218 / 218 |
| missing frames (pose / depth / either) | 0 / 0 / 0 |
| valid (non-vignette) pixels | 1,355,951 |
| valid pixels with no depth | 27,926 (**2.060%**): 13,380 above the grid (top crop), 13,389 below it, 577 left of the first grid column center, 580 right of the last (first version: 28,898, 2.131%) |
| round trip: full-res depth vs independent scalar bilinear of the 512x400 grid at the mapped coordinates, 10 random valid pixels (seed 20260930) x 2 frames (0, 129) | max abs diff **2.2e-16** (native units) |
| unavailable pixels all NaN / available pixels all finite | yes / yes |
| adapter mapping formula vs `scripts/mast3r_slam_resolution_mapping.py::original_to_model` (1,000 random points) | max abs diff 0.0 |

The 2.060% is identical for every sequence (fixed 1350x1080 input;
measured min = max over 169 sequences).

Bilinear rule (unchanged): a 1350x1080 pixel has depth iff its mapped grid
coordinate lies in [0, 511] x [0, 399]; then its four bilinear neighbours
are grid cells holding a prediction. Outside that box the pixel is
unavailable (never extrapolated, never clamped). With the center-aligned
formula the first and last original columns fall outside it (px = -0.31
and 511.31). Unit tests: `tests/conventions/test_mast3r_grid_mapping.py`
(5 tests, including a PIL marker test of the formula; full suite 65/65
pass).

Pose convention check (descriptive, not a metric; poses do not depend on
the grid mapping): after position-only
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
three (`logs/pipeline2_rerun_gpu_status.log`: GPU 4, 48.5 GB free, 0%
utilized at launch), with the corrected mapping (code as of commit
ac03529; manifests record 22ee00f, the head when each sequence finished,
whose later commits touch neither the driver nor the adapter).
**169/169 sequences status ok, 0 failures.** Wall time 7 h 27 min
(2026-10-01T13:27:05Z to 20:54:12Z); summed per-sequence compute 21.8 h.
Logs: `logs/pipeline2_rerun_center_mast3r_shard{0,1,2}.log`. The first
(corner-aligned) run, 169/169 ok, 8 h 55 min, is archived in
`results/pipeline2_eval_corner_aligned/`.

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
| valid pixels with no depth, frames with depth | 2.060% in every sequence |
| valid pixels with no depth, all GT frames, pooled | 2.060% |

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
0.5997 / 0.9436 / 0.7024 / 0.8756.

Corpus-wide segment_intersects_mesh fraction (all configs and taus
pooled): 3.89%.

### Corpus table, tau = 0.25 (point estimate, mesh-level 95% CI)

| config | recall m+l @50% | area_fraction | calibration (raw / ignore-excl.) | false_reassurance | false_alarm | loc_err median mm (n) | segment_intersect |
|---|---|---|---|---|---|---|---|
| oracle | 1.0000 [1.0000, 1.0000] (n=719, 103 meshes) | 0.2628 [0.2427, 0.2829] | 1.154 / 0.999 | 0.00150 [0.00125, 0.00176] | 0.0006 [0.0003, 0.0009] | 0.008 (n=957) | 0.0021 |
| pred_depth_only | 1.0000 [1.0000, 1.0000] (n=719, 103 meshes) | 0.4909 [0.4610, 0.5199] | 2.156 / 2.001 | 0.00020 [0.00014, 0.00027] | 0.5003 [0.4735, 0.5262] | 15.750 (n=957) | 0.0334 |
| pred_pose_only | 0.9054 [0.8738, 0.9340] (n=719, 103 meshes) | 0.3346 [0.3023, 0.3697] | 1.470 / 1.357 | 0.11357 [0.09425, 0.13531] | 0.3466 [0.2986, 0.3943] | 5.964 (n=946) | 0.0772 |
| fully_predicted | 0.9583 [0.9385, 0.9763] (n=719, 103 meshes) | 0.5703 [0.5441, 0.5969] | 2.504 / 2.362 | 0.05189 [0.04027, 0.06530] | 0.5986 [0.5733, 0.6250] | 19.155 (n=944) | 0.0477 |

### All taus: recall (m+l, 50%), area fraction, false reassurance, false alarm (point estimates)

| config | tau | recall m+l | recall small | recall large | area_fraction | false_reassurance | false_alarm |
|---|---|---|---|---|---|---|---|
| oracle | 0.15 | 1.0000 | 1.0000 | 1.0000 | 0.2629 | 0.00142 | 0.0008 |
| oracle | 0.25 | 1.0000 | 1.0000 | 1.0000 | 0.2628 | 0.00150 | 0.0006 |
| oracle | 0.35 | 1.0000 | 1.0000 | 1.0000 | 0.2627 | 0.00152 | 0.0004 |
| oracle | 0.5 | 1.0000 | 1.0000 | 1.0000 | 0.2627 | 0.00154 | 0.0003 |
| pred_depth_only | 0.15 | 1.0000 | 1.0000 | 1.0000 | 0.5410 | 0.00012 | 0.5498 |
| pred_depth_only | 0.25 | 1.0000 | 1.0000 | 1.0000 | 0.4909 | 0.00020 | 0.5003 |
| pred_depth_only | 0.35 | 1.0000 | 1.0000 | 1.0000 | 0.4429 | 0.00029 | 0.4415 |
| pred_depth_only | 0.5 | 1.0000 | 1.0000 | 1.0000 | 0.3687 | 0.00051 | 0.3173 |
| pred_pose_only | 0.15 | 0.9235 | 0.8319 | 0.9706 | 0.3743 | 0.09414 | 0.4053 |
| pred_pose_only | 0.25 | 0.9054 | 0.7605 | 0.9548 | 0.3346 | 0.11357 | 0.3466 |
| pred_pose_only | 0.35 | 0.8915 | 0.7395 | 0.9434 | 0.3125 | 0.12637 | 0.3085 |
| pred_pose_only | 0.5 | 0.8818 | 0.7059 | 0.9344 | 0.2904 | 0.13859 | 0.2644 |
| fully_predicted | 0.15 | 0.9708 | 0.9076 | 0.9864 | 0.6283 | 0.03947 | 0.6325 |
| fully_predicted | 0.25 | 0.9583 | 0.9034 | 0.9774 | 0.5703 | 0.05189 | 0.5986 |
| fully_predicted | 0.35 | 0.9513 | 0.8908 | 0.9706 | 0.5063 | 0.06573 | 0.5517 |
| fully_predicted | 0.5 | 0.9221 | 0.8571 | 0.9434 | 0.4051 | 0.09078 | 0.4472 |

### Sensitivity intervals, tau = 0.25: (Colon, Segment) and (Colon, Segment, Phantom Number)

| config | metric | mesh CI | (Colon, Segment) CI | (Colon, Segment, Phantom) CI |
|---|---|---|---|---|
| oracle | recall m+l | [1.0000, 1.0000] | [1.0000, 1.0000] | [1.0000, 1.0000] |
| oracle | false_reassurance_rate | [0.00125, 0.00176] | [0.00106, 0.00206] | [0.00121, 0.00180] |
| oracle | false_alarm_rate | [0.00031, 0.00091] | [0.00029, 0.00094] | [0.00031, 0.00089] |
| oracle | area_fraction | [0.24267, 0.28289] | [0.22032, 0.30508] | [0.23882, 0.28698] |
| pred_depth_only | recall m+l | [1.0000, 1.0000] | [1.0000, 1.0000] | [1.0000, 1.0000] |
| pred_depth_only | false_reassurance_rate | [0.00014, 0.00027] | [0.00011, 0.00031] | [0.00013, 0.00028] |
| pred_depth_only | false_alarm_rate | [0.47347, 0.52617] | [0.44064, 0.55297] | [0.46914, 0.53092] |
| pred_depth_only | area_fraction | [0.46097, 0.51985] | [0.42356, 0.55129] | [0.45492, 0.52574] |
| pred_pose_only | recall m+l | [0.8738, 0.9340] | [0.8743, 0.9353] | [0.8750, 0.9342] |
| pred_pose_only | false_reassurance_rate | [0.09425, 0.13531] | [0.08534, 0.15335] | [0.09169, 0.14016] |
| pred_pose_only | false_alarm_rate | [0.29856, 0.39434] | [0.28058, 0.40345] | [0.29608, 0.39712] |
| pred_pose_only | area_fraction | [0.30235, 0.36967] | [0.27618, 0.39745] | [0.29729, 0.37280] |
| fully_predicted | recall m+l | [0.9385, 0.9763] | [0.9362, 0.9797] | [0.9373, 0.9775] |
| fully_predicted | false_reassurance_rate | [0.04027, 0.06530] | [0.03271, 0.07638] | [0.03888, 0.06716] |
| fully_predicted | false_alarm_rate | [0.57326, 0.62503] | [0.55417, 0.64725] | [0.56927, 0.62762] |
| fully_predicted | area_fraction | [0.54411, 0.59692] | [0.52442, 0.61565] | [0.53902, 0.59979] |

### Diagnostic b (over 169 sequences: mean / median / max)

| config | ray_miss_frac | d_pred_unavailable_frac_of_evaluable |
|---|---|---|
| oracle | 0.0164 / 0.0106 / 0.0943 (c2_transverse2_t2_v1) | 0.0000 / 0.0000 / 0.0010 |
| pred_depth_only | 0.0164 / 0.0106 / 0.0943 (c2_transverse2_t2_v1) | 0.0211 / 0.0214 / 0.0226 |
| pred_pose_only | 0.0332 / 0.0194 / 0.8998 (c2_rectum_t4_v1) | 0.0205 / 0.0205 / 0.0733 |
| fully_predicted | 0.0332 / 0.0194 / 0.8998 (c2_rectum_t4_v1) | 0.0213 / 0.0212 / 0.0498 |

### Diagnostic c (pooled faces)

| tau | only-unobserved under fully_predicted, not pred_pose_only | also unobserved under pred_depth_only | fraction |
|---|---|---|---|
| 0.15 | 19,799,074 | 13,258,450 | 0.670 |
| 0.25 | 17,901,952 | 11,688,960 | 0.653 |
| 0.35 | 14,800,175 | 9,571,345 | 0.647 |
| 0.5 | 9,266,309 | 5,669,120 | 0.612 |

Notes on diagnostic b: in pred_pose_only, "d_pred" is GT depth, so its
unavailable fraction counts GT-invalid pixels hit under the predicted pose
(as in D1 Stage B). In pred_depth_only and fully_predicted it is
dominated by the 2.06% crop. ray_miss_frac in the predicted-pose
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
| pred_depth_only | 0.25 | 0.3583 [0.3253, 0.3914] | 0.2420 | 0.4348 (0.4321-0.4379) | 0.4702 | -0.0766 [-0.1195, -0.0310] | [-0.1468, -0.0020] | [-0.1262, -0.0243] | 0.5003 (0.7450) | 0.4909 | 719 / 103 |
| pred_pose_only | 0.25 | 0.4829 [0.4448, 0.5226] | 0.5271 | 0.3142 (0.3132-0.3151) | 0.3123 | +0.1687 [0.1217, 0.2186] | [0.0857, 0.2642] | [0.1165, 0.2278] | 0.3466 (0.7403) | 0.3346 | 719 / 103 |
| fully_predicted | 0.25 | 0.2631 [0.2444, 0.2827] | 0.1696 | 0.4141 (0.4103-0.4184) | 0.4792 | -0.1510 [-0.1884, -0.1108] | [-0.2029, -0.0885] | [-0.1904, -0.1094] | 0.5986 (0.7527) | 0.5703 | 719 / 103 |
| fully_predicted | 0.15 | 0.2283 [0.2094, 0.2478] | 0.1395 | 0.3629 (0.3575-0.3674) | 0.4562 | -0.1346 [-0.1778, -0.0908] | [-0.1993, -0.0654] | [-0.1819, -0.0877] | 0.6325 (0.7549) | 0.6283 | 719 / 103 |
| fully_predicted | 0.35 | 0.3058 [0.2833, 0.3287] | 0.2191 | 0.4415 (0.4397-0.4431) | 0.4766 | -0.1357 [-0.1699, -0.0999] | [-0.1836, -0.0822] | [-0.1725, -0.0963] | 0.5517 (0.7494) | 0.5063 | 719 / 103 |
| fully_predicted | 0.5 | 0.3779 [0.3503, 0.4055] | 0.3180 | 0.3874 (0.3865-0.3883) | 0.4151 | -0.0095 [-0.0482, 0.0310] | [-0.0633, 0.0571] | [-0.0515, 0.0359] | 0.4472 (0.7452) | 0.4051 | 719 / 103 |

#### mast3r_slam: by size class, fully_predicted tau = 0.25

| size class | IoU mean | random mean | paired diff [mesh CI] | n |
|---|---|---|---|---|
| small | 0.2312 | 0.3504 | -0.1192 [-0.1747, -0.0595] | 238 |
| medium | 0.1798 | 0.4027 | -0.2230 [-0.2713, -0.1713] | 277 |
| large | 0.3153 | 0.4211 | -0.1059 [-0.1471, -0.0634] | 442 |
| medium_plus_large | 0.2631 | 0.4141 | -0.1510 [-0.1884, -0.1108] | 719 |

### Position relative to the baseline, at both cluster levels

Reporting rule (`docs/eval_protocol.md` lines 448-463) applied with both
intervals: a direction ("above" / "below" its baseline) is stated only
when the mesh-level CI (103 meshes) and the (Colon, Segment) CI (15 molds)
of the paired difference both exclude zero on the same side. Where they
disagree, that is stated and no direction is declared. Medium + large
regions, 719 regions, 103 meshes, 113 trajectories, 15 molds in every row.

| pipeline | config | tau | paired diff | mesh CI (103 meshes) | (Colon, Segment) CI (15 molds) | statement |
|---|---|---|---|---|---|---|
| endodac | oracle | 0.25 | +0.7269 | [0.7086, 0.7455] | [0.6951, 0.7657] | above its baseline at both levels |
| endodac | pred_depth_only | 0.25 | -0.0851 | [-0.1251, -0.0413] | [-0.1528, -0.0050] | below its baseline at both levels |
| endodac | pred_pose_only | 0.25 | +0.1057 | [0.0562, 0.1561] | [0.0353, 0.1852] | above its baseline at both levels |
| endodac | fully_predicted | 0.25 | -0.1409 | [-0.1811, -0.0980] | [-0.1972, -0.0746] | below its baseline at both levels |
| endodac | fully_predicted | 0.15 | -0.1661 | [-0.2047, -0.1269] | [-0.2160, -0.1120] | below its baseline at both levels |
| endodac | fully_predicted | 0.35 | -0.1004 | [-0.1431, -0.0550] | [-0.1640, -0.0252] | below its baseline at both levels |
| endodac | fully_predicted | 0.5 | +0.0576 | [0.0086, 0.1065] | [-0.0130, 0.1427] | levels disagree: mesh CI is above zero, (Colon, Segment) CI includes zero; no direction declared |
| mast3r_slam | oracle | 0.25 | +0.7269 | [0.7085, 0.7461] | [0.6946, 0.7648] | above its baseline at both levels |
| mast3r_slam | pred_depth_only | 0.25 | -0.0766 | [-0.1195, -0.0310] | [-0.1468, -0.0020] | below its baseline at both levels |
| mast3r_slam | pred_pose_only | 0.25 | +0.1687 | [0.1217, 0.2186] | [0.0857, 0.2642] | above its baseline at both levels |
| mast3r_slam | fully_predicted | 0.25 | -0.1510 | [-0.1884, -0.1108] | [-0.2029, -0.0885] | below its baseline at both levels |
| mast3r_slam | fully_predicted | 0.15 | -0.1346 | [-0.1778, -0.0908] | [-0.1993, -0.0654] | below its baseline at both levels |
| mast3r_slam | fully_predicted | 0.35 | -0.1357 | [-0.1699, -0.0999] | [-0.1836, -0.0822] | below its baseline at both levels |
| mast3r_slam | fully_predicted | 0.5 | -0.0095 | [-0.0482, 0.0310] | [-0.0633, 0.0571] | not distinguishable from its baseline at either level |

In words:
- Below the baseline at both levels: both pipelines at pred_depth_only
  0.25 and at fully_predicted 0.15, 0.25, 0.35.
- Above the baseline at both levels: both pipelines at pred_pose_only
  0.25 and at oracle.
- **Levels disagree: EndoDAC fully_predicted at tau = 0.50.** The
  mesh-level CI [0.009, 0.107] excludes zero; the (Colon, Segment) CI
  [-0.013, 0.143] includes it. No direction is declared for this cell.
- Not distinguishable from the baseline at either level: MASt3R-SLAM
  fully_predicted at tau = 0.50 (mesh CI [-0.048, 0.031], (Colon,
  Segment) CI [-0.063, 0.057]).
- By size class (tables above) only the mesh-level CI was computed; those
  rows carry no (Colon, Segment) interval and no direction is declared
  from them here.

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

| tau | role | A | B | A - B | mesh CI | (C,S) CI | (C,S,P) CI | mesh CI excludes zero in the stated direction | (C,S) CI does |
|---|---|---|---|---|---|---|---|---|---|
| 0.15 | alongside | 0.03947 | 0.09414 | -0.05467 | [-0.06812, -0.04293] | [-0.07438, -0.03881] | [-0.07020, -0.04171] | yes | yes |
| 0.25 | **primary** | 0.05189 | 0.11357 | -0.06168 | [-0.07511, -0.04962] | [-0.08055, -0.04642] | [-0.07799, -0.04799] | yes | yes |
| 0.35 | alongside | 0.06573 | 0.12637 | -0.06064 | [-0.07317, -0.04896] | [-0.07817, -0.04552] | [-0.07636, -0.04756] | yes | yes |
| 0.5 | alongside | 0.09078 | 0.13859 | -0.04781 | [-0.05962, -0.03614] | [-0.06593, -0.02903] | [-0.06058, -0.03537] | yes | yes |

**H6** (false alarm, A - B > 0; A = pred_depth_only, B = pred_pose_only)

| tau | role | A | B | A - B | mesh CI | (C,S) CI | (C,S,P) CI | mesh CI excludes zero in the stated direction | (C,S) CI does |
|---|---|---|---|---|---|---|---|---|---|
| 0.15 | alongside | 0.54978 | 0.40533 | +0.14445 | [0.09920, 0.18982] | [0.07836, 0.21072] | [0.09891, 0.19063] | yes | yes |
| 0.25 | **primary** | 0.50028 | 0.34662 | +0.15366 | [0.10109, 0.20519] | [0.07354, 0.23075] | [0.09920, 0.20738] | yes | yes |
| 0.35 | alongside | 0.44148 | 0.30846 | +0.13303 | [0.07719, 0.18733] | [0.04915, 0.21439] | [0.07496, 0.18928] | yes | yes |
| 0.5 | alongside | 0.31732 | 0.26436 | +0.05296 | [-0.00360, 0.10617] | [-0.03010, 0.13134] | [-0.00662, 0.10963] | no | no |

**Result, per the 2026-10-01 clarification** (`docs/success_criteria.md`
section 6, lines 449-467: "a criterion or hypothesis that names no tau is
judged at the protocol's primary value, tau = 0.25 [...] The other three
tau values are reported alongside as sensitivity, always, including any
value at which the criterion is not met"). The clarification was written
after the first version of these results was seen, as it states itself.

- **H5 holds** at the primary tau = 0.25: false reassurance is lower
  under fully_predicted than under pred_pose_only, difference -0.0617,
  mesh-level CI [-0.0751, -0.0496] (103 meshes), (Colon, Segment) CI
  [-0.0806, -0.0464] (15 molds). Alongside: the mesh-level CI excludes
  zero at tau = 0.15, 0.35 and 0.50 as well; so does the (Colon, Segment)
  CI at every tau.
- **H6 holds** at the primary tau = 0.25: false alarm is higher under
  pred_depth_only than under pred_pose_only, difference +0.1537,
  mesh-level CI [0.1011, 0.2052], (Colon, Segment) CI [0.0735, 0.2308].
  Alongside: the criterion is also met at tau = 0.15 and 0.35, and is
  **not met at tau = 0.50** (difference +0.0530, mesh-level CI [-0.0036,
  0.1062], (Colon, Segment) CI [-0.0301, 0.1313]).

The grid-mapping correction changed neither outcome at any tau (section
8).

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
| pred_depth_only | 0.25 | false_reassurance_rate | 0.00020 | 0.00043 | -0.00024 | [-0.00034, -0.00015] | [-0.00040, -0.00011] | [-0.00035, -0.00014] |
| pred_depth_only | 0.25 | false_alarm_rate | 0.50028 | 0.49817 | +0.00211 | [-0.01075, 0.01496] | [-0.01564, 0.02100] | [-0.01238, 0.01655] |
| pred_depth_only | 0.25 | area_fraction | 0.49087 | 0.48885 | +0.00202 | [-0.00976, 0.01391] | [-0.01417, 0.01932] | [-0.01115, 0.01499] |
| pred_depth_only | 0.25 | recall_medium_plus_large_at_50pct | 1.00000 | 1.00000 | +0.00000 | [0.00000, 0.00000] | [0.00000, 0.00000] | [0.00000, 0.00000] |
| pred_depth_only | 0.25 | region_iou_mean_medium_plus_large | 0.35827 | 0.35178 | +0.00649 | [-0.01093, 0.02377] | [-0.00830, 0.02198] | [-0.01167, 0.02417] |
| pred_pose_only | 0.25 | false_reassurance_rate | 0.11357 | 0.05517 | +0.05840 | [0.04227, 0.07557] | [0.03787, 0.08344] | [0.04122, 0.07806] |
| pred_pose_only | 0.25 | false_alarm_rate | 0.34662 | 0.36475 | -0.01813 | [-0.07065, 0.03371] | [-0.09282, 0.04772] | [-0.07167, 0.03422] |
| pred_pose_only | 0.25 | area_fraction | 0.33464 | 0.36870 | -0.03406 | [-0.06164, -0.00469] | [-0.07653, 0.00716] | [-0.06391, -0.00222] |
| pred_pose_only | 0.25 | recall_medium_plus_large_at_50pct | 0.90542 | 0.96523 | -0.05981 | [-0.08976, -0.03220] | [-0.08716, -0.03549] | [-0.08832, -0.03319] |
| pred_pose_only | 0.25 | region_iou_mean_medium_plus_large | 0.48289 | 0.46223 | +0.02066 | [-0.02269, 0.06596] | [-0.03578, 0.08271] | [-0.02067, 0.06262] |
| fully_predicted | 0.25 | false_reassurance_rate | 0.05189 | 0.03074 | +0.02115 | [0.01053, 0.03277] | [0.00943, 0.03503] | [0.01025, 0.03339] |
| fully_predicted | 0.25 | false_alarm_rate | 0.59856 | 0.54276 | +0.05580 | [0.03786, 0.07372] | [0.02702, 0.08627] | [0.03696, 0.07629] |
| fully_predicted | 0.25 | area_fraction | 0.57025 | 0.51673 | +0.05353 | [0.03240, 0.07477] | [0.02187, 0.08583] | [0.03093, 0.07655] |
| fully_predicted | 0.25 | recall_medium_plus_large_at_50pct | 0.95828 | 0.98331 | -0.02503 | [-0.04447, -0.00769] | [-0.04209, -0.00703] | [-0.04386, -0.00705] |
| fully_predicted | 0.25 | region_iou_mean_medium_plus_large | 0.26307 | 0.29079 | -0.02772 | [-0.05244, -0.00380] | [-0.06090, 0.00140] | [-0.05633, -0.00211] |
| fully_predicted | 0.15 | false_reassurance_rate | 0.03947 | 0.02347 | +0.01601 | [0.00678, 0.02597] | [0.00674, 0.02619] | [0.00624, 0.02651] |
| fully_predicted | 0.15 | false_alarm_rate | 0.63250 | 0.58817 | +0.04433 | [0.03072, 0.05827] | [0.02200, 0.06739] | [0.02929, 0.06069] |
| fully_predicted | 0.15 | area_fraction | 0.62832 | 0.57421 | +0.05411 | [0.03517, 0.07422] | [0.02373, 0.08405] | [0.03224, 0.07638] |
| fully_predicted | 0.15 | recall_medium_plus_large_at_50pct | 0.97079 | 0.98748 | -0.01669 | [-0.03253, -0.00150] | [-0.03050, -0.00257] | [-0.03276, -0.00144] |
| fully_predicted | 0.15 | region_iou_mean_medium_plus_large | 0.22830 | 0.24594 | -0.01764 | [-0.03928, 0.00327] | [-0.05055, 0.01054] | [-0.04327, 0.00635] |
| fully_predicted | 0.35 | false_reassurance_rate | 0.06573 | 0.03816 | +0.02757 | [0.01474, 0.04138] | [0.01134, 0.04680] | [0.01439, 0.04260] |
| fully_predicted | 0.35 | false_alarm_rate | 0.55173 | 0.49075 | +0.06097 | [0.03987, 0.08300] | [0.03015, 0.09382] | [0.03789, 0.08452] |
| fully_predicted | 0.35 | area_fraction | 0.50630 | 0.46388 | +0.04243 | [0.02174, 0.06413] | [0.01140, 0.07221] | [0.01923, 0.06543] |
| fully_predicted | 0.35 | recall_medium_plus_large_at_50pct | 0.95132 | 0.98192 | -0.03060 | [-0.05270, -0.01049] | [-0.05122, -0.00990] | [-0.05147, -0.01169] |
| fully_predicted | 0.35 | region_iou_mean_medium_plus_large | 0.30579 | 0.33062 | -0.02484 | [-0.04987, 0.00017] | [-0.05261, 0.00709] | [-0.05304, 0.00322] |
| fully_predicted | 0.5 | false_reassurance_rate | 0.09078 | 0.04953 | +0.04124 | [0.02414, 0.06023] | [0.01618, 0.07402] | [0.02346, 0.06204] |
| fully_predicted | 0.5 | false_alarm_rate | 0.44720 | 0.38584 | +0.06137 | [0.03361, 0.08986] | [0.02044, 0.10154] | [0.03179, 0.09171] |
| fully_predicted | 0.5 | area_fraction | 0.40510 | 0.38557 | +0.01953 | [-0.00058, 0.04082] | [-0.01344, 0.05121] | [-0.00320, 0.04311] |
| fully_predicted | 0.5 | recall_medium_plus_large_at_50pct | 0.92211 | 0.97079 | -0.04868 | [-0.07413, -0.02459] | [-0.07391, -0.02483] | [-0.07347, -0.02532] |
| fully_predicted | 0.5 | region_iou_mean_medium_plus_large | 0.37789 | 0.44160 | -0.06371 | [-0.10024, -0.02756] | [-0.12711, -0.00737] | [-0.10729, -0.02107] |

## 8. Record of the grid-mapping correction: old vs new values

Old = corner-aligned mapping (commit 64e61e1, outputs archived in
`results/pipeline2_eval_corner_aligned/`). New = pixel-center aligned
(this document). Rendered by `scripts/pipeline2_report_tables.py
--old-root results/pipeline2_eval_corner_aligned`.

Unchanged, exactly (difference 0 in every row; these use no predicted
depth): all oracle and pred_pose_only rows at every tau (false
reassurance, false alarm, area fraction, recall) and their region IoU
rows. Missing-frame counts are unchanged (they depend on poses only).
All EndoDAC numbers are unchanged.

Changed (pooled point estimates, 169 sequences, 103 meshes, 113
trajectories):

| quantity | old (corner aligned) | new (center aligned) | new - old |
|---|---|---|---|
| pred_depth_only tau 0.15 false_reassurance_rate | 0.00012 | 0.00012 | -0.00000 |
| pred_depth_only tau 0.15 false_alarm_rate | 0.54991 | 0.54978 | -0.00013 |
| pred_depth_only tau 0.15 area_fraction | 0.54114 | 0.54100 | -0.00014 |
| pred_depth_only tau 0.15 recall m+l @50% | 1.00000 | 1.00000 | +0.00000 |
| pred_depth_only tau 0.25 false_reassurance_rate | 0.00020 | 0.00020 | -0.00000 |
| pred_depth_only tau 0.25 false_alarm_rate | 0.50043 | 0.50028 | -0.00015 |
| pred_depth_only tau 0.25 area_fraction | 0.49101 | 0.49087 | -0.00014 |
| pred_depth_only tau 0.25 recall m+l @50% | 1.00000 | 1.00000 | +0.00000 |
| pred_depth_only tau 0.35 false_reassurance_rate | 0.00029 | 0.00029 | -0.00000 |
| pred_depth_only tau 0.35 false_alarm_rate | 0.44168 | 0.44148 | -0.00020 |
| pred_depth_only tau 0.35 area_fraction | 0.44302 | 0.44287 | -0.00014 |
| pred_depth_only tau 0.35 recall m+l @50% | 1.00000 | 1.00000 | +0.00000 |
| pred_depth_only tau 0.5 false_reassurance_rate | 0.00051 | 0.00051 | -0.00000 |
| pred_depth_only tau 0.5 false_alarm_rate | 0.31763 | 0.31732 | -0.00031 |
| pred_depth_only tau 0.5 area_fraction | 0.36882 | 0.36867 | -0.00015 |
| pred_depth_only tau 0.5 recall m+l @50% | 1.00000 | 1.00000 | +0.00000 |
| fully_predicted tau 0.15 false_reassurance_rate | 0.03932 | 0.03947 | +0.00016 |
| fully_predicted tau 0.15 false_alarm_rate | 0.63253 | 0.63250 | -0.00003 |
| fully_predicted tau 0.15 area_fraction | 0.62847 | 0.62832 | -0.00015 |
| fully_predicted tau 0.15 recall m+l @50% | 0.97079 | 0.97079 | +0.00000 |
| fully_predicted tau 0.25 false_reassurance_rate | 0.05169 | 0.05189 | +0.00020 |
| fully_predicted tau 0.25 false_alarm_rate | 0.59857 | 0.59856 | -0.00001 |
| fully_predicted tau 0.25 area_fraction | 0.57039 | 0.57025 | -0.00014 |
| fully_predicted tau 0.25 recall m+l @50% | 0.95828 | 0.95828 | +0.00000 |
| fully_predicted tau 0.35 false_reassurance_rate | 0.06547 | 0.06573 | +0.00026 |
| fully_predicted tau 0.35 false_alarm_rate | 0.55172 | 0.55173 | +0.00001 |
| fully_predicted tau 0.35 area_fraction | 0.50644 | 0.50630 | -0.00013 |
| fully_predicted tau 0.35 recall m+l @50% | 0.95132 | 0.95132 | +0.00000 |
| fully_predicted tau 0.5 false_reassurance_rate | 0.09048 | 0.09078 | +0.00029 |
| fully_predicted tau 0.5 false_alarm_rate | 0.44725 | 0.44720 | -0.00004 |
| fully_predicted tau 0.5 area_fraction | 0.40526 | 0.40510 | -0.00016 |
| fully_predicted tau 0.5 recall m+l @50% | 0.92350 | 0.92211 | -0.00139 |
| mast3r_slam|pred_depth_only|0.25 IoU mean | 0.35784 | 0.35827 | +0.00043 |
| mast3r_slam|pred_depth_only|0.25 random mean | 0.43486 | 0.43484 | -0.00002 |
| mast3r_slam|pred_depth_only|0.25 paired diff | -0.07702 | -0.07657 | +0.00045 |
| mast3r_slam|fully_predicted|0.25 IoU mean | 0.26303 | 0.26307 | +0.00004 |
| mast3r_slam|fully_predicted|0.25 random mean | 0.41389 | 0.41405 | +0.00016 |
| mast3r_slam|fully_predicted|0.25 paired diff | -0.15086 | -0.15099 | -0.00013 |
| mast3r_slam|fully_predicted|0.15 IoU mean | 0.22742 | 0.22830 | +0.00088 |
| mast3r_slam|fully_predicted|0.15 random mean | 0.36271 | 0.36295 | +0.00024 |
| mast3r_slam|fully_predicted|0.15 paired diff | -0.13529 | -0.13465 | +0.00064 |
| mast3r_slam|fully_predicted|0.35 IoU mean | 0.30472 | 0.30579 | +0.00106 |
| mast3r_slam|fully_predicted|0.35 random mean | 0.44147 | 0.44149 | +0.00002 |
| mast3r_slam|fully_predicted|0.35 paired diff | -0.13674 | -0.13570 | +0.00105 |
| mast3r_slam|fully_predicted|0.5 IoU mean | 0.37737 | 0.37789 | +0.00052 |
| mast3r_slam|fully_predicted|0.5 random mean | 0.38754 | 0.38740 | -0.00013 |
| mast3r_slam|fully_predicted|0.5 paired diff | -0.01017 | -0.00951 | +0.00066 |
| H5 tau 0.15 A - B | -0.05482 | -0.05467 | +0.00016 |
| H5 tau 0.15 mesh CI | [-0.06832, -0.04305] | [-0.06812, -0.04293] | |
| H5 tau 0.25 A - B | -0.06188 | -0.06168 | +0.00020 |
| H5 tau 0.25 mesh CI | [-0.07534, -0.04982] | [-0.07511, -0.04962] | |
| H5 tau 0.35 A - B | -0.06090 | -0.06064 | +0.00026 |
| H5 tau 0.35 mesh CI | [-0.07345, -0.04922] | [-0.07317, -0.04896] | |
| H5 tau 0.5 A - B | -0.04811 | -0.04781 | +0.00029 |
| H5 tau 0.5 mesh CI | [-0.05994, -0.03642] | [-0.05962, -0.03614] | |
| H6 tau 0.15 A - B | 0.14458 | 0.14445 | -0.00013 |
| H6 tau 0.15 mesh CI | [0.09932, 0.18994] | [0.09920, 0.18982] | |
| H6 tau 0.25 A - B | 0.15381 | 0.15366 | -0.00015 |
| H6 tau 0.25 mesh CI | [0.10123, 0.20533] | [0.10109, 0.20519] | |
| H6 tau 0.35 A - B | 0.13322 | 0.13303 | -0.00020 |
| H6 tau 0.35 mesh CI | [0.07734, 0.18754] | [0.07719, 0.18733] | |
| H6 tau 0.5 A - B | 0.05327 | 0.05296 | -0.00031 |
| H6 tau 0.5 mesh CI | [-0.00328, 0.10642] | [-0.00360, 0.10617] | |
| valid pixels with no depth (fraction) | 0.02131 | 0.02060 | -0.00072 |

Largest absolute changes: region recall (m+l, 50%) fully_predicted at
tau = 0.50, -0.0014 (one region of 719 changes detection status); region
IoU mean fully_predicted at tau = 0.35, +0.0011; false reassurance
fully_predicted at tau = 0.50, +0.0003. No baseline statement of section
5 and no H5/H6 outcome of section 6 changed. In section 7, one interval
changed sides of zero: region IoU MASt3R - EndoDAC, fully_predicted at
tau = 0.35, mesh CI [-0.0508, -0.0010] before and [-0.0499, +0.0002] now
(point -0.0259 before, -0.0248 now).

## 9. Complete-sequence sensitivity (post-hoc, descriptive)

**Everything in this section is a post-hoc descriptive sensitivity
analysis. It was not pre-registered, carries no pass/fail, and replaces
no full-corpus number.** It restricts both pipelines to the 143 sequences
on which MASt3R-SLAM predicted every frame (pose and depth), tau = 0.25,
all configurations. Re-aggregated from the saved per-sequence outputs
and region IoU rows; no ray casting.
`scripts/pipeline2_complete_sequence_sensitivity.py`,
`results/pipeline2_eval/complete_sequence_sensitivity/`. Mesh-level
cluster bootstrap, 10,000 replicates, seed 20260925, with the (Colon,
Segment) interval alongside. The "all-169" columns are the values of
sections 4, 5 and 7 (the script checks that its own all-169 point
estimates equal them: max abs difference 1.1e-16).

### 9.1 What the 143 sequences cover

| set | sequences | meshes | trajectories | molds (Colon, Segment) | (Colon, Segment, Phantom) groups |
|---|---|---|---|---|---|
| complete | 143 | 90 | 99 | 15 | 56 |
| incomplete | 26 | 19 | 18 | 8 | 14 |
| all | 169 | 103 | 113 | 15 | 58 |

84 meshes appear only among the complete sequences, 13 only among the
incomplete ones, 6 in both. All 15 molds remain represented in the 143,
unevenly: c2_cecum keeps 4 of 11 sequences, c2_transverse1 6 of 12,
c2_rectum 7 of 12; the eight c1 molds keep 9 to 12 of 11 to 12. The 143
contain 604 medium + large regions (of 719).

### 9.2 Both pipelines on the 143 sequences

**endodac: 143 complete sequences, tau = 0.25 (post-hoc descriptive)**

| config | false_reassurance [mesh CI] ((C,S) CI) | false_alarm [mesh CI] ((C,S) CI) | area_fraction [mesh CI] ((C,S) CI) | all-169: FR / FA / area |
|---|---|---|---|---|
| oracle | 0.00152 [0.00126, 0.00182] ([0.00106, 0.00211]) | 0.0006 [0.0003, 0.0010] ([0.0003, 0.0009]) | 0.2622 [0.2411, 0.2841] ([0.2221, 0.3029]) | 0.00150 / 0.0006 / 0.2628 |
| pred_depth_only | 0.00041 [0.00029, 0.00056] ([0.00021, 0.00070]) | 0.5085 [0.4806, 0.5365] ([0.4488, 0.5633]) | 0.4983 [0.4690, 0.5284] ([0.4388, 0.5502]) | 0.00043 / 0.4982 / 0.4888 |
| pred_pose_only | 0.04932 [0.04179, 0.05766] ([0.04194, 0.05933]) | 0.3602 [0.3252, 0.3950] ([0.3133, 0.4070]) | 0.3686 [0.3397, 0.3983] ([0.3167, 0.4170]) | 0.05517 / 0.3648 / 0.3687 |
| fully_predicted | 0.02467 [0.01856, 0.03176] ([0.01795, 0.03412]) | 0.5465 [0.5205, 0.5723] ([0.4994, 0.5971]) | 0.5240 [0.4945, 0.5545] ([0.4644, 0.5723]) | 0.03074 / 0.5428 / 0.5167 |

**endodac: region IoU (medium + large) vs area-matched random, 143 complete sequences, tau = 0.25 (post-hoc descriptive)**

| config | n regions / meshes / molds | IoU mean [mesh CI] | random mean (seed range) | paired diff [mesh CI] | (C,S) CI | all-169 paired diff [mesh CI] ((C,S) CI) |
|---|---|---|---|---|---|---|
| oracle | 604 / 90 / 15 | 0.9980 [0.9977, 0.9983] | 0.2697 (0.2690-0.2706) | +0.7283 [+0.7098, +0.7468] | [+0.6979, +0.7627] | +0.7269 [+0.7086, +0.7455] ([+0.6951, +0.7657]) |
| pred_depth_only | 604 / 90 / 15 | 0.3508 [0.3187, 0.3840] | 0.4423 (0.4402-0.4442) | -0.0915 [-0.1358, -0.0458] | [-0.1644, -0.0061] | -0.0851 [-0.1251, -0.0413] ([-0.1528, -0.0050]) |
| pred_pose_only | 604 / 90 / 15 | 0.4716 [0.4277, 0.5153] | 0.3576 (0.3563-0.3585) | +0.1141 [+0.0601, +0.1716] | [+0.0428, +0.1951] | +0.1057 [+0.0562, +0.1561] ([+0.0353, +0.1852]) |
| fully_predicted | 604 / 90 / 15 | 0.3013 [0.2697, 0.3349] | 0.4385 (0.4360-0.4421) | -0.1373 [-0.1814, -0.0919] | [-0.1992, -0.0649] | -0.1409 [-0.1811, -0.0980] ([-0.1972, -0.0746]) |

**mast3r_slam: 143 complete sequences, tau = 0.25 (post-hoc descriptive)**

| config | false_reassurance [mesh CI] ((C,S) CI) | false_alarm [mesh CI] ((C,S) CI) | area_fraction [mesh CI] ((C,S) CI) | all-169: FR / FA / area |
|---|---|---|---|---|
| oracle | 0.00152 [0.00126, 0.00183] ([0.00107, 0.00209]) | 0.0006 [0.0003, 0.0010] ([0.0003, 0.0009]) | 0.2622 [0.2409, 0.2842] ([0.2222, 0.3022]) | 0.00150 / 0.0006 / 0.2628 |
| pred_depth_only | 0.00018 [0.00012, 0.00026] ([0.00009, 0.00029]) | 0.5180 [0.4924, 0.5424] ([0.4610, 0.5646]) | 0.5076 [0.4760, 0.5392] ([0.4445, 0.5629]) | 0.00020 / 0.5003 / 0.4909 |
| pred_pose_only | 0.10864 [0.09184, 0.12754] ([0.08519, 0.14107]) | 0.2940 [0.2577, 0.3318] ([0.2387, 0.3415]) | 0.3130 [0.2860, 0.3411] ([0.2628, 0.3654]) | 0.11357 / 0.3466 / 0.3346 |
| fully_predicted | 0.04420 [0.03354, 0.05686] ([0.02677, 0.06712]) | 0.5925 [0.5688, 0.6152] ([0.5498, 0.6337]) | 0.5673 [0.5386, 0.5959] ([0.5178, 0.6155]) | 0.05189 / 0.5986 / 0.5703 |

**mast3r_slam: region IoU (medium + large) vs area-matched random, 143 complete sequences, tau = 0.25 (post-hoc descriptive)**

| config | n regions / meshes / molds | IoU mean [mesh CI] | random mean (seed range) | paired diff [mesh CI] | (C,S) CI | all-169 paired diff [mesh CI] ((C,S) CI) |
|---|---|---|---|---|---|---|
| oracle | 604 / 90 / 15 | 0.9980 [0.9977, 0.9983] | 0.2697 (0.2690-0.2706) | +0.7283 [+0.7098, +0.7465] | [+0.6977, +0.7634] | +0.7269 [+0.7085, +0.7461] ([+0.6946, +0.7648]) |
| pred_depth_only | 604 / 90 / 15 | 0.3513 [0.3176, 0.3865] | 0.4411 (0.4376-0.4447) | -0.0898 [-0.1347, -0.0429] | [-0.1640, -0.0083] | -0.0766 [-0.1195, -0.0310] ([-0.1468, -0.0020]) |
| pred_pose_only | 604 / 90 / 15 | 0.5123 [0.4764, 0.5504] | 0.3198 (0.3188-0.3206) | +0.1925 [+0.1427, +0.2437] | [+0.1066, +0.2877] | +0.1687 [+0.1217, +0.2186] ([+0.0857, +0.2642]) |
| fully_predicted | 604 / 90 / 15 | 0.2709 [0.2506, 0.2922] | 0.4152 (0.4109-0.4200) | -0.1442 [-0.1852, -0.1014] | [-0.2020, -0.0814] | -0.1510 [-0.1884, -0.1108] ([-0.2029, -0.0885]) |

Position relative to the baseline on the 143 (both levels, same rule as
section 5): for both pipelines, pred_depth_only and fully_predicted are
below the baseline at both levels, pred_pose_only and oracle above it at
both levels. Same statements as on all 169 at tau = 0.25.

### 9.3 Paired MASt3R-SLAM minus EndoDAC: 143 next to 169

| config | metric | 143: MASt3R | 143: EndoDAC | 143: diff [mesh CI] | 143: (C,S) CI | 169: diff [mesh CI] | 169: (C,S) CI |
|---|---|---|---|---|---|---|---|
| oracle | false_reassurance_rate | 0.00152 | 0.00152 | +0.00000 [+0.00000, +0.00000] | [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] | [+0.00000, +0.00000] |
| oracle | false_alarm_rate | 0.00057 | 0.00057 | +0.00000 [+0.00000, +0.00000] | [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] | [+0.00000, +0.00000] |
| oracle | area_fraction | 0.26224 | 0.26224 | +0.00000 [+0.00000, +0.00000] | [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] | [+0.00000, +0.00000] |
| oracle | region_iou_mean_medium_plus_large | 0.99799 | 0.99799 | +0.00000 [+0.00000, +0.00000] | [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] | [+0.00000, +0.00000] |
| pred_depth_only | false_reassurance_rate | 0.00018 | 0.00041 | -0.00023 [-0.00034, -0.00015] | [-0.00041, -0.00011] | -0.00024 [-0.00034, -0.00015] | [-0.00040, -0.00011] |
| pred_depth_only | false_alarm_rate | 0.51804 | 0.50854 | +0.00949 [-0.00392, +0.02225] | [-0.00720, +0.02665] | +0.00211 [-0.01075, +0.01496] | [-0.01564, +0.02100] |
| pred_depth_only | area_fraction | 0.50759 | 0.49834 | +0.00925 [-0.00329, +0.02158] | [-0.00720, +0.02558] | +0.00202 [-0.00976, +0.01391] | [-0.01417, +0.01932] |
| pred_depth_only | region_iou_mean_medium_plus_large | 0.35129 | 0.35085 | +0.00044 [-0.01825, +0.01861] | [-0.01371, +0.01602] | +0.00649 [-0.01093, +0.02377] | [-0.00830, +0.02198] |
| pred_pose_only | false_reassurance_rate | 0.10864 | 0.04932 | +0.05932 [+0.04346, +0.07599] | [+0.03933, +0.08525] | +0.05840 [+0.04227, +0.07557] | [+0.03787, +0.08344] |
| pred_pose_only | false_alarm_rate | 0.29401 | 0.36016 | -0.06615 [-0.10564, -0.03045] | [-0.13206, -0.01094] | -0.01813 [-0.07065, +0.03371] | [-0.09282, +0.04772] |
| pred_pose_only | area_fraction | 0.31300 | 0.36861 | -0.05561 [-0.07758, -0.03609] | [-0.09529, -0.02257] | -0.03406 [-0.06164, -0.00469] | [-0.07653, +0.00716] |
| pred_pose_only | region_iou_mean_medium_plus_large | 0.51230 | 0.47164 | +0.04066 [-0.00261, +0.08529] | [-0.01178, +0.09865] | +0.02066 [-0.02269, +0.06596] | [-0.03578, +0.08271] |
| fully_predicted | false_reassurance_rate | 0.04420 | 0.02467 | +0.01953 [+0.00865, +0.03084] | [+0.00577, +0.03522] | +0.02115 [+0.01053, +0.03277] | [+0.00943, +0.03503] |
| fully_predicted | false_alarm_rate | 0.59248 | 0.54650 | +0.04599 [+0.02891, +0.06255] | [+0.01859, +0.07280] | +0.05580 [+0.03786, +0.07372] | [+0.02702, +0.08627] |
| fully_predicted | area_fraction | 0.56733 | 0.52402 | +0.04332 [+0.02297, +0.06267] | [+0.01143, +0.07150] | +0.05353 [+0.03240, +0.07477] | [+0.02187, +0.08583] |
| fully_predicted | region_iou_mean_medium_plus_large | 0.27092 | 0.30126 | -0.03034 [-0.05795, -0.00409] | [-0.07007, +0.00245] | -0.02772 [-0.05244, -0.00380] | [-0.06090, +0.00140] |

Rows where the 143-sequence and 169-sequence results differ in whether an
interval excludes zero:
- pred_pose_only false alarm: 143: -0.066, mesh CI [-0.106, -0.030],
  (Colon, Segment) CI [-0.132, -0.011], both exclude zero; 169: -0.018,
  both intervals include zero.
- pred_pose_only area fraction: 143: -0.056, both intervals exclude zero;
  169: -0.034, mesh CI excludes zero, (Colon, Segment) CI includes it.

Every other row has the same sign and the same interval-vs-zero status at
both levels on 143 and on 169. In particular, fully_predicted false
reassurance (+0.020 on 143, +0.021 on 169), false alarm (+0.046, +0.056)
and area fraction (+0.043, +0.054) keep both intervals above zero;
pred_pose_only false reassurance stays +0.059 (143) vs +0.058 (169);
fully_predicted region IoU stays negative with the mesh CI below zero and
the (Colon, Segment) CI including zero on both sets.

### 9.4 How the 26 incomplete sequences differ (descriptive only)

Stage A covariates from `results/d1/stage_a_summary.csv`. No test is
attached; 26 sequences from 19 meshes and 8 molds. `ate_mm`,
`endpoint_error_frac_of_gt_path` and `depth_scale_relative_iqr` are
EndoDAC's Stage A quantities. The two `mast3r_*` rows are not Stage A
covariates: they are MASt3R-SLAM's own values from this evaluation, and
for incomplete sequences they cover posed frames only. Tag spellings are
the sheet's own ("zizag", "straight  line").

| covariate | level | complete | incomplete |
|---|---|---|---|
| Colon | c1 | 90 (63%) | 3 (12%) |
| Colon | c2 | 53 (37%) | 23 (88%) |
| Segment | ascending | 21 (15%) | 2 (8%) |
| Segment | cecum | 13 (9%) | 9 (35%) |
| Segment | descending | 17 (12%) | 0 (0%) |
| Segment | rectum | 19 (13%) | 5 (19%) |
| Segment | sigmoid | 11 (8%) | 1 (4%) |
| Segment | sigmoid1 | 11 (8%) | 1 (4%) |
| Segment | sigmoid2 | 12 (8%) | 0 (0%) |
| Segment | transverse1 | 18 (13%) | 6 (23%) |
| Segment | transverse2 | 21 (15%) | 2 (8%) |
| physical_segment_id | c1_ascending | 11 (8%) | 0 (0%) |
| physical_segment_id | c1_cecum | 9 (6%) | 2 (8%) |
| physical_segment_id | c1_descending | 11 (8%) | 0 (0%) |
| physical_segment_id | c1_rectum | 12 (8%) | 0 (0%) |
| physical_segment_id | c1_sigmoid1 | 11 (8%) | 1 (4%) |
| physical_segment_id | c1_sigmoid2 | 12 (8%) | 0 (0%) |
| physical_segment_id | c1_transverse1 | 12 (8%) | 0 (0%) |
| physical_segment_id | c1_transverse2 | 12 (8%) | 0 (0%) |
| physical_segment_id | c2_ascending | 10 (7%) | 2 (8%) |
| physical_segment_id | c2_cecum | 4 (3%) | 7 (27%) |
| physical_segment_id | c2_descending | 6 (4%) | 0 (0%) |
| physical_segment_id | c2_rectum | 7 (5%) | 5 (19%) |
| physical_segment_id | c2_sigmoid | 11 (8%) | 1 (4%) |
| physical_segment_id | c2_transverse1 | 6 (4%) | 6 (23%) |
| physical_segment_id | c2_transverse2 | 9 (6%) | 2 (8%) |
| Video Number | v1 | 49 (34%) | 6 (23%) |
| Video Number | v2 | 48 (34%) | 9 (35%) |
| Video Number | v3 | 46 (32%) | 11 (42%) |
| Debris | no | 99 (69%) | 15 (58%) |
| Debris | yes | 44 (31%) | 11 (42%) |
| Edge Enhancement | 1.0 | 8 (6%) | 0 (0%) |
| Edge Enhancement | 2.0 | 50 (35%) | 8 (31%) |
| Edge Enhancement | 3.0 | 84 (59%) | 18 (69%) |
| Edge Enhancement | 5.0 | 1 (1%) | 0 (0%) |
| Open End Visible | no | 35 (24%) | 10 (38%) |
| Open End Visible | yes | 108 (76%) | 16 (62%) |
| Tags (sequences carrying the tag) | debris on lens | 0 (0%) | 5 (19%) |
| Tags (sequences carrying the tag) | exploratory | 10 (7%) | 2 (8%) |
| Tags (sequences carrying the tag) | fast | 16 (11%) | 10 (38%) |
| Tags (sequences carrying the tag) | loop | 18 (13%) | 3 (12%) |
| Tags (sequences carrying the tag) | mirrored path | 3 (2%) | 5 (19%) |
| Tags (sequences carrying the tag) | polyp | 38 (27%) | 10 (38%) |
| Tags (sequences carrying the tag) | saturation | 22 (15%) | 0 (0%) |
| Tags (sequences carrying the tag) | straight  line | 1 (1%) | 0 (0%) |
| Tags (sequences carrying the tag) | straight line | 6 (4%) | 2 (8%) |
| Tags (sequences carrying the tag) | textureless enface | 22 (15%) | 12 (46%) |
| Tags (sequences carrying the tag) | textureless surface | 1 (1%) | 0 (0%) |
| Tags (sequences carrying the tag) | water jet | 8 (6%) | 5 (19%) |
| Tags (sequences carrying the tag) | water on lens | 5 (3%) | 16 (62%) |
| Tags (sequences carrying the tag) | zigzag | 20 (14%) | 4 (15%) |
| Tags (sequences carrying the tag) | zizag | 1 (1%) | 0 (0%) |
| Tags (sequences carrying the tag) | (no tag) | 39 (27%) | 0 (0%) |

| covariate | complete: median (q25-q75) [min, max] | incomplete: median (q25-q75) [min, max] |
|---|---|---|
| Total Frames | 376 (250.5-465) [117, 959] | 427 (313.5-552) [191, 853] |
| Camera Speed | 15 (10-20) [7, 61] | 20 (17-40) [10, 60] |
| Brightness | 0 (-2-4) [-8, 8] | -0.5 (-3-0.75) [-5, 8] |
| gt_path_length_mm | 124.5 (98.56-144.9) [40.64, 348.3] | 171.9 (155.4-245.4) [104.6, 366.2] |
| unobserved_area_total_mm2 | 4135 (3239-6098) [1540, 9185] | 4445 (2910-7236) [854.2, 9063] |
| n_unobserved_components | 86 (52.5-136) [16, 1550] | 84.5 (63.5-111.2) [26, 445] |
| Qualitative Score | 1 (1-3) [1, 3] | 2 (1-2) [1, 3] |
| Quantitative Score | 0.1815 (-0.6216-0.8025) [-3.41, 1.904] | -0.2529 (-0.99-0.1641) [-1.435, 0.9719] |
| ate_mm | 4.575 (2.836-7.132) [0.8682, 13.8] | 10.14 (7.685-13.97) [3.961, 17.26] |
| endpoint_error_frac_of_gt_path | 0.04987 (0.03009-0.08539) [0.005784, 0.1852] | 0.04711 (0.0404-0.07168) [0.01399, 0.1602] |
| depth_scale_relative_iqr | 0.154 (0.1011-0.2326) [0.03527, 0.8905] | 0.3552 (0.1883-0.5865) [0.1004, 0.8941] |
| mast3r_ate_after_alignment_mm | 7.041 (4.95-8.937) [2.03, 18.13] | 6.416 (3.416-8.671) [1.554, 11.19] |
| mast3r_rotation_residual_median_deg | 16.39 (11.72-21.26) [3.916, 143.4] | 29.62 (17.94-50.22) [6.098, 178] |

Largest contrasts: 23 of the 26 are colon c2 (88%, vs 37% of the 143); 16
of 26 carry the tag "water on lens" (62% vs 3%), 12 "textureless enface"
(46% vs 15%), 10 "fast" (38% vs 11%), 5 "debris on lens" (19% vs 0%);
median camera speed 20 vs 15; median GT path length 172 mm vs 125 mm;
EndoDAC's own ATE is also higher on them (median 10.1 mm vs 4.6 mm), as
is its depth-scale relative IQR (0.36 vs 0.15).

## 10. D2b


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
   scores above it (each at both the mesh and the (Colon, Segment) level,
   section 5), and false alarm is about 0.5 to 0.6 wherever
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
   observe nothing. Section 9 (post-hoc, descriptive) is consistent with
   this: on the 143 complete sequences MASt3R-SLAM's pred_pose_only false
   alarm is 0.294 and area fraction 0.313, against 0.347 and 0.335 on all
   169, while EndoDAC's barely move (0.360 vs 0.365; 0.369 vs 0.369), and
   the paired MASt3R - EndoDAC false-alarm difference goes from -0.018
   (intervals include zero) to -0.066 (both intervals exclude zero). It
   is not confirmed: the 26 sequences also differ in content (section
   9.4: mostly colon c2, water on lens, faster camera), so removing them
   changes the sample as well as the missing frames. Would be confirmed
   by ray-casting only the frames MASt3R-SLAM posed for BOTH pipelines on
   the 26 sequences and finding the difference closes; refuted if
   EndoDAC restricted to those same frames still shows much lower false
   alarm than MASt3R-SLAM on them.

   The same section bears on item 2: the pred_pose_only false-reassurance
   excess of MASt3R-SLAM over EndoDAC is +0.059 on the 143 complete
   sequences (+0.058 on 169), so it is not produced by the incomplete
   sequences.

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

1. Two Stage 2/3 descriptive numbers were computed with the corner-aligned
   formula and were not rerun: the MASt3R / EndoDAC ray-direction check
   (median 19.8 / 15.4 deg, frame 0 of one sequence) and Stage 3's
   nearest-neighbour depth-scale descriptives
   (`docs/pipelines/mast3r_slam.md`, correction block of 2026-10-01).
   Neither enters any number in this document.
2. MASt3R-SLAM's relative rotations over 30 frames are about three times
   GT's on c1_cecum_t1_v1 (13.9 vs 4.9 deg median). Not examined on
   other sequences.
3. Interpretations 1, 2 and 4 each name an analysis that was not run;
   interpretation 3 names one beyond section 9.
4. Section 5's by-size-class rows have mesh-level intervals only.

Resolved since the first version: the tau for H5/H6 (author's
clarification of 2026-10-01, section 6); the half-pixel question about
the grid mapping (section 1.2).
