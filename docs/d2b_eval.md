# D2b evaluation: CUT3R on 169 sequences, D2b verdicts, H5/H6 on CUT3R, run-to-run variability

Three pipelines evaluated (EndoDAC, MASt3R-SLAM, CUT3R), all four
configurations x tau {0.15, 0.25, 0.35, 0.50}, 169 registered sequences,
103 meshes, 113 trajectories, 15 (Colon, Segment) molds, 58 (Colon,
Segment, Phantom Number) groups. No sequence excluded.

**Every number below inherits the GT-based scale recovery of
`docs/eval_protocol.md` section 1 (depth scale fit to GT depth, Umeyama
alignment to the GT trajectory): it is an upper bound on what a deployed
system without GT could achieve, not an estimate of deployed performance.**

Code (committed): `scripts/pipeline_adapters.py` (`Cut3rAdapter`,
invalid-depth rule), `scripts/eval_pipeline_sequence.py` (`--pred-root`),
`scripts/pipeline2_region_iou.py` and `scripts/pipeline2_aggregate.py`
(parametrized), `scripts/d2b_aggregate.py`, `scripts/d2b_report_tables.py`,
`scripts/d2b_preflight.py`, `scripts/d2b_invalid_depth_counts.py`,
`scripts/d2b_eval_chain.sh`, `tests/conventions/test_mast3r_grid_mapping.py`
(invalid-depth test; 66/66 pass). Results (not committed):
`results/d2b_eval/`. No file under `src/eval/`, `src/gt/`,
`src/eval_ext/region_iou.py`, `docs/success_criteria.md` or
`docs/eval_protocol.md` was modified.

## Protocol entries applied (quoted)

`docs/eval_protocol.md` lines 196-216, "2026-09-29: Missing predictions,
internal crops, confidence maps":

> ### 2026-09-29: Missing predictions, internal crops, confidence maps (before any metric for a second pipeline)
>
> Written after MASt3R-SLAM full-corpus inference and before any
> evaluation metric was computed on its output.
>
> - A frame for which a pipeline outputs no pose or no depth (for example a
>   tracker's skipped frame) contributes no observations. The sequence
>   stays in the analysis. Missing frames are never filled by
>   interpolation or copied from neighbours. The number of such frames and
>   the number of affected sequences are reported per pipeline.
> - A pixel with no predicted depth (for example outside a pipeline's
>   internal crop) is treated exactly like a pixel whose d_pred is
>   unavailable: it does not enter the tau test and marks nothing
>   observed. The fraction of valid (non-vignette) pixels affected is
>   reported per pipeline.
> - Depth predicted on a pipeline's internal grid is mapped to the
>   1350x1080 input grid by the documented per-pipeline mapping with
>   bilinear interpolation; interpolation never crosses into pixels that
>   have no prediction.
> - Confidence maps are not used, for any pipeline. No confidence
>   threshold, weighting, or masking is applied.

`docs/eval_protocol.md` lines 465-493, "2026-10-02: Numerical run-to-run
variability":

> ### 2026-10-02: Numerical run-to-run variability (before any CUT3R metric)
>
> Written after CUT3R Stage 2 showed that its per-frame outputs change
> substantially under numerically irrelevant perturbations (input noise of
> 1e-6, batch composition, TF32), and before any evaluation metric was
> computed on CUT3R output.
>
> 1. Pinned configuration. Each pipeline's primary run uses one pinned
>    numerical configuration, recorded in its pipeline document: entry
>    point, batch size, precision settings (vendor defaults are kept, not
>    changed), and GPU model. All decision points (D2b) and hypotheses are
>    judged on the primary runs only.
> 2. Variability runs. For every pipeline, K = 5 additional runs, each
>    adding i.i.d. Gaussian noise with standard deviation 1e-6 (on the
>    0-1 image scale) to the input frames, seeds 1 to 5, everything else
>    pinned. Run on a fixed subset: for each (Colon, Segment) mold, the
>    first registered sequence in alphabetical order (15 sequences).
> 3. Reported for each pipeline, configuration fully_predicted,
>    tau = 0.25, on the 15-sequence subset: false reassurance, false
>    alarm, predicted-unobserved area fraction and region IoU (medium +
>    large) for the primary run and each variability run; the standard
>    deviation and range across the six runs, per sequence and pooled.
>    The pooled range is reported next to every between-pipeline
>    difference and next to the D2b thresholds (0.01 for false
>    reassurance, 0.10 for region IoU).
> 4. These runs are descriptive. They do not change any verdict. A D2b
>    endpoint whose primary-run difference is smaller than the pooled
>    run-to-run range of either pipeline involved is reported with that
>    statement attached.

`docs/eval_protocol.md` lines 495-502, "2026-10-03: Invalid predicted
depth":

> ### 2026-10-03: Invalid predicted depth (before any CUT3R metric)
>
> A predicted depth that is non-finite or not strictly positive is treated
> exactly like an unavailable d_pred: the pixel does not enter the tau test
> and marks nothing observed. This applies to every pipeline. The count of
> such pixels is reported per pipeline and per sequence. Written after the
> CUT3R primary run showed 10 such pixels (all in c2_transverse1_t1_v2) and
> before any evaluation metric was computed on CUT3R output.

`docs/eval_protocol.md` lines 504-520, "2026-10-03: Noise injection point
for the variability runs":

> ### 2026-10-03: Noise injection point for the variability runs (clarifies the 2026-10-02 entry)
>
> The 2026-10-02 entry specifies noise of standard deviation 1e-6 "on the
> 0-1 image scale to the input frames". As implemented, before any
> variability metric was computed:
> - MASt3R-SLAM and CUT3R: noise is added to the tensor the network
>   receives, after the loader's resizing and normalization to [-1, 1], at
>   standard deviation 2e-6 (equal to 1e-6 on the 0-1 scale). For
>   MASt3R-SLAM, earlier injection is not possible: its loader converts the
>   image back to 8-bit before resizing, which would erase the noise or flip
>   whole grey levels.
> - EndoDAC: noise is added right after loading, on the 0-1 scale, before
>   the loader's resizing. Resizing averages neighbouring pixels and may
>   attenuate the noise by a small factor relative to the other two
>   pipelines. This is reported next to every EndoDAC variability result,
>   together with a one-sequence check (c1_cecum_t1_v1, seed 1) with the
>   noise added instead to EndoDAC's network input tensor.

Also applied as before: the aggregation rule (lines 360-380) and the
region IoU reporting rule (lines 448-463).

How the invalid-depth rule is implemented: in the adapters
(`scripts/pipeline_adapters.py`), not in locked code. A non-finite or
non-positive predicted depth is `available=False`; for a pipeline with an
internal grid, a full-resolution pixel is available only if all four of
its bilinear neighbours hold a valid depth. Such pixels are passed to
`src/eval/fusion.py` exactly like any other unavailable d_pred.

---

# MEASURED

## 1. Pre-flight

### 1.1 Reproduction: PASS

The driver and adapters used for every run below
(`results/d2b_eval/preflight/preflight1_reproduction.json`):

| check | result |
|---|---|
| EndoDAC `c1_cecum_t1_v1` vs `results/eval_stage4/summary.json` | **980/980 leaves, 0 missing, 0 mismatches, bit-identical** |
| EndoDAC `predicted_observed_packed.bin`, `regions.csv` vs D1 Stage B | byte-identical |
| MASt3R-SLAM `c1_cecum_t1_v1` `metrics.json` vs `results/pipeline2_eval/` | **1192 leaves (timings excluded), 0 missing, 0 mismatches**; new leaves only: `missing_predictions/adapter/invalid_depth_counted_on`, `missing_predictions/adapter/n_frames_with_invalid_depth`, `missing_predictions/adapter/n_invalid_depth_pixels`, `missing_predictions/pred_root` |
| MASt3R-SLAM `predicted_observed_packed.bin`, `regions.csv` | byte-identical |

In aggregation, two further checks: EndoDAC's and MASt3R-SLAM's region IoU
rows recomputed with the parametrized script equal the pipeline 2
evaluation's (281,358 rows, max abs difference 0.0), and EndoDAC
fully_predicted tau = 0.25 random rows reproduce the validity gate's
(19,140 rows, max abs difference 0.0).

### 1.2 CUT3R adapter (no metric): PASS

`scripts/d2b_preflight.py` (`results/d2b_eval/preflight/preflight2_cut3r_adapter.json`):

| item | `c1_cecum_t1_v1` | `c2_transverse1_t1_v2` (the sequence with invalid depth) |
|---|---|---|
| GT frames / with pose / with depth / missing either | 218 / 218 / 218 / 0 | 627 / 627 / 627 / 0 |
| adapter mapping vs the 63 markers located on CUT3R's own grid (Stage 1) | max abs residual 0.016 grid px (corner-aligned formula: mean -0.310) | same |
| grid parameters | 512x400, scale 2.63672 / 2.63415, crop 0 / 5 rows, pixel-center; identical to MASt3R-SLAM's | same |
| valid (non-vignette) pixels with no depth from the crop | 27,926 of 1,355,951 per frame (2.060%) | same |
| invalid predicted depth (grid pixels) | 0 | 10 (frames 192-195) |
| valid full-resolution pixels made unavailable by them | 0 | 75 (over the sequence) |
| full-res depth vs independent scalar bilinear (10 pixels, 2 or 3 frames) | max abs diff 2.2e-16 | 2.2e-16 |
| unavailable pixels NaN / available pixels finite and > 0 | yes / yes | yes / yes |

### 1.3 Invalid predicted depth in the primary outputs, all 169 sequences: PASS

`scripts/d2b_invalid_depth_counts.py`
(`results/d2b_eval/preflight/invalid_depth_counts_<pipeline>.csv`), every
saved depth pixel on each pipeline's own grid:

| pipeline | frames | pixels | non-finite | finite and <= 0 | sequences affected |
|---|---|---|---|---|---|
| EndoDAC | 67,886 | 98,977,788,000 | 0 | 0 | none |
| MASt3R-SLAM | 67,886 | 13,903,052,800 | 0 | 0 | none |
| CUT3R | 67,886 | 13,903,052,800 | 0 | 10 | `c2_transverse1_t1_v2` |

EndoDAC and MASt3R-SLAM have no invalid predicted depth, so their earlier
evaluations are unaffected by the 2026-10-03 rule (and pre-flight 1 shows
the updated adapters reproduce them bit-identically).

### 1.4 EndoDAC noise injection point (2026-10-03 entry): discrepancy with the protocol text

The entry states that for EndoDAC "noise is added right after loading,
on the 0-1 scale, before the loader's resizing" (line 515). The code adds
it after the resize, to the tensor the networks receive
(`scripts/endodac_inference.py:208-212`):

```
resized = im.resize((FEED_W, FEED_H), Image.LANCZOS)
t = torch.from_numpy(np.array(resized)).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
...
t = t + torch.from_numpy((noise_sigma * rng.standard_normal(t.shape)).astype(np.float32))
```

`t` is what `depther(tensor)` (line 226) and the pose encoder (lines
244-245) receive. My earlier description in `docs/noise_sensitivity.md`
("the 320x256 float tensor fed to the networks, 0-1 scale, right after
loading") was accurate about the tensor, and "right after loading" was
misleading. Consequence: the check the entry asks for ("noise added
instead to EndoDAC's network input tensor") is the same code path as the
variability runs. Rerun on `c1_cecum_t1_v1`, seed 1
(`results/pipelines/variability/endodac/seed1_network_input_check/`):
poses and depth **bit-identical** to the seed-1 variability run (max abs
difference 0.0 over all 218 frames). There is no resize attenuation to
report: all three pipelines received the noise on their network input
tensor (std 1e-6 on EndoDAC's 0-1 tensor, 2e-6 on the [-1, 1] tensors of
the other two).

Proposed replacement for the EndoDAC bullet of the 2026-10-03 entry, for
the author to add (not edited here):

> - EndoDAC: noise is added to the tensor the networks receive, after
>   the loader's resizing to 320x256 and scaling to the 0-1 range, at
>   standard deviation 1e-6. All three pipelines therefore receive the
>   noise on their network input tensor. A rerun with the noise added at
>   that point (c1_cecum_t1_v1, seed 1) is bit-identical to the
>   variability run.

## 2. CUT3R main run

`scripts/d2b_eval_chain.sh`: `scripts/eval_pipeline_sequence.py
--pipeline cut3r --all`, six processes on GPU 0 (RTX 6000 Ada;
`logs/d2b_gpu_status.log`: 48.5 GB free, 0% utilized). **169/169
sequences status ok, 0 failures.** Wall time 5 h 02 min
(2026-10-02T20:36:28Z to 2026-10-03T01:38:14Z). Per-sequence checks on all
169: oracle via `src/eval/fusion.py` bit-identical to `src/eval/oracle.py`
at every tau; oracle bits and ignore set identical to D1 Stage B's (0
faces differ).

| quantity | value |
|---|---|
| GT frames | 67,886 |
| frames with no predicted pose / depth | 0 / 0 |
| sequences with every frame predicted | 169 |
| valid pixels with no depth, pooled over all frames | 2.0595% (crop 2.0595% in every sequence; the 10 invalid pixels add 75 full-res pixels in `c2_transverse1_t1_v2`) |
| invalid predicted depth pixels (non-finite or <= 0) | 10, all in `c2_transverse1_t1_v2` (frames 192-195), treated as unavailable |

### 2.1 CUT3R corpus tables (layout of `docs/d1_stage_b.md`)

Pooled point estimates; mesh-level cluster bootstrap, 10,000 replicates,
seed 20260925; 169 sequences, 103 meshes, 113 trajectories, 15 (Colon,
Segment) and 58 (Colon, Segment, Phantom) groups behind every row.
`results/d2b_eval/cut3r_corpus_tables.json`.

### Corpus table, tau = 0.25 (point estimate, mesh-level 95% CI)

| config | recall m+l @50% | area_fraction | calibration (raw / ignore-excl.) | false_reassurance | false_alarm | loc_err median mm (n) | segment_intersect |
|---|---|---|---|---|---|---|---|
| oracle | 1.0000 [1.0000, 1.0000] (n=719, 103 meshes) | 0.2628 [0.2427, 0.2829] | 1.154 / 0.999 | 0.00150 [0.00125, 0.00176] | 0.0006 [0.0003, 0.0009] | 0.008 (n=957) | 0.0021 |
| pred_depth_only | 1.0000 [1.0000, 1.0000] (n=719, 103 meshes) | 0.5277 [0.4961, 0.5576] | 2.318 / 2.163 | 0.00017 [0.00013, 0.00023] | 0.5377 [0.5137, 0.5619] | 18.396 (n=957) | 0.0460 |
| pred_pose_only | 0.9054 [0.8836, 0.9266] (n=719, 103 meshes) | 0.6066 [0.5781, 0.6342] | 2.664 / 2.535 | 0.08845 [0.07521, 0.10350] | 0.6404 [0.6103, 0.6701] | 27.038 (n=927) | 0.1218 |
| fully_predicted | 0.9430 [0.9244, 0.9596] (n=719, 103 meshes) | 0.6561 [0.6297, 0.6819] | 2.881 / 2.744 | 0.07067 [0.05620, 0.08719] | 0.6613 [0.6391, 0.6835] | 26.357 (n=938) | 0.1130 |

### All taus: recall (m+l, 50%), area fraction, false reassurance, false alarm (point estimates)

| config | tau | recall m+l | recall small | recall large | area_fraction | false_reassurance | false_alarm |
|---|---|---|---|---|---|---|---|
| oracle | 0.15 | 1.0000 | 1.0000 | 1.0000 | 0.2629 | 0.00142 | 0.0008 |
| oracle | 0.25 | 1.0000 | 1.0000 | 1.0000 | 0.2628 | 0.00150 | 0.0006 |
| oracle | 0.35 | 1.0000 | 1.0000 | 1.0000 | 0.2627 | 0.00152 | 0.0004 |
| oracle | 0.5 | 1.0000 | 1.0000 | 1.0000 | 0.2627 | 0.00154 | 0.0003 |
| pred_depth_only | 0.15 | 1.0000 | 1.0000 | 1.0000 | 0.5799 | 0.00009 | 0.5820 |
| pred_depth_only | 0.25 | 1.0000 | 1.0000 | 1.0000 | 0.5277 | 0.00017 | 0.5377 |
| pred_depth_only | 0.35 | 1.0000 | 1.0000 | 1.0000 | 0.4783 | 0.00024 | 0.4861 |
| pred_depth_only | 0.5 | 1.0000 | 1.0000 | 1.0000 | 0.4026 | 0.00046 | 0.3804 |
| pred_pose_only | 0.15 | 0.9277 | 0.8067 | 0.9638 | 0.6638 | 0.07116 | 0.6661 |
| pred_pose_only | 0.25 | 0.9054 | 0.7941 | 0.9457 | 0.6066 | 0.08845 | 0.6404 |
| pred_pose_only | 0.35 | 0.8929 | 0.7857 | 0.9367 | 0.5570 | 0.10309 | 0.6135 |
| pred_pose_only | 0.5 | 0.8651 | 0.7605 | 0.9095 | 0.4898 | 0.12467 | 0.5694 |
| fully_predicted | 0.15 | 0.9569 | 0.8866 | 0.9638 | 0.7095 | 0.05732 | 0.6832 |
| fully_predicted | 0.25 | 0.9430 | 0.8655 | 0.9548 | 0.6561 | 0.07067 | 0.6613 |
| fully_predicted | 0.35 | 0.9277 | 0.8361 | 0.9434 | 0.6051 | 0.08222 | 0.6363 |
| fully_predicted | 0.5 | 0.9082 | 0.8067 | 0.9367 | 0.5341 | 0.09699 | 0.5929 |

### Sensitivity intervals, tau = 0.25: (Colon, Segment) and (Colon, Segment, Phantom Number)

| config | metric | mesh CI | (Colon, Segment) CI | (Colon, Segment, Phantom) CI |
|---|---|---|---|---|
| oracle | recall m+l | [1.0000, 1.0000] | [1.0000, 1.0000] | [1.0000, 1.0000] |
| oracle | false_reassurance_rate | [0.00125, 0.00176] | [0.00106, 0.00206] | [0.00121, 0.00180] |
| oracle | false_alarm_rate | [0.00031, 0.00091] | [0.00029, 0.00094] | [0.00031, 0.00089] |
| oracle | area_fraction | [0.24267, 0.28289] | [0.22032, 0.30508] | [0.23882, 0.28698] |
| pred_depth_only | recall m+l | [1.0000, 1.0000] | [1.0000, 1.0000] | [1.0000, 1.0000] |
| pred_depth_only | false_reassurance_rate | [0.00013, 0.00023] | [0.00011, 0.00025] | [0.00012, 0.00023] |
| pred_depth_only | false_alarm_rate | [0.51373, 0.56190] | [0.48840, 0.58413] | [0.50896, 0.56624] |
| pred_depth_only | area_fraction | [0.49607, 0.55762] | [0.45949, 0.59018] | [0.49158, 0.56360] |
| pred_pose_only | recall m+l | [0.8836, 0.9266] | [0.8809, 0.9274] | [0.8850, 0.9254] |
| pred_pose_only | false_reassurance_rate | [0.07521, 0.10350] | [0.06808, 0.11624] | [0.07450, 0.10488] |
| pred_pose_only | false_alarm_rate | [0.61033, 0.67006] | [0.58535, 0.69805] | [0.60505, 0.67464] |
| pred_pose_only | area_fraction | [0.57811, 0.63420] | [0.55708, 0.65000] | [0.57236, 0.63867] |
| fully_predicted | recall m+l | [0.9244, 0.9596] | [0.9201, 0.9634] | [0.9246, 0.9596] |
| fully_predicted | false_reassurance_rate | [0.05620, 0.08719] | [0.04873, 0.09840] | [0.05582, 0.08720] |
| fully_predicted | false_alarm_rate | [0.63911, 0.68354] | [0.62551, 0.70200] | [0.63690, 0.68648] |
| fully_predicted | area_fraction | [0.62975, 0.68187] | [0.59633, 0.71004] | [0.62304, 0.68747] |

### Diagnostic b (over 169 sequences: mean / median / max)

| config | ray_miss_frac | d_pred_unavailable_frac_of_evaluable |
|---|---|---|
| oracle | 0.0164 / 0.0106 / 0.0943 (c2_transverse2_t2_v1) | 0.0000 / 0.0000 / 0.0010 |
| pred_depth_only | 0.0164 / 0.0106 / 0.0943 (c2_transverse2_t2_v1) | 0.0211 / 0.0214 / 0.0226 |
| pred_pose_only | 0.0337 / 0.0196 / 0.3469 (c2_transverse2_t4_v3) | 0.0317 / 0.0386 / 0.1133 |
| fully_predicted | 0.0337 / 0.0196 / 0.3469 (c2_transverse2_t4_v3) | 0.0205 / 0.0207 / 0.0241 |

### Diagnostic c (pooled faces)

| tau | only-unobserved under fully_predicted, not pred_pose_only | also unobserved under pred_depth_only | fraction |
|---|---|---|---|
| 0.15 | 10,082,356 | 5,625,234 | 0.558 |
| 0.25 | 9,757,747 | 5,306,140 | 0.544 |
| 0.35 | 8,930,672 | 4,742,517 | 0.531 |
| 0.5 | 7,152,105 | 3,412,494 | 0.477 |

## 3. Region IoU, three pipelines, with matched random baselines

`src/eval_ext/region_iou.py` unchanged; 20 area-matched random sets per
(pipeline, sequence, cell), seeds and construction of
`scripts/diagnose_d1_3.py`. Rows: `results/d2b_eval/region_iou/`. 719
medium + large regions, 103 meshes, 113 trajectories in every row.
EndoDAC and MASt3R-SLAM rows are identical to the pipeline 2 evaluation's
(section 1.1).

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

### cut3r: region IoU (medium + large) vs area-matched random baseline

| config | tau | IoU mean [mesh CI] | IoU median | random mean, seed-avg (range over 20 seeds) | random median | paired diff (pipeline - random) [mesh CI] | (C,S) CI | (C,S,P) CI | false_alarm (random) | area_fraction | n regions / meshes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| oracle | 0.25 | 0.9979 [0.9975, 0.9983] | 0.9991 | 0.2711 (0.2704-0.2719) | 0.2784 | +0.7269 [0.7084, 0.7457] | [0.6951, 0.7663] | [0.7059, 0.7490] | 0.0006 (0.7311) | 0.2628 | 719 / 103 |
| pred_depth_only | 0.25 | 0.3140 [0.2857, 0.3437] | 0.1917 | 0.4204 (0.4186-0.4223) | 0.4758 | -0.1065 [-0.1467, -0.0661] | [-0.1636, -0.0468] | [-0.1500, -0.0605] | 0.5377 (0.7453) | 0.5277 | 719 / 103 |
| pred_pose_only | 0.25 | 0.1942 [0.1695, 0.2211] | 0.0885 | 0.3313 (0.3292-0.3335) | 0.3823 | -0.1372 [-0.1693, -0.1038] | [-0.1750, -0.1046] | [-0.1699, -0.1028] | 0.6404 (0.7562) | 0.6066 | 719 / 103 |
| fully_predicted | 0.25 | 0.1523 [0.1348, 0.1711] | 0.0749 | 0.2650 (0.2597-0.2707) | 0.2159 | -0.1128 [-0.1508, -0.0757] | [-0.1837, -0.0539] | [-0.1557, -0.0726] | 0.6613 (0.7525) | 0.6561 | 719 / 103 |
| fully_predicted | 0.15 | 0.1311 [0.1167, 0.1470] | 0.0656 | 0.2083 (0.2065-0.2108) | 0.0898 | -0.0772 [-0.1190, -0.0394] | [-0.1535, -0.0183] | [-0.1222, -0.0365] | 0.6832 (0.7548) | 0.7095 | 719 / 103 |
| fully_predicted | 0.35 | 0.1793 [0.1590, 0.2011] | 0.0879 | 0.3307 (0.3245-0.3359) | 0.4049 | -0.1513 [-0.1890, -0.1135] | [-0.2094, -0.0922] | [-0.1926, -0.1114] | 0.6363 (0.7507) | 0.6051 | 719 / 103 |
| fully_predicted | 0.5 | 0.2323 [0.2058, 0.2613] | 0.1206 | 0.4126 (0.4078-0.4159) | 0.4591 | -0.1804 [-0.2165, -0.1415] | [-0.2283, -0.1251] | [-0.2177, -0.1406] | 0.5929 (0.7498) | 0.5341 | 719 / 103 |

#### cut3r: by size class, fully_predicted tau = 0.25

| size class | IoU mean | random mean | paired diff [mesh CI] | n |
|---|---|---|---|---|
| small | 0.1352 | 0.1931 | -0.0579 [-0.1072, -0.0095] | 238 |
| medium | 0.0983 | 0.2449 | -0.1466 [-0.2010, -0.0970] | 277 |
| large | 0.1861 | 0.2776 | -0.0915 [-0.1299, -0.0556] | 442 |
| medium_plus_large | 0.1523 | 0.2650 | -0.1128 [-0.1508, -0.0757] | 719 |

### Position relative to the baseline, both cluster levels

A direction is stated only when the mesh-level CI and the (Colon,
Segment) CI of the paired difference exclude zero on the same side.

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
| cut3r | oracle | 0.25 | +0.7269 | [0.7084, 0.7457] | [0.6951, 0.7663] | above its baseline at both levels |
| cut3r | pred_depth_only | 0.25 | -0.1065 | [-0.1467, -0.0661] | [-0.1636, -0.0468] | below its baseline at both levels |
| cut3r | pred_pose_only | 0.25 | -0.1372 | [-0.1693, -0.1038] | [-0.1750, -0.1046] | below its baseline at both levels |
| cut3r | fully_predicted | 0.25 | -0.1128 | [-0.1508, -0.0757] | [-0.1837, -0.0539] | below its baseline at both levels |
| cut3r | fully_predicted | 0.15 | -0.0772 | [-0.1190, -0.0394] | [-0.1535, -0.0183] | below its baseline at both levels |
| cut3r | fully_predicted | 0.35 | -0.1513 | [-0.1890, -0.1135] | [-0.2094, -0.0922] | below its baseline at both levels |
| cut3r | fully_predicted | 0.5 | -0.1804 | [-0.2165, -0.1415] | [-0.2283, -0.1251] | below its baseline at both levels |

## 4. Three-pipeline summary

Every region IoU value next to its false alarm, area fraction and matched
random baseline (reporting rule, lines 448-463).

| pipeline | config | tau | false reassurance | false alarm | area fraction | recall m+l @50% | region IoU mean | matched random IoU | IoU - random [mesh CI] | (C,S) CI |
|---|---|---|---|---|---|---|---|---|---|---|
| EndoDAC | oracle | 0.25 | 0.00150 | 0.0006 | 0.2628 | 1.0000 | 0.9979 | 0.2711 | +0.7269 [0.7086, 0.7455] | [0.6951, 0.7657] |
| EndoDAC | pred_depth_only | 0.25 | 0.00043 | 0.4982 | 0.4888 | 1.0000 | 0.3518 | 0.4369 | -0.0851 [-0.1251, -0.0413] | [-0.1528, -0.0050] |
| EndoDAC | pred_pose_only | 0.25 | 0.05517 | 0.3648 | 0.3687 | 0.9652 | 0.4622 | 0.3565 | +0.1057 [0.0562, 0.1561] | [0.0353, 0.1852] |
| EndoDAC | fully_predicted | 0.25 | 0.03074 | 0.5428 | 0.5167 | 0.9833 | 0.2908 | 0.4317 | -0.1409 [-0.1811, -0.0980] | [-0.1972, -0.0746] |
| EndoDAC | fully_predicted | 0.15 | 0.02347 | 0.5882 | 0.5742 | 0.9875 | 0.2459 | 0.4121 | -0.1661 [-0.2047, -0.1269] | [-0.2160, -0.1120] |
| EndoDAC | fully_predicted | 0.35 | 0.03816 | 0.4908 | 0.4639 | 0.9819 | 0.3306 | 0.4310 | -0.1004 [-0.1431, -0.0550] | [-0.1640, -0.0252] |
| EndoDAC | fully_predicted | 0.5 | 0.04953 | 0.3858 | 0.3856 | 0.9708 | 0.4416 | 0.3840 | +0.0576 [0.0086, 0.1065] | [-0.0130, 0.1427] |
| MASt3R-SLAM | oracle | 0.25 | 0.00150 | 0.0006 | 0.2628 | 1.0000 | 0.9979 | 0.2711 | +0.7269 [0.7085, 0.7461] | [0.6946, 0.7648] |
| MASt3R-SLAM | pred_depth_only | 0.25 | 0.00020 | 0.5003 | 0.4909 | 1.0000 | 0.3583 | 0.4348 | -0.0766 [-0.1195, -0.0310] | [-0.1468, -0.0020] |
| MASt3R-SLAM | pred_pose_only | 0.25 | 0.11357 | 0.3466 | 0.3346 | 0.9054 | 0.4829 | 0.3142 | +0.1687 [0.1217, 0.2186] | [0.0857, 0.2642] |
| MASt3R-SLAM | fully_predicted | 0.25 | 0.05189 | 0.5986 | 0.5703 | 0.9583 | 0.2631 | 0.4141 | -0.1510 [-0.1884, -0.1108] | [-0.2029, -0.0885] |
| MASt3R-SLAM | fully_predicted | 0.15 | 0.03947 | 0.6325 | 0.6283 | 0.9708 | 0.2283 | 0.3629 | -0.1346 [-0.1778, -0.0908] | [-0.1993, -0.0654] |
| MASt3R-SLAM | fully_predicted | 0.35 | 0.06573 | 0.5517 | 0.5063 | 0.9513 | 0.3058 | 0.4415 | -0.1357 [-0.1699, -0.0999] | [-0.1836, -0.0822] |
| MASt3R-SLAM | fully_predicted | 0.5 | 0.09078 | 0.4472 | 0.4051 | 0.9221 | 0.3779 | 0.3874 | -0.0095 [-0.0482, 0.0310] | [-0.0633, 0.0571] |
| CUT3R | oracle | 0.25 | 0.00150 | 0.0006 | 0.2628 | 1.0000 | 0.9979 | 0.2711 | +0.7269 [0.7084, 0.7457] | [0.6951, 0.7663] |
| CUT3R | pred_depth_only | 0.25 | 0.00017 | 0.5377 | 0.5277 | 1.0000 | 0.3140 | 0.4204 | -0.1065 [-0.1467, -0.0661] | [-0.1636, -0.0468] |
| CUT3R | pred_pose_only | 0.25 | 0.08845 | 0.6404 | 0.6066 | 0.9054 | 0.1942 | 0.3313 | -0.1372 [-0.1693, -0.1038] | [-0.1750, -0.1046] |
| CUT3R | fully_predicted | 0.25 | 0.07067 | 0.6613 | 0.6561 | 0.9430 | 0.1523 | 0.2650 | -0.1128 [-0.1508, -0.0757] | [-0.1837, -0.0539] |
| CUT3R | fully_predicted | 0.15 | 0.05732 | 0.6832 | 0.7095 | 0.9569 | 0.1311 | 0.2083 | -0.0772 [-0.1190, -0.0394] | [-0.1535, -0.0183] |
| CUT3R | fully_predicted | 0.35 | 0.08222 | 0.6363 | 0.6051 | 0.9277 | 0.1793 | 0.3307 | -0.1513 [-0.1890, -0.1135] | [-0.2094, -0.0922] |
| CUT3R | fully_predicted | 0.5 | 0.09699 | 0.5929 | 0.5341 | 0.9082 | 0.2323 | 0.4126 | -0.1804 [-0.2165, -0.1415] | [-0.2283, -0.1251] |

## 5. H5 and H6 on CUT3R

Criterion text, `docs/success_criteria.md` lines 424-431, quoted:

> 4. New pre-stated hypotheses, tested only on pipelines evaluated after
>    this entry (EndoDAC excluded, since these were derived from its data).
>    H5 (masking): for each new pipeline, false reassurance under
>        fully_predicted is lower than under pred_pose_only, with the
>        mesh-level CI of the paired difference excluding zero.
>    H6 (depth drives false alarm): for each new pipeline, false alarm
>        under pred_depth_only exceeds that under pred_pose_only, with the
>        mesh-level CI of the paired difference excluding zero.

Tau rule, `docs/success_criteria.md` lines 457-463 (2026-10-01
clarification), quoted:

> Rule: a criterion or hypothesis that names no tau is judged at the
> protocol's primary value, tau = 0.25 (docs/eval_protocol.md, section on
> the tau gate, "Proposed primary value: tau = 0.25", which predates this
> entry). The other three tau values are reported alongside as
> sensitivity, always, including any value at which the criterion is not
> met. This rule is chosen because it existed before the results; it is
> not chosen for its effect on H6.

CUT3R is a pipeline evaluated after the 2026-09-27 entry. Paired by
sequence, pooled ratios, mesh-cluster bootstrap (10,000, seed 20260925;
`results/d2b_eval/h5_h6_cut3r.json`). 169 sequences, 103 meshes, 113
trajectories.

**H5** (false reassurance, A - B < 0; A = fully_predicted, B = pred_pose_only)

| tau | role | A | B | A - B | mesh CI | (C,S) CI | (C,S,P) CI | mesh CI excludes zero in the stated direction | (C,S) CI does |
|---|---|---|---|---|---|---|---|---|---|
| 0.15 | alongside | 0.05732 | 0.07116 | -0.01385 | [-0.02364, -0.00385] | [-0.02564, -0.00454] | [-0.02378, -0.00349] | yes | yes |
| 0.25 | **primary** | 0.07067 | 0.08845 | -0.01778 | [-0.02691, -0.00841] | [-0.02743, -0.00989] | [-0.02743, -0.00827] | yes | yes |
| 0.35 | alongside | 0.08222 | 0.10309 | -0.02087 | [-0.02983, -0.01187] | [-0.02965, -0.01323] | [-0.03023, -0.01125] | yes | yes |
| 0.5 | alongside | 0.09699 | 0.12467 | -0.02768 | [-0.03560, -0.02012] | [-0.03678, -0.02034] | [-0.03600, -0.01960] | yes | yes |

**H6** (false alarm, A - B > 0; A = pred_depth_only, B = pred_pose_only)

| tau | role | A | B | A - B | mesh CI | (C,S) CI | (C,S,P) CI | mesh CI excludes zero in the stated direction | (C,S) CI does |
|---|---|---|---|---|---|---|---|---|---|
| 0.15 | alongside | 0.58196 | 0.66611 | -0.08416 | [-0.10283, -0.06448] | [-0.11026, -0.05585] | [-0.10429, -0.06373] | no | no |
| 0.25 | **primary** | 0.53769 | 0.64043 | -0.10275 | [-0.12652, -0.07778] | [-0.13931, -0.06627] | [-0.12889, -0.07607] | no | no |
| 0.35 | alongside | 0.48613 | 0.61355 | -0.12742 | [-0.15593, -0.09729] | [-0.17423, -0.08198] | [-0.15972, -0.09404] | no | no |
| 0.5 | alongside | 0.38036 | 0.56937 | -0.18901 | [-0.22403, -0.15316] | [-0.24076, -0.13730] | [-0.22858, -0.14977] | no | no |

**Result (mechanical, mesh-level CI at the primary tau = 0.25):**
- **H5 holds for CUT3R**: false reassurance is lower under
  fully_predicted than under pred_pose_only, -0.01778, mesh CI [-0.02691,
  -0.00841]. Alongside: also at tau = 0.15, 0.35, 0.50; the (Colon,
  Segment) and (Colon, Segment, Phantom) CIs also exclude zero at every
  tau.
- **H6 does not hold for CUT3R**: false alarm under pred_depth_only does
  not exceed that under pred_pose_only; the difference is negative,
  -0.10275, mesh CI [-0.12652, -0.07778], which excludes zero on the
  opposite side. Alongside: the same sign and the same exclusion at tau =
  0.15, 0.35 and 0.50, at all three cluster levels.

With MASt3R-SLAM (`docs/pipeline2_eval.md` section 6): H5 holds for both
new pipelines; H6 holds for MASt3R-SLAM and not for CUT3R, at tau = 0.25.

## 6. Pairwise differences (descriptive)

Paired by sequence (169) or by region (719); 103 meshes. No pass/fail
attached except through D2b (section 7). Files:
`results/d2b_eval/pairwise_<a>_minus_<b>.json`.

### CUT3R minus EndoDAC

| config | tau | metric | CUT3R | EndoDAC | CUT3R - EndoDAC | mesh CI | (C,S) CI | (C,S,P) CI |
|---|---|---|---|---|---|---|---|---|
| oracle | 0.25 | false_reassurance_rate | 0.00150 | 0.00150 | +0.00000 | [0.00000, 0.00000] | [0.00000, 0.00000] | [0.00000, 0.00000] |
| oracle | 0.25 | false_alarm_rate | 0.00056 | 0.00056 | +0.00000 | [0.00000, 0.00000] | [0.00000, 0.00000] | [0.00000, 0.00000] |
| oracle | 0.25 | area_fraction | 0.26279 | 0.26279 | +0.00000 | [0.00000, 0.00000] | [0.00000, 0.00000] | [0.00000, 0.00000] |
| oracle | 0.25 | recall_medium_plus_large_at_50pct | 1.00000 | 1.00000 | +0.00000 | [0.00000, 0.00000] | [0.00000, 0.00000] | [0.00000, 0.00000] |
| oracle | 0.25 | region_iou_mean_medium_plus_large | 0.99793 | 0.99793 | +0.00000 | [0.00000, 0.00000] | [0.00000, 0.00000] | [0.00000, 0.00000] |
| pred_depth_only | 0.25 | false_reassurance_rate | 0.00017 | 0.00043 | -0.00026 | [-0.00038, -0.00016] | [-0.00049, -0.00010] | [-0.00040, -0.00014] |
| pred_depth_only | 0.25 | false_alarm_rate | 0.53769 | 0.49817 | +0.03952 | [0.02554, 0.05459] | [0.01796, 0.06695] | [0.02340, 0.05730] |
| pred_depth_only | 0.25 | area_fraction | 0.52775 | 0.48885 | +0.03890 | [0.02495, 0.05375] | [0.01728, 0.06427] | [0.02285, 0.05561] |
| pred_depth_only | 0.25 | recall_medium_plus_large_at_50pct | 1.00000 | 1.00000 | +0.00000 | [0.00000, 0.00000] | [0.00000, 0.00000] | [0.00000, 0.00000] |
| pred_depth_only | 0.25 | region_iou_mean_medium_plus_large | 0.31397 | 0.35178 | -0.03781 | [-0.06043, -0.01679] | [-0.07180, -0.01100] | [-0.06172, -0.01502] |
| pred_pose_only | 0.25 | false_reassurance_rate | 0.08845 | 0.05517 | +0.03328 | [0.01986, 0.04785] | [0.01830, 0.05238] | [0.01806, 0.04992] |
| pred_pose_only | 0.25 | false_alarm_rate | 0.64043 | 0.36475 | +0.27568 | [0.24087, 0.30987] | [0.22660, 0.32528] | [0.23644, 0.31492] |
| pred_pose_only | 0.25 | area_fraction | 0.60657 | 0.36870 | +0.23787 | [0.20601, 0.26854] | [0.18574, 0.28338] | [0.20041, 0.27339] |
| pred_pose_only | 0.25 | recall_medium_plus_large_at_50pct | 0.90542 | 0.96523 | -0.05981 | [-0.08355, -0.03571] | [-0.08481, -0.03698] | [-0.08211, -0.03776] |
| pred_pose_only | 0.25 | region_iou_mean_medium_plus_large | 0.19416 | 0.46223 | -0.26807 | [-0.31428, -0.22092] | [-0.33059, -0.20505] | [-0.31314, -0.22274] |
| fully_predicted | 0.25 | false_reassurance_rate | 0.07067 | 0.03074 | +0.03993 | [0.02613, 0.05488] | [0.02466, 0.05871] | [0.02682, 0.05471] |
| fully_predicted | 0.25 | false_alarm_rate | 0.66133 | 0.54276 | +0.11856 | [0.10147, 0.13549] | [0.09107, 0.14709] | [0.10055, 0.13776] |
| fully_predicted | 0.25 | area_fraction | 0.65611 | 0.51673 | +0.13938 | [0.11885, 0.15919] | [0.11276, 0.16569] | [0.11849, 0.16120] |
| fully_predicted | 0.25 | recall_medium_plus_large_at_50pct | 0.94298 | 0.98331 | -0.04033 | [-0.05913, -0.02244] | [-0.05985, -0.02362] | [-0.05898, -0.02239] |
| fully_predicted | 0.25 | region_iou_mean_medium_plus_large | 0.15228 | 0.29079 | -0.13851 | [-0.16696, -0.11100] | [-0.17422, -0.09977] | [-0.16904, -0.10881] |
| fully_predicted | 0.15 | false_reassurance_rate | 0.05732 | 0.02347 | +0.03385 | [0.02217, 0.04649] | [0.02142, 0.04925] | [0.02257, 0.04669] |
| fully_predicted | 0.15 | false_alarm_rate | 0.68320 | 0.58817 | +0.09503 | [0.08249, 0.10757] | [0.07571, 0.11451] | [0.08138, 0.10928] |
| fully_predicted | 0.15 | area_fraction | 0.70954 | 0.57421 | +0.13533 | [0.11790, 0.15232] | [0.11518, 0.15502] | [0.11699, 0.15354] |
| fully_predicted | 0.15 | recall_medium_plus_large_at_50pct | 0.95688 | 0.98748 | -0.03060 | [-0.04854, -0.01417] | [-0.04832, -0.01546] | [-0.04878, -0.01456] |
| fully_predicted | 0.15 | region_iou_mean_medium_plus_large | 0.13113 | 0.24594 | -0.11481 | [-0.13716, -0.09306] | [-0.14461, -0.08335] | [-0.14129, -0.09082] |
| fully_predicted | 0.35 | false_reassurance_rate | 0.08222 | 0.03816 | +0.04407 | [0.02854, 0.06098] | [0.02661, 0.06622] | [0.02875, 0.06107] |
| fully_predicted | 0.35 | false_alarm_rate | 0.63632 | 0.49075 | +0.14556 | [0.12489, 0.16751] | [0.11282, 0.18083] | [0.12284, 0.16904] |
| fully_predicted | 0.35 | area_fraction | 0.60511 | 0.46388 | +0.14123 | [0.11837, 0.16259] | [0.11051, 0.16923] | [0.11738, 0.16460] |
| fully_predicted | 0.35 | recall_medium_plus_large_at_50pct | 0.92768 | 0.98192 | -0.05424 | [-0.07482, -0.03409] | [-0.07496, -0.03631] | [-0.07467, -0.03434] |
| fully_predicted | 0.35 | region_iou_mean_medium_plus_large | 0.17934 | 0.33062 | -0.15128 | [-0.18397, -0.11900] | [-0.18665, -0.11265] | [-0.18480, -0.11821] |
| fully_predicted | 0.5 | false_reassurance_rate | 0.09699 | 0.04953 | +0.04746 | [0.03060, 0.06625] | [0.02706, 0.07364] | [0.03041, 0.06710] |
| fully_predicted | 0.5 | false_alarm_rate | 0.59291 | 0.38584 | +0.20707 | [0.18014, 0.23330] | [0.17062, 0.24443] | [0.17927, 0.23492] |
| fully_predicted | 0.5 | area_fraction | 0.53407 | 0.38557 | +0.14850 | [0.12530, 0.17021] | [0.11668, 0.17667] | [0.12347, 0.17227] |
| fully_predicted | 0.5 | recall_medium_plus_large_at_50pct | 0.90821 | 0.97079 | -0.06259 | [-0.08614, -0.03969] | [-0.08939, -0.03886] | [-0.08526, -0.04111] |
| fully_predicted | 0.5 | region_iou_mean_medium_plus_large | 0.23226 | 0.44160 | -0.20934 | [-0.24572, -0.17324] | [-0.25701, -0.16226] | [-0.24727, -0.17212] |

### CUT3R minus MASt3R-SLAM

| config | tau | metric | CUT3R | MASt3R-SLAM | CUT3R - MASt3R-SLAM | mesh CI | (C,S) CI | (C,S,P) CI |
|---|---|---|---|---|---|---|---|---|
| oracle | 0.25 | false_reassurance_rate | 0.00150 | 0.00150 | +0.00000 | [0.00000, 0.00000] | [0.00000, 0.00000] | [0.00000, 0.00000] |
| oracle | 0.25 | false_alarm_rate | 0.00056 | 0.00056 | +0.00000 | [0.00000, 0.00000] | [0.00000, 0.00000] | [0.00000, 0.00000] |
| oracle | 0.25 | area_fraction | 0.26279 | 0.26279 | +0.00000 | [0.00000, 0.00000] | [0.00000, 0.00000] | [0.00000, 0.00000] |
| oracle | 0.25 | recall_medium_plus_large_at_50pct | 1.00000 | 1.00000 | +0.00000 | [0.00000, 0.00000] | [0.00000, 0.00000] | [0.00000, 0.00000] |
| oracle | 0.25 | region_iou_mean_medium_plus_large | 0.99793 | 0.99793 | +0.00000 | [0.00000, 0.00000] | [0.00000, 0.00000] | [0.00000, 0.00000] |
| pred_depth_only | 0.25 | false_reassurance_rate | 0.00017 | 0.00020 | -0.00002 | [-0.00009, 0.00002] | [-0.00010, 0.00002] | [-0.00010, 0.00002] |
| pred_depth_only | 0.25 | false_alarm_rate | 0.53769 | 0.50028 | +0.03741 | [0.02328, 0.05186] | [0.01556, 0.06075] | [0.02103, 0.05449] |
| pred_depth_only | 0.25 | area_fraction | 0.52775 | 0.49087 | +0.03688 | [0.02303, 0.05114] | [0.01563, 0.05830] | [0.02060, 0.05316] |
| pred_depth_only | 0.25 | recall_medium_plus_large_at_50pct | 1.00000 | 1.00000 | +0.00000 | [0.00000, 0.00000] | [0.00000, 0.00000] | [0.00000, 0.00000] |
| pred_depth_only | 0.25 | region_iou_mean_medium_plus_large | 0.31397 | 0.35827 | -0.04430 | [-0.07550, -0.01824] | [-0.08079, -0.01429] | [-0.07561, -0.01661] |
| pred_pose_only | 0.25 | false_reassurance_rate | 0.08845 | 0.11357 | -0.02512 | [-0.04753, -0.00337] | [-0.05134, -0.00241] | [-0.05074, -0.00259] |
| pred_pose_only | 0.25 | false_alarm_rate | 0.64043 | 0.34662 | +0.29381 | [0.24430, 0.34429] | [0.22168, 0.36726] | [0.24116, 0.34678] |
| pred_pose_only | 0.25 | area_fraction | 0.60657 | 0.33464 | +0.27194 | [0.23459, 0.30760] | [0.21111, 0.32534] | [0.23121, 0.31217] |
| pred_pose_only | 0.25 | recall_medium_plus_large_at_50pct | 0.90542 | 0.90542 | +0.00000 | [-0.02875, 0.03226] | [-0.02361, 0.02577] | [-0.02680, 0.02829] |
| pred_pose_only | 0.25 | region_iou_mean_medium_plus_large | 0.19416 | 0.48289 | -0.28873 | [-0.33187, -0.24702] | [-0.35261, -0.22401] | [-0.33146, -0.24699] |
| fully_predicted | 0.25 | false_reassurance_rate | 0.07067 | 0.05189 | +0.01878 | [0.00345, 0.03442] | [0.00491, 0.03429] | [0.00477, 0.03334] |
| fully_predicted | 0.25 | false_alarm_rate | 0.66133 | 0.59856 | +0.06277 | [0.04770, 0.07689] | [0.04686, 0.07794] | [0.04791, 0.07654] |
| fully_predicted | 0.25 | area_fraction | 0.65611 | 0.57025 | +0.08586 | [0.06217, 0.10713] | [0.05674, 0.10943] | [0.06211, 0.10826] |
| fully_predicted | 0.25 | recall_medium_plus_large_at_50pct | 0.94298 | 0.95828 | -0.01530 | [-0.03814, 0.00843] | [-0.03488, 0.00252] | [-0.03776, 0.00657] |
| fully_predicted | 0.25 | region_iou_mean_medium_plus_large | 0.15228 | 0.26307 | -0.11079 | [-0.12997, -0.09216] | [-0.13340, -0.08585] | [-0.13060, -0.09109] |
| fully_predicted | 0.15 | false_reassurance_rate | 0.05732 | 0.03947 | +0.01784 | [0.00453, 0.03135] | [0.00767, 0.03002] | [0.00613, 0.03045] |
| fully_predicted | 0.15 | false_alarm_rate | 0.68320 | 0.63250 | +0.05069 | [0.03803, 0.06197] | [0.03668, 0.06412] | [0.03866, 0.06231] |
| fully_predicted | 0.15 | area_fraction | 0.70954 | 0.62832 | +0.08122 | [0.05907, 0.10177] | [0.05436, 0.10563] | [0.05738, 0.10326] |
| fully_predicted | 0.15 | recall_medium_plus_large_at_50pct | 0.95688 | 0.97079 | -0.01391 | [-0.03602, 0.00775] | [-0.03521, 0.00283] | [-0.03602, 0.00765] |
| fully_predicted | 0.15 | region_iou_mean_medium_plus_large | 0.13113 | 0.22830 | -0.09717 | [-0.11552, -0.07928] | [-0.11935, -0.07484] | [-0.11735, -0.07781] |
| fully_predicted | 0.35 | false_reassurance_rate | 0.08222 | 0.06573 | +0.01649 | [-0.00143, 0.03489] | [-0.00260, 0.03652] | [-0.00039, 0.03415] |
| fully_predicted | 0.35 | false_alarm_rate | 0.63632 | 0.55173 | +0.08459 | [0.06574, 0.10192] | [0.06327, 0.10475] | [0.06523, 0.10230] |
| fully_predicted | 0.35 | area_fraction | 0.60511 | 0.50630 | +0.09881 | [0.07437, 0.11978] | [0.06784, 0.12164] | [0.07390, 0.12130] |
| fully_predicted | 0.35 | recall_medium_plus_large_at_50pct | 0.92768 | 0.95132 | -0.02364 | [-0.05034, 0.00287] | [-0.04769, -0.00129] | [-0.04944, 0.00138] |
| fully_predicted | 0.35 | region_iou_mean_medium_plus_large | 0.17934 | 0.30579 | -0.12644 | [-0.14897, -0.10315] | [-0.15083, -0.09883] | [-0.14974, -0.10337] |
| fully_predicted | 0.5 | false_reassurance_rate | 0.09699 | 0.09078 | +0.00621 | [-0.01723, 0.02891] | [-0.02516, 0.03411] | [-0.01772, 0.02954] |
| fully_predicted | 0.5 | false_alarm_rate | 0.59291 | 0.44720 | +0.14570 | [0.11800, 0.17054] | [0.11084, 0.18181] | [0.11728, 0.17266] |
| fully_predicted | 0.5 | area_fraction | 0.53407 | 0.40510 | +0.12898 | [0.10521, 0.15054] | [0.09667, 0.15710] | [0.10371, 0.15219] |
| fully_predicted | 0.5 | recall_medium_plus_large_at_50pct | 0.90821 | 0.92211 | -0.01391 | [-0.04410, 0.01786] | [-0.03953, 0.00872] | [-0.04451, 0.01797] |
| fully_predicted | 0.5 | region_iou_mean_medium_plus_large | 0.23226 | 0.37789 | -0.14563 | [-0.17529, -0.11534] | [-0.18245, -0.09163] | [-0.17769, -0.11175] |

### MASt3R-SLAM minus EndoDAC

| config | tau | metric | MASt3R-SLAM | EndoDAC | MASt3R-SLAM - EndoDAC | mesh CI | (C,S) CI | (C,S,P) CI |
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

## 7. D2b

Criterion text, `docs/success_criteria.md` lines 407-422, quoted:

> 3. D2b: new decision point, primary endpoints.
>    Trigger: 3 pipelines evaluated, including EndoDAC (reduced from the
>    4 to 6 of section 3; recorded here as a scope reduction). Original D2
>    is evaluated if and when 4 or more pipelines are available; otherwise
>    it is reported as not triggered.
>    Configuration fully_predicted, tau = 0.25.
>    D2b.1: best-minus-worst pooled false reassurance >= 0.01 (absolute),
>           and the mesh-level 95% bootstrap CI of the paired difference
>           (same sequences, clustered by mesh) excludes zero.
>    D2b.2: best-minus-worst mean region IoU (medium + large) >= 0.10,
>           with the same CI requirement.
>    D2b.3 (record only): top and bottom pipelines on each endpoint do not
>           swap across tau in {0.15, 0.25, 0.35, 0.50}.
>    D2b.1 and D2b.2 are judged and reported separately. EndoDAC's false
>    reassurance (0.031) was known when these thresholds were set;
>    between-pipeline differences were not.

**Triggered**: 3 pipelines evaluated, including EndoDAC. Judged on the
primary runs only (2026-10-02 entry, item 1). Best and worst per endpoint
at fully_predicted, tau = 0.25: for false reassurance, lowest = best; for
region IoU, highest = best. Paired difference worst minus best (false
reassurance, paired by sequence, pooled ratios) and best minus worst
(region IoU, paired per region), mesh-cluster bootstrap 10,000 replicates,
seed 20260925 (`results/d2b_eval/d2b.json`).

| tau | FR EndoDAC | FR MASt3R-SLAM | FR CUT3R | FR best / worst | IoU EndoDAC | IoU MASt3R-SLAM | IoU CUT3R | IoU best / worst |
|---|---|---|---|---|---|---|---|---|
| 0.15 | 0.02347 | 0.03947 | 0.05732 | EndoDAC / CUT3R | 0.2459 | 0.2283 | 0.1311 | EndoDAC / CUT3R |
| 0.25 | 0.03074 | 0.05189 | 0.07067 | EndoDAC / CUT3R | 0.2908 | 0.2631 | 0.1523 | EndoDAC / CUT3R |
| 0.35 | 0.03816 | 0.06573 | 0.08222 | EndoDAC / CUT3R | 0.3306 | 0.3058 | 0.1793 | EndoDAC / CUT3R |
| 0.5 | 0.04953 | 0.09078 | 0.09699 | EndoDAC / CUT3R | 0.4416 | 0.3779 | 0.2323 | EndoDAC / CUT3R |

| criterion | best | worst | best value | worst value | abs difference | threshold | mesh CI of the paired difference | (C,S) CI | (C,S,P) CI | difference >= threshold | mesh CI excludes zero | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| D2b.1 (pooled false reassurance) | EndoDAC | CUT3R | 0.03074 | 0.07067 | 0.03993 | 0.01 | [0.02650, 0.05469] | [0.02497, 0.05927] | [0.02633, 0.05470] | yes | yes | **PASS** |
| D2b.2 (mean region IoU, medium + large) | EndoDAC | CUT3R | 0.29079 | 0.15228 | 0.13851 | 0.1 | [0.11140, 0.16634] | [0.10057, 0.17449] | [0.10826, 0.16946] | yes | yes | **PASS** |

- D2b.1: |difference| 0.03993; pooled run-to-run range (six runs, 15-sequence subset): EndoDAC 0.00001, CUT3R 0.01407; difference smaller than the range of either pipeline: no; (C,S) CI excludes zero: yes; (C,S,P) CI excludes zero: yes
- D2b.2: |difference| 0.13851; pooled run-to-run range (six runs, 15-sequence subset): EndoDAC 0.00002, CUT3R 0.01890; difference smaller than the range of either pipeline: no; (C,S) CI excludes zero: yes; (C,S,P) CI excludes zero: yes

**Verdicts (mechanical; mesh-level CI is the verdict basis):**

- **D2b.1: PASS.** Best EndoDAC (0.03074), worst CUT3R (0.07067).
  Best-minus-worst difference 0.03993 >= 0.01; mesh-level CI of the paired
  difference [0.02650, 0.05469] excludes zero (169 sequences, 103 meshes).
  Alongside: (Colon, Segment) CI [0.02497, 0.05927] and (Colon, Segment,
  Phantom) CI [0.02633, 0.05470] also exclude zero.
- **D2b.2: PASS.** Best EndoDAC (0.2908), worst CUT3R (0.1523).
  Best-minus-worst difference 0.1385 >= 0.10; mesh-level CI [0.1114,
  0.1663] excludes zero (719 regions, 103 meshes). Alongside: (Colon,
  Segment) CI [0.1006, 0.1745] and (Colon, Segment, Phantom) CI [0.1083,
  0.1695] also exclude zero.
- **D2b.3 (record only):** top EndoDAC and bottom CUT3R on both
  endpoints at every tau in {0.15, 0.25, 0.35, 0.50}; **no swap**. The
  middle pipeline is MASt3R-SLAM on both endpoints at every tau.

Item 4 of the 2026-10-02 entry: both primary-run differences (0.03993 and
0.1385) exceed the pooled run-to-run range of each pipeline involved
(false reassurance: EndoDAC 0.00001, CUT3R 0.01407; region IoU: EndoDAC
0.00002, CUT3R 0.01890; section 8). The statement of item 4 is therefore
not attached to either verdict.

Next to the thresholds (item 3): the pooled run-to-run range of CUT3R's
false reassurance (0.0141) exceeds the D2b.1 threshold (0.01); its region
IoU range (0.0189) is below the D2b.2 threshold (0.10). EndoDAC's and
MASt3R-SLAM's ranges are below both thresholds.

**Original D2: not triggered (3 of 4 pipelines).** `docs/success_criteria.md`
line 113: "Trigger: 4 to 6 pipelines evaluated."

## 8. Run-to-run variability (2026-10-02 entry, item 3)

fully_predicted, tau = 0.25, on the 15-sequence subset (first registered
sequence per mold: `c1_ascending_t1_v1`, `c1_cecum_t1_v1`,
`c1_descending_t1_v1`, `c1_rectum_t1_v1`, `c1_sigmoid1_t1_v1`,
`c1_sigmoid2_t1_v1`, `c1_transverse1_t1_v1`, `c1_transverse2_t1_v1`,
`c2_ascending_t1_v1`, `c2_cecum_t1_v1`, `c2_descending_t3_v1`,
`c2_rectum_t1_v1`, `c2_sigmoid_t1_v1`, `c2_transverse1_t1_v1`,
`c2_transverse2_t1_v1`); 15 meshes, 15 molds, 65 medium + large regions.
Primary run plus five noise runs (std 1e-6 on the 0-1 scale, seeds 1-5),
each evaluated with the same driver as the primary
(`results/d2b_eval/variability/<pipeline>/<run>/`), region IoU with 20
matched random sets per run (`results/d2b_eval/variability_region_iou/`).
240 sequence evaluations, all status ok; wall time 4 h 43 min
(2026-10-03T01:38:14Z to 06:21:41Z), plus region IoU 3 min.
`results/d2b_eval/variability_summary.json`.

MASt3R-SLAM's unperturbed rerun is a separate descriptive line, not one
of the six runs; it was run on all 15 subset sequences for this report
(14 of them on 2026-10-02, 15/15 ok). **EndoDAC note required by the
2026-10-03 entry:** EndoDAC's noise was added to its network input tensor
after resizing (section 1.4), the same point as the other two pipelines;
the one-sequence check is bit-identical.

### Pooled over the 15-sequence subset, per run

| pipeline | run | false reassurance | false alarm | area fraction | region IoU mean (m+l) | matched random IoU | n regions |
|---|---|---|---|---|---|---|---|
| EndoDAC | primary | 0.04771 | 0.54368 | 0.50058 | 0.22802 | 0.42438 | 65 |
| EndoDAC | seed1 | 0.04771 | 0.54368 | 0.50057 | 0.22803 | 0.42438 | 65 |
| EndoDAC | seed2 | 0.04770 | 0.54368 | 0.50057 | 0.22803 | 0.42438 | 65 |
| EndoDAC | seed3 | 0.04771 | 0.54368 | 0.50057 | 0.22803 | 0.42438 | 65 |
| EndoDAC | seed4 | 0.04770 | 0.54368 | 0.50058 | 0.22801 | 0.42438 | 65 |
| EndoDAC | seed5 | 0.04770 | 0.54368 | 0.50058 | 0.22803 | 0.42437 | 65 |
| MASt3R-SLAM | primary | 0.03247 | 0.60317 | 0.58003 | 0.21523 | 0.43588 | 65 |
| MASt3R-SLAM | seed1 | 0.03214 | 0.60373 | 0.58102 | 0.21541 | 0.43631 | 65 |
| MASt3R-SLAM | seed2 | 0.03245 | 0.60306 | 0.57991 | 0.21528 | 0.43653 | 65 |
| MASt3R-SLAM | seed3 | 0.03246 | 0.60337 | 0.58037 | 0.21514 | 0.43395 | 65 |
| MASt3R-SLAM | seed4 | 0.03245 | 0.60305 | 0.57990 | 0.21953 | 0.43632 | 65 |
| MASt3R-SLAM | seed5 | 0.03218 | 0.60368 | 0.58091 | 0.21543 | 0.43760 | 65 |
| MASt3R-SLAM | rerun_unperturbed | 0.03247 | 0.60305 | 0.57987 | 0.21521 | 0.43580 | 65 |
| CUT3R | primary | 0.05340 | 0.66605 | 0.66871 | 0.13037 | 0.19342 | 65 |
| CUT3R | seed1 | 0.04518 | 0.66385 | 0.67052 | 0.11921 | 0.20573 | 65 |
| CUT3R | seed2 | 0.04202 | 0.66121 | 0.66756 | 0.11591 | 0.22085 | 65 |
| CUT3R | seed3 | 0.04356 | 0.66125 | 0.66633 | 0.12739 | 0.22713 | 65 |
| CUT3R | seed4 | 0.05573 | 0.66595 | 0.66645 | 0.13480 | 0.21276 | 65 |
| CUT3R | seed5 | 0.05609 | 0.65389 | 0.64448 | 0.13169 | 0.25460 | 65 |

### Pooled: SD and range across the six runs (primary + seeds 1 to 5)

| pipeline | metric | SD | range | min | max |
|---|---|---|---|---|---|
| EndoDAC | false_reassurance | 3.09e-06 | 8.63e-06 | 0.04770 | 0.04771 |
| EndoDAC | false_alarm | 2.28e-06 | 5.57e-06 | 0.54368 | 0.54368 |
| EndoDAC | area_fraction | 3.08e-06 | 7.70e-06 | 0.50057 | 0.50058 |
| EndoDAC | region_iou_mean | 7.35e-06 | 1.92e-05 | 0.22801 | 0.22803 |
| MASt3R-SLAM | false_reassurance | 1.53e-04 | 3.27e-04 | 0.03214 | 0.03247 |
| MASt3R-SLAM | false_alarm | 3.03e-04 | 6.78e-04 | 0.60305 | 0.60373 |
| MASt3R-SLAM | area_fraction | 5.04e-04 | 1.12e-03 | 0.57990 | 0.58102 |
| MASt3R-SLAM | region_iou_mean | 1.73e-03 | 4.39e-03 | 0.21514 | 0.21953 |
| CUT3R | false_reassurance | 6.44e-03 | 1.41e-02 | 0.04202 | 0.05609 |
| CUT3R | false_alarm | 4.52e-03 | 1.22e-02 | 0.65389 | 0.66605 |
| CUT3R | area_fraction | 9.69e-03 | 2.60e-02 | 0.64448 | 0.67052 |
| CUT3R | region_iou_mean | 7.44e-03 | 1.89e-02 | 0.11591 | 0.13480 |

### Per sequence: range (SD) across the six runs

| sequence | pipeline | primary FR | FR range (SD) | primary FA | FA range (SD) | primary area | area range (SD) | primary IoU (n regions) | IoU range (SD) |
|---|---|---|---|---|---|---|---|---|---|
| c1_ascending_t1_v1 | EndoDAC | 0.00014 | 1.13e-05 (4.85e-06) | 0.67228 | 1.02e-05 (3.95e-06) | 0.40107 | 8.38e-06 (3.23e-06) | 0.1414 (5) | 4.48e-06 (1.93e-06) |
| c1_ascending_t1_v1 | MASt3R-SLAM | 0.00314 | 3.29e-05 (1.40e-05) | 0.77700 | 3.89e-05 (1.82e-05) | 0.57383 | 1.02e-04 (4.33e-05) | 0.1281 (5) | 7.61e-05 (2.71e-05) |
| c1_ascending_t1_v1 | CUT3R | 0.02525 | 2.34e-02 (8.49e-03) | 0.84265 | 2.37e-02 (8.34e-03) | 0.78162 | 8.85e-02 (3.24e-02) | 0.0308 (5) | 4.73e-03 (1.67e-03) |
| c1_cecum_t1_v1 | EndoDAC | 0.01951 | 4.71e-05 (1.67e-05) | 0.57298 | 8.42e-05 (3.26e-05) | 0.22246 | 4.00e-05 (1.38e-05) | 0.4719 (1) | 1.00e-04 (4.01e-05) |
| c1_cecum_t1_v1 | MASt3R-SLAM | 0.01216 | 2.44e-04 (1.04e-04) | 0.68599 | 2.46e-04 (8.81e-05) | 0.29354 | 1.99e-04 (6.88e-05) | 0.4124 (1) | 2.96e-04 (1.16e-04) |
| c1_cecum_t1_v1 | CUT3R | 0.12169 | 1.21e-01 (4.11e-02) | 0.81350 | 1.51e-02 (5.62e-03) | 0.42461 | 2.52e-02 (9.88e-03) | 0.2143 (1) | 3.63e-02 (1.31e-02) |
| c1_descending_t1_v1 | EndoDAC | 0.01629 | 4.81e-05 (1.91e-05) | 0.51062 | 1.25e-05 (4.44e-06) | 0.48443 | 3.32e-05 (1.17e-05) | 0.3389 (3) | 5.85e-06 (2.50e-06) |
| c1_descending_t1_v1 | MASt3R-SLAM | 0.00887 | 1.74e-05 (7.11e-06) | 0.55850 | 1.38e-04 (5.83e-05) | 0.53953 | 1.65e-04 (7.14e-05) | 0.3156 (3) | 7.78e-05 (2.77e-05) |
| c1_descending_t1_v1 | CUT3R | 0.00407 | 0.00e+00 (9.50e-19) | 0.62706 | 1.12e-02 (4.67e-03) | 0.63936 | 1.94e-02 (8.05e-03) | 0.2477 (3) | 1.06e-02 (4.50e-03) |
| c1_rectum_t1_v1 | EndoDAC | 0.03566 | 3.46e-05 (1.43e-05) | 0.72224 | 1.09e-05 (4.09e-06) | 0.56582 | 2.39e-05 (8.65e-06) | 0.1700 (5) | 1.32e-05 (4.65e-06) |
| c1_rectum_t1_v1 | MASt3R-SLAM | 0.08706 | 4.87e-03 (2.48e-03) | 0.70602 | 6.70e-03 (3.43e-03) | 0.50669 | 1.44e-02 (7.37e-03) | 0.1732 (5) | 5.74e-04 (2.80e-04) |
| c1_rectum_t1_v1 | CUT3R | 0.00302 | 2.06e-03 (7.91e-04) | 0.72411 | 1.11e-02 (4.63e-03) | 0.58869 | 2.34e-02 (9.76e-03) | 0.1145 (5) | 7.45e-03 (2.53e-03) |
| c1_sigmoid1_t1_v1 | EndoDAC | 0.04217 | 9.78e-06 (3.67e-06) | 0.47872 | 1.42e-05 (5.32e-06) | 0.69214 | 1.02e-05 (4.24e-06) | 0.2943 (3) | 2.04e-05 (8.40e-06) |
| c1_sigmoid1_t1_v1 | MASt3R-SLAM | 0.03839 | 9.39e-05 (3.87e-05) | 0.52603 | 5.48e-05 (2.14e-05) | 0.76048 | 1.21e-04 (4.36e-05) | 0.2808 (3) | 7.28e-05 (3.02e-05) |
| c1_sigmoid1_t1_v1 | CUT3R | 0.00131 | 2.47e-03 (1.05e-03) | 0.53258 | 2.28e-02 (7.43e-03) | 0.80279 | 3.65e-02 (1.19e-02) | 0.1534 (3) | 7.57e-03 (2.45e-03) |
| c1_sigmoid2_t1_v1 | EndoDAC | 0.00132 | 1.66e-05 (8.57e-06) | 0.43905 | 4.07e-05 (1.45e-05) | 0.44432 | 3.20e-05 (1.22e-05) | 0.5254 (2) | 3.26e-05 (1.17e-05) |
| c1_sigmoid2_t1_v1 | MASt3R-SLAM | 0.00133 | 1.00e-04 (3.65e-05) | 0.55893 | 3.04e-03 (1.22e-03) | 0.56082 | 3.79e-03 (1.52e-03) | 0.4558 (2) | 3.06e-03 (1.22e-03) |
| c1_sigmoid2_t1_v1 | CUT3R | 0.01537 | 1.41e-02 (4.83e-03) | 0.68605 | 1.10e-02 (4.32e-03) | 0.77090 | 2.00e-02 (7.99e-03) | 0.1562 (2) | 6.05e-03 (2.31e-03) |
| c1_transverse1_t1_v1 | EndoDAC | 0.03446 | 1.24e-05 (6.63e-06) | 0.61557 | 1.14e-05 (4.67e-06) | 0.73699 | 2.41e-05 (1.00e-05) | 0.4308 (4) | 3.67e-05 (1.44e-05) |
| c1_transverse1_t1_v1 | MASt3R-SLAM | 0.02550 | 1.32e-04 (4.86e-05) | 0.63876 | 3.23e-05 (1.18e-05) | 0.79008 | 9.26e-05 (3.75e-05) | 0.2424 (4) | 5.33e-05 (2.02e-05) |
| c1_transverse1_t1_v1 | CUT3R | 0.04514 | 1.84e-02 (6.93e-03) | 0.66344 | 1.02e-03 (3.99e-04) | 0.82810 | 1.50e-02 (5.54e-03) | 0.2202 (4) | 9.98e-03 (4.01e-03) |
| c1_transverse2_t1_v1 | EndoDAC | 0.05378 | 3.90e-05 (1.52e-05) | 0.56280 | 8.47e-06 (3.33e-06) | 0.68880 | 3.05e-05 (1.27e-05) | 0.0623 (7) | 2.26e-06 (8.02e-07) |
| c1_transverse2_t1_v1 | MASt3R-SLAM | 0.02673 | 2.71e-04 (1.07e-04) | 0.49421 | 2.23e-04 (8.99e-05) | 0.61065 | 2.26e-04 (8.54e-05) | 0.3550 (7) | 1.06e-03 (3.77e-04) |
| c1_transverse2_t1_v1 | CUT3R | 0.05976 | 4.34e-02 (1.66e-02) | 0.58824 | 3.55e-02 (1.21e-02) | 0.71697 | 5.51e-02 (2.14e-02) | 0.1080 (7) | 5.86e-02 (2.83e-02) |
| c2_ascending_t1_v1 | EndoDAC | 0.03387 | 4.75e-05 (2.00e-05) | 0.58338 | 8.00e-06 (2.93e-06) | 0.53139 | 2.09e-05 (8.73e-06) | 0.1705 (5) | 9.85e-06 (3.37e-06) |
| c2_ascending_t1_v1 | MASt3R-SLAM | 0.00010 | 2.38e-05 (9.44e-06) | 0.68337 | 6.31e-04 (2.40e-04) | 0.69770 | 1.98e-03 (7.37e-04) | 0.1284 (5) | 3.77e-04 (1.41e-04) |
| c2_ascending_t1_v1 | CUT3R | 0.00388 | 3.88e-03 (1.58e-03) | 0.71687 | 2.83e-02 (1.22e-02) | 0.77166 | 6.68e-02 (2.83e-02) | 0.0557 (5) | 6.06e-02 (2.21e-02) |
| c2_cecum_t1_v1 | EndoDAC | 0.17266 | 1.50e-05 (5.81e-06) | 0.40711 | 3.40e-05 (1.23e-05) | 0.24904 | 9.85e-06 (3.61e-06) | 0.1464 (4) | 5.37e-06 (1.93e-06) |
| c2_cecum_t1_v1 | MASt3R-SLAM | 0.00006 | 4.80e-08 (2.48e-08) | 0.52036 | 1.91e-04 (7.18e-05) | 0.35891 | 1.29e-04 (4.68e-05) | 0.1207 (4) | 5.91e-05 (2.35e-05) |
| c2_cecum_t1_v1 | CUT3R | 0.26768 | 4.82e-01 (1.72e-01) | 0.46621 | 1.60e-01 (5.34e-02) | 0.23563 | 1.64e-01 (7.00e-02) | 0.2441 (4) | 1.29e-01 (5.49e-02) |
| c2_descending_t3_v1 | EndoDAC | 0.01074 | 1.37e-04 (5.16e-05) | 0.55513 | 3.36e-05 (1.32e-05) | 0.56101 | 3.21e-05 (1.10e-05) | 0.1383 (9) | 6.49e-05 (2.71e-05) |
| c2_descending_t3_v1 | MASt3R-SLAM | 0.00422 | 2.41e-04 (1.00e-04) | 0.62764 | 1.57e-03 (7.70e-04) | 0.66113 | 2.37e-03 (1.16e-03) | 0.1755 (9) | 3.15e-02 (1.27e-02) |
| c2_descending_t3_v1 | CUT3R | 0.00761 | 9.37e-03 (3.64e-03) | 0.66903 | 1.28e-02 (5.48e-03) | 0.73429 | 2.02e-02 (8.27e-03) | 0.1333 (9) | 1.81e-02 (6.28e-03) |
| c2_rectum_t1_v1 | EndoDAC | 0.27322 | 4.08e-05 (1.58e-05) | 0.56790 | 4.20e-05 (1.62e-05) | 0.50324 | 3.39e-05 (1.41e-05) | 0.1086 (4) | 7.38e-06 (2.90e-06) |
| c2_rectum_t1_v1 | MASt3R-SLAM | 0.14150 | 3.06e-04 (1.54e-04) | 0.57930 | 2.70e-04 (9.69e-05) | 0.61773 | 3.23e-04 (1.33e-04) | 0.1841 (4) | 3.07e-04 (1.14e-04) |
| c2_rectum_t1_v1 | CUT3R | 0.12979 | 5.15e-02 (1.74e-02) | 0.60090 | 1.19e-02 (4.59e-03) | 0.66031 | 4.82e-02 (1.83e-02) | 0.0967 (4) | 2.40e-03 (8.57e-04) |
| c2_sigmoid_t1_v1 | EndoDAC | 0.01373 | 2.51e-05 (1.01e-05) | 0.45487 | 2.58e-05 (1.10e-05) | 0.56925 | 3.76e-05 (1.36e-05) | 0.4259 (3) | 1.87e-05 (7.19e-06) |
| c2_sigmoid_t1_v1 | MASt3R-SLAM | 0.00468 | 1.10e-03 (4.30e-04) | 0.56611 | 7.43e-04 (2.85e-04) | 0.70874 | 2.38e-03 (9.61e-04) | 0.2915 (3) | 2.19e-03 (8.56e-04) |
| c2_sigmoid_t1_v1 | CUT3R | 0.01131 | 4.47e-02 (1.92e-02) | 0.51038 | 1.05e-01 (4.51e-02) | 0.63103 | 1.35e-01 (5.21e-02) | 0.1626 (3) | 1.20e-01 (4.44e-02) |
| c2_transverse1_t1_v1 | EndoDAC | 0.02038 | 3.44e-05 (1.36e-05) | 0.44273 | 3.52e-05 (1.40e-05) | 0.51736 | 3.24e-05 (1.31e-05) | 0.3056 (6) | 1.20e-04 (4.09e-05) |
| c2_transverse1_t1_v1 | MASt3R-SLAM | 0.03464 | 6.31e-05 (2.53e-05) | 0.53755 | 1.75e-04 (6.42e-05) | 0.60876 | 2.52e-04 (9.21e-05) | 0.1615 (6) | 2.83e-04 (1.32e-04) |
| c2_transverse1_t1_v1 | CUT3R | 0.12336 | 3.45e-02 (1.24e-02) | 0.68082 | 7.17e-02 (2.68e-02) | 0.79116 | 1.35e-01 (4.84e-02) | 0.0519 (6) | 1.15e-02 (4.28e-03) |
| c2_transverse2_t1_v1 | EndoDAC | 0.01222 | 1.36e-04 (5.10e-05) | 0.34033 | 7.81e-05 (3.15e-05) | 0.48618 | 4.61e-05 (2.00e-05) | 0.3638 (4) | 1.34e-05 (5.04e-06) |
| c2_transverse2_t1_v1 | MASt3R-SLAM | 0.07227 | 2.25e-04 (8.51e-05) | 0.47413 | 1.28e-04 (4.38e-05) | 0.57054 | 1.40e-04 (6.11e-05) | 0.1578 (4) | 4.38e-05 (1.62e-05) |
| c2_transverse2_t1_v1 | CUT3R | 0.05155 | 9.12e-02 (3.87e-02) | 0.59096 | 1.16e-01 (4.34e-02) | 0.73990 | 1.46e-01 (5.17e-02) | 0.1852 (4) | 1.21e-01 (4.72e-02) |

---

# INTERPRETATION

Each item is a reading of the numbers above, not a finding.

1. **CUT3R ranks last on both D2b endpoints because of its poses, not
   its depth.** Its pred_depth_only cells are close to the other two
   pipelines' (false alarm 0.538 vs 0.498 and 0.500; region IoU 0.314 vs
   0.352 and 0.358), while its pred_pose_only cells are far apart (false
   alarm 0.640 vs 0.365 and 0.347; region IoU 0.194 vs 0.462 and 0.483),
   and its trajectory descriptives were already the worst (median ATE
   13.1 vs 5.2 and 7.0 mm, `docs/pipelines/cut3r.md` S3.5). Would be
   confirmed by a per-sequence association between CUT3R's ATE and its
   fully_predicted false reassurance / region IoU deficit relative to
   EndoDAC; refuted if the deficit is unrelated to ATE or concentrated in
   a few sequences.

2. **H6 fails for CUT3R because its pose error inflates false alarm
   under pred_pose_only beyond what its depth error adds under
   pred_depth_only.** H6 assumed (from EndoDAC) that depth drives false
   alarm; for CUT3R pred_pose_only false alarm (0.640) exceeds
   pred_depth_only (0.538). A misplaced camera casts rays into the wrong
   part of the mesh, marking observed surface the camera never saw as
   unobserved where GT depth then fails the tau test. Would be confirmed
   by the per-frame fraction of evaluable pixels rejected by the tau test
   under pred_pose_only (CUT3R vs EndoDAC) tracking the per-frame pose
   error; refuted if CUT3R's pred_pose_only false alarm is driven by a
   few sequences with gross alignment failure (diagnostic b's maximum
   ray-miss fraction, 0.347 in `c2_transverse2_t4_v3`, is a candidate).

3. **The D2b verdicts are not artefacts of numerical run-to-run
   variability.** Both differences are about three (false reassurance)
   to seven (region IoU) times CUT3R's pooled six-run range and orders of
   magnitude above EndoDAC's. The variability subset is 15 sequences and
   65 regions, so its pooled range is a small-sample estimate; would be
   refuted if the range measured on the full corpus approached the
   differences.

4. **CUT3R's per-sequence variability is very uneven.** Its six-run
   range of false reassurance is 0.48 on `c2_cecum_t1_v1` and a median of
   0.023 across the 15 sequences. Consistent with its rollout amplifying
   perturbations differently per sequence (`docs/noise_sensitivity.md`).
   Would be explored by relating the per-sequence range to sequence
   length and to CUT3R's per-frame pose differences between runs; not
   done.

5. **All three pipelines score below their matched random baseline at
   fully_predicted tau = 0.25** (EndoDAC -0.141, MASt3R-SLAM -0.151,
   CUT3R -0.113, all below at both cluster levels). CUT3R's random
   baseline is lower (0.265 vs 0.43 and 0.41) because its predicted
   unobserved area is larger (0.656 of the mesh), which makes random
   components merge more; its gap to its own baseline is the smallest of
   the three even though its IoU is the lowest. This is the reporting
   rule doing its job, not a ranking: D2b compares pipelines by region
   IoU directly, as the criterion states.

---

# Open issues

1. The EndoDAC bullet of the 2026-10-03 protocol entry describes the
   injection point incorrectly (section 1.4). Proposed text above; the
   entry is the author's to correct.
2. MASt3R-SLAM's primary run is not bit-reproducible (two-process
   timing; its unperturbed rerun differs from the primary by about as
   much as a noise run, section 8); its variability numbers mix the
   perturbation with that nondeterminism.
3. The variability subset (15 sequences, 65 regions) is small; ranges
   on the full corpus are not known.
4. Interpretations 1, 2 and 4 each name an analysis that was not run.
5. D2 (4 or more pipelines) is not triggered.
