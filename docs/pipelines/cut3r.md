# CUT3R feasibility, Stage 1: `c1_cecum_t1_v1`

Third-pipeline feasibility only. **No evaluation metric of any kind
(coverage, recall, false reassurance, false alarm, region IoU,
localization error) was computed, and nothing under `src/eval/` or
`src/eval_ext/` was run on CUT3R output.** One sequence, no full-corpus
run.

Pipeline integration checklist (there is no pipeline doc template in this
repository; the checklist starts here and is to be copied into the next
pipeline document):

- [x] Training data checked against docs/training_data_audit.md before
      Stage 1. (CUT3R: 32 general-domain datasets, no C3VD.)

Fixed before any result (task statement, 2026-10-02):

- checkpoint `cut3r_512_dpt_4_64.pth` (the README's final checkpoint),
  `--size 512`;
- online inference entry point (`demo.py`), not global alignment
  (`demo_ga.py`);
- confidence outputs are recorded and not used (`docs/eval_protocol.md`,
  2026-09-29); `--vis_threshold` is visualization only and is not passed
  anywhere;
- input: raw fisheye frames, no undistortion.

Code (committed): `scripts/cut3r_run.py` (copied entry script, vendored
code imported and never edited), `scripts/cut3r_grid_alignment_test.py`,
`scripts/cut3r_dump_mast3r_confident_pixels.py`,
`scripts/cut3r_stage1_checks.py`. Environment record:
`docs/pipelines/cut3r_env_pip_freeze.txt`. Outputs (not committed):
`results/pipelines/cut3r_stage1/`. Logs: `logs/cut3r_*.log`.

---

# MEASURED

## 1. Install

| item | value |
|---|---|
| CUT3R repository | https://github.com/CUT3R/CUT3R, cloned with `--recursive` to `scratch/pipelines/CUT3R` (gitignored) |
| CUT3R commit | `8bc15dc92a6d7fd92920b4ec81540d3dec7d3ecf` (2025-08-27); working tree unmodified apart from the built RoPE extension and the downloaded checkpoint |
| environment | conda env `cut3r`, Python 3.11.16; `torch==2.5.1+cu124`, `torchvision==0.20.1+cu124` (pip, CUDA wheel index, same build as the `mast3r-slam` env); then `pip install -r requirements.txt gdown`. `numpy==1.26.4`, `pillow==10.3.0` (both pinned by CUT3R's requirements.txt), `opencv-python==4.11.0.86`, `roma==1.6.1`. Full list: `docs/pipelines/cut3r_env_pip_freeze.txt` (114 packages) |
| RoPE CUDA kernels | `src/croco/models/curope`, `python setup.py build_ext --inplace`: built; the run logs do not contain the "cannot find cuda-compiled version of RoPE2D" fallback warning |
| checkpoint | `scratch/pipelines/CUT3R/src/cut3r_512_dpt_4_64.pth`, 3,173,761,006 bytes, Google Drive id `1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD` (README) |
| checkpoint SHA-256 | `45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103` |
| license | `scratch/pipelines/CUT3R/LICENSE` (5 lines): "CUT3R is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 License." CC BY-NC-SA 4.0, as expected |
| model load | `<All keys matched successfully>` |

Not installed (not needed for inference): gsplat, evo, open3d, the
`llvm-openmp<16` conda pin. One install error: the README's
`gdown --fuzzy <url>` fails with gdown 6.4.1 ("unrecognized arguments:
--fuzzy"); the file id was passed directly instead. Install log:
`logs/cut3r_install.log`.

## 2. End-to-end run, runtime, memory

`scripts/cut3r_run.py`: the inference path of `demo.py` (its own
`prepare_input`, then `ARCroco3DStereo.from_pretrained` and
`src.dust3r.inference.inference`), with the viser viewer removed. GPU 0
(`logs/cut3r_stage1_gpu_status.log`: 2 MiB used, 0% utilized at launch).

**Runs end to end on all 218 frames: yes.** 218 poses, 218 pointmaps, no
non-finite value. Two runs gave bit-identical poses and depth (max abs
difference 0.0).

| stage, 218 frames | seconds |
|---|---|
| frame loading and resize (CUT3R's `load_images`, CPU) | 16.9 |
| model load | 11.0 |
| inference | 12.0 (0.055 s/frame; second run 13.1) |
| saving depth + confidence (`np.savez_compressed`) | 20.1 |
| whole process, wall (second run, incl. imports and checkpoint hashing) | 83 |

**Memory does not stay constant with sequence length. It grows linearly.**
`torch.cuda.max_memory_allocated` / `max_memory_reserved`, model alone
3,039 MiB:

| frames | source | peak allocated (MiB) | peak reserved (MiB) | inference (s) |
|---|---|---|---|---|
| 27 | first 27 frames | 4,421 | 5,260 | 2.1 |
| 55 | first 55 frames | 5,843 | 7,146 | 3.3 |
| 109 | first 109 frames | 8,585 | 11,552 | 6.1 |
| 218 | the sequence | 14,125 | 19,908 | 12.0 |
| 436 | the sequence replayed forward-backward (length probe only) | 25,195 | 36,632 | 23.9 |
| 654 | same | 36,271 | 47,988 | 38.0 |

Least-squares line through the six points: **50.8 MiB per frame + 3,049
MiB** (largest residual 2 MiB). The README states this itself (line
121): "we accelerate the feedforward process by processing inputs in
parallel within the encoder, which results in linear memory consumption
as the number of frames increases." Reserved memory reached the card's
49,140 MiB at 654 frames without failing; reserved includes the
allocator's cache.

Extrapolated with that line, a 48 GB card holds about 895 to 907 frames
of allocated memory. Three of the 169 sequences are longer:
`c1_cecum_t4_v2`, `c1_cecum_t4_v3` (959 frames, 51.8 GB predicted) and
`c2_transverse2_t4_v3` (940 frames). 19 sequences have more than 654 frames, the
longest length actually tested. No run above 654 frames was attempted.

## 3. Outputs

Per input frame (`outputs["pred"][i]`, keys `camera_pose`, `conf`,
`conf_self`, `pts3d_in_other_view`, `pts3d_in_self_view`, `rgb`):

| output | per frame? | shape | notes |
|---|---|---|---|
| pose | **yes, every frame** | `camera_pose` (1, 7): translation + quaternion; `pose_encoding_to_camera` gives a 4x4 named `c2w_mats` (`src/dust3r/utils/camera.py:396-420`) | relative to the first frame; scale is the model's own |
| pointmap, camera frame | **yes, every frame** | `pts3d_in_self_view` (1, 400, 512, 3) | per-pixel 3D point in that frame's camera coordinates. Z-depth is its third component (this is what `demo.py` saves as depth, `depths_tosave = pts3ds_self_tosave[..., 2]`) |
| pointmap, first-frame coordinates | yes | `pts3d_in_other_view` (1, 400, 512, 3) | not needed for the protocol; used here only in section 5 |
| confidence | yes | `conf_self`, `conf` (1, 400, 512) | saved, never used |
| intrinsics | **no intrinsics output** | | `demo.py` estimates one focal per frame from the pointmap after the fact (`estimate_focal_knowing_depth`, Weiszfeld, principal point fixed at `(W // 2, H // 2)`): median 709.7 grid px over the 218 frames, range 645.8 to 801.3. Recorded, not used |
| RGB | yes | `rgb` (1, 400, 512, 3) | not saved |

No intrinsics are required as input. No pixel with z <= 0 in any frame.

**Internal resolution: 512x400.** `src/dust3r/utils/image.py:99-116`
(the DUSt3R loader, same code as MASt3R-SLAM's):
`img = exif_transpose(PIL.Image.open(...)).convert("RGB")`;
`img = _resize_pil_image(img, size)` with
`interp = PIL.Image.LANCZOS`,
`new_size = tuple(int(round(x * long_edge_size / S)) for x in img.size)`,
`return img.resize(new_size, interp)` (lines 65-72): 1350x1080 ->
512x410; then `halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8`
and `img.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))`: rows
5..404 kept, 5 resized rows cropped at top and at bottom, no horizontal
crop. The loader prints `1350x1080 --> 512x400`. 512x400 is not one of
the checkpoint's listed training resolutions (512x384, 512x336, ...);
`demo.py` uses it as is.

Saved by `scripts/cut3r_run.py`: `depth/<i>.npz` (`z`, `conf`),
`poses_c2w.npy`, `focal_weiszfeld.npy`, `MANIFEST.json`.

## 4. Grid mapping, verified externally

`scripts/cut3r_grid_alignment_test.py`
(`results/pipelines/cut3r_stage1/grid_alignment/grid_alignment_test.json`,
`logs/cut3r_grid_alignment_test.log`). A synthetic 1350x1080 PNG with 63
Gaussian markers (sigma 8 px, seed 20261001, the markers of the
MASt3R-SLAM test) is passed through `demo.py::prepare_input` ->
`load_images`. Markers are located on the 512x400 tensor the encoder
receives by thresholding and connected components, with no mapping
formula involved. This is not a forward-then-inverse round trip.

Free least-squares fit `grid = a * original + b`, per axis:

| axis | fitted a | fitted b | a expected (resized / original) | b if pixel-center aligned | b if corner aligned |
|---|---|---|---|---|---|
| x | 0.379260 | **-0.3107** | 0.379259 | -0.3104 | 0.0 |
| y | 0.379627 | **-5.3093** | 0.379630 | -5.3102 | -5.0 |

Residual (located - predicted, grid pixels):

| formula | mean dx | mean dy | std dx / dy | max abs dx / dy |
|---|---|---|---|---|
| pixel-center, `px = (ox + 0.5) / scale_w - 0.5 - half_crop_w` | +0.0001 | -0.0004 | 0.0040 / 0.0036 | 0.0156 / 0.0090 |
| corner, `px = ox / scale_w - half_crop_w` | -0.3102 | -0.3105 | 0.0040 / 0.0036 | 0.3202 / 0.3173 |

**Pixel-center aligned; crop 5 resized rows top and bottom, none
horizontally.** Parameters: `scale_w = 2.63672`, `scale_h = 2.63415`,
`half_crop_w = 0`, `half_crop_h = 5`, grid 512x400. This is the mapping
`scripts/pipeline_adapters.py::original_to_model` already implements for
MASt3R-SLAM (corrected 2026-10-01); the test asserts that function
against the located markers (tolerance 0.02 grid px). Pillow is 10.3.0
here and 11.3.0 in the MASt3R-SLAM environment; both give the same
alignment.

**Valid (non-vignette) pixels with no depth: 27,926 of 1,355,951,
2.060%** (13,380 above the grid, 13,389 below, 577 left of the first grid
column center, 580 right of the last), under the protocol's bilinear rule
(`BilinearGridMap`: a pixel has depth iff its mapped coordinate lies in
[0, 511] x [0, 399]). Same value as MASt3R-SLAM, as the grids are
identical.

What the test does not cover: that the DPT head's output pixel (r, c)
corresponds to input pixel (r, c). The output has the input's shape
(400x512); the correspondence itself is assumed, as it was for
MASt3R-SLAM.

## 5. Pose convention

`scripts/cut3r_stage1_checks.py`
(`results/pipelines/cut3r_stage1/c1_cecum_t1_v1/stage1_checks.json`).
Per frame pair, GT relative transform against the predicted relative
transform with the saved matrix read as camera-to-world (A) and as
world-to-camera (B).

| pairs | reading | mean rot. error | mean translation-direction cosine | median cosine | pairs with cosine > 0 |
|---|---|---|---|---|---|
| stride 1 (217) | A, camera-to-world | 1.35 deg | **+0.488** | +0.647 | 81.1% |
| stride 1 (217) | B, world-to-camera | 1.25 deg | -0.489 | -0.657 | 18.9% |
| stride 10 (21) | A, camera-to-world | 11.33 deg | **+0.577** | +0.787 | 90.5% |
| stride 10 (21) | B, world-to-camera | 10.05 deg | -0.578 | -0.781 | 14.3% |

Same check on the same sequence, earlier pipelines (their own documents):
EndoDAC +0.976 / +0.886 for the chosen reading (frames 0-19 and 30-49,
`docs/pipelines/endodac.md` section 7); MASt3R-SLAM +0.930
(`docs/pipelines/mast3r_slam.md` Q3, keyframe pairs). CUT3R's cosine is
positive for reading A but is not unanimous across pairs (41 of 217
stride-1 pairs are negative).

Because of that, a second check that does not involve GT: CUT3R also
predicts each frame's points in first-frame coordinates
(`pts3d_in_other_view`). Median distance between `T @ pts3d_in_self_view`
and `pts3d_in_other_view`, relative to the median point norm:

| frame | A: T = saved pose | B: T = inverse of saved pose |
|---|---|---|
| 0 | 0.021 | 0.021 |
| 50 | **0.068** | 0.440 |
| 100 | **0.046** | 0.789 |
| 150 | **0.024** | 0.431 |
| 217 | **0.036** | 0.962 |

**The saved pose is camera-to-world** (frame 0's pose is near identity, so
it does not discriminate). This agrees with the code's own name
(`c2w_mats`) and with `demo.py`, which applies the pose to self-view
points to get world points.

## 6. Ray-direction check

Angle between each model's implied per-pixel ray and this project's
Scaramuzza ray (`src/geometry/camera.py::unproject`), frame 0, on the
185,725 pixels MASt3R is confident on (the population of the Stage 2
MASt3R check and of the EndoDAC check; regenerated with one MASt3R forward
pass by `scripts/cut3r_dump_mast3r_confident_pixels.py`). MASt3R-SLAM and
CUT3R share the 512x400 grid, so the three models are compared at
identical pixels. Original-frame coordinates from the pixel-center
mapping. CUT3R's ray is the direction of `pts3d_in_self_view`; EndoDAC's
is the pinhole ray of the intrinsics it was run with.

| | EndoDAC | MASt3R-SLAM | CUT3R |
|---|---|---|---|
| median angle | 15.42 deg | 19.82 deg | **24.09 deg** |
| mean | 16.12 deg | 20.26 deg | 24.23 deg |
| p95 | 30.61 deg | 36.33 deg | 41.82 deg |
| max | 36.66 deg | 42.43 deg | 48.31 deg |

EndoDAC and MASt3R-SLAM values previously recorded with the
corner-aligned coordinates: 15.42 and 19.82 deg median. Recomputed here
with the pixel-center coordinates they are 15.422 and 19.816 (previously
15.420 and 19.815): the open item of the 2026-10-01 correction block in
`docs/pipelines/mast3r_slam.md` changes these by under 0.003 deg.

CUT3R against MASt3R-SLAM directly, same pixels: median 4.22 deg, p95
5.82 deg. CUT3R over all 204,800 grid pixels of frame 0 (no confidence
selection): median 24.66 deg, p95 46.51 deg, max 62.24 deg. CUT3R's
pointmap is close to a pinhole of focal about 715 grid px (median of
radius / tan(angle) over pixels: 715.5; `demo.py`'s Weiszfeld estimate
for frame 0: 713.3), i.e. about 39 deg horizontal field of view for the
512-px-wide grid.

## 7. Trajectory quality (not an evaluation metric)

Position-only Umeyama (Sim(3)) alignment to GT, all 218 frames; depth
scale per frame = median(GT mm) / median(prediction) over GT-valid pixels
that have a prediction, sequence value = median over frames, relative IQR
= (q75 - q25) / median. Same sequence for all three.

| | EndoDAC | MASt3R-SLAM | CUT3R |
|---|---|---|---|
| frames with pose | 218 | 218 | 218 |
| ATE RMSE after Sim(3) | 9.23 mm | 13.40 mm | **20.62 mm** |
| endpoint drift / GT path length (348.3 mm) | 2.78% | 5.41% | **0.93%** (3.26 mm) |
| Sim(3) scale `s_pose` | 557.87 | 50.01 | 22.97 |
| depth scale median | 177.01 | 38.24 | **30.52** |
| depth scale relative IQR | 0.134 | 0.239 | **0.395** |
| depth scale per-frame range | | | 19.01 to 41.93 |

Sources: EndoDAC `results/d1/stage_a_summary.csv`; MASt3R-SLAM
`results/pipelines/mast3r_slam_full_run/c1_cecum_t1_v1/` (ATE, drift,
`s_pose`), depth scale median from the pipeline 2 evaluation's
`metrics.json` (pixel-center mapping), relative IQR from Stage 3's
descriptives (nearest-neighbour lookup, corner-aligned mapping; not
recomputed). CUT3R: `stage1_checks.json`, bilinear pixel-center mapping.

CUT3R's depth scale (30.5) and pose scale (23.0) differ by a factor of
1.33; for MASt3R-SLAM the two are 38.2 and 50.0.

## 8. Missing frames

**None.** 218 of 218 frames have a pose and a pointmap; 0 non-finite
poses, 0 non-finite pointmaps, 0 pixels with z <= 0. The online path has
no tracking state that can be lost and no keyframe selection: every input
frame gets an output by construction. Pixels with no depth exist only
through the crop (section 4).

## 9. Projected runtime and storage, 169 sequences

Basis: this one 218-frame sequence, per-frame costs assumed constant
(inference was 0.055 to 0.060 s/frame from 55 to 654 frames), 67,886
frames in the corpus.

| component | measured, 218 frames | per frame | corpus |
|---|---|---|---|
| frame extraction from the archive (zip read, 388 MB) | 1.7 s | 0.008 s | 0.15 h |
| loading and resize | 16.9 s | 0.078 s | 1.46 h |
| inference | 12.0 s | 0.055 s | 1.04 h |
| saving z + conf | 20.1 s | 0.092 s | 1.74 h |
| per-sequence fixed (model load 11 s, imports and start-up about 15 s) | 26 s | | 1.2 h |
| **total, one process, one GPU** | | | **about 5.6 h** |

Earlier projections in this project were low by up to 1.8x; with that
factor the range is **5.6 to 10 h**. Not included: any evaluation (the
MASt3R-SLAM evaluation took 21.8 GPU-process hours and is independent of
the inference method), a per-sequence checkpoint hash (10 s, can be done
once), and anything done about the memory limit below.

Storage: 1.36 MB per frame with confidence (297.5 MB for 218 frames),
**about 93 GB** for the corpus; 0.67 MB per frame and **about 45 GB**
with z only. Compressibility of other sequences is not known; MASt3R-SLAM
on the same grid took 85 GB with confidence.

**The projection does not hold for the whole corpus as the pipeline is
fixed now.** By the memory line of section 2, three sequences (959, 959
and 940 frames) need more than a 48 GB card, and every
sequence needs a mostly free GPU (median sequence, 383 frames: about
22.5 GB allocated; 61 sequences above 436 frames need over 25 GB). On
this shared machine two GPUs were free at the time of the run.

---

# INTERPRETATION

Each item is a reading, not a finding.

1. **CUT3R's high ATE with low endpoint drift suggests the trajectory's
   shape is distorted while its ends stay registered.** The sequence is a
   loop tagged "fast"; a persistent-state model can re-register to
   surfaces it has seen without keeping metric path shape in between.
   Would be confirmed by the per-frame aligned position error peaking
   mid-sequence and falling towards both ends; refuted if the error is
   flat or grows monotonically.

2. **The non-unanimous translation cosines reflect noisy per-step
   translation, not a mixed pose convention.** The pointmap consistency
   check separates the two readings at every tested frame after the
   first, and the stride-10 cosine is higher than the stride-1 one.
   Would be confirmed if the negative-cosine pairs are those with the
   smallest GT step length; refuted if they cluster in one part of the
   sequence regardless of step length.

3. **The depth scale's relative IQR of 0.40 (per-frame scale between 19
   and 42) would matter under the protocol's single per-sequence depth
   scale.** A frame whose own scale is far from the sequence median gets
   depths off by that ratio before the tau test. Whether it matters is
   an evaluation question and was not measured. The per-frame scale
   series is in `stage1_checks.json` only as summary statistics; plotting
   it against frame index would show whether it drifts or jumps.

4. **The 24 deg ray disagreement is the pinhole-like pointmap meeting a
   fisheye image**, as for MASt3R-SLAM (4 deg apart from it): the model
   places a ~39 deg field of view on an image whose true field of view is
   much wider. The protocol uses only Z-depth and pose with the project's
   own camera model, so this does not enter by itself; it does say that
   CUT3R's Z is the Z of a point along a different ray than the one the
   evaluation casts. Would be quantified by comparing radial profiles of
   predicted and GT depth; not done here.

---

# Open issues

1. **Memory limit (decision needed before any corpus run).** The fixed
   entry point cannot process the three longest sequences on a 48 GB
   card, by extrapolation; lengths above 654 frames are untested. Options,
   none tried: (a) report those sequences as pipeline failures (out of
   memory), per the failure-reporting rule; (b) feed frames through the
   vendored `inference_recurrent` / `inference_step` path, which exists in
   `src/dust3r/inference.py` and may avoid the parallel encoder pass, in
   which case its outputs would first have to be shown equal to the
   `inference` path's on a sequence that fits; (c) a larger card. (b)
   changes the entry point fixed for this stage.
2. The DPT output-to-input pixel correspondence is assumed (section 4).
3. 512x400 is outside the checkpoint's listed training resolutions
   (section 3). No alternative was tried; changing it would be a
   per-method tuning choice.
4. One sequence only. Missing-frame behaviour, depth-scale spread and
   trajectory quality on other sequences are unknown.
5. Stage 3's nearest-neighbour depth-scale descriptives for MASt3R-SLAM
   still use the corner-aligned mapping (the relative IQR of 0.239 in
   section 7 is that value).
6. `demo.py` seeds only Python's `random`; `scripts/cut3r_run.py` also
   seeds torch and numpy. Two runs were bit-identical on this GPU;
   determinism across GPUs or drivers was not tested.
