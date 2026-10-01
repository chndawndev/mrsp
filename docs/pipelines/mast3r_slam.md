# MASt3R-SLAM feasibility, Stage 1: `c1_cecum_t1_v1`

Second-pipeline feasibility only. **No evaluation metric of any kind
(coverage, recall, false reassurance, false alarm, IoU, localization error)
is computed here, and no script under `src/eval/` or `src/gt/` is run on
any MASt3R-SLAM output.** This stage computes no metric, so it needs no
D2 pre-registration entry to proceed. **Correction (2026-09-28, Stage
2):** an earlier version of this paragraph said no D2 pre-registration
entry exists at all — wrong. `docs/success_criteria.md` section 6 (the
2026-09-27 deviation, commit `14894c3`) already defines D2's primary
endpoints and was pushed before this Stage 1 document's own commit
(`a13211a`) — a factual miss on my part, not a timeline issue. Rules on
the distinction between "trajectory quality" (computed here,
self-contained) and "evaluation" (not computed here) follow the task's
own hard constraint.

MEASURED sections report only what was directly observed (code read, files
opened, commands run). INTERPRETATION sections are marked explicitly, with
what would confirm or refute them.

---

## Install record

- **Repo**: `github.com/rmurai0610/MASt3R-SLAM` (CVPR 2025), cloned
  `--recursive`.
- **Commit**: `e6f4e3d474fad0e11f561482012be864ba8c3f17` (2025-11-08
  23:36:29 -0800).
- **Submodules**: `thirdparty/eigen` @ `bddaa99e151244402c6e804e01b970288650da6b`,
  `thirdparty/in3d/thirdparty/pyimgui` @ `c79137d6c47040e6ce226f8109868a3018d5354b`.
  (`thirdparty/mast3r` and `thirdparty/in3d` are plain vendored
  directories, not git submodules.)
- **Environment**: conda env `mast3r-slam`, Python 3.11 (isolated,
  matching the `endodac` env precedent). PyTorch installed via pip with an
  explicit CUDA wheel index (not conda — see Errors below), matching the
  EndoDAC precedent:
  `pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124`.
  Confirmed: `torch 2.5.1+cu124`, `torch.version.cuda == '12.4'`,
  `torch.cuda.is_available() == True` on GPU 7.
- **Install commands** (from the repo's own README):
  ```
  pip install --no-build-isolation -e thirdparty/mast3r
  pip install --no-build-isolation -e thirdparty/in3d
  pip install --no-build-isolation -e .
  ```
  All three succeeded. `--no-build-isolation` was needed on all three (not
  just the root package, which documents it) because `curope`'s and
  `mast3r_slam`'s setup scripts `import torch` at build time, invisible to
  pip's isolated build venv otherwise.
- **Custom CUDA extensions built from source**: `curope` (7 architectures,
  compute_50–compute_90), `mast3r_slam_backends` (6 architectures,
  compute_60–compute_86), `lietorch` (1 architecture, compute_89, matching
  this box's actual GPU). All built and imported successfully.
- **Post-install import check** (clean, no exceptions):
  `torch`, `cv2` (`opencv-python-5.0.0.93`), `numpy` (`1.26.4`),
  `mast3r_slam`, `lietorch`, `mast3r_slam.mast3r_utils.load_mast3r`.
- **Flagged, tested, not breaking**: `opencv-python 5.0.0.93` declares
  `numpy>=2`, but `in3d`'s `trimesh[easy]` dependency chain pulled numpy
  down to `1.26.4`. Import of both together succeeds; no runtime failure
  observed in the actual run (Q1). Not exercised beyond that.
- **Full logs**: `logs/mast3r_slam_env_setup.log`,
  `logs/mast3r_slam_pytorch_install.log`, `logs/mast3r_slam_pip_install.log`.

### Errors hit and fixes

1. **Conda channel-priority bug**: `conda install pytorch==2.5.1 ...
   pytorch-cuda=12.4 -c pytorch -c nvidia` silently resolved `pytorch`
   itself from `conda-forge` as a CPU-only build, despite the explicit
   channel flags (an earlier unrelated `git-lfs` conda install into base
   had already enabled conda-forge). Symptom:
   `torch.cuda.is_available()` False, `torch.version.cuda` None.
   **Fix**: abandoned conda for PyTorch, used the pip
   `--index-url` command above instead (same fix already established for
   EndoDAC).
2. **`curope` build-isolation failure**: `ModuleNotFoundError: No module
   named 'torch'` during "Getting requirements to build wheel". **Fix**:
   `--no-build-isolation`.
3. **`pyimgui` build failure**: `core.h: No such file or directory` —
   Cython wasn't installed, so `core.pyx` was never compiled to
   `core.h`/`core.cpp` before the C++ build. **Fix**: `pip install
   "cython<3"` (pinned below 3.x; pyimgui's build is known incompatible
   with Cython 3's changed semantics).

### License

- Repo `LICENSE.md`: **CC BY-NC-SA 4.0**.
- `thirdparty/mast3r/LICENSE`: **CC BY-NC-SA 4.0** (Naver/DUSt3R).
- `thirdparty/mast3r/CHECKPOINTS_NOTICE`: checkpoints additionally carry
  training-data license terms restricting use to **non-commercial
  academic research**. Fine for this research use; a real constraint on
  any future commercial framing.

### Checkpoints

Downloaded from the Naver Labs server named in the README, no auth
required. SHA-256 recorded for *this specific download*'s reproducibility
(no upstream-published checksum exists to verify against, same caveat as
EndoDAC's weights) in `logs/mast3r_slam_checkpoint_sha256.txt`:

| File | Size | SHA-256 |
|---|---|---|
| `MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth` | 2.75 GB | `e28f91b488554653e2b46ddae9c78c1143e0bcb2e27d3e26cdb0b717f1568eb2` |
| `..._retrieval_trainingfree.pth` | 8.4 MB | `000a324f06f9369ab3e32d03dd4b8a838d48e98e3f3bf9d7dd034dce1bfd4c50` |
| `..._retrieval_codebook.pkl` | 256 MB | `700af9e31ad0ca6159955dc80b4fd6e14234780b52f5c91884f6553991933c1a` |

Stored in `scratch/pipelines/MASt3R-SLAM/checkpoints/`, gitignored, never
committed.

---

## Q1: Does it run end to end on `c1_cecum_t1_v1`? Runtime, peak GPU memory.

**MEASURED. Yes.** `scripts/mast3r_slam_run.py --sequence c1_cecum_t1_v1
--gpu 7` (uncalibrated mode, `config/base.yaml`, `--no-viz`, headless),
GPU 7 selected via `scripts/gpu_status.py` (32.1 GB free, 0% util at
selection time; full `nvidia-smi` logged in `logs/mast3r_slam_run.log`).

- **Return code**: 0. No exception, no crash, no reloc/track-loss message
  in stdout or stderr (`logs/mast3r_slam_run_c1_cecum_t1_v1.{stdout,stderr}.log`).
- **Input**: 218 raw fisheye frames (1350x1080, RGB), no undistortion, via
  `scripts/mast3r_slam_prepare_input.py` (extracts only `rgb/*.png` zip
  members, same selective-extraction pattern as
  `scripts/endodac_inference.py`).
- **Wall time**: 70.2 s for 218 frames (≈5.6 FPS steady-state, per the
  run's own `FPS:` log lines — well below the paper's >15 FPS claim; see
  INTERPRETATION below).
- **Peak GPU memory**: 27,282 MiB total on the device (sampled every 2 s
  during the run). GPU 7 had another process already using 16,428 MiB at
  baseline (not preemptible, not ours), so this job's own incremental
  peak is **≈10.9 GB**.

**INTERPRETATION**: 5.6 FPS vs. the paper's >15 FPS claim is plausibly
explained by GPU 7 sharing memory bandwidth/SM time with another
already-running 16.4 GB process throughout this run, not by a MASt3R-SLAM
performance problem — unconfirmed, would need a rerun on an otherwise-idle
GPU to isolate.

---

## Q2: Outputs — per-frame or keyframe-only? Depth/pointmaps? Resolution, confidence?

**MEASURED, from opening the actual files (`scripts/
mast3r_slam_inspect_outputs` step folded into direct inspection below) and
reading `mast3r_slam/evaluate.py`'s `save_traj`/`save_reconstruction`/
`save_keyframes` and `mast3r_slam/tracker.py`'s `FrameTracker.track`:**

- **Poses**: `logs/c1_cecum_t1_v1/c1_cecum_t1_v1.txt`, TUM format
  (`timestamp tx ty tz qx qy qz qw`), **keyframe-only**: 20 lines for 218
  input frames. Timestamp is `frame_index / 30.0` (from `RGBFiles`'s fixed
  30 fps assumption — this project's actual frame rate is unrelated;
  `frame_index = round(timestamp * 30)` recovers the original index
  exactly for all 20 keyframes, verified). Keyframe frame-indices selected
  by the run: `0, 30, 42, 45, 48, 51, 68, 71, 82, 87, 93, 105, 111, 122,
  126, 130, 144, 156, 161, 168` — the last 49 input frames (169-217) never
  produced a new keyframe.
- **Point cloud / depth**: `logs/c1_cecum_t1_v1/c1_cecum_t1_v1.ply`, one
  **fused, confidence-filtered, world-frame point cloud across all
  keyframes** (`save_reconstruction`, `mast3r_slam/evaluate.py`) —
  per-frame or per-keyframe **depth maps are not saved to disk by
  default**. Internally, `Frame.X` (pointmap, camera frame) and `Frame.C`
  (confidence) exist for **every** incoming frame, not just keyframes —
  `FrameTracker.track` (`mast3r_slam/tracker.py:27`) calls
  `mast3r_match_asymmetric` and `frame.update_pointmap(Xff, Cff)` on every
  frame, matched against the last keyframe — but this per-frame
  X/C is transient (overwritten each loop iteration, never persisted)
  unless the frame is promoted to a keyframe.
- **Keyframe RGB copies**: `logs/c1_cecum_t1_v1/keyframes/c1_cecum_t1_v1/
  <timestamp>.png`, 20 files, plain RGB re-dumps of the input keyframe
  images (`save_keyframes`) — no depth, no confidence map saved alongside
  them.
- **Resolution**: MASt3R itself operates at a fixed internal resolution
  from the model config (`img_size=(512, 512)` in the loaded checkpoint's
  instantiation line, confirmed in stdout); this is separate from the
  1350x1080 input, and the point cloud is in metric world coordinates
  (not tied to a pixel grid) so no direct "output resolution" applies to
  it the way it would to a depth image.
- **Confidence**: `Frame.C` exists per-frame internally and is used
  (thresholded) when building the fused `.ply`, but is **not saved
  per-frame/per-keyframe as a standalone confidence map file**.

**UNKNOWN**: why keyframe selection stopped at frame 168 with 49 more
frames remaining. No error, warning, or relocalization message appears in
the logs. Could be (a) genuinely little viewpoint change in that segment
(consistent with the trajectory-quality numbers below, where the fused
GT path length is short), or (b) a keyframe-selection heuristic interacting
with this data in a way not otherwise diagnosed. Settling this would need
either visual inspection of frames 168-217 or per-frame instrumentation
(see Q4 Option A).

---

## Q3: Pose convention

**MEASURED**, same per-keyframe-pair method as EndoDAC's decisive check
(`docs/pipelines/endodac.md` §7, `scratch/pipelines/
endodac_pose_validation_perstep.py`), run via `scripts/
mast3r_slam_pose_check.py`. For each of the 19 consecutive keyframe pairs,
compared the GT relative transform (`inv(C2W_i) @ C2W_j` from `pose.txt`)
against both the predicted relative transform as saved
(`inv(T_i) @ T_j`, "Hypothesis A") and its inverse ("Hypothesis B"), using
rotation angle and translation-direction cosine similarity — decisive
because translation direction flips sign under inversion while
near-identity rotations do not.

**Result** (`logs/mast3r_slam_pose_check.log`):

| | mean rotation error | mean translation-direction cosine similarity |
|---|---|---|
| Hypothesis A (`T_WC` as saved) | 5.71° | **+0.9298** |
| Hypothesis B (`inv(T_WC)`) | 5.94° | -0.9279 |

**Conclusion: `T_WC` as saved is camera-to-world**, consistent with both
the variable's own name in the codebase and the CLAUDE.md-frozen
convention already used for GT poses in this project. No inversion is
needed to compare against `pose.txt`.

---

## Q4: Options to get every frame (not just keyframes)

Feasibility-level listing only, per instructions — **no option chosen or
implemented**.

| Option | What it gives | Trade-off |
|---|---|---|
| **A. Log every frame's tracked pose** (`states.set_frame(frame)` already runs every iteration in `main.py`; a small addition would append `frame.T_WC`/`frame.X`/`frame.C` to disk each iteration instead of discarding them) | Per-frame pose, pointmap, and confidence for all 218 frames, at zero extra model inference cost (the forward pass already happens for every frame in `FrameTracker.track`) | Requires a code change to `main.py`'s loop (a copy/wrapper script, not editing the vendored repo in place); per-frame poses are tracked against the *last keyframe*, not globally optimized the way keyframe poses are (keyframes get backend/global bundle adjustment via `FactorGraph`, non-keyframes do not) — so per-frame poses would likely be noisier than keyframe poses |
| **B. Interpolate between keyframe poses** (e.g. SE(3)/Sim(3) spherical linear interpolation using the already-saved 20 keyframe poses) | Full coverage from data already on disk, no rerun | No sensing of actual per-frame motion — pure geometric guess, will be systematically wrong wherever a non-keyframe frame's real trajectory deviates from constant-velocity between its neighboring keyframes; unusable for depth (interpolation has no way to produce plausible surface geometry) |
| **C. Force every frame to be a keyframe** (bypass or defeat the keyframe-selection heuristic, e.g. via a config knob if one exists, or by treating each frame as needing relocalization) | Real, tracked pose+depth+confidence for every frame, globally optimized like a keyframe | Not confirmed this is exposed as a supported mode without a deeper code change; almost certainly changes runtime substantially (Q6's projection assumes the as-shipped keyframe rate) and changes what the pipeline is actually doing relative to its designed operating point |
| **D. Re-run a dedicated per-frame MASt3R depth pass** (call `mast3r_inference_mono` directly per frame, outside the SLAM loop, for depth only; pair with Option A or C for pose) | Per-frame depth without touching the tracking/keyframe logic at all | Doubles compute for depth (extra forward pass per frame); still needs a pose source per frame, so doesn't solve the pose half alone |

---

## Q5: Trajectory quality (not an evaluation metric) vs. EndoDAC

**MEASURED**, self-contained closed-form Umeyama (duplicated, not
imported, from `scratch/pipelines/endodac_scale_analysis.py:29-43` — same
reasoning as that script: this task's hard constraint bars `src/eval/`
even for non-metric numbers), via `scripts/
mast3r_slam_trajectory_quality.py`. Aligns the 20 predicted keyframe
positions (Hypothesis A / camera-to-world, per Q3) to their corresponding
GT positions with a similarity transform, computes ATE (RMSE
post-alignment) and endpoint drift as a fraction of GT path length —
same definitions `results/pipelines/endodac_scale_analysis.json` uses.

| | MASt3R-SLAM (this run, 20 keyframes) | EndoDAC (`c1_cecum_t1_v1`, `results/d1/stage_a_summary.csv`) |
|---|---|---|
| Sim(3) scale recovered | 50.57 | 557.87 (`s_pose`) |
| ATE (RMSE, post-alignment) | 9.26 mm | see `results/d1/stage_a_summary.csv` row for this sequence |
| GT path length compared against | 302.2 mm (over frames 0-168 only) | full 218-frame path |
| Endpoint drift, fraction of GT path | 4.70% | 2.78% (already recorded) |

**Not directly comparable without a caveat**: MASt3R-SLAM's 20 keyframes
only span input frames 0-168 (Q2's open question), so its GT path length
denominator (302.2 mm) covers less of the sequence than EndoDAC's
full-218-frame comparison. The endpoint-drift percentages are on
different underlying path lengths, not the same held-out track.

**INTERPRETATION**: on the portion of the trajectory it did track,
MASt3R-SLAM's endpoint drift (4.70%) is in the same order of magnitude as
EndoDAC's (2.78%), not dramatically worse — but this is not a controlled
comparison given the different endpoints, and neither number is an
evaluation metric under this project's pre-registered definitions.

---

## Q6: Projected runtime, all 169 sequences

**MEASURED extrapolation**, same method as `docs/pipelines/endodac.md` §4
and `docs/d1_stage_a.md`'s pre-flight timing: measured per-frame rate ×
corpus-wide total frame count.

- Measured: 70.227 s / 218 frames = 0.3221 s/frame.
- Corpus total: **67,886 frames** across 169 registered sequences
  (`docs/dataset_official_description.md:9`).
- Projected: 67,886 × 0.3221 s ≈ **21,869 s ≈ 6.07 hours**, single GPU,
  sequential, assuming this sequence's per-frame rate is representative
  (it is the only sequence measured so far — not confirmed representative
  of longer/shorter sequences or different keyframe densities).

**INTERPRETATION**: 6.07 hours is in the same range as EndoDAC's
full-corpus run (4h41m) and well under the "ask before any run expected
to take over an hour" threshold *for this single-sequence stage*, but a
full 169-sequence run would need separate approval per CLAUDE.md.

---

## Open issues

1. ~~Why keyframe selection stops at frame 168/217 for this sequence~~
   — **diagnosed in Stage 2** (below): little camera motion, not
   tracking degradation.
2. ~~No D2 pre-registration entry exists in `docs/success_criteria.md`~~
   — **corrected, 2026-09-28 (Stage 2)**: this was wrong. `docs/
   success_criteria.md` section 6 (2026-09-27 deviation, commit
   `14894c3`) already defines D2's primary endpoints (region IoU, false
   reassurance) and a validity gate, and was pushed before this Stage 1
   document's own commit (`a13211a`). This stage still computed no
   metric, so the entry wasn't load-bearing for anything done here, but
   the claim itself was factually wrong and is corrected, not removed,
   for the record.
3. The FPS-vs-paper gap (Q1 INTERPRETATION) is unconfirmed — would need a
   rerun on an idle GPU to isolate contention from a genuine performance
   difference.
4. Q4's four options are unranked and unimplemented by design (the task
   asked not to choose); a future stage would need to pick one before any
   full-corpus MASt3R-SLAM run.

---

# Stage 2: per-frame outputs and the frame 169-217 gap

Still `c1_cecum_t1_v1` only. Still **no evaluation metric of any kind
computed, and no script under `src/eval/`/`src/gt/` run on any
MASt3R-SLAM output** — the region-IoU validity gate passed
(`docs/region_iou_gate.md`, commit `2e38f37`) before this stage started,
per the task's own precondition, but that gate result is not used here;
this stage still produces no metric. Implements Option A from Stage 1's
Q4 (log every frame's tracked pose relative to its reference keyframe,
re-anchor after the run using the keyframe's final globally-optimized
pose) and diagnoses the frame 169-217 gap with real evidence.

## Method: instrumented tracker + copied entry script

Two new files, neither an in-place edit of the vendored MASt3R-SLAM
clone:

- `scripts/mast3r_slam_perframe_tracker.py`:
  `InstrumentedFrameTracker(FrameTracker)`, a subclass that copies
  `mast3r_slam/tracker.py::FrameTracker.track()`'s body verbatim (read in
  full before writing) and records, at both of its return points,
  `frame_id`, `ref_keyframe_frame_id` (`keyframe.frame_id`, read at the
  top of `track()`, before this frame's own tracking/promotion logic
  runs), `T_ref_to_frame` (the raw 8-float Sim3 `T_CkCf = T_WCk.inv() *
  T_WCf`, already computed internally, scale preserved, not dropped),
  `match_frac`, `match_frac_k`, and `skipped`.
- `scripts/mast3r_slam_run_perframe.py`: a copy of `main.py`'s driver
  (read in full before writing), with three changes: (1) uses
  `InstrumentedFrameTracker`; (2) always headless; (3) after every
  `tracker.track(frame)` call, saves that frame's camera-frame Z depth
  and confidence at MASt3R's internal resolution to
  `depth/{frame_id:04d}.npz`; (4) at the very end, dumps every
  keyframe's FINAL Sim3 pose and FINAL (fully-fused) Z/confidence,
  overwriting that frame_id's tracking-time dump — a keyframe's depth is
  only correct once fully fused (confirmed by reading `mast3r_slam/
  global_opt.py`: the backend only ever calls `keyframes.update_T_WCs`,
  poses only; a keyframe's `X_canon`/`C` are updated exclusively inside
  `track()`'s `keyframe.update_pointmap(Xkk, Ckf)` call, every time a
  later frame is tracked against it).

**MEASURED — rerun**: `logs/mast3r_slam_run_perframe.log`. GPU 7 (32.1GB
free, 0% util, via `scripts/gpu_status.py`). Same checkpoints/config/
uncalibrated mode as Stage 1. Returncode 0. Wall time **54.9s** (faster
than Stage 1's 70.2s single-GPU run of the unmodified pipeline — plausibly
less GPU contention this time, not investigated further). **Manifest**:
`n_frames_total=218, n_keyframes=20, n_tracking_records=217` (218 frames −
1 anchor frame that never calls `track()` = 217, exactly as expected).
**Keyframe frame-index set is bit-identical to Stage 1's**
(`[0, 30, 42, 45, 48, 51, 68, 71, 82, 87, 93, 105, 111, 122, 126, 130,
144, 156, 161, 168]`) — the pipeline is deterministic across reruns, not
assumed. Per-frame depth+confidence storage: 293,822,448 bytes for 218
frames (**1.348 MB/frame**, float32, `np.savez_compressed`).

## Step 4: pose reconstruction

`scripts/mast3r_slam_reconstruct_poses.py`: for each of the 218 frames,
a keyframe uses its own final Sim3 pose directly; a non-keyframe frame is
reconstructed as `T_WC_ref_final * T_ref_to_frame` (`lietorch.Sim3`
composition, the same operator `track()` itself uses, substituting the
reference keyframe's later/final pose — exact, not an approximation,
because `T_ref_to_frame` doesn't depend on which `T_WC_ref` produced it),
converted to SE3 via `mast3r_slam.lietorch_utils.as_SE3` only at this
final step (same helper `save_traj` uses for the keyframe-only TUM file —
same convention, not a new one).

**MEASURED**: `logs/mast3r_slam_reconstruct_poses.log`.
**`n_total_frames=218 n_reconstructed=218 n_failed=0`** — every frame got
a pose, none skipped/unreconstructable. **Sanity check**: the 20
keyframes' reconstructed poses (direct passthrough) against the
already-saved `c1_cecum_t1_v1_perframe/c1_cecum_t1_v1.txt` TUM file:
**`max_abs_diff=0.0`** across all 20 — bit-exact, confirming the
reconstruction script's keyframe passthrough path and the TUM-writing
path agree exactly (a correctness check on this script, not a new
metric).

## Step 5: resolution mapping, vignette crop fraction, ray-direction check

`scripts/mast3r_slam_resolution_mapping.py`. Calls MASt3R-SLAM's own
`resize_img(img, 512, return_transformation=True)` (imported, not
reimplemented) on one real frame.

**MEASURED — mapping** (`results/pipelines/mast3r_slam_perframe/
c1_cecum_t1_v1/resolution_mapping.json`): original 1350×1080 → MASt3R
grid **512×400**. `scale_w=2.63672, scale_h=2.63415, half_crop_w=0.0,
half_crop_h=5.0`. All cropping is vertical only (the long side, width,
is resized with zero crop; height loses a 5px band top and bottom of the
already-resized 410px-tall image). Forward (original→model) and inverse
(model→original) formulas implemented and documented in the script.
**Round-trip test**, 2000 random points + 5 named corners/center: **max
error 1.6e-13 px, mean 2.1e-14 px** — the two formulas are exact inverses
of each other (floating-point-only residual).

~~**MEASURED — vignette crop fraction**: of the fixed 102,049-pixel
vignette mask (`docs/eval_protocol.md` §8, loaded by duplicating
`src/eval/evaluable.py`'s 3-line unpack rather than importing that
locked file), **12,354 pixels (12.11%) fall outside the region the
MASt3R grid covers** — i.e. get no depth at all, purely from the
vertical crop. Vignette pixels concentrate near the image's physical
edges (a hardware property), so they are disproportionately affected by
even a small (5px-of-410, ~1.2%-of-height) crop relative to the frame as
a whole.~~

**Correction (2026-09-28, Stage 3 pre-flight)**: the paragraph above used
the vignette mask **backwards**. `src/eval/evaluable.py::load_vignette_mask`'s
own docstring (read, not modified): *"True = vignette (non-evaluable)."*
`evaluable_pixel_mask` computes evaluability from `~vignette_mask` — mask
**False** is the valid/imaged population. The number above was computed
on `np.nonzero(vignette_mask)` (mask **True**, the small 102,049px black
border itself), answering "how much of the black border does the crop
cut into" — not "how much of the actually-imaged area loses depth to
cropping," which is what the surrounding prose claimed and what matters
for anything downstream. **Corrected, computed on the ~1,355,951-pixel
valid (`~vignette_mask`) population**
(`scripts/mast3r_slam_check_vignette_and_endodac_ray.py`,
`results/pipelines/mast3r_slam_perframe/c1_cecum_t1_v1/
vignette_and_endodac_ray_check.json`): **28,898 of 1,355,951 valid pixels
(2.13%) fall outside the region the MASt3R grid covers** — much closer to
the naive whole-frame estimate (~2.4% of rows lost to a 5px-of-410
vertical crop) than the wrong 12.11% figure, which makes sense now that
the population matches: a crop band near the top/bottom edges removes a
proportional slice of the (much larger, roughly uniformly-distributed)
valid region, not a disproportionate slice of the small edge-concentrated
vignette border.

**MEASURED — ray-direction check**, frame 0 (arbitrary, documented
choice — the run's anchor frame, simplest to reproduce standalone
without needing the tracking loop). Confident pixels (`C` above
`max(Q_conf, C_conf)` = 1.5, the same threshold `track()` uses):
185,725 of 204,800. For each, MASt3R's implied ray direction
(`normalize(X_canon[px])`) vs. this project's own Scaramuzza ray
(`geometry.camera.unproject`, imported from the unlocked `src/geometry/`)
at the same pixel mapped back to original coordinates:
**median 19.82°, p95 36.35°, mean 20.26°, max 42.49°** angular
difference.

**INTERPRETATION**: a ~20° median disagreement is large — MASt3R has no
notion of this project's Scaramuzza camera model at all; it regresses a
pointmap end-to-end from pixels, with no explicit camera model or
calibration in uncalibrated mode, so its implied per-pixel ray directions
are a learned approximation, not a projection of a shared geometric
model. This is exactly why the protocol (per the task's own framing)
uses only MASt3R's Z (camera-frame depth along *our* model's ray) and
not its raw XY/pointmap directions — the two models' rays genuinely
point differently, confirmed here with numbers rather than assumed.
Whether 20° is "a lot" for the eventual fusion-stage error budget is not
assessed here — that would require comparing against the same
project's already-quantified EndoDAC/GT error budgets, not attempted in
this stage.

## Step 6: the frame 169-217 gap, diagnosed

`scripts/mast3r_slam_diagnose_gap.py`. Two independent signals.

**MEASURED — GT motion** (`pose.txt` only, no MASt3R data):

| | path length | rotation | per-step path | per-step rotation |
|---|---|---|---|---|
| Frames 169-217 (48 steps) | 14.37mm | 9.50° | **0.299 mm/step** | **0.198°/step** |
| Inter-keyframe intervals, 0-168 (19 intervals, median) | — | — | **15.57 mm/step** (range 8.99-25.78) | **1.615°/step** (range 0.00-4.80) |

Frames 169-217 moved **~52x less per step** in translation and **~8x
less per step** in rotation than the median inter-keyframe interval
that *did* trigger a new keyframe earlier in the same sequence.

**MEASURED — MASt3R's own tracking signal for frames 169-217**
(`per_frame_tracking.csv`): **49 tracking records, 0 skipped**.
`match_frac` and `match_frac_k` (identical here, both computed from the
same `valid_kf` mask): median **0.556**, min **0.520** — comfortably
above both the skip threshold (`min_match_frac=0.05`) and the new-keyframe
trigger threshold (`match_frac_thresh=0.333`, and `new_kf` requires this
value to fall *below* that threshold). Tracking quality stayed healthy
and essentially flat throughout the gap.

**Which explanation the numbers support**: **little camera motion**, not
tracking degradation. There is no evidence of degraded tracking (0
skipped frames, match fractions staying well above every relevant
threshold) — the pipeline simply never needed a new keyframe because the
camera barely moved relative to keyframe 168 for the rest of the
sequence. This is a MEASURED conclusion from the two signals above, not
an inference from absence of data.

## Step 7: trajectory quality, all 218 frames, vs. EndoDAC

`scripts/mast3r_slam_trajectory_quality_perframe.py` (self-contained
Umeyama, same as Stage 1's script, still not importing `src/eval/`) —
now over all 218 reconstructed frames instead of Stage 1's 20 keyframes,
so both trajectories cover the identical span for the first time.

| | MASt3R-SLAM (218 frames) | EndoDAC (218 frames, `results/d1/stage_a_summary.csv`) |
|---|---|---|
| GT path length compared against | 348.35mm | 348.35mm (same — full sequence, both now) |
| ATE (RMSE, post-alignment) | **13.40mm** | **9.23mm** (`ate_mm`) |
| Endpoint drift, fraction of GT path | **5.41%** | **2.78%** (`endpoint_error_frac_of_gt_path`) |

Unlike Stage 1's comparison (flagged there as not directly comparable —
MASt3R-SLAM's 20 keyframes only spanned frames 0-168), **this is now a
controlled, same-span comparison**: both trajectories are aligned against
and measured over the identical 218-frame, 348.35mm GT path.

**INTERPRETATION**: on this one sequence, MASt3R-SLAM's ATE is ~45%
higher and endpoint drift ~95% higher (roughly double) than EndoDAC's.
Neither number is an evaluation metric under this project's pre-registered
definitions — this is trajectory quality only, one sequence, no
statistical claim about which pipeline is "better" is being made or
supported by a single data point.

## Step 8: storage + runtime extrapolation, 169 sequences

**MEASURED extrapolation**, same method as Stage 1's Q6: measured
per-frame rate × corpus-wide total frame count
(`docs/dataset_official_description.md:9`: 67,886 frames, 169 sequences).
**Numbers only — no full-corpus run**, per the task's explicit
instruction.

- Runtime: 54.94s / 218 frames = 0.2520 s/frame → 67,886 × 0.2520s ≈
  **17,109s ≈ 4.75 hours**, single GPU, sequential.
- Storage (per-frame depth+confidence `.npz` only, not counting the
  keyframe-only `.ply`/TUM outputs which are much smaller): 293,822,448
  bytes / 218 frames = 1.348 MB/frame → 67,886 × 1.348MB ≈ **91.5 GB**.

**INTERPRETATION**: 4.75 hours is in the same range as EndoDAC's
full-corpus run (4h41m) and Stage 1's own projection for the unmodified
pipeline (6.07h) — this instrumented version is not meaningfully slower
despite the added per-frame disk I/O. 91.5GB of per-frame depth data for
the full corpus is a real storage cost that would need explicit approval
(CLAUDE.md: "ask before any run expected to take over an hour," and this
is well over an hour) before any full-corpus run — not requested or
started in this stage.

## Stage 2 open issues

1. The ray-direction disagreement (median 19.8°) is measured on one
   frame only (frame 0); whether it's representative of other frames/
   sequences, or varies systematically with position in the frame (e.g.
   worse toward the periphery, where the Scaramuzza model's distortion
   is strongest) is not assessed here.
2. ~~The 12.11% vignette-crop-from-cropping figure~~ — **corrected to
   2.13% in Stage 3** (wrong population, see the correction block above)
   — for `c1_cecum_t1_v1` only; not confirmed representative of other
   sequences (all share the same 1350×1080 input size and thus the same
   resize/crop geometry, so it likely generalizes, but this is
   INTERPRETATION, not measured on other sequences).
3. Storage/runtime extrapolation (step 8) assumes this one sequence's
   per-frame rate and per-frame storage are representative of the
   169-sequence corpus — same caveat Stage 1's Q6 already carried,
   unresolved.
4. Which of Stage 1 Q4's four per-frame options to standardize on for
   any future full-corpus run is still unranked — this stage implemented
   Option A specifically (as instructed) but didn't compare it against
   the other three empirically.

---

# Stage 3: full-corpus inference (no metrics)

All 169 registered sequences. Still **no evaluation metric of any kind
computed, and no script under `src/eval/`/`src/eval_ext/` run on any
MASt3R-SLAM output.**

## Pre-flight

**1. Disk**: 1.7TB free on `/data1_ycao/chua` at start — well above the
500GB floor. PASS.

**2. Vignette mask convention — Stage 2 had it backwards.** Confirmed by
reading (not modifying) `src/eval/evaluable.py`:
`load_vignette_mask`'s own docstring says *"True = vignette
(non-evaluable)"*; `evaluable_pixel_mask` computes evaluability from
`~vignette_mask` (mask **False** is the valid/imaged population). Stage
2's `vignette_crop_fraction` selected the **True** (invalid,
102,049-pixel black-border) population instead and reported the
crop-loss fraction of *that* — backwards from "how much of the actually
imaged area loses depth to cropping." **Corrected** (see the correction
block under Stage 2's Step 5, above): recomputed on the ~1,355,951-pixel
valid population, **2.13%** (28,898 pixels) fall outside the MASt3R
grid's coverage, not 12.11%. The evaluator itself (`evaluable_pixel_mask`)
uses the correct convention — no locked code has this bug, only Stage
2's diagnostic script did, and it was never used for anything downstream
of that one reported number.

**3. EndoDAC ray-direction comparison** (`scripts/
mast3r_slam_check_vignette_and_endodac_ray.py`, GPU 7). Reran MASt3R's
own confident-pixel selection on frame 0 (deterministic — the rerun's
median angle matched Stage 2's saved value exactly, diff 0.0°) to get
the identical pixel set both checks compare against, rather than a
redefined population:

| | MASt3R (vs. our camera model) | EndoDAC (vs. our camera model) |
|---|---|---|
| n pixels compared | 185,725 | 185,725 (same set) |
| median angle | 19.82° | 15.42° |
| p95 angle | 36.35° | 30.62° |
| mean angle | 20.26° | 16.12° |
| max angle | 42.49° | 36.70° |

**INTERPRETATION**: both methods disagree substantially with this
project's Scaramuzza model — neither EndoDAC nor MASt3R uses it, so
neither "should" agree, and this is exactly the expected shape of the
result, not a red flag for either method. EndoDAC's disagreement is
somewhat smaller (15.4° vs. 19.8° median) but both are large enough that
neither pipeline's raw per-pixel ray geometry is a substitute for this
project's own camera model — consistent with Stage 2's already-stated
conclusion that only Z (not the raw pointmap direction) should be used
downstream.

## Full run

`scripts/mast3r_slam_run_corpus.py`, GPU 7 (selected once via `scripts/
gpu_status.py`, 32.1GB free, 0% util at start), resumable, subprocess-
per-sequence (Stage 1/2's proven scripts, not a shared-process refactor).

**One real bug found and fixed mid-run, not retried with different
settings**: sequence 14/169 (`c1_cecum_t1_v3`) failed at the
trajectory-quality step with `KeyError: "There is no item named
'pose.txt' in the archive"`. This is the already-documented
`c1_cecum_t1_v3.zip`-wraps-everything-in-a-subfolder exception
(`src/geometry/coverage_mesh.py::_resolve_zip_member`'s docstring: *"at
least one (c1_cecum_t1_v3.zip, confirmed the only exception found)
wraps everything in a `<video_name>/` subfolder"*) — a genuine data-
loading defect in this stage's own new code (`load_gt_poses` in two
scripts hardcoded `"pose.txt"` at archive root), not a MASt3R-SLAM
performance issue on that sequence. Fixed by matching on basename
(`resolve_pose_txt_member`, duplicated in both affected scripts,
verified against the real archive: resolves to
`c1_cecum_t1_v3/pose.txt`). The run was stopped cleanly (no orphaned
processes), the fix applied, and resumed — already-`"ok"` sequences were
skipped, `c1_cecum_t1_v3` got a fresh attempt under the corrected code.
This is a code-defect fix, not "retrying with different settings" (which
would mean tuning MASt3R-SLAM's own behavior to make a struggling
sequence pass) — same class of fix as the `git-lfs`/build-isolation
fixes in Stage 1's install.

**Result: 169/169 processed, 0 errors after the fix** (156 sequences run
fresh in the resumed pass + 13 already-complete from before the
interruption, matching the earlier validation run). Total per-sequence
compute time (sum of each sequence's own `elapsed_seconds`, the true
compute cost — wall-clock time undercounts this because of the
mid-run interruption/restart): **30,754.8s ≈ 8.54 hours**, single GPU,
sequential. Every manifest records `gpu_index=7` and the git commit that
produced it. Storage: **85GB actual** (`results/pipelines/
mast3r_slam_full_run/`), close to Stage 2's 91.5GB extrapolation.

## Per-sequence descriptives (no thresholds, no metrics)

### Completion

**169/169 (100%) completed without crashing** (manifest `status="ok"`,
every subprocess step returned 0). Under the **strict D1.1 operational
definition** (`docs/success_criteria.md`, 2026-09-23 clarification, read
not modified: *"every frame... has finite predicted depth and a finite
predicted pose, and the Sim(3) trajectory alignment succeeds"*):
**143/169 (84.6%) complete**.

**MEASURED, not just asserted**: all 26 non-complete sequences are
missing **exactly one frame** each (`n_reconstructed = n_total_frames -
1` in every one of the 26 cases, confirmed by inspection, not assumed),
and every one of the 26 still has `sim3_alignment_succeeded=True`. Each
missing frame traces to MASt3R-SLAM's own `FrameTracker.track()`
"Skipped frame" path (`match_frac < min_match_frac`, a transient
tracking hiccup against the current keyframe) — not a script bug, not a
permanent track loss, and not a crash. This is a real, if small,
difference from `c1_cecum_t1_v1` (which Stage 1/2 already established
has 0 skips) — **26/169 sequences (15.4%) have at least one transient
per-frame tracking failure**, MASt3R-SLAM's own documented failure mode,
correctly distinguished here from the crash-free-completion count per
the same D1.1 clarification entry's own distinction (originally written
for EndoDAC, which "cannot lose track by construction" — MASt3R-SLAM
can, and sometimes transiently does).

**Correction (2026-10-01, pipeline 2 evaluation pre-flight,
`docs/pipeline2_eval.md`)**: the two paragraphs above are wrong about the
size of the gap. `check_completion` counted frames from the rows of
`poses_per_frame.csv`, not from the GT frame count. Frames that MASt3R-SLAM
processed in RELOC mode after the skipped frame never reach `track()`, so
they have no row at all and were not counted as missing. Against the GT
frame count (`pose.txt` lines in each archive,
`scripts/pipeline2_preflight.py adapter-check`,
`results/pipeline2_eval/preflight/gate2_adapter_check.json`): the same 26
sequences lack a pose for **6,188 frames in total** (of 67,886), from 61
to 441 frames per sequence, not 1. In 21 of the 26 every frame from the
skipped frame to the end of the sequence has no pose (relocalization never
succeeded; e.g. `results/pipelines/mast3r_slam_full_run/c1_cecum_t1_v2/step1_run_perframe.log`: "Skipped frame 170", then repeated
"Failed to relocalize"); in the other 5 tracking resumed later. Depth
(`depth/*.npz`) exists for every GT frame of every sequence. The 143/169
strict-D1.1 count is unchanged (the same 26 sequences fail it); the
statements "missing exactly one frame" and "not a permanent track loss" do
not hold. Text above left in place.

### Trajectory-quality and depth-scale distributions (169 sequences, no threshold attached)

| quantity | min | 25% | median | 75% | max |
|---|---|---|---|---|---|
| ATE after Sim(3) alignment (mm) | 1.554 | 4.675 | 6.969 | 8.861 | 18.13 |
| endpoint drift / GT path length | 1.25% | 3.84% | 5.66% | 8.31% | 26.93% |
| `s_pose` (Sim3 scale) | 10.38 | 33.19 | 44.93 | 61.56 | 106.7 |
| depth scale median | 13.02 | 24.80 | 29.12 | 34.90 | 52.60 |
| depth scale relative IQR | 0.063 | 0.167 | 0.236 | 0.340 | 1.355 |
| n keyframes | 2 | 7 | 9 | 13 | 27 |
| longest no-new-keyframe run (frames) | 28 | 61 | 98 | 176 | 444 |

**Next to EndoDAC's already-published corpus distributions**
(`docs/d1_stage_a.md`, same 169 sequences):

| quantity | MASt3R-SLAM median | EndoDAC median |
|---|---|---|
| ATE (mm) | 6.969 | 5.157 |
| endpoint drift / GT path | 5.66% | 4.97% |
| depth scale median | 29.12 | 118.80 |
| depth scale relative IQR | 0.236 | 0.170 |

**INTERPRETATION**: MASt3R-SLAM's median ATE and endpoint drift are
somewhat worse than EndoDAC's but the same order of magnitude — neither
pipeline is dramatically better on this purely descriptive trajectory
measure, and neither number is an evaluation metric under this project's
pre-registered definitions. The depth-scale-median gap (29 vs. 119) is
expected, not a finding: the two methods' depth outputs are in
different, method-specific native units (MASt3R's "metric" checkpoint
and EndoDAC's scale-free network are not calibrated to the same
constant), which is exactly why both require this per-sequence GT-based
rescaling in the first place — the ratio's *absolute* value carries no
meaning across methods, only each method's own dispersion (relative IQR)
is comparable, and MASt3R's is somewhat higher (0.236 vs. 0.170),
suggesting its scale is modestly less stable frame-to-frame than
EndoDAC's. The keyframe-count and longest-gap rows have no EndoDAC
equivalent (frame-to-frame method, no keyframe concept) — reported as
MASt3R-specific descriptives only.

### Covariates

103 distinct `mesh_hash` values, 15 distinct `physical_segment_id`
values across the 169 sequences — matches the already-established
figures (`CLAUDE.md`: "103 distinct meshes," `docs/d1_stage_a.md`: "15
distinct" physical segments). Joined directly from `results/
d1/stage_a_summary.csv` (plain CSV merge on `sequence`), not re-derived.

## Stage 3 outputs

`results/pipelines/mast3r_slam_full_run/<sequence>/` (gitignored, 85GB
total, never committed): `depth/*.npz`, `per_frame_tracking.csv`,
`keyframes_final.csv`, `poses_per_frame.csv`,
`trajectory_quality_perframe.json`, `descriptives.json`,
`MANIFEST.json`, per-step subprocess logs. `results/mast3r/
stage3_summary.csv` (169 rows, gitignored): one row per sequence,
including its status, descriptives, and joined covariates.

## Stage 3 open issues

1. The 26 sequences with a single transiently-skipped frame are not
   further characterized here (e.g. whether the skip correlates with
   fast motion, low texture, or a specific anatomical segment) — flagged
   as a follow-up candidate, not chased down, per this stage's own scope
   (descriptives only, no analysis of *why*).
2. The EndoDAC ray-direction comparison (pre-flight item 3) is frame 0
   only, same single-frame caveat as Stage 2's own MASt3R check.
3. No metric has been computed on any of this corpus's output yet —
   evaluation is explicitly a separate task, per instructions.
