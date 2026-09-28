# MASt3R-SLAM feasibility, Stage 1: `c1_cecum_t1_v1`

Second-pipeline feasibility only. **No evaluation metric of any kind
(coverage, recall, false reassurance, false alarm, IoU, localization error)
is computed here, and no script under `src/eval/` or `src/gt/` is run on
any MASt3R-SLAM output.** A D2 pre-registration entry does not exist yet;
this stage does not need one because it produces no metric. Rules on the
distinction between "trajectory quality" (computed here, self-contained)
and "evaluation" (not computed here) follow the task's own hard
constraint.

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

1. Why keyframe selection stops at frame 168/217 for this sequence
   (Q2's UNKNOWN) — not diagnosed, needs either visual inspection or
   per-frame instrumentation to settle.
2. No D2 pre-registration entry exists in `docs/success_criteria.md` —
   correctly out of scope for this feasibility-only stage, which computes
   no metric; required before any D2 comparison metric is computed.
3. The FPS-vs-paper gap (Q1 INTERPRETATION) is unconfirmed — would need a
   rerun on an idle GPU to isolate contention from a genuine performance
   difference.
4. Q4's four options are unranked and unimplemented by design (the task
   asked not to choose); a future stage would need to pick one before any
   full-corpus MASt3R-SLAM run.
