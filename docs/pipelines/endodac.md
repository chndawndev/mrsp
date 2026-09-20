# EndoDAC pipeline setup and single-sequence inference

Repo: https://github.com/BeileiCui/EndoDAC (MICCAI 2024, "Efficient Adapting
Foundation Model for Self-Supervised Depth Estimation from Any Endoscopic
Camera"). Cloned to `scratch/pipelines/EndoDAC` at commit
`22ec911b270d056fdd7f430474e4be407cce8287` (2025-01-23), the current tip of
`main` as of this survey (2026-09-19).

Target sequence: `c1_cecum_t1_v1`, 218 RGB frames, 1350x1080, already
extracted at `scratch/c1_cecum_t1_v1/` (rgb/, depth/, pose.txt, etc.).

Status: **environment built, weights downloaded, sanity-check inference
run and diagnosed, including a quantitative correction to the initial
finding.** Runtime looks fine (~16 ms/frame). §5's visual sanity check
initially looked concerning (PNGs appeared flat, no visible lumen
structure). §6 diagnosed this: not a preprocessing bug, not a botched
checkpoint load, not the numpy ABI issue hit along the way (all ruled out,
MEASURED) — and, most importantly (§6 Check 5), **not actually a flat
prediction at all**: the quantitative check the earlier visual read was
missing shows baseline predicted inverse depth already correlates
strongly with GT depth (Spearman ρ ≈ -0.87 over ~1.16M pixels/frame). The
checkpoint generalizes to C3VDv2 zero-shot with reduced sharpness/contrast
relative to GT, not with no signal. The vignette border (black frame
corners) is a real but secondary drag on that correlation (~0.04-0.06 ρ).
Training-data domain gap (SCARED/Hamlyn only, confirmed from the paper)
remains the best explanation for the *fidelity* gap, not for a failure —
because there wasn't one. **Correction (2026-09-20, §6 Check 3):** C3VDv2's
RGB is real clinical-endoscope footage of a real silicone phantom, not a
synthetic rendering — only the GT side is rendered. The domain gap is
real-tissue-camera → real-phantom-camera, smaller than originally
described, which fits Check 5's positive correlation result better than
the original "real vs. synthetic" framing did.

---

## 1. Environment

Separate conda env `endodac`, isolated from the project's own env. Built
from the repo's own `conda.yaml` / `requirements.txt`
(`scratch/pipelines/EndoDAC/conda.yaml:1-16`,
`scratch/pipelines/EndoDAC/requirements.txt:1-15`), with two packages
dropped:

- `cuml-cu11` (RAPIDS GPU ML library) and `git+.../submitit` (SLURM job
  submission helper) are listed in both `conda.yaml` and `requirements.txt`
  but are not imported anywhere in the repo's `.py` files
  (`grep -rln "cuml\|submitit" --include=*.py .` returns nothing). Installing
  them would add a large, unused RAPIDS dependency for no benefit to
  inference. Flagging this as a deviation from the literal declared spec,
  not a silent one.

Actual install command (Python 3.9, matches `conda.yaml:8`):

```
conda create -n endodac python=3.9
conda activate endodac
pip install --extra-index-url https://download.pytorch.org/whl/cu117 \
    torch==2.0.0 torchvision==0.15.0 omegaconf torchmetrics==0.10.3 \
    fvcore iopath xformers==0.0.18 tensorboardX tqdm opencv-python \
    open3d scikit-image scipy matplotlib pillow
```

`scipy`, `matplotlib`, `pillow` are added explicitly: they're imported by
`evaluate_pose.py` (`scipy.stats`) and `test_simple.py` (`matplotlib`, `PIL`)
but are not listed in `requirements.txt` — presumably pulled in transitively
by another package in the author's own environment, not guaranteed here, so
listed explicitly rather than relying on that.

Full package list actually resolved and installed (from `pip install`'s own
summary, `logs/endodac_pip_install.log`): `torch-2.0.0+cu117`,
`torchvision-0.15.0+cu117`, `xformers-0.0.18`, `torchmetrics-0.10.3`,
`fvcore-0.1.5.post20221221`, `iopath-0.1.10`, `opencv-python-5.0.0.93`,
`open3d-0.19.0`, `scikit-image-0.24.0`, `scipy-1.13.1`,
`matplotlib-3.9.4`, `pillow-11.3.0`, plus transitive deps. One thing to
watch, not yet hit: `numpy-2.0.2` was pulled in, and `torch==2.0.0` predates
NumPy 2's release — this combination is known to occasionally break in
other projects (removed aliases like `np.float_`); not yet observed to
fail here, flagged as a risk to check if a numpy-related `AttributeError`
shows up during inference.

`gdown==5.2.2` installed separately (not part of the repo's own spec) to
fetch the two Google Drive checkpoints below.

Full env creation and pip install logs: `logs/endodac_env_setup.log`,
`logs/endodac_pip_install.log` (`EXITCODE:0` in both).

---

## 2. Weights — downloaded and extracted

Two checkpoints, both linked from the README
(`scratch/pipelines/EndoDAC/README.md`), downloaded with the user's
explicit approval (asked and confirmed in-conversation before fetching):

| What | Source URL | SHA-256 of downloaded bytes | Extracted to |
|---|---|---|---|
| EndoDAC checkpoint (DV-LoRA adapted weights) | https://drive.google.com/file/d/1qzAYBtwYJDN7hEi6pApqBOOz6pUhyY70/view (README, "Results" table) | `9c3f12faecdb9888bc6d64e01ebdace5947ac4b881751fe9e88128cf14b9e23a` (of the 737 MB zip before extraction) | `scratch/pipelines/EndoDAC/checkpoints/endodac/` |
| `depth_anything_vitb14` backbone | https://drive.google.com/file/d/163ILZcnz_-IUoIgy1UF_r7PAQBqgDbll/view (README, "Initialization" section) | `08aebe79906475327361aa99b78c5e208dac57553d06af6cd7e446c1ddc8f067` | `scratch/pipelines/EndoDAC/pretrained_model/depth_anything_vitb14.pth` (renamed to match the exact filename `models/endodac/endodac.py:187-189` expects: `depth_anything_{backbone_arch}.pth`) |

Neither download is script-driven by the repo itself (no `gdown` call, no
checksum published anywhere in the repo) — both are manual Google Drive
links; the hashes above are recorded for reproducibility of *this specific
download*, not verified against any upstream-published value, because the
authors didn't publish one.

**Resolved open question** (previously unknown without downloading): the
main checkpoint archive is a zip bundling **9** files, not just
`depth_model.pth`:

```
adam.pth              228 MB   optimizer state (training only, unused for inference)
depth_model.pth       397 MB   depth network — what test_simple.py loads
intrinsics_head.pth   0.5 MB   camera-intrinsics decoder — CONFIRMED PRESENT
pose.pth              5.3 MB   pose decoder
pose_encoder.pth      47 MB    pose encoder
position.pth          13 MB    "PositionDecoder" — training-only auxiliary net (trainer_end_to_end.py:58-61), not referenced by any eval/inference script
position_encoder.pth  47 MB    encoder for the above
transform.pth         13 MB    "TransformDecoder" — same story (trainer_end_to_end.py:68-71), training-only
transform_encoder.pth 47 MB    encoder for the above
```

So **pose and intrinsics inference is possible** from the public checkpoint
— `evaluate_pose.py`'s `--learn_intrinsics` path has everything it needs
(`pose_encoder.pth`, `pose.pth`, `intrinsics_head.pth` all present). The
`position*`/`transform*` files are a same-architecture-family auxiliary
pair (grep-traced to `trainer_end_to_end.py` only, likely a non-rigid /
appearance-flow correction module used solely in the self-supervised
training loss, per the AF-SfMLearner lineage this repo is built on) — not
imported by `test_simple.py`, `evaluate_depth.py`, or `evaluate_pose.py`,
so not needed for our inference-only use and not investigated further; this
is inference about their role, not confirmed from a docstring, since the
repo has none for them.

---

## 3. Preprocessing and output format (from static code inspection)

### Preprocessing

Two different, **mutually inconsistent** preprocessing paths exist in the
repo, and neither is obviously "the" default — this needs to be resolved
before running, not silently picked:

**Path A — `test_simple.py` (single-image / folder inference,
`scratch/pipelines/EndoDAC/test_simple.py:139-142`):** loads the full
original image, resizes directly to `(feed_width, feed_height)` with
`PIL.Image.LANCZOS`, then `ToTensor()` (scales to [0,1], no further
normalization). `feed_height`/`feed_width` are read from the checkpoint
dict at runtime (`test_simple.py:80-81`), not hardcoded in this script;
`options.py:86-93` defaults them to height=256, width=320 (aspect ratio
1.25) for training/eval. **No crop.** This path does not read camera
intrinsics or apply any C3VD-specific logic.

**Path B — `datasets/c3vd_dataset.py` (the repo's own C3VD loader, used by
`evaluate_depth.py` when `--dataset c3vd`):** applies a **hardcoded pixel
crop** `self.box = (200, 180, 1150, 900)`
(`scratch/pipelines/EndoDAC/datasets/c3vd_dataset.py:99`) — a 950x720 region
cut from the frame — *before* resizing to `(height, width)`. This box
matches our frame size well: our extracted `c1_cecum_t1_v1` RGB frames are
exactly 1350x1080, the resolution this crop was evidently written against.
This crop most likely exists to strip the black vignette border outside the
circular fisheye FOV (a common C3VD preprocessing step), but the repo
documents no rationale for the exact numbers, so this is inference, not a
confirmed fact.

**Aspect ratio note:** Path A resizes the *full* 1350x1080 frame directly to
320x256 (feed aspect 1.25) with no crop, so it silently *stretches* the
image, changing the effective aspect ratio from 1.25 (1350/1080 = 1.25,
which happens to already equal 320/256 — so in this specific case Path A
does **not** distort aspect ratio, only downsamples). This is coincidental
to our frame's specific resolution, not a general guarantee — flagging in
case a different sequence's frames differ. Path B's crop (950/720 = 1.319)
is a *different* aspect ratio than the final resize target (320/256=1.25),
so Path B *does* mildly distort the image during resize, on top of
discarding the cropped-out border pixels.

**Filename convention mismatch:** Path B's loader expects files named
`<frame>_color.png` / `<frame>_depth.tiff` in one folder per sequence
(`c3vd_dataset.py:87-91`). Our extracted layout is `rgb/0000.png`,
`depth/0000_depth.tiff` in separate subfolders (confirmed by directory
listing). Path B would need path adaptation to run on our extracted data
as-is; Path A takes any image path directly and needs no adaptation.

**No mean/std (ImageNet-style) normalization is applied anywhere** in
either path or inside the model's forward pass — checked
`models/endodac/endodac.py` and `models/backbones/vision_transformer.py`
for `Normalize`/`mean=`/`std=` and found none. Input to the DINOv2-based
backbone is raw `[0,1]` `ToTensor()` output. This is unusual for a
ViT/DINOv2 backbone (which is normally trained on ImageNet-normalized
input) but is what the released code actually does — reported as fact, not
flagged as a bug to fix.

**Frozen-decision compliance:** per `CLAUDE.md`'s frozen "Pipeline input
format" decision, we feed the **original fisheye frames, no undistortion**.
Path A satisfies this directly (resize only). Path B's fixed crop is *not*
an undistortion, but it does discard the frame's outer region — whether
that outer region is genuinely fisheye image content, or a black vignette
border with no information, has not been checked yet for this dataset's
specific archives (`docs/conventions.md`/`docs/pinhole_tradeoff.md` don't
address it, since those docs are about the omnidirectional-vs-pinhole
choice, not vignette cropping). **Recommendation to decide, not yet
decided:** use Path A (no crop) for this sanity check, since it's the
simpler path, requires no adaptation, and clearly does not undistort or
discard fisheye content. Path B is noted here because it's what the
authors themselves used for their own C3VD numbers, so a future decision to
match their reported metrics exactly might require adopting it — that's a
separate future call, not made here.

### Output format

**Depth** (from `test_simple.py:150-156` and `utils/layers.py`'s
`disp_to_depth`): the network outputs a sigmoid disparity map per frame,
converted with `min_depth=0.1`, `max_depth=150` (`test_simple.py:156`;
`options.py:115-122` gives the same defaults for train/eval).
This is the standard Monodepth2 self-supervised convention: **scale-
ambiguous relative depth**, not metric. `evaluate_depth.py:190-194` confirms
this explicitly — it always median-scales predicted depth against GT depth
before computing error metrics (`ratio = median(gt_depth) /
median(pred_depth)`) unless `--disable_median_scaling` is passed. So: **per-
frame relative depth up to an unknown global scale factor**, not directly
comparable in mm to our GT depth without a scale-recovery step. For the C3VD
dataset specifically, `evaluate_depth.py:84` sets `MAX_DEPTH = 100` (matches
our own 0-100mm convention), but this only changes the metric clamping range
during evaluation, not the network's own output units.

**Pose** (from `evaluate_pose.py:130-135`, `utils/layers.py`'s
`transformation_from_parameters`): the pose network takes exactly **two
adjacent frames** as input (`opt.frame_ids = [0, 1]`,
`evaluate_pose.py:126`) and outputs one **relative** transform between them
via axis-angle + translation, converted to a 4x4 matrix
(`transformation_from_parameters`, monodepth2 convention — need to verify
sign/direction convention against our `.T`-transposed pose.txt convention
before using it, not yet checked). There is no absolute/global pose output;
absolute trajectories are only ever built by chaining relative transforms
(`evaluate_pose.py:19-26`, `dump_xyz`). Like depth, the translation
magnitude is scale-ambiguous — `compute_ate` (`evaluate_pose.py:44-51`)
explicitly solves for an optimal scale factor against GT before computing
trajectory error, confirming translations aren't metric out of the box.

**Intrinsics**: an `IntrinsicsHead` module exists
(`evaluate_pose.py:106-109`, gated behind `--learn_intrinsics`), predicting
a 3x3 camera matrix from the pose encoder's features. It is trained
**jointly with, and stored as, a separate checkpoint file**
(`intrinsics_head.pth`, `evaluate_pose.py:97`) — **confirmed present** in
the downloaded checkpoint (§2). The emitted intrinsics are a plain pinhole
3x3 matrix (fx, fy, cx, cy read off the diagonal/last column,
`evaluate_pose.py:210-213`) — **not** compatible with our Scaramuzza
omnidirectional model without treating it as a rough pinhole approximation
only, same caveat as our own frozen "shared pinhole approximation" decision
for methods needing explicit intrinsics.

---

## 4. Runtime

Measured on GPU index 7 of the shared box (RTX 6000 Ada Generation, chosen
via `scripts/gpu_status.py`, otherwise idle — 32.1 GB free at the time),
`CUDA_VISIBLE_DEVICES=7`, batch size 1, depth-only forward pass
(`scratch/pipelines/run_endodac_sanity_check.py`,
`logs/endodac_sanity_check.log`):

| Frame | Time |
|---|---|
| 0000 (first — includes CUDA context/kernel warmup) | 1830.7 ms |
| 0109 | 16.1 ms |
| 0217 | 15.6 ms |

**Steady-state: ~16 ms/frame** (62.5 fps), single GPU, single frame at a
time, no batching attempted. Extrapolated cost for the full 169-sequence /
67,886-frame registered corpus at this rate, compute only (excludes I/O,
model load, and the one-time ~1.8 s warmup which amortizes to
negligible at this scale): **67,886 × 16 ms ≈ 18.1 minutes**. This is a
lower bound — no batching, no I/O overlap, and pose/intrinsics inference
(not measured here, depth-only) would add additional forward passes per
frame pair.

---

## 5. Sanity-check depth PNGs — done; see §6 Check 5 for a correction

**Superseded, keep reading**: this section's purely visual read (from an
8-bit PNG at a fixed display range) turned out to be too pessimistic. §6
Check 5 adds the quantitative check this section was missing (Pearson/
Spearman correlation against GT) and finds real, strong, GT-correlated
structure in the same prediction described as "flat" below. Left in place
verbatim for the record of what was actually observed at the time; treat
§6 Check 5 as the corrected conclusion.

Ran Path A inference (§3) on 3 frames from `c1_cecum_t1_v1` (0000, 0109,
0217 — first/middle/last of 218), saved alongside GT depth as PNGs for
visual comparison, no metric computed:
`results/pipelines/endodac_sanity_check/{frame}_pred_depth_medianscaled.png`
vs `{frame}_gt_depth.png` (predicted depth is median-scaled to GT's median
purely so both images share a comparable display range — this scaling is
for visualization only, not a claim about metric accuracy; see §3 on scale
ambiguity).

**Finding: the predicted depth does not show the lumen/tunnel geometry that
is clearly visible in both the RGB frame and the GT depth.** GT depth
(and the RGB) both show a sharp, high-contrast tunnel structure — a bright
(far/deep) lumen opening, darkening toward the near walls, with visible
haustral fold ridges. EndoDAC's prediction, at all 3 frames tested, is an
almost-flat, low-contrast gray gradient with no visible lumen opening and
no fold structure — it does not look like a plausible reconstruction of
this scene's geometry.

**Follow-up diagnostic run** (`run_endodac_sanity_check_pathb.py`,
`logs/endodac_sanity_check_pathb.log`): re-ran the same 3 frames through
"Path B" preprocessing instead (the repo's own `c3vd_dataset.py` crop,
§3) to check whether Path A's raw full-frame-with-black-vignette input was
the cause. **It was not the (sole) cause**: Path B's prediction
(`{frame}_pred_depth_pathB_medianscaled.png` vs
`{frame}_gt_depth_pathB_cropped.png`) is similarly flat and
low-contrast — some improvement in that a directional gradient is visible,
but still no lumen opening, no fold structure, nothing resembling the GT's
sharp geometry. Ruling out the crop as the explanation narrows, but does
not resolve, the concern.

**This is reported as a measured observation, not a diagnosed cause.**
Plausible explanations not yet checked (not confirmed, listed only as
what would need to be checked next): a domain gap between whatever this
released checkpoint was actually trained on (repo doesn't state whether
it's SCARED, C3VD v1, or something else) and this C3VDv2 sequence's
appearance (lighting, texture, debris) or camera geometry (fisheye FOV,
octagonal vignette shape); a preprocessing detail still not matched
(e.g. the missing ImageNet normalization noted in §3 might matter more
than the repo's own code implies); or the checkpoint genuinely
underperforming out-of-domain. **Not concluding pipeline failure or bug**
— this is exactly the kind of finding this sanity check exists to catch,
reported for a decision on whether EndoDAC is still a viable candidate
before any further investment.

---

## 6. Diagnosis: why no lumen structure

Cheapest checks first, per instructions. Findings below in the order run;
each is marked MEASURED (directly observed/read) or INTERPRETATION
(reasoned conclusion, not directly proven).

### Check 0 — ruled out: was the numpy 2.x/torch ABI break the actual cause?

Before trusting anything in §5, verified this explicitly rather than from
memory, since it's the kind of thing that's easy to get wrong by
recollection. **MEASURED, timeline reconstructed from logs and file
timestamps, not memory:**

- `pip install` initially resolved `numpy==2.0.2` as a transitive
  dependency of `torchvision` (`logs/endodac_pip_install.log:369`).
- The **first** sanity-check attempt (numpy 2.0.2 active) **crashed with
  `RuntimeError: Numpy is not available`** at the `torch.from_numpy(...)`
  call — before any image was written. **Zero PNGs exist from that
  attempt.** This was a loud, hard failure, not a silent bad-output one.
- `numpy<2` was then installed (resolved to 1.26.4).
- The sanity-check script was rerun **only after** that fix. This is the
  *only* run that ever produced output — both the Path A set (files
  timestamped 2026-09-19 21:59:10) and the Path B set (timestamped
  2026-09-19 22:00:20) were generated with numpy 1.26.4 already active.
  There is no "pre-fix" image set to compare against, because the pre-fix
  run never got far enough to produce one.
- **Independent re-verification, 2026-09-20**: confirmed `numpy==1.26.4`
  active in the env right now, then reran the identical Path A script
  end-to-end from scratch (fresh process, `results/pipelines/
  endodac_sanity_check_rerun/`). Compared the rerun's raw predicted-depth
  array against the original run's saved `.npy` for all 3 frames:
  **max absolute difference = 0.0, mean absolute difference = 0.0** —
  bit-for-bit identical, not just visually similar. Confirms the original
  result was already fully deterministic and reproducible under the fixed
  numpy version, with no residual effect from the earlier ABI break.

**Conclusion: the numpy ABI break is ruled out as the cause of the flat
prediction.** It was a hard crash that happened before the fix, and
produced no output at all — not a silent corruption that leaked into the
images we've been analyzing. The rest of this diagnosis (Checks 1-4 below)
stands.

### Check 0b — raw network output and weight-loading verification

Requested before pursuing Checks 1-4 further: inspect the raw network
output directly (not the rendered PNG), and verify the checkpoint actually
loaded. Script: `scratch/pipelines/endodac_raw_output_diagnosis.py`, log:
`logs/endodac_raw_output_diagnosis.log`. **All MEASURED, no interpretation
in this section.**

**Weight-loading audit.** Compared `depth_model.pth`'s keys against the
live model's `state_dict()` keys directly (not just visually, counted):
389 tensor keys in the checkpoint, 389 keys in the model, **389/389
intersect — 0 model keys left unmatched, 0 checkpoint keys silently
dropped.** The `{k: v for k, v in depther_dict.items() if k in
model_dict}` filtering pattern in `test_simple.py` (flagged earlier as a
pattern that *could* silently drop weights on a key mismatch) did not
actually drop anything here — full key-name agreement.

**Parameter value comparison** (backbone-only-init model, i.e. before
`depth_model.pth` is loaded, vs. the same tensors after loading — 96 LoRA
tensors total in the model, 8 sampled directly, spanning `lora_A`,
`lora_B`, `lora_U` across 2 transformer blocks, plus 3 `depth_head`
projection weights):

| Parameter | Identical to backbone-only init? | Backbone-only (min/max/mean/std) | Loaded (min/max/mean/std) |
|---|---|---|---|
| `encoder.blocks.0.mlp.fc1.lora_A` | **No** | -0.036/0.036/-0.0001/0.021 | -0.108/0.107/-0.0016/0.040 |
| `encoder.blocks.0.mlp.fc1.lora_B` | **No** | 0/0/0/0 (exact zero-init) | -0.077/0.077/0.0001/0.017 |
| `encoder.blocks.0.mlp.fc1.lora_U` | **No** | -0.656/0.929/0.159/0.652 | -0.849/-0.004/-0.261/0.398 |
| `encoder.blocks.0.mlp.fc2.lora_B` | **No** | 0/0/0/0 | -0.055/0.060/-0.0002/0.015 |
| `encoder.blocks.1.mlp.fc1.lora_B` | **No** | 0/0/0/0 | -0.063/0.061/-0.0002/0.016 |
| `depth_head.projects.0.weight` | **Yes** | -0.186/0.211/-0.00009/0.026 | (same) |
| `depth_head.projects.1.weight` | **Yes** | -0.134/0.115/0.00001/0.026 | (same) |
| `depth_head.projects.2.weight` | **Yes** | -0.102/0.093/0.00005/0.024 | (same) |

Every sampled LoRA tensor differs from its backbone-only value — most
tellingly, every `lora_B` tensor is **exactly** `0.0` (the DVLoRA
zero-init scheme, `models/backbones/mylora/layers.py:359-360`,
`nn.init.zeros_(self.lora_B)`) before loading and clearly non-zero after.
**Conclusion: the EndoDAC checkpoint's LoRA adapter weights were
genuinely applied — this is not the bare `depth_anything_vitb14` backbone
running unmodified.** The 3 `depth_head.projects.*` samples came back
identical, meaning those specific projection layers hold the same value
before and after loading `depth_model.pth` — consistent with the paper's
claim of "remarkably few trainable parameters" (most of the depth head may
be frozen, inherited unchanged from the DepthAnything backbone); flagged
as measured but not further investigated, since it doesn't bear on the
main question.

**Raw sigmoid disparity, before `disp_to_depth`** (`outputs[("disp",0)]`,
shape `(1,1,256,320)` for all 3 frames):

| Frame | min | max | mean | std | std/mean |
|---|---|---|---|---|---|
| 0000 | 0.3338 | 0.6771 | 0.4391 | 0.0836 | 0.190 |
| 0109 | 0.2914 | 0.8242 | 0.4753 | 0.1331 | 0.280 |
| 0217 | 0.3302 | 0.6722 | 0.4387 | 0.0808 | 0.184 |

10-bin histograms (all 3 frames, `logs/endodac_raw_output_diagnosis.log`)
show mass spread across the full observed range, not piled into 1-2 bins —
e.g. frame 0109: 5933 / 27656 / 16273 / 5055 / 3811 / 5275 / 6685 / 5718 /
4021 / 1493 pixels per bin (81,920 pixels total, 256×320).

**Converted depth** (`disp_to_depth(disp, 0.1, 150)`):

| Frame | min | max | mean | std |
|---|---|---|---|---|
| 0000 | 0.1477 | 0.2992 | 0.2347 | 0.0385 |
| 0109 | 0.1213 | 0.3426 | 0.2250 | 0.0538 |
| 0217 | 0.1487 | 0.3024 | 0.2345 | 0.0371 |

**Input tensor** (all 3 frames): shape `(1, 3, 256, 320)`, dtype
`torch.float32`, min `0.0`, max `1.0`. Per-frame mean/std: 0000 →
0.4324/0.1893; 0109 → 0.4733/0.2380; 0217 → 0.4332/0.1897 (matches §6
Check 1's separately-computed numbers exactly, as expected — same
transform).

**Answering the specific question — is the disparity constant, or is
variation being lost in the 8-bit PNG render?** relative std (std/mean) is
0.18-0.28 across the 3 frames: **not near-constant, real spatial
variation is present in the raw output.** And it is **not** being lost in
the median-scaled 8-bit rendering: the converted-depth range for frame
0109 (0.121-0.343, a ~99% relative span) scales to roughly 20-57mm after
the ~166x median-scale factor used for the PNG — a ~94-level spread out of
256, which is a clearly visible gradient, not something 8-bit quantization
would wash out. Looking back at the rendered PNG (§5) with this in mind:
it does show a real gradient (darker toward the right/far side, lighter
upper-left) — **the earlier description "almost-flat" undersold the
measured variation.** The precise, corrected finding: **the network
produces a real, non-degenerate, spatially-varying prediction — it is
just the wrong pattern.** It does not reproduce the lumen "hole" or fold
ridges visible in GT; it produces a smooth directional gradient instead.
This is a more specific finding than "flat/degenerate output" and is
consistent with — though does not by itself prove — a model that is
confidently predicting the wrong geometry rather than failing to predict
anything, which is exactly the kind of failure a train/test domain gap
would produce (as opposed to, say, a numerical bug that would more likely
produce NaNs, exact zeros, or a genuinely constant map).

### Check 1 — training preprocessing vs. our sanity script

**MEASURED.** The dataset class actually used for training is
`SCAREDRAWDataset` (`trainer_end_to_end.py:131`,
`datasets_dict = {"endovis": datasets.SCAREDRAWDataset}` — the *only*
entry; `options.py:78-81`'s `--dataset` choices are literally
`["endovis"]`, no other value is accepted). Read its full `__getitem__`
chain end to end: `SCAREDRAWDataset`/`SCAREDDataset`
(`datasets/scared_dataset.py`) → `MonoDataset.__getitem__`/`.preprocess`
(`datasets/mono_dataset.py:91-108, 133-183`).

Result: **training preprocessing is resize-only, no crop** —
`transforms.Resize((height, width), interpolation=Image.LANCZOS)`
(`mono_dataset.py:80-82`), then `transforms.ToTensor()` (`mono_dataset.py:63`,
`:100-102`) — no `Normalize(mean, std)` call anywhere in
`mono_dataset.py`, `scared_dataset.py`, or `trainer_end_to_end.py` (checked
by full read, not just grep this time). Color jitter (`ColorJitter`) is
applied to a separate `color_aug` copy only when `is_train=True` and a coin
flip hits (`mono_dataset.py:139-140, 152-156`); at inference/eval
(`is_train=False`) `do_color_aug` is always `False`
(`mono_dataset.py:139`), so `color_aug` degenerates to the identity
function and `("color_aug", 0, 0)` — which is what
`trainer_end_to_end.py:533` actually feeds the depth model
(`self.models["depth_model"](inputs["color_aug", 0, 0])`), not
`("color", 0, 0)` — is pixel-identical to the plain resized/ToTensor'd
image at eval time.

**Conclusion: "Path A" (our sanity script — direct resize, `ToTensor()`,
no crop, no extra normalization) is not just *a* reasonable preprocessing
choice, it is an exact structural match to what training actually used.**
"Path B" (the repo's C3VD-specific crop) was never part of training at
all — it exists only in `evaluate_depth.py`'s C3VD *evaluation* path,
apparently added later for users evaluating on C3VD, unconnected to how
the released checkpoint was trained. This flips §3's open question:
Path A, not Path B, is the training-consistent choice. **Preprocessing
mismatch is ruled out** as the explanation for the flat prediction.

**Tensor statistics.** Requested comparison of exact min/max/mean/std:
since the transform itself (`ToTensor()`, no normalization) is identical
in both cases, there is no separate "training tensor statistics" to
recover independently of the training images themselves, which we don't
have access to (SCARED access is gated, see Check 2). What's measurable is
our own input tensor, computed directly from the 3 test frames after the
same resize used at inference:

| Frame | min | max | mean | std | per-channel mean (R,G,B) |
|---|---|---|---|---|---|
| 0000 | 0.0 | 1.0 | 0.432 | 0.189 | 0.564, 0.395, 0.338 |
| 0109 | 0.0 | 1.0 | 0.473 | 0.238 | 0.602, 0.440, 0.378 |
| 0217 | 0.0 | 1.0 | 0.433 | 0.190 | 0.566, 0.396, 0.338 |

One concrete, measured structural fact worth flagging: **6.4-7.0% of
pixels in every frame are pure black** (exact `(0,0,0)`, the octagonal
vignette border — measured on frame 0109, both at native 1350x1080 and
after resize to 320x256). Whether SCARED training frames contain a
comparable fraction of pure-black border pixels is **UNKNOWN** (no access
to SCARED to check) — flagged as a specific, checkable difference in input
statistics, not confirmed as causal.

### Check 2 — reproduce an authors' own reported number

**Not attempted — infeasible to do cheaply.** The README's results table
only reports SCARED-benchmark-style numbers (comparing against Fang et al./
Endo-SfM/AF-SfMLearner, all SCARED-benchmark baselines); no C3VD or Hamlyn
numbers appear in the README at all. To reproduce *any* published number
requires the SCARED dataset, which is **not freely downloadable**:
checked the AF-SfMLearner repo (linked from EndoDAC's README as the SCARED
prep procedure to follow) — it states "You can download the Endovis or
SCARED dataset by signing the challenge rules and emailing them to
max.allan@intusurg.com" (github.com/ShuweiShao/AF-SfMLearner). This is a
data-use-agreement/email-request process, not a cheap same-session check.
**Stopping here rather than pursuing it further without your direction** —
this would need you to hold or obtain SCARED access already; not something
to pursue unilaterally.

### Check 3 — what was the checkpoint actually trained on

**MEASURED, from the paper itself** (fetched arXiv HTML full text,
arxiv.org/abs/2405.08672 / arxiv.org/html/2405.08672): EndoDAC was trained
on **SCARED only** ("15351, 1705, and 551 frames for the training,
validation and test sets, respectively"), with **Hamlyn used only for
zero-shot validation** ("21 videos for validation"). **The paper does not
mention C3VD anywhere** — no C3VD training, no C3VD results, no C3VD
cross-dataset test. This matches Check 1's code-level finding
(`options.py` only ever accepts `--dataset endovis`): C3VD support in
`evaluate_depth.py`/`c3vd_dataset.py` is code the repo maintainer added for
general usability, disconnected from what the authors actually trained or
reported on.

**Correction (2026-09-20): the paragraph below originally called C3VDv2
"synthetic, rendered" — that was a factual error, caught and corrected
before it propagated further. Rewritten with the corrected premise.**

**What's actually rendered vs. real, per this project's own documentation**
(`docs/dataset_official_description.md:5`, "acquired with a static,
undeformed colon phantom"; `docs/conventions.md:3`, citing
`C3VDv3` as "the rendering/capture pipeline that produced this dataset";
`docs/gpu_validation.md:20`'s finding that `RenderingModule.cpp` outputs
`fbRgb` alongside the rendered GT channels, consistent with a real-capture
RGB stream being masked/packaged through the same pipeline rather than
being synthesized): **C3VDv2's RGB frames are captured with a real
clinical endoscope imaging a real (silicone) colon phantom.** Only the
GT side — depth, normals, coverage mesh, optical flow — is rendered,
against a separately-scanned 3D model of the phantom, registered to the
real camera's recovered trajectory (`tools/Handeye.cpp`'s hand-eye
calibration, cited in `docs/conventions.md`, is a real-camera/real-robot
calibration technique, not something a synthetic pipeline would need).
The black octagonal vignette (Check 1) is very plausibly genuine hardware
vignetting from the real endoscope's optics, not a rendering artifact —
and if so, it's a characteristic SCARED and Hamlyn footage (also captured
with real clinical endoscopes) may well share, not something unique to
C3VD. This last point is **inference, not verified** — I have not checked
SCARED/Hamlyn frames directly (no access, Check 2) for a comparable
vignette.

**Revised interpretation:** the actual domain shift is real tissue
(ex-vivo porcine for SCARED, presumably human or animal tissue for
Hamlyn) → a real silicone phantom, both imaged with real clinical
endoscopy hardware — not "real → synthetic." This is a substantially
*smaller* shift than the original (incorrect) framing claimed: same
general imaging modality, same rough optical/vignetting characteristics,
same category of specular highlights and lighting (real light on a wet or
lubricated surface, not a rendering-engine lighting model) — the
difference is the imaged material (organic tissue vs. silicone) and
phantom geometry, not the whole appearance domain. This revision is
**consistent with, and better explains, Check 5's finding** (§6): the
baseline prediction already correlates strongly with GT depth (Spearman
ρ ≈ -0.87) — a real endoscope-to-real-endoscope generalization gap, not a
sim-to-real one, is exactly the kind of gap a SCARED/Hamlyn-trained model
would be expected to cross reasonably well. **Still INTERPRETATION where
marked**: the precise degree of hardware/lighting similarity between the
C3VD acquisition rig and the SCARED/Hamlyn rigs hasn't been verified from
any source — Check 2 (SCARED access) or the C3VD paper's own acquisition
methodology section (not yet read) would be the way to check this
further.

### Check 4 — C3VD v1 vs C3VDv2 comparison

**Not run — would need your approval first**, per workflow rules (new
external download, more than the one sequence already extracted). Checked
feasibility only (no download, no extraction):

- **Not available locally**: `/data1_ycao/chua/datasets/C3VDv2` has no
  `c0_*` (C3VD v1) archives — confirmed by directory listing, 0 matches.
- **Externally available, and NOT gated** (unlike SCARED): checked
  durrlab.github.io/C3VD/ — C3VD v1 is distributed via direct Google Drive
  links, CC BY-NC-SA 4.0, no registration/data-use-agreement mentioned. A
  single registered-video archive is 0.6-11 GB depending on which
  cecum/sigmoid/etc. sequence is picked.

This would be a genuinely informative next check (same checkpoint, same
"Path A" script, C3VD v1 vs our C3VDv2 sequence — if v1 looks visually
plausible and v2 doesn't, that's much stronger evidence than the
interpretation above; if both look equally flat, the domain-gap story
weakens and points somewhere else). **Have not downloaded anything — this
needs your go-ahead first**, both because it's a new external dataset
download and because it's an additional sequence beyond the one already
extracted.

### Check 5 — vignette hypothesis, and the quantitative signal we'd been missing

Motivation: the frames have 6.4-7.0% pure-black vignette pixels (§0b);
`docs/gpu_validation.md:18` gives the exact geometry — optical center
`cx=677.74, cy=543.06` (native 1350x1080), all-valid radius <706px,
all-invalid radius >864px, mixed in between (the mask isn't a perfect
circle, consistent with the visibly octagonal border in the RGB frames). A
ViT backbone trained only on SCARED/Hamlyn (Check 3) may never have seen a
hard black border like this. No new downloads needed — tested by modifying
only the input, same checkpoint, same 3 frames.
Script: `scratch/pipelines/endodac_vignette_test.py`, log:
`logs/endodac_vignette_test.log`, images:
`results/pipelines/endodac_vignette_test/`.

Three input variants, all MEASURED:

- **(a) baseline** — as-is, full frame (same as §5).
- **(b) vignette inpainted** — pure-black pixels filled by nearest-valid-
  pixel extrapolation (`scipy.ndimage.distance_transform_edt`), full frame
  size kept.
- **(c) tight circular crop** — cropped to the largest axis-aligned square
  fully inside the confirmed-all-valid r=706 circle: a 998x998 box
  (`left=179, upper=44, right=1177, lower=1042` for this frame size),
  verified 0.000000 black pixels remaining inside the crop.

**The requested quantitative signal** — Pearson/Spearman correlation
between predicted inverse depth (raw sigmoid disparity, pre
`disp_to_depth`) and GT depth (mm), over GT-valid pixels only:

| Frame | Variant | n valid px | Pearson r | Spearman ρ |
|---|---|---|---|---|
| 0000 | a baseline | 1,161,423 | -0.756 | -0.876 |
| 0000 | b inpainted | 1,161,423 | -0.802 | -0.916 |
| 0000 | c tight crop | 801,476 | -0.836 | -0.908 |
| 0109 | a baseline | 1,166,039 | -0.782 | -0.851 |
| 0109 | b inpainted | 1,166,039 | -0.801 | -0.902 |
| 0109 | c tight crop | 806,092 | -0.880 | -0.959 |
| 0217 | a baseline | 1,161,414 | -0.751 | -0.873 |
| 0217 | b inpainted | 1,161,414 | -0.795 | -0.915 |
| 0217 | c tight crop | 801,467 | -0.841 | -0.912 |
| **mean** | **a baseline** | | **-0.763** | **-0.867** |
| **mean** | **b inpainted** | | **-0.800** | **-0.911** |
| **mean** | **c tight crop** | | **-0.852** | **-0.926** |

All correlations negative as expected (higher predicted disparity =
closer = lower GT depth, if the prediction tracks real geometry at all),
all p-values effectively 0 at ~1M pixels. **This is a major correction to
§5's framing.**

**§5 said "no lumen/tunnel structure... does not look like a plausible
reconstruction."** That was a visual read of an 8-bit PNG rendered with a
fixed 0-100mm display range, and it was too pessimistic. The Spearman
correlation for the *baseline* (no fix, exactly what §5 already ran) is
**-0.85 to -0.88** — strong, not weak, monotonic agreement with GT depth
over 1.16M pixels per frame. Looking again at the baseline PNGs with this
in mind, and directly at the tight-crop PNG (`0109_c_tightcrop_pred.png`
vs `0109_c_tightcrop_gt.png`): a blurred, low-contrast, but real analog of
the GT's bright lumen region (upper-left-of-center) versus darker
right-hand wall *is* visible on closer inspection in all three variants,
most legibly in (c). It is much lower-contrast and far less sharp than
GT — no distinct fold ridges, a soft blob instead of a crisp boundary —
but it is not the "no signal, wrong pattern" read from §5. **§5's
qualitative description was misleading; this section's numbers supersede
it.**

**Vignette hypothesis: partially supported, not the dominant effect.**
Correlation improves monotonically a → b → c (mean Spearman -0.867 →
-0.911 → -0.926), consistent with the black border hurting the
prediction somewhat. But the effect size is modest — roughly 0.04-0.06 of
Spearman ρ, on top of an already-strong baseline correlation — not the
difference between "broken" and "working" that would be expected if the
vignette were the primary cause of a genuinely flat/degenerate output.
Since the baseline was never actually degenerate (Check 0b already showed
real variation; this check now shows that variation is meaningfully
GT-correlated), the vignette looks like a real but secondary contributor,
not the explanation for a failure that, on this quantitative measure,
isn't as severe as §5 described.

### Summary

| Check | Status | Result |
|---|---|---|
| 0. numpy ABI break as the cause | Done | MEASURED: ruled out — pre-fix run crashed and produced no output; post-fix rerun bit-identical to original |
| 0b. Raw output + weight-loading verification | Done | MEASURED: checkpoint genuinely loaded (LoRA weights differ from zero-init); raw disparity has real spatial variation (std/mean 0.18-0.28), not constant — wrong pattern, not no signal |
| 1. Training vs. sanity-script preprocessing | Done | MEASURED: identical transform pipeline; preprocessing mismatch ruled out |
| 2. Reproduce authors' number | Blocked | SCARED is gated (data-use agreement), not attempted |
| 3. What was it trained on | Done | MEASURED (paper + code): SCARED only, zero-shot-validated on Hamlyn only, never C3VD |
| 4. C3VD v1 vs v2 domain-gap test | Not run | Feasible (ungated, ~1-11GB), needs your approval to download + extract |
| 5. Vignette hypothesis (quantitative correlation) | Done | MEASURED: baseline Spearman ρ already -0.87 (strong); crop/inpaint improve it modestly to -0.93. §5's "no structure" read corrected — real GT-correlated signal present, vignette a secondary, not dominant, contributor |

**Revised bottom line.** §5's visual impression of a flat, implausible
prediction does not survive quantitative check: baseline predicted inverse
depth already correlates strongly with GT depth (Spearman ρ ≈ -0.87 across
3 frames, ~1.16M pixels each). The checkpoint is producing real,
GT-correlated geometric structure zero-shot on C3VDv2, just at much lower
contrast/sharpness than GT and without resolving fine structure (fold
ridges). The vignette border is a real, measurable, secondary drag on
correlation (~0.04-0.06 Spearman ρ), not the explanation for a "broken"
output — because the output was not actually broken. The domain-gap
interpretation (Check 3) is still relevant to explain the *gap in
sharpness/scale*, but "the checkpoint fails to generalize to C3VDv2" is no
longer an accurate summary; "the checkpoint generalizes with reduced
fidelity" is. Check 4 (C3VD v1 comparison) would still be informative for
quantifying that fidelity gap, but the original framing that motivated it
("does EndoDAC even produce plausible geometry here") is now answered:
yes, with a measurable, strong, quantitative correlation.

---

## 7. Pose convention: empirical validation

Before any trajectory work, per instructions: verify
`transformation_from_parameters`'s output convention against our frozen
`.reshape(4,4).T` camera-to-world convention **empirically, not from
reading alone** — the same way the pose.txt transpose ambiguity was
originally settled. Baseline input (no crop, no inpaint) throughout, per
instructions. Scripts: `scratch/pipelines/endodac_pose_validation.py`
(global chain+alignment) and
`scratch/pipelines/endodac_pose_validation_perstep.py` (per-step,
written after the first test came back inconclusive — see below). Logs:
`logs/endodac_pose_validation*.log`.

### First attempt: chain + Umeyama + ATE — inconclusive, and why

Ran pose inference on 20 consecutive frames (0-19), chained the 19
relative transforms into a trajectory two ways — (A) as predicted,
right-multiplied (`world_T_cam_{i+1} = world_T_cam_i @ T_i`, matching
`evaluate_pose.py`'s own `dump_xyz` convention), (B) with each `T_i`
inverted before chaining — Umeyama-aligned (rotation+translation+scale)
each to the GT trajectory from `pose.txt`, and compared ATE (RMSE after
alignment).

**MEASURED, frames 0-19**: ATE_A = 0.0047mm, ATE_B = 0.0047mm — a
complete tie. Investigated why before trusting it: this window's GT motion
totals only **0.077mm over 19 steps** (checked directly from `pose.txt`) —
the sequence opens on a near-static settling period, not real motion. Too
small a signal to discriminate anything.

Reran on frames 30-49 (GT path length 45.7mm, reasonable motion) and
frames 139-158 (path length 37.8mm, picked for higher curvature —
straightness ratio 0.60 vs. 0.89 for the 30-49 window, by scanning all
198 possible 20-frame windows for one with real curvature, not just a
near-straight segment). **Still inconclusive**: ATE_A=1.45mm vs.
ATE_B=1.36mm (frames 30-49); ATE_A=1.22mm vs. ATE_B=1.21mm (frames
139-158) — real motion now, but still no clear winner.

**Why the global test doesn't discriminate, established directly, not
assumed:** per-step predicted rotations are tiny (see §7's per-step table
below, sub-degree to low-single-digit-degree GT relative rotation over
one frame step) — near-identity rotations satisfy `R ≈ Rᵀ`, so inverting a
step barely changes its rotational contribution. What it does change is
the translation's sign. For a chain of near-identity-rotation,
sign-flippable steps, a **global proper 3D rotation** (which Umeyama
alignment is free to choose) can often convert the "wrong-direction" chain
into something close to congruent with the correct one, particularly over
a short window — this is a known blind spot of chain+Umeyama ATE, not a
property specific to this checkpoint. The fix: stop giving the alignment
step that much freedom.

### Second attempt: per-step comparison — decisive

For each consecutive pair `(i, i+1)` in a window, computed the GT relative
transform directly, `G = inv(C2W_i) @ C2W_{i+1}` (in the same "chains via
right-multiplication" convention as hypothesis A above, so `G` is the
correct-convention target for the network's raw output `T`), and compared
**per pair, with no global alignment freedom**: rotation angle of
`Gᵀ @ R` (degrees) and cosine similarity between GT and predicted
translation *directions* (magnitude is scale-ambiguous, §3, so only
direction is meaningful) — for both `T` as-is and `inv(T)`.

**MEASURED, frames 139-158 (19 pairs):**

| | mean rotation error | mean translation-direction cosine similarity |
|---|---|---|
| Hypothesis A (`T` as predicted) | 0.36° | **-0.976** |
| Hypothesis B (`inv(T)`) | 0.19° | **+0.976** |

**MEASURED, frames 30-49 (19 pairs), independent confirmation:**

| | mean rotation error | mean translation-direction cosine similarity |
|---|---|---|
| Hypothesis A (`T` as predicted) | 0.49° | **-0.887** |
| Hypothesis B (`inv(T)`) | 0.42° | **+0.886** |

Every single pair in both windows shows the same sign pattern (checked
individually in the logs, not just the means — e.g. frames 139-158's 19
pairs are unanimous, cosine similarity negative for A and positive for B
in every one). Rotation error is small and similar for both hypotheses
(expected, given `R ≈ Rᵀ` for near-identity rotations — not
discriminative, consistent with why the global test failed). Translation
direction is completely decisive: **~-0.9 (nearly anti-parallel) for the
raw network output, ~+0.9 (nearly parallel) for the inverted output,
unanimous across 38 total pairs in 2 independent windows.**

**Conclusion: `transformation_from_parameters`'s raw output must be
inverted before being chained as a camera-to-world step.** Equivalently:
the raw output `T` (from the color-channel order `cat([frame_{i+1},
frame_i])`, no `invert` flag passed, matching `evaluate_pose.py`'s own
call) represents the transform from frame `i`'s local camera frame into
frame `i+1`'s local camera frame — call it `T_{(i+1)←i}` — not the
`C2W_i → C2W_{i+1}` step needed for chaining, which is its inverse,
`T_{i←(i+1)}`.

**Worth flagging, INTERPRETATION not proven:** `evaluate_pose.py`'s own
`dump_xyz`/ATE computation (used to produce the paper's reported pose
numbers) chains the raw, non-inverted output — i.e., it uses what this
check found to be hypothesis A, the wrong one for matching our GT
convention. Two readings are possible, and I can't distinguish them
without SCARED access (Check 2, blocked): either (a) their own script has
the same sign issue, silently invisible in their reported ATE because
chain+Umeyama+ATE is insensitive to it for the reason established above
(their own evaluation methodology would share this blind spot), or (b)
SCARED's pose.txt-equivalent ground truth uses the opposite
camera-pose convention from ours, in which case what's "wrong" for us is
right for them and there's no bug at all — just two projects' differing
GT conventions. Not resolved, not blocking (our own empirical test is
sound regardless of which explanation is true), flagged for completeness.

### What this means going forward

Any code that chains EndoDAC's predicted relative poses into an absolute
trajectory for comparison against this project's GT must invert each
`transformation_from_parameters` output first. Applied in §8's
full-sequence run below.

---

## Open questions

1. ~~Does the public checkpoint include pose/intrinsics weights?~~
   **Resolved (§2): yes**, both `intrinsics_head.pth` and the pose
   encoder/decoder are present alongside `depth_model.pth`.
2. ~~Path A vs Path B preprocessing — which to use going forward?~~
   **Resolved differently than either earlier answer (§6 Check 5):** Path A
   (baseline, no crop) already gives a strong GT correlation (Spearman
   ρ ≈ -0.87); Path B-style tight cropping improves it only modestly
   (ρ ≈ -0.93). Both "produce plausible depth" in the quantitative sense —
   the September 19 note that "neither produces plausible-looking depth"
   was based on the superseded flat-prediction finding (§5) and no longer
   holds. Path A (baseline) is used going forward per the user's
   instruction to use baseline input for pose work and the full-sequence
   run.
3. Whether the crop box in `c3vd_dataset.py` (`200, 180, 1150, 900`) is
   removing a black vignette border or discarding real fisheye image
   content — unverified, would need visual inspection of a cropped-out
   region. Low priority: Check 5 shows the crop only matters modestly for
   correlation either way.
4. ~~The relative-pose matrix convention...~~ **Resolved empirically (§7):**
   `transformation_from_parameters`'s raw output must be **inverted**
   before chaining as a camera-to-world step to match our
   `.reshape(4,4).T` convention — confirmed by per-step translation-
   direction cosine similarity (+0.88 to +0.98 inverted vs. -0.88 to -0.98
   as-is, unambiguous across 2 independent 19-pair windows). The global
   chain+Umeyama+ATE test that was tried first was inconclusive (near-tied
   ATE both ways) for a specific, identified reason — see §7 — not because
   the convention doesn't matter.
5. What `position*`/`transform*` auxiliary networks in the checkpoint
   actually do — traced to training-only code (`trainer_end_to_end.py`),
   not imported by any inference/eval script, not investigated further
   since out of scope for this sanity check.
6. **Resolved, and the premise was wrong (§6 Checks 0b and 5)**: "why does
   predicted depth show no lumen/fold structure" assumed a flat/failed
   prediction that quantitative measurement does not support. MEASURED:
   raw disparity has real spatial variation (Check 0b), and that variation
   correlates strongly with GT depth (Spearman ρ ≈ -0.87 baseline, Check
   5) — not a preprocessing bug (Check 1), not a botched weight load
   (Check 0b), not primarily the vignette (Check 5, secondary effect
   only). Remaining open item, now narrower: whether the domain gap
   (Check 3: SCARED/Hamlyn-only training) explains the *sharpness/contrast*
   gap versus GT, which Check 4 (C3VD v1 comparison, needs approval) could
   still speak to — but this is no longer a question of basic viability.
