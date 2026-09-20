# EndoDAC pipeline setup and single-sequence inference

Repo: https://github.com/BeileiCui/EndoDAC (MICCAI 2024, "Efficient Adapting
Foundation Model for Self-Supervised Depth Estimation from Any Endoscopic
Camera"). Cloned to `scratch/pipelines/EndoDAC` at commit
`22ec911b270d056fdd7f430474e4be407cce8287` (2025-01-23), the current tip of
`main` as of this survey (2026-09-19).

Target sequence: `c1_cecum_t1_v1`, 218 RGB frames, 1350x1080, already
extracted at `scratch/c1_cecum_t1_v1/` (rgb/, depth/, pose.txt, etc.).

Status: **environment built, weights downloaded, sanity-check inference
run, cause diagnosed to the extent possible without new data access.**
Runtime looks fine (~16 ms/frame). Sanity-check depth PNGs raised a real
concern (§5): no lumen/tunnel structure. §6's diagnosis: not a
preprocessing bug (ruled out, MEASURED) and not something we can pin down
further without either gated SCARED access or your approval to pull an
ungated C3VD v1 sequence for comparison. Leading interpretation: the
checkpoint was trained/validated only on real tissue video (SCARED,
Hamlyn) and never on anything resembling C3VD's synthetic rendering — a
plausible but unproven domain-gap explanation.

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

## 5. Sanity-check depth PNGs — done, and the result is a concern

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

**Interpretation carried forward, not proven:** the released checkpoint has
only ever seen real endoscopic tissue video (SCARED, ex-vivo porcine;
zero-shot-tested on Hamlyn, also real tissue). C3VDv2 is a synthetic,
rendered phantom dataset with a different appearance model entirely
(rendering-engine lighting/shading, no real specular-moisture tissue look,
different noise characteristics, and the octagonal black-vignette FOV mask
quantified in Check 1). This is a substantially larger domain shift than
"real tissue A → real tissue B" (SCARED → Hamlyn), which is the only
cross-domain generalization the paper actually demonstrates. **This is a
plausible, well-supported explanation for the flat/near-degenerate
prediction, but it is INTERPRETATION, not proven** — Checks 2 and 4 are
the ways to actually confirm it, and neither has been run.

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

### Summary

| Check | Status | Result |
|---|---|---|
| 1. Training vs. sanity-script preprocessing | Done | MEASURED: identical transform pipeline; preprocessing mismatch ruled out |
| 2. Reproduce authors' number | Blocked | SCARED is gated (data-use agreement), not attempted |
| 3. What was it trained on | Done | MEASURED (paper + code): SCARED only, zero-shot-validated on Hamlyn only, never C3VD |
| 4. C3VD v1 vs v2 domain-gap test | Not run | Feasible (ungated, ~1-11GB), needs your approval to download + extract |

**Leading interpretation, not proven**: a real-tissue-only-trained
checkpoint applied zero-shot to a synthetic rendered dataset with a
different appearance model and a partially-black-vignetted FOV — a domain
shift well outside anything the paper's own generalization claims cover.
Not a bug in our setup (Check 1 rules that out); not confirmed as *the*
cause without Check 4.

---

## Open questions

1. ~~Does the public checkpoint include pose/intrinsics weights?~~
   **Resolved (§2): yes**, both `intrinsics_head.pth` and the pose
   encoder/decoder are present alongside `depth_model.pth`.
2. ~~Path A vs Path B preprocessing — which to use going forward?~~
   **Partially settled (§5): neither produces plausible-looking depth.**
   The choice between them is no longer the leading question; *why both
   fail* is.
3. Whether the crop box in `c3vd_dataset.py` (`200, 180, 1150, 900`) is
   removing a black vignette border or discarding real fisheye image
   content — unverified, would need visual inspection of a cropped-out
   region. Lower priority now given finding 5 below.
4. The relative-pose matrix convention (`transformation_from_parameters`)
   has not been checked against our `.reshape(4,4).T` pose.txt convention
   for sign/handedness compatibility — needed before any pose comparison,
   not needed for the depth-only sanity check. Also now secondary to
   finding 5: no point checking pose convention on a method whose depth
   output already looks implausible on this dataset.
5. What `position*`/`transform*` auxiliary networks in the checkpoint
   actually do — traced to training-only code (`trainer_end_to_end.py`),
   not imported by any inference/eval script, not investigated further
   since out of scope for this sanity check.
6. **Partially resolved (§6)**: why does predicted depth show no lumen/fold
   structure? MEASURED: not a preprocessing bug (Check 1 — our script
   exactly matches training preprocessing) and not the missing ImageNet
   normalization (Check 1 — training never used it either, so it's not a
   mismatch). MEASURED: checkpoint trained only on SCARED, zero-shot
   validated only on Hamlyn, never on C3VD (Check 3, from the paper
   itself). **Still open**: whether train/test domain gap is *the* actual
   cause (INTERPRETATION only) — Check 2 (reproduce on SCARED) is blocked
   by gated access, Check 4 (C3VD v1 vs v2 comparison) is feasible but
   needs approval to download. Until one of those runs, EndoDAC's
   viability as a candidate remains genuinely open, not resolved.
