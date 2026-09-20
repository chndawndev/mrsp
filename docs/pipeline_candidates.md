# Pipeline candidate survey

Purpose: survey availability of candidate depth/pose pipelines before committing
to one for the missed-region localization benchmark, per request. **Pure
research — nothing installed, cloned, or run.** All facts below were checked
against public GitHub API metadata, README/raw-file fetches, and arXiv pages
on 2026-09-19/20; URLs are given so every claim can be rechecked.

Two methods on the requested list (**ColonCrafter**, **CoGE**) have no public
code as of this check — searched GitHub, GitHub Topics, and the arXiv HTML
full text of both papers directly; no repository, project page, or "code will
be released" statement was found in either paper. They are marked NOT FOUND
rather than given a guessed URL.

**Naming note — EndoDAC vs EndoDAV**: these are two distinct, real, separately
published methods, not a typo of one another. Both riff on "Depth Anything":
EndoDAC = "Efficient Adapting foundation model... from **Any** Endoscopic
**C**amera" (MICCAI'24, [BeileiCui/EndoDAC](https://github.com/BeileiCui/EndoDAC)).
EndoDAV = "Depth **A**ny **V**ideo in Endoscopy" (MICCAI'25,
[Zanue/EndoDAV](https://github.com/Zanue/EndoDAV)). A third repo,
`RicardoEspinosaLoera/EndoDAC`, also turned up in search — it is not the
official repo (no MICCAI badge, low activity); the official EndoDAC repo is
BeileiCui's. Flagging this because the near-identical acronyms are easy to
mix up in config/script names later.

## Summary table

| Method | Repo status | Weights downloadable | Output type | GT pose injectable | GT depth injectable | Intrinsics | Last commit | Runs out of box? |
|---|---|---|---|---|---|---|---|---|
| EndoDAC | Found, official | Yes (Google Drive, checkpoint + backbone) | depth + pose + intrinsics | UNKNOWN (not documented) | UNKNOWN (not documented) | Estimated internally | 2025-01-23 | No blocking issues found; open issues are C3VD-adaptation questions |
| EndoDAV | Found, official | Yes — verified via git-lfs blob (464 MB), despite terse README | depth + pose (+ intrinsics decoder in code) | UNKNOWN (not documented) | UNKNOWN (not documented) | Estimated internally (code has `intrinsics_decoder.py`) | 2025-09-25 | No bug-report issues found (2 open, both usage questions); README self-describes code as "messy" |
| ColonAdapter | Found, official | Yes (OneDrive link) + needs external DUSt3R weights | depth (+ pose, pointmaps per paper claims) | UNKNOWN (not documented) | UNKNOWN (paper claims no GT depth needed, not an injection API) | Estimated internally (paper: "without requiring ground-truth intrinsic parameters") | 2026-04-08 (most recent of all 10) | 0 open issues, but default branch is literally named `temp` — immaturity signal |
| ColonCrafter | **NOT FOUND** | N/A | depth (per paper) | N/A | N/A | UNKNOWN | N/A | N/A — no code to run |
| CoGE | **NOT FOUND** | N/A | depth + reconstruction (per paper) | N/A | N/A | UNKNOWN | N/A | N/A — no code to run |
| Endo3R | Found, official | Yes (gdown link) + needs external DUSt3R weights | pointmaps + scale-consistent depth + pose + intrinsics | UNKNOWN (GT pose appears only in eval scripts, per issue #6) | UNKNOWN (not documented) | Estimated internally | 2025-09-25 | Inference works; training code explicitly NOT released (3 of 5 open issues ask for it); 1 issue is a license-clarification request |
| DROID-SLAM | Found, official | Yes (Google Drive / `download_model.sh`) | pose + depth | UNKNOWN / no documented mode | UNKNOWN / no documented mode | **Required as input**, pinhole + OpenCV distortion string (`fx fy cx cy [k1 k2 p1 p2 ...]`) | 2025-05-05 | Large accumulated issue backlog (104 open); the newest non-PR issues sampled are minor, not blocking |
| MASt3R-SLAM | Found, official | Yes (direct Naver Labs Europe downloads) | dense pointmaps + poses (per-pixel confidence) | UNKNOWN / no documented mode | UNKNOWN / no documented mode | **Both modes supported**: calibrated (`intrinsics.yaml`) or uncalibrated (`--no-calib`); claims no fixed/parametric camera-model assumption | 2025-11-09 (most recently pushed) | 68 open issues incl. "CUDA 12.6" and "Not able to run on RTX 5070" — real GPU/CUDA friction on newer hardware |
| CUT3R | Found, official | Yes (2 checkpoints via gdown) | pointmaps (README does not explicitly restate depth/pose decomposition) | UNKNOWN (not documented) | UNKNOWN (not documented) | UNKNOWN (not documented in README) | 2025-08-27 | 65 open issues; recent ones sampled are clarifying questions, not bug reports |
| MonST3R | Found, official | Yes (Google Drive or Hugging Face, `download_ckpt.sh`) | per-frame pointmaps + per-frame pose + per-frame intrinsics | UNKNOWN (not documented) | UNKNOWN (not documented) | Estimated internally | 2025-06-16 | 30 open issues incl. a reproducibility complaint (#83) and evaluation-script errors (#82, #87) |

UNKNOWN means: not stated in the README/paper text I could fetch, and I did
not infer or guess. Where I do infer something from architecture rather than
explicit documentation, it is labeled "(inference)" in the per-method
paragraph below, not stated as fact.

## Per-method detail

### EndoDAC
Repo: [BeileiCui/EndoDAC](https://github.com/BeileiCui/EndoDAC) (MICCAI 2024,
official; [paper](https://papers.miccai.org/miccai-2024/272-Paper0225.html)).
`pushed_at` from the GitHub API is 2025-01-23T13:27:24Z; 3 open issues, all
in Chinese, asking about C3VD dataset adaptation/training — not bug reports.
README ([raw](https://raw.githubusercontent.com/BeileiCui/EndoDAC/main/README.md))
gives a Google Drive checkpoint link plus a required
`depth_anything_vitb14` backbone checkpoint (also Google Drive). Outputs
depth, pose, and camera intrinsics jointly, self-supervised
("a self-supervised adaptation strategy that estimates camera intrinsics
using the pose encoder" — README). No documented option to inject GT pose or
GT depth as a conditioning input; the repo ships `export_gt_depth`/GT-pose
scripts but those are for evaluation comparison, not model input.

### EndoDAV
Repo: [Zanue/EndoDAV](https://github.com/Zanue/EndoDAV) (MICCAI 2025,
official). `pushed_at` 2025-09-25T00:31:53Z; 2 open issues (#2 "Details about
training", #3 "About dataset9 keyframe4" — usage questions, not bugs). The
README itself is minimal — one line: "9/24: release code and model weights.
the code is messy and i am still working on it" — with no explicit download
link in the README text. However, the repo's git tree
([recursive listing](https://api.github.com/repos/Zanue/EndoDAV/git/trees/main?recursive=1))
shows `ckpts/models.zip` as a git-lfs pointer (`size 464347092` = ~443 MB); I
verified the LFS blob is actually servable with `curl -I
https://media.githubusercontent.com/media/Zanue/EndoDAV/main/ckpts/models.zip`
→ HTTP 200, `content-type: application/zip`. So weights are real and
downloadable, just not documented as clearly as the README's tone suggests.
The code tree includes `models/decoders/pose_decoder.py`,
`models/decoders/pose_cnn.py`, and `models/decoders/intrinsics_decoder.py`,
consistent with the paper abstract's "simultaneously learns depth and camera
pose" and a SSB-LoRA fine-tuning scheme. No documented GT-pose/GT-depth
injection API.

### ColonAdapter
Repo: [JayJiang99/ColonAdapter](https://github.com/JayJiang99/ColonAdapter)
(RAL journal, official; [paper](https://arxiv.org/abs/2511.22250)).
`pushed_at` 2026-04-08T01:14:08Z — the most recently updated repo of all ten
candidates — but the **default branch is named `temp`**, which reads as a
pre-release/unstable state rather than a finished main branch. 0 open issues
(low sample size given how new/low-traffic the repo is; not strong evidence
either way). README gives a OneDrive weight download link plus a dependency
on external DUSt3R pretrained weights
(https://download.europe.naverlabs.com/ComputerVision/DUSt3R/). The paper
abstract claims SOTA in "camera pose estimation, monocular depth prediction,
and dense 3D point map reconstruction, without requiring ground-truth
intrinsic parameters," and reports ATE 0.0062 on SimCol3D — but the
inference script excerpt I fetched only shows depth/disparity/depth-viz
outputs, so the pose/pointmap outputs claimed in the paper are not confirmed
to be exposed through the same documented inference entry point (flagged as
a gap, not resolved).

### ColonCrafter — NOT FOUND
Paper: ["ColonCrafter: A Depth Estimation Model for Colonoscopy Videos Using
Diffusion Priors"](https://arxiv.org/abs/2509.13525) (Biocomputing 2026). I
fetched the full arXiv HTML text
([html](https://arxiv.org/html/2509.13525v1)) and found **no mention of code,
GitHub, or a project page anywhere in the paper** — not even a "code will be
released" statement. Searched GitHub directly for "ColonCrafter" — nothing
relevant. No usable public implementation exists as of this check; do not
plan around this one without rechecking later.

### CoGE — NOT FOUND
Paper: ["CoGE: Sim-to-Real Online Geometric Estimation for Monocular
Colonoscopy"](https://arxiv.org/abs/2605.13038) (2026, Shao/Cui/Ren, CUHK).
Same result as ColonCrafter: full arXiv HTML text
([html](https://arxiv.org/html/2605.13038v1)) has no code/GitHub/project-page
mention, and the GitHub Topics page for `coge-colonoscopy` is explicitly
empty ("hasn't been used on any public repositories, yet"). No usable public
implementation as of this check.

### Endo3R
Repo: [wrld/Endo3R](https://github.com/wrld/Endo3R) (MICCAI 2025 Oral,
official; [project page](https://wrld.github.io/Endo3R/)). `pushed_at`
2025-09-25T22:18:01Z; 5 open issues — notably #7 "Apply for the training
code", #5 "may I ask when the training code will be released?", #4 "Inquiry
About the Release of Training Code" — i.e. **only inference code and weights
are public; training code is explicitly withheld** as of this check. Issue
#3 is a license-clarification request, meaning the license terms are
apparently unclear to users too. #6 asks about GT camera pose for SCARED
dataset 8 & 9, which is evaluation-only usage, not evidence of a GT-pose
injection API. Weights: `gdown` link to a Google Drive checkpoint, plus a
required external DUSt3R base checkpoint (wget from Naver Labs). Outputs
"globally aligned pointmaps, scale-consistent video depth, camera poses and
intrinsics" (README/abstract) — the broadest single-method output of any
candidate surveyed. Intrinsics are estimated, not required
("without any prior information or extra optimization").

### DROID-SLAM
Repo: [princeton-vl/DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM)
(Teed & Deng, NeurIPS 2021, official). `pushed_at` 2025-05-05T05:29:42Z; 104
open issues — the largest backlog of any candidate, reflecting the repo's
age and popularity more than necessarily indicating breakage; the two most
recently created non-PR issues sampled (#172, #167) are minor/feature-request
in nature, not "it doesn't run" reports. Weights downloadable via Google
Drive / `tools/download_model.sh`. Outputs camera pose trajectory and depth
maps (`--reconstruction_path`). **Camera intrinsics are a required input**,
in an explicit pinhole + optional OpenCV radial/tangential distortion string
format: `fx fy cx cy [k1 k2 p1 p2 [k3 [k4 k5 k6]]]`
([README](https://raw.githubusercontent.com/princeton-vl/DROID-SLAM/main/README.md)).
This project's camera is a Scaramuzza omnidirectional model (per
`docs/conventions.md`), so DROID-SLAM cannot consume it directly — it would
need an upstream undistort-to-virtual-pinhole step. This project has already
investigated that specific cost/tradeoff (see `docs/pinhole_tradeoff.md` and
commit `36b95d0`), so the groundwork for evaluating this cost already exists.
No documented mode to inject GT pose or GT depth in place of its own
estimates.

### MASt3R-SLAM
Repo: [rmurai0610/MASt3R-SLAM](https://github.com/rmurai0610/MASt3R-SLAM)
(CVPR 2025, official; [paper](https://arxiv.org/abs/2412.12392)). `pushed_at`
2025-11-09T07:36:29Z, the most recently pushed repo among the SLAM-style
candidates. 68 open issues; representative recent titles include #140 "CUDA
12.6" and #137 "Not able to run on RTX 5070" — concrete evidence of
GPU/CUDA-version friction on newer hardware, worth checking against this
lab's GPU generation before committing. Weights: three files hosted directly
at `download.europe.naverlabs.com/ComputerVision/MASt3R/` (no gating, direct
download). Outputs dense pointmaps with per-pixel confidence plus globally
consistent per-frame camera poses; the paper states 15 FPS on the reported
hardware (arXiv 2412.12392). Camera intrinsics: **uniquely among the ten
candidates, this repo documents two explicit input modes** — calibrated
(`intrinsics.yaml`) or uncalibrated (`--no-calib` self-calibrating). The
paper also states the method "makes no assumption on a fixed or parametric
camera model beyond a unique camera centre," which is suggestive for
omnidirectional-camera use — but this is the paper's general claim, not a
confirmation that the Scaramuzza model specifically is supported; treat as
**inference**, not fact, until checked against the code. No documented mode
to inject GT pose or GT depth.

### CUT3R
Repo: [CUT3R/CUT3R](https://github.com/CUT3R/CUT3R) (official implementation
of ["Continuous 3D Perception Model with Persistent
State"](https://arxiv.org/abs/2501.12387)). `pushed_at`
2025-08-27T15:30:07Z; 65 open issues, recent ones sampled (#114, #113, #112,
#110) are clarifying/architecture questions, not bug reports. Weights: two
checkpoints (`cut3r_224_linear_4.pth`, `cut3r_512_dpt_4_64.pth`) via `gdown`,
both confirmed present with working Google Drive file IDs in the README.
Output type: the fetched
[README](https://github.com/CUT3R/CUT3R/blob/main/README.md) does not
explicitly restate its output decomposition; based on the paper title/abstract
and its stated lineage from DUSt3R/MonST3R/Spann3R, it is a pointmap-based
model (**inference**, not confirmed by README text alone). Whether GT pose or
GT depth can be injected, and whether intrinsics are required or estimated,
are UNKNOWN — not addressed in the README I fetched. No inference-speed
figure found; the only performance note is that the encoder processes frames
in parallel for linear (not quadratic) memory growth with frame count.

### MonST3R
Repo: [Junyi42/monst3r](https://github.com/Junyi42/monst3r) (official;
[project page](https://monst3r-project.github.io/);
[paper](https://arxiv.org/abs/2410.03825)). `pushed_at`
2025-06-16T21:01:48Z; 30 open issues, with concrete friction signals sampled:
#87 "error occurs in sam2 video segmentation: NotImplementedError: Only MP4
video and JPEG folder are supported", #83 "Cannot reproduce static scene
performance of DUSt3R and MonST3R on NYU and ScanNet" (a reproducibility
complaint against the authors' own reported numbers), and #82 "Raising
errors when run evaluation example on davis dataset." Weights downloadable
via Google Drive or Hugging Face
([Junyi42/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt](https://huggingface.co/Junyi42/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt)),
with a `data/download_ckpt.sh` convenience script. Outputs a time-varying
dynamic point cloud plus per-frame camera poses and per-frame intrinsics,
which the README states can be used to derive video depth and dynamic/static
segmentation downstream. Intrinsics are estimated internally, not required.
Memory: ~33 GB VRAM stated for a 65-frame 16:9 video, reducible to ~23 GB
with optimized settings — worth checking against this lab's shared GPU
capacity (`nvidia-smi`) before committing, per workflow rules. No documented
mode to inject GT pose or GT depth.

## Observations relevant to the planned four-way decomposition (descriptive, not a recommendation)

The plan under discussion is a four-way ablation: {pipeline depth, GT depth}
× {pipeline pose, GT pose}. This requires a documented way to feed GT pose
into a method while it still predicts depth, and vice versa.

- **None of the ten candidates documents a built-in API for this.** The
  foundation-model group (EndoDAC, EndoDAV, ColonAdapter, Endo3R, CUT3R,
  MonST3R) jointly predicts depth/pose/intrinsics end-to-end with no
  documented conditioning input for GT pose or GT depth. The classical-SLAM
  group (DROID-SLAM, MASt3R-SLAM) couples pose and depth inside a bundle-
  adjustment/optimization loop, which is architecturally a different kind of
  obstacle to "injecting" one half — it is not a simple flag either. GT pose
  appearing in a repo (e.g. Endo3R issue #6, DROID-SLAM/MASt3R-SLAM eval
  scripts) is consistently for **evaluation/metric comparison**, never as a
  documented model input.
- Practically, this pushes the decomposition from "which pipeline supports
  this out of the box" (none do) to "whose code is simple/well-documented
  enough to safely modify to add GT-conditioning without that modification
  itself becoming a source of error" — a code-risk question this survey did
  not attempt to answer and that would need a closer read of each candidate's
  actual model-forward code, not just its README.
- **ColonCrafter and CoGE are disqualified outright right now** — no public
  code found despite direct searches and full-paper-text checks. This could
  change; recheck before ruling them out permanently.
- **DROID-SLAM is the only candidate with a hard pinhole-intrinsics
  requirement** baked into its input format. Every other candidate either
  estimates intrinsics internally or (MASt3R-SLAM) supports both calibrated
  and uncalibrated modes. Given this project's Scaramuzza omnidirectional
  camera and the frozen convention against substituting a pinhole model
  (`docs/conventions.md`), DROID-SLAM would require an explicit
  undistort-to-pinhole preprocessing step, whose cost this project has
  already partially quantified elsewhere (`docs/pinhole_tradeoff.md`).
- **Method-family overlap, for context on independence**: EndoDAC, EndoDAV,
  Endo3R, CUT3R, MonST3R, and MASt3R-SLAM all build on or fine-tune from the
  DUSt3R/vision-foundation-model lineage rather than being independently
  designed architectures; ColonAdapter also explicitly adapts a "geometric
  foundation model." Only DROID-SLAM is architecturally unrelated to that
  lineage (classical recurrent bundle adjustment). This is worth keeping in
  mind if multiple candidates are later used together as if they were
  independent evidence.
- **Maturity/stability flags worth carrying forward**: ColonAdapter's default
  branch is named `temp`; EndoDAV's README self-describes the code as
  "messy"; Endo3R withholds training code and has an open license-
  clarification issue; MASt3R-SLAM has recent CUDA/GPU-generation compat
  issues open. None of these are confirmed blockers, but each is a concrete,
  citable reason to budget extra setup time if that candidate is chosen.
