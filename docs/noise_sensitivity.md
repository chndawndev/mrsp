# Noise sensitivity of the three pipelines, and variability-run inference

Descriptive. **No evaluation metric was computed and nothing under
`src/eval/` or `src/eval_ext/` was run on any output described here.**
Inference only, plus trajectory and depth-scale descriptives.

Protocol text followed, `docs/eval_protocol.md` lines 465-495
("2026-10-02: Numerical run-to-run variability (before any CUT3R
metric)"), items 1 and 2 quoted:

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

Items 3 and 4 (what is reported, and that the runs change no verdict)
concern the evaluation and are not acted on here.

Code (committed): `scripts/cut3r_recurrent.py` (`batched_encoder`),
`scripts/cut3r_run.py` (`--path recurrent_batchenc`),
`scripts/cut3r_run_corpus.py`, `scripts/endodac_inference.py` and
`scripts/mast3r_slam_run_perframe.py` / `scripts/mast3r_slam_run_corpus.py`
(`--noise-sigma`, `--noise-seed`, `--out-root`; defaults leave the
primary runs' code path unchanged), `scripts/noise_sensitivity_compare.py`,
`scripts/pinned_and_variability_runs.sh`. Outputs (not committed):
`results/pipelines/variability/`, `results/noise_sensitivity/`. Logs:
`logs/pinvar_chain.log`, `logs/variability_*.log`,
`logs/noise_sensitivity_compare.log`.

---

# MEASURED

## 1. Confirmation test (CUT3R)

Question left open by `docs/pipelines/cut3r.md` Stage 2, interpretation
1: do the parallel and recurrent paths differ only through how frames are
batched in the image encoder?

Test (`scripts/cut3r_run.py --path recurrent_batchenc`,
`logs/cut3r_confirmation_batchenc.log`): the image encoder is called once
on all 218 frames, as the parallel path does
(`model._encode_views` -> `_encode_image`), and the vendored
`forward_recurrent` loop is run with its per-frame `_encode_image` calls
answered from that batch. Nothing after the encoder is changed.

Result on `c1_cecum_t1_v1`, against the Stage 1 parallel outputs
(`results/pipelines/cut3r_stage1/compare__c1_cecum_t1_v1__vs__c1_cecum_t1_v1_recurrent_batchenc.json`):
**bit-identical.** Translation, rotation and Z differences are exactly 0
in all 218 frames.

So on this sequence the entire parallel-vs-recurrent difference (up to
21.3 deg, Stage 2) enters through the encoder's batch dependence; the
decoder, state update and heads of the two paths compute the same thing.

## 2. Pinned configurations and where the noise is added

| | EndoDAC | MASt3R-SLAM | CUT3R |
|---|---|---|---|
| entry point | `scripts/endodac_inference.py` | `scripts/mast3r_slam_run_perframe.py` via `scripts/mast3r_slam_run_corpus.py` | `scripts/cut3r_run_corpus.py` (recurrent wrapper) |
| batch size | 1 frame (depth), 1 pair (pose) | 1 frame | 1 frame |
| precision settings | PyTorch defaults (matmul TF32 off, cuDNN TF32 on); `cudnn.deterministic = True`, `cudnn.benchmark = False` (`endodac_inference.py:184-185`, part of the primary run since D1 Stage A) | matmul TF32 on, as the vendored `main.py:147` sets it (`mast3r_slam_run_perframe.py:187`); cuDNN TF32 on (default) | matmul TF32 on, set by the vendored `src/croco/models/croco.py:13`; cuDNN TF32 on (default) |
| GPU model, primary run | NVIDIA RTX 6000 Ada Generation (index 7) | NVIDIA RTX 6000 Ada Generation (index 7) | NVIDIA RTX 6000 Ada Generation (index 0) |
| GPU, variability runs | same model, index 0 | same model, index 0 | same model, index 0 |
| noise added to | the 320x256 float tensor fed to the networks, 0-1 scale, right after loading; std 1e-6 | `frame.img`, the 512x400 normalized tensor MASt3R encodes, [-1, 1] scale; std 2e-6 | the 512x400 normalized tensor CUT3R encodes, [-1, 1] scale; std 2e-6 |

All eight GPUs of the machine are the same model. EndoDAC's and
MASt3R-SLAM's primary runs used a different card (index 7) than their
variability runs (index 0).

Noise of frame i with seed K is drawn from
`numpy.random.default_rng([K, i])`, so a frame receives the same noise
every time it is loaded. For MASt3R-SLAM the noise is not added to the
0-1 float image its dataset returns, because `create_frame` ->
`resize_img` converts that image back to 8 bits
(`np.uint8(img * 255)`, `mast3r_slam/mast3r_utils.py:247`) before
resizing: noise added there would be discarded or would flip whole grey
levels. A std of 2e-6 on a [-1, 1] tensor is 1e-6 on the 0-1 scale. The
CUT3R Stage 2 noise test used 1e-6 on the [-1, 1] scale and a different
seeding, so its numbers are not those below.

## 3. Noise sensitivity on `c1_cecum_t1_v1`, three pipelines side by side

`scripts/noise_sensitivity_compare.py`
(`results/noise_sensitivity/noise_sensitivity__c1_cecum_t1_v1.json`).
Each pipeline's primary run against (a) an unperturbed rerun and (b) the
seed-1 noise run. Translation relative to the primary run's trajectory
extent, in the pipeline's own units; Z on the pipeline's own grid
(EndoDAC 1350x1080, the other two 512x400). CUT3R's "unperturbed rerun"
is the Stage 2 standalone recurrent run.

**Unperturbed rerun vs primary run**, per frame, `c1_cecum_t1_v1` (218 frames)

| quantity | statistic | EndoDAC | MASt3R-SLAM | CUT3R |
|---|---|---|---|---|
| translation difference / extent | min | 0 | 0 | 0 |
| translation difference / extent | median | 0 | 0.0018 | 0 |
| translation difference / extent | p95 | 0 | 0.0065 | 0 |
| translation difference / extent | max | 0 | 0.0086 | 0 |
| rotation difference (deg) | min | 0 | 0 | 0 |
| rotation difference (deg) | median | 0 | 0.14 | 0 |
| rotation difference (deg) | p95 | 0 | 0.22 | 0 |
| rotation difference (deg) | max | 0 | 0.27 | 0 |
| Z, max relative difference | min | 0 | 0 | 0 |
| Z, max relative difference | median | 0 | 0 | 0 |
| Z, max relative difference | p95 | 0 | 1.3e-06 | 0 |
| Z, max relative difference | max | 0 | 0.00022 | 0 |
| Z, median relative difference | min | 0 | 0 | 0 |
| Z, median relative difference | median | 0 | 0 | 0 |
| Z, median relative difference | p95 | 0 | 8.2e-07 | 0 |
| Z, median relative difference | max | 0 | 8.8e-05 | 0 |
| bit-identical | | yes | no | yes |

**Noise std 1e-6 (seed 1) vs primary run**, per frame, `c1_cecum_t1_v1` (218 frames)

| quantity | statistic | EndoDAC | MASt3R-SLAM | CUT3R |
|---|---|---|---|---|
| translation difference / extent | min | 0 | 0 | 1.1e-06 |
| translation difference / extent | median | 6.1e-06 | 0.00045 | 0.065 |
| translation difference / extent | p95 | 1.8e-05 | 0.00066 | 0.19 |
| translation difference / extent | max | 1.9e-05 | 0.00097 | 0.23 |
| rotation difference (deg) | min | 0 | 0 | 0.00018 |
| rotation difference (deg) | median | 0.00023 | 0.0088 | 4.1 |
| rotation difference (deg) | p95 | 0.0004 | 0.016 | 7.4 |
| rotation difference (deg) | max | 0.00049 | 0.021 | 8.9 |
| Z, max relative difference | min | 7.9e-05 | 0.0005 | 0.00038 |
| Z, max relative difference | median | 0.00014 | 0.0012 | 0.067 |
| Z, max relative difference | p95 | 0.0002 | 0.0031 | 0.19 |
| Z, max relative difference | max | 0.00036 | 0.0071 | 0.2 |
| Z, median relative difference | min | 8.4e-06 | 2.6e-05 | 3e-05 |
| Z, median relative difference | median | 1.2e-05 | 0.00011 | 0.028 |
| Z, median relative difference | p95 | 1.7e-05 | 0.00038 | 0.13 |
| Z, median relative difference | max | 2.4e-05 | 0.0014 | 0.14 |
| bit-identical | | no | no | no |

**Sequence level**, `c1_cecum_t1_v1` (not evaluation metrics)

| pipeline | run | ATE (mm) | endpoint drift | `s_pose` | depth scale median | relative IQR |
|---|---|---|---|---|---|---|
| EndoDAC | primary | 9.230 | 2.78% | 557.87 | 177.013 | 0.134 |
| EndoDAC | rerun_unperturbed | 9.230 | 2.78% | 557.87 | 177.013 | 0.134 |
| EndoDAC | noise_1e-6_seed1 | 9.230 | 2.78% | 557.87 | 177.013 | 0.134 |
| MASt3R-SLAM | primary | 13.398 | 5.41% | 50.01 | 38.239 | 0.240 |
| MASt3R-SLAM | rerun_unperturbed | 13.290 | 5.33% | 50.10 | 38.239 | 0.240 |
| MASt3R-SLAM | noise_1e-6_seed1 | 13.396 | 5.41% | 50.02 | 38.233 | 0.240 |
| CUT3R | primary | 19.666 | 2.49% | 25.39 | 30.885 | 0.388 |
| CUT3R | rerun_unperturbed | 19.666 | 2.49% | 25.39 | 30.885 | 0.388 |
| CUT3R | noise_1e-6_seed1 | 19.846 | 1.43% | 27.43 | 31.542 | 0.348 |

Reading the tables, without interpretation:

- EndoDAC reruns bit-identically. With noise, its largest per-frame
  changes are 1.9e-05 of the extent, 0.0005 deg and 3.6e-04 relative Z;
  its sequence-level descriptives are unchanged to the printed precision.
- MASt3R-SLAM does not rerun bit-identically (two-process system; the
  same 20 keyframes are selected). Rerun vs primary: up to 0.27 deg and
  0.0086 of the extent. With noise: up to 0.021 deg, 0.00097 of the
  extent and 0.0071 relative Z. The noise run is closer to the primary
  than the unperturbed rerun is; the rerun and the noise run were made on
  a different card from the primary and at different times. ATE moves by
  0.11 mm between the primary and the rerun.
- CUT3R reruns bit-identically. With noise: up to 8.9 deg (median 4.1),
  0.23 of the extent, 0.20 relative Z (median of the per-frame median
  relative Z difference 0.028). Sequence level: ATE 19.67 -> 19.85 mm,
  endpoint drift 2.49% -> 1.43%, depth scale 30.89 -> 31.54, relative IQR
  0.388 -> 0.348.

## 4. Variability-run inference

Subset, computed from the archives
(`scripts/cut3r_run_corpus.py::variability_subset`; 15 molds, 3,811
frames):

`c1_ascending_t1_v1`, `c1_cecum_t1_v1`, `c1_descending_t1_v1`,
`c1_rectum_t1_v1`, `c1_sigmoid1_t1_v1`, `c1_sigmoid2_t1_v1`,
`c1_transverse1_t1_v1`, `c1_transverse2_t1_v1`, `c2_ascending_t1_v1`,
`c2_cecum_t1_v1`, `c2_descending_t3_v1`, `c2_rectum_t1_v1`,
`c2_sigmoid_t1_v1`, `c2_transverse1_t1_v1`, `c2_transverse2_t1_v1`.

`c2_descending` has no `t1` or `t2` sequence; its first is `t3_v1`.

5 runs per pipeline, seeds 1 to 5, noise std 1e-6 (0-1 scale), everything
else as in the primary run. Outputs in
`results/pipelines/variability/<pipeline>/seed<k>/<sequence>/`, separate
from the primary runs. **225 of 225 sequence runs finished with status
ok; no failure.**

| pipeline | per run: sequences ok | per run: runtime (sum over sequences) | per run: storage | 5 runs: wall time | 5 runs: storage |
|---|---|---|---|---|---|
| CUT3R | 15 / 15 | 11.6 to 14.5 min | 2.52 GB | 1 h 04 min | 12.6 GB |
| EndoDAC | 15 / 15 | 15.0 to 15.9 min | 22.23 GB | 1 h 18 min | 111.2 GB |
| MASt3R-SLAM | 15 / 15 | 24.6 to 25.9 min | 5.12 GB | 2 h 07 min | 25.6 GB |

Total: 141 GB including the two unperturbed reruns of `c1_cecum_t1_v1`;
4 h 29 min of wall time for the 15 runs on GPU 0, one job at a time.

MASt3R-SLAM, frames with a pose in each variability run: identical to the
primary run's count on all 15 sequences and all 5 seeds (including
`c2_cecum_t1_v1`, 173 of 304 in every run).

Saved per run: EndoDAC full-resolution depth and poses as in its primary
run; MASt3R-SLAM depth (with its confidence, as its entry script saves
it), per-frame poses and keyframes; CUT3R Z and camera-to-world pose.

---

# INTERPRETATION

1. **The sensitivity is specific to CUT3R among the three, on this
   sequence.** Under the same perturbation its per-frame outputs move
   about three orders of magnitude more than MASt3R-SLAM's and four to
   five more than EndoDAC's. EndoDAC has no state carried across frames;
   MASt3R-SLAM has tracking state but anchors every frame to a keyframe
   and optimizes poses against matches. One sequence, one seed: the
   variability runs on the 15-sequence subset contain what is needed to
   check this on the other 14 sequences and 4 seeds (per-frame
   comparison only, no metric); not done here.

2. **MASt3R-SLAM's own rerun variability is larger than its response to
   the noise.** Its unperturbed rerun differs from the primary more than
   the seed-1 noise run does. Candidate causes, not separated: timing of
   the tracking and backend processes, and the different physical card.
   Would be separated by two unperturbed reruns on the same card
   (differences between them = process timing alone).

3. **For the evaluation this means the protocol's noise runs measure
   different things per pipeline**: for CUT3R mostly the model's
   amplification, for MASt3R-SLAM a mixture with its rerun
   nondeterminism, for EndoDAC almost nothing. That is a reading of
   section 3, and the protocol reports ranges per pipeline, which
   accommodates it.

---

# Open issues

1. The pinned-configuration record of EndoDAC and MASt3R-SLAM is in
   section 2 here and referenced from their pipeline documents; their
   primary runs were made before the protocol entry existed, so the
   record describes what was run, it was not chosen under the protocol.
2. EndoDAC's and MASt3R-SLAM's primary runs used card index 7, the
   variability runs card index 0 (same model). EndoDAC reran
   bit-identically on index 0; for MASt3R-SLAM a card effect cannot be
   separated from process timing (interpretation 2).
3. MASt3R-SLAM's variability manifests do not carry the GPU model string
   (the corpus driver's manifest replaces the entry script's); the model
   is taken from `nvidia-smi` in `logs/pinvar_chain.log`.
4. Section 3 covers one sequence and seed 1.
