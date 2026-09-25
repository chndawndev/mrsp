# D1 Stage A: EndoDAC inference on all 169 registered sequences

Produces and sanity-checks EndoDAC predicted depth + chained
camera-to-world poses for all 169 registered C3VDv2 sequences. **No
evaluation, no D1 pass/fail call** -- that is Stage B's job
(`docs/success_criteria.md` §2). Everything here is descriptive.

Code: `scripts/endodac_inference.py` (per-sequence procedure, generalized
from the validated single-sequence run, `docs/pipelines/endodac.md` §8),
`scripts/endodac_full_corpus_run.py` (GPU-sharded launcher),
`scripts/d1_stage_a_summary.py` (this doc's numbers). Logs:
`logs/endodac_full_run.log`, `logs/endodac_full_run_shard0.log`,
`logs/d1_stage_a_summary.log`.

---

## MEASURED

### Pre-flight

1. **Disk**: 2.0TB free on `/data1_ycao/chua` before starting (86% used
   of 14T). Extrapolated need from the existing validated
   `c1_cecum_t1_v1` run (1.2GB / 218 frames) to 67,886 frames: ~374GB --
   well under the 500GB stop threshold. **Actual usage after the full
   run: 369GB** (`results/pipelines/endodac_full_run/`), matching the
   extrapolation closely. Free disk after the run: 1.7TB.

2. **`.gitignore`**, applied before any code was committed:
   ```diff
   -/scratch/
   +/scratch/*
   +!/scratch/pipelines/
   +/scratch/pipelines/*
   +!/scratch/pipelines/*.py
   +!/scratch/pipelines/*.md
   +!/scratch/pipelines/*.json
   +!/scratch/pipelines/*.csv
    /results/
    *.ply
   ```
   This re-included 21 pre-existing top-level `scratch/pipelines/*.py`
   scripts (all previously-uncommitted diagnostic/driver code already
   referenced by `EXPERIMENTS.md`) plus the two new committed scripts
   under `scripts/`. `scratch/pipelines/EndoDAC/` (the nested third-party
   clone -- its own `.git`, 760MB checkpoints + 372MB pretrained
   weights) stays fully ignored: the re-include patterns only match
   `*.py`/`*.md`/`*.json`/`*.csv` directly inside `scratch/pipelines/`,
   not recursively.

   **Staged-file size check before commit** (`git diff --cached
   --name-only | xargs du -b`, aborts on anything > 1MB): largest staged
   file was `scripts/endodac_inference.py` at 16.7KB. Nothing aborted.

3. **Reproduction gate**: backed up `results/eval_stage4/summary.json`,
   reran `c1_cecum_t1_v1` through the new version-controlled code
   (`scripts/endodac_inference.py --sequence c1_cecum_t1_v1`, overwriting
   `results/pipelines/endodac_full_run/c1_cecum_t1_v1/` in place), reran
   `scripts/eval_stage4.py` UNCHANGED, and did an exact (not
   tolerance-based) recursive, NaN-aware diff of every leaf value between
   the new and backed-up `summary.json`. **Result: 980/980 leaf values
   compared, 0 mismatches -- bit-identical.** The determinism safeguards
   added to the new script (`torch.manual_seed(0)`,
   `cudnn.deterministic=True`, `cudnn.benchmark=False`) were sufficient
   on this hardware; this was not guaranteed in advance and the run would
   have stopped here (before the full corpus) had it failed.

### Full run

169/169 sequences processed: 168 run fresh, 1
(`c1_cecum_t1_v1`) skipped as already-complete-and-verified by the
resumability check (its manifest, from the reproduction gate rerun,
already had `status: "ok"` and matching frame/shape counts). Single GPU
(index 7, the only one with >=8GB free at launch time -- GPUs 0-3/5/6
were at 1.5GB free / 87-100% util under other users' jobs, GPU 4 had
5.2GB free, below the threshold; none were preempted). Total wall time:
2026-09-24T21:39:29Z to 2026-09-25T02:20:20Z, **4h41m**. **0 failures,
0 non-finite depth or pose values across all 169 sequences.** Every
sequence's manifest records `git_commit
6c149f38fca4a4038b954b79dbaabcc4d3c7dbca` (the code state that produced
it) and `gpu_index 7`.

### Storage format

`results/pipelines/endodac_full_run/<sequence>/`: `depth/{i:04d}_pred_depth.npy`
(float32, native resolution, network-native scale-ambiguous units),
`poses_pred.npy` / `poses_pred.txt` (camera-to-world, frame 0 =
identity), `intrinsics_pred_per_pair.npy` + `_summary.json` (feed
resolution, per-consecutive-pair), `MANIFEST.json` (frame count, dtypes,
shapes, non-finite counts, runtime, GPU index, git commit, status/error)
-- same layout as the original single-sequence run
(`docs/pipelines/endodac.md` §8), MANIFEST.json schema extended per this
stage's needs.

### D1.1 completion

**169/169 (100%)**, per the operational definition in
`docs/success_criteria.md` §6, "2026-09-23: Clarification: operational
definition of D1.1 completion": *"every frame of the sequence has finite
predicted depth and a finite predicted pose, and the Sim(3) trajectory
alignment succeeds."* That same entry's own caveat applies and is
repeated here: **EndoDAC is a frame-to-frame pose pipeline and cannot
lose track by construction; for it, this count measures crash-free
completion only**, not the "no permanent track loss" half of §2's
original qualitative D1.1 text. 0 sequences had a non-`ok` manifest
status, a non-finite depth/pose value, or a failed Sim(3) alignment.

### Trajectory-quality distributions (169 sequences, no threshold attached)

| quantity | min | 25% | median | 75% | max |
|---|---|---|---|---|---|
| ATE after Sim(3) alignment (mm) | 0.868 | 3.151 | 5.157 | 8.222 | 17.264 |
| endpoint drift / GT path length | 0.58% | 3.17% | 4.97% | 8.51% | 18.52% |
| `s_pose` (Umeyama scale) | 51.72 | 266.41 | 327.25 | 409.17 | 984.29 |
| depth scale median | 72.23 | 107.06 | 118.80 | 128.13 | 205.79 |
| depth scale relative IQR | 0.035 | 0.108 | 0.170 | 0.258 | 0.894 |

`c1_cecum_t1_v1`'s own row reproduces the already-published reference
values exactly: `s_pose=557.8711809619352` (saved: 557.87117574),
`endpoint_error_frac_of_gt_path=0.027793` (2.78%, the exact figure
already cited in the D1.1 clarification entry), `depth_scale_median=177.01337448048218`
(saved: 177.01336670) -- consistent with, not identical to, the
originally-saved `results/pipelines/endodac_scale_analysis.json` values
(those came from a slightly earlier run before the reproduction gate
rerun; diffs are ~1e-5/1e-6, the same order already documented for
scale-recovery reruns elsewhere in this project).

### Covariates

Joined `results/mesh_identity.csv` (169 rows: `mesh_hash`, `Colon`,
`Segment`, `Phantom Number`, `Video Number`, `Debris`) and
`docs/release_v1.csv` (the full summary-sheet extract, 192 rows) onto the
169 by `Video Name`/`sequence`, exact string match, **0 unmatched** (both
joins are `validate="one_to_one"`; consistent with
`docs/data_inventory.md`'s already-established 192/192 match).

**Columns found in `C3VDv2_Data_Summary_Sheet_v1.xlsx`** (sheet
`release_v1`, read directly from the dataset root, read-only): `Colon,
Segment, Phantom Number, Video Number, Video Name, Camera Speed, Edge
Enhancement, Brightness, Debris, Deformation, Open End Visible, Tags,
Comments, Qualitative Score, Quantitative Score, Youtube Preview URL ID,
Total Frames`. **17 columns, all read; no phantom production date or
batch field exists anywhere in the sheet** -- not inferred from Colon,
Segment, Phantom Number, or anything else.

- `mesh_hash`: **103 distinct** values across the 169 sequences (matches
  the already-established figure, `CLAUDE.md`'s "103 distinct meshes and
  113 trajectories").
- `physical_segment_id` (`Colon + "_" + Segment`, the task's own
  definition): **15 distinct** values (8 `c1_*`, 7 `c2_*`).
- `Debris`: 114 `no` / 55 `yes`. `Deformation`: 169 `no` (all registered
  sequences; deformation is a separate video category,
  `deformation_videos/`, not part of this run). `Open End Visible`: 124
  `yes` / 44 `no` / **1 `"no "`** (trailing space, a raw data-entry
  quirk in the sheet itself -- reported as found, not silently
  normalized).

---

## INTERPRETATION

- **Depth-scale relative IQR is wide (median 17.0%, max 89.4%)** --
  consistent with a single global per-sequence scale factor being a
  coarse fit for sequences with more scene-depth or lighting variation
  across frames, since the depth network's own scale drifts frame to
  frame relative to GT. Not confirmed further here (would need a
  per-frame breakdown correlated against per-sequence covariates like
  Debris or Camera Speed); Stage B's job is metric evaluation, not this
  diagnosis.
- **`s_pose` and depth-scale-median both vary roughly 3-4x across the
  corpus** (s_pose 51.7-984.3, depth scale 72.2-205.8) -- plausible given
  sequences differ in physical scale (segment length, camera speed) and
  the network's monocular self-supervised training never saw an absolute
  scale reference. Whether `s_pose` and depth-scale-median move together
  within a sequence (as the single-sequence `endodac_scale_analysis.py`
  cross-check found for `c1_cecum_t1_v1`, within ~20%) or diverge for
  some sequences is not checked here -- would need a per-sequence
  ratio-and-threshold pass, left to Stage B or a follow-up if it matters
  to a specific method comparison.
