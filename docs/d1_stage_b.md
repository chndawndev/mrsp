# D1 Stage B: full-corpus evaluation and D1.1-D1.5

Evaluates all 169 registered sequences, 4 configurations x 4 taus,
aggregates per the pre-registered rule, and reports D1.1-D1.5 as MEASURED
numbers. **No interpretation of what the D1 outcome means for the
project** -- that's a separate decision, not made here.

Code: `scripts/eval_d1_sequence.py` (per-sequence, generalizes
`scripts/eval_stage4.py`), `scripts/eval_d1_run_corpus.py` (GPU-sharded
launcher), `scripts/eval_d1_mesh_identity.py` (pre-flight 4),
`scripts/eval_d1_aggregate.py` (corpus tables, bootstrap, D1.1-D1.5).

---

## MEASURED

### Pre-flight 1: doc citations

Confirmed present, quoted at the cited lines:
- `docs/success_criteria.md` §6 line 256: "2026-09-23: Pre-run note:
  anticipated risk to D1.3 (no change to criteria)" -- predicts D1.3 "may
  fail on the full corpus because predicted-depth over-flagging inflates
  fully_predicted recall." (Confirmed below: it did.)
- `docs/success_criteria.md` §6 line 281: "2026-09-23: Deviation:
  localization error operationalized as Euclidean" -- "Trigger fixed in
  advance: if more than 20% of matched GT-predicted pairs on the full
  corpus have a segment that intersects the mesh, geodesic localization
  error will be computed..."
- `docs/success_criteria.md` §6 line 302: "2026-09-23: Clarification:
  operational definition of D1.1 completion" -- finite depth/pose every
  frame + Sim(3) alignment succeeds; "for [frame-to-frame pipelines] D1.1
  measures crash-free completion only."
- `docs/eval_protocol.md` line 338: "2026-09-23: Aggregation rule for
  corpus-level metrics" -- pooled point estimate (region recall:
  detected/total per class; area metrics: summed numerator/summed
  denominator), mesh-level cluster bootstrap (10,000 replicates, 95%
  percentile interval), two sensitivity clusterings `(Colon, Segment)`
  and `(Colon, Segment, Phantom Number)`, reported not decision-driving.
  "All D1 to D2 pass/fail decisions use the point estimate."
- `docs/eval_protocol.md` line 389: Euclidean-not-geodesic decision, same
  trigger restated.

### Pre-flight 2: reproduction gate

`scripts/eval_d1_sequence.py --sequence c1_cecum_t1_v1`, exact (NaN-aware)
diff against `results/eval_stage4/summary.json`: **980/980 reference
keys, 0 missing, 0 mismatches, bit-identical.** (Run twice: once before
and once after a mid-implementation schema fix -- see "Corrections"
below -- both passed identically.)

### Pre-flight 3: timing

Measured on the shortest (`c1_transverse1_t1_v3`, 117 frames, 91.8s),
median (`c2_rectum_t2_v3`, 383 frames, 275.4s), and longest
(`c1_cecum_t4_v3`, 959 frames, 694.3s) sequences: linear fit **0.715
s/frame + 8.3s fixed overhead per sequence**, projecting ~13.9 hours for
the full 67,886-frame corpus on 1 GPU. **Actual full-run wall time:
14h00m** (2026-09-25T12:15:58Z to 2026-09-26T02:16:26Z, single GPU, index
7 pinned by explicit choice since it was the only GPU reliably idle at
launch) -- within 1% of the pre-flight projection.

### Pre-flight 4: mesh geometry identity (descriptive, changes no analysis)

15 distinct `(Colon, Segment)` groups, 58 distinct `(Colon, Segment,
Phantom Number)` groups, among the 169 sequences (103 distinct mesh
hashes total). 360 pairwise mesh-hash comparisons (308 at the coarser
level, 52 at the finer level), each rigidly aligned (PCA-plus-centroid
initial alignment, refined with `trimesh.registration.icp` on a 20,000-
point subsample) and scored by symmetric point-to-surface distance
(20,000-point Monte Carlo subsample per direction, pooled):

| level | n pairs | mean_mm (median of per-pair means) | p95_mm (median of per-pair p95s) | mean_mm range | p95_mm range |
|---|---|---|---|---|---|
| (Colon, Segment) | 308 | 0.0199 | 0.0494 | 0.0046 - 4.565 | 0.0106 - 14.41 |
| (Colon, Segment, Phantom Number) | 52 | 0.0156 | 0.0374 | 0.0054 - 2.144 | 0.0130 - 6.045 |

Most pairs are near-identical (median mean distance ~0.02mm, i.e.
registration/reconstruction noise, not a different physical shape); a
minority of pairs at both levels have mean distances in the 1-4.6mm
range (visible in the max column). Full per-pair data:
`results/d1/mesh_identity_check.csv`. **Not concluding which grouping
level is "independent"** -- both are reported as sensitivity clusterings
below, per the aggregation rule.

### Corrections made during implementation (before the main run)

Two issues found and fixed while building the corpus driver, both before
they cost meaningful compute:
1. **Schema gap**: `metrics.json` initially stored only the final ratio
   for each area-based metric (false reassurance, false alarm, area
   fraction, calibration ratio). The aggregation rule requires pooling
   via summed numerator/summed denominator across sequences, which the
   ratio alone can't reconstruct. Added the raw per-sequence numerator/
   denominator areas (mm²) to `metrics.json` before any sequence beyond
   the reproduction-gate pilot had been (re)computed; verified the raw
   areas reconstruct every reported ratio exactly (0.0 diff) and that
   the reproduction gate still passed bit-identically after the change.
2. **Race condition, my own error**: deleted an in-progress sequence's
   output directory to force a schema-fix rerun while the corpus shard
   process was still writing to it, crashing the whole shard (not just
   that sequence) after ~1 sequence's worth of GPU time. Added defensive
   directory recreation and a shard-loop-level exception guard so one
   sequence's failure (of any kind) can no longer take down the rest of
   the shard; relaunched cleanly, no further incidents.

### Main run

169/169 sequences, single GPU (index 7), **0 failures, exit code 0**.
Per-sequence outputs: `results/d1/per_sequence/<seq>/` (packed
`predicted_observed_bits`, `regions.csv`, `metrics.json`,
`MANIFEST.json`).

### D1.1-D1.5

| # | value | threshold | result |
|---|---|---|---|
| D1.1 | 169/169 (100%) | >= 70% of 169 | **pass** |
| D1.2 | 1.0 (oracle, medium+large recall@50%, tau=0.25) | >= 0.80 | **pass** |
| D1.3 | 1.67 percentage points (oracle 1.0 - fully_predicted 0.9833) | >= 10 pp | **FAIL** |
| D1.4 | 3.30 pp (large 0.9910 - small 0.9580, fully_predicted, tau=0.25) | >= 15 pp (record-only) | record-only: does not meet the gap |
| D1.5 | 0.0307 (false reassurance, fully_predicted, tau=0.25) | strictly between 0.02 and 0.98 | **pass** |

D1.1's caveat (from the §6 entry): for a frame-to-frame pipeline like
EndoDAC this measures crash-free completion only, not track-loss
detection.

**Per your instruction: D1.3 failed. Reporting it and stopping here --
no further analysis run to explain or rescue it.** This matches the
direction `docs/success_criteria.md` §6's 2026-09-23 pre-run note
predicted before the corpus run: "predicted-depth over-flagging inflates
fully_predicted recall," which the area-fraction numbers below are
directly consistent with (fully_predicted's area fraction, 0.517, is
about 2x oracle's, 0.263).

### Segment-intersect trigger (Euclidean-vs-geodesic)

Corpus-wide `segment_intersects_mesh` fraction, pooled across every
matched GT-predicted pair with a non-null flag, every configuration,
every tau together: **4.13%**. (The per-config-per-tau breakdown in the
corpus table below is much lower for oracle alone, 0.21% at tau=0.25 --
oracle's near-perfect predictions rarely produce a matched pair whose
straight-line centroid segment tunnels through the mesh; the pooled
figure that decides the trigger includes the noisier predicted
configurations too.) **4.13% is below the 20% trigger -- Euclidean
stands, no geodesic computation performed.**

### Corpus tables (point estimate, mesh-level 95% bootstrap CI, n regions/meshes)

Primary tau = 0.25 shown; `results/d1/corpus_tables.json` has all 4 taus
x 4 configs x every metric, plus both sensitivity-clustering CIs.

| config | recall (medium+large, 50%) | area_fraction | calibration_ratio (raw / ignore-excluded) | false_reassurance | false_alarm | loc_err median (mm) | segment_intersect |
|---|---|---|---|---|---|---|---|
| oracle | 1.0000 [1.000, 1.000] (n=719, 103 meshes) | 0.2628 [0.243, 0.283] | 1.154 / 0.999 | 0.00150 [0.00125, 0.00176] | 0.0006 | 0.0079 (n=957) | 0.0021 |
| pred_depth_only | 1.0000 | 0.4888 | 2.147 / 1.992 | 0.00043 | 0.4982 | 16.487 | 0.0387 |
| pred_pose_only | 0.9652 | 0.3687 | 1.619 / 1.487 | 0.05517 | 0.3648 | 8.147 (n=951) | 0.0726 |
| fully_predicted | 0.9833 | 0.5167 | 2.269 / 2.120 | 0.03074 | 0.5428 | 17.667 (n=956) | 0.0502 |

`recall` region counts: 719 medium+large regions is the pooled
denominator when the oracle configuration detects all of them (every
region evaluable at 719); the 957-region headline total (matching
`CLAUDE.md`'s frozen "957 regions with equivalent diameter > 5mm")
appears in the localization-error row counts, and drops slightly for
non-oracle configs (951, 956) where a small number of headline regions
go undetected (no predicted component overlaps them, so they contribute
no localization error).

### Diagnostics a-c

**(a) calibration ratio, ignore-excluded** -- reported in the table above
next to the raw value; consistently lower than raw (e.g. fully_predicted
2.269 raw vs 2.120 ignore-excluded at tau=0.25) since the ignore set
(structurally-unreachable-but-GT-observed faces) inflates the raw
denominator's complement.

**(b) evaluable-pixel diagnostics**, by configuration (mean / median /
max across the 169 sequences):

| config | ray_miss_frac | d_pred_unavailable_frac_of_evaluable |
|---|---|---|
| oracle | 0.0164 / 0.0106 / 0.0943 | 0.000016 / 0.000007 / 0.00103 |
| pred_depth_only | 0.0164 / 0.0106 / 0.0943 | 0 / 0 / 0 |
| pred_pose_only | 0.0229 / 0.0126 / **0.5430** | 0.0169 / 0.0142 / 0.1083 |
| fully_predicted | 0.0229 / 0.0126 / **0.5430** | 0 / 0 / 0 |

**(c) fully_predicted-only-unobserved faces vs pred_depth_only overlap**,
pooled across the corpus, by tau:

| tau | faces only-unobserved under fully_predicted, not pred_pose_only | also unobserved under pred_depth_only | fraction |
|---|---|---|---|
| 0.15 | 14,427,880 | 12,202,534 | 84.6% |
| 0.25 | 12,927,697 | 10,611,164 | 82.1% |
| 0.35 | 10,622,128 | 8,441,848 | 79.5% |
| 0.50 | 6,184,845 | 4,509,952 | 72.9% |

---

## INTERPRETATION

- **The pred_pose_only/fully_predicted ray_miss_frac max of 54.3%** (vs a
  median around 1.3%) suggests at least one sequence has severely
  misaligned predicted poses that push many camera rays off the mesh
  entirely. Not traced to a specific sequence here (would need sorting
  `results/d1/diagnostic_b.csv` by `ray_miss_frac` and inspecting that
  sequence's trajectory, e.g. against `results/d1/stage_a_summary.csv`'s
  `endpoint_error_frac_of_gt_path`) -- flagged as a candidate follow-up,
  not investigated further per the "no extra analysis this session"
  instruction attached to the D1.3 stop.
- **The 72-85% overlap between "fully_predicted-only" and
  "pred_depth_only"-flagged faces (diagnostic c), decreasing as tau
  loosens**, is consistent with the same mechanism the Stage 1 viewer
  export's single-sequence check found (72/73 = 98.6% on one region) --
  predicted depth's own over-flagging behavior accounts for most, but a
  shrinking majority as tau loosens, of the extra area fully_predicted
  marks unobserved beyond pred_pose_only. This would be confirmed further
  by checking whether the residual 15-27% (not explained by
  pred_depth_only) correlates with specific covariates (e.g. Debris) in
  `results/d1/per_sequence_metrics.csv`; not done here.
- **D1.3's failure direction matches the pre-registered prediction
  exactly** (oracle recall saturates at 1.0, fully_predicted's recall
  stays high at 0.983 rather than dropping the way a naive "gap" read
  might expect) -- the area-fraction numbers (fully_predicted ~2x
  oracle's) are consistent with over-flagging inflating recall rather
  than harming it, exactly as the pre-run note anticipated. This is a
  restatement of what was already predicted before the run, not a new
  finding; confirming the underlying mechanism further (e.g. that
  specific regions are only detected because of over-flagging, not
  genuine coverage) is explicitly out of scope for this stage.
