# CLAUDE.md

Project: missed-region localization benchmark on C3VDv2.
Goal: evaluate how accurately depth + pose pipelines localize unobserved colon
surface, against the released GT coverage mesh.

---

## Hard rules

**Never edit `docs/success_criteria.md`.** It is pre-registration. Do not
reword, clarify, or extend it, and **do not append a deviation entry either**.
Deviation entries are written by the author only. If a criterion looks wrong,
unmeasurable, or infeasible, draft the proposed text in your reply, say what
would have to change, and stop. Do not touch the file.

**Never change metric or GT code to make a result work.** `src/eval/` and
`src/gt/` hold the metric definitions, the detection threshold, the registration
procedure, and the GT loaders. Do not modify them without explicit approval in
the conversation, even to fix what looks like a bug. Report the suspected bug,
with evidence, and wait. "The metric was wrong so I fixed it and reran" is a
protocol violation.

**Patient data is off limits.** The lab's real colonoscopy videos are not
cleared for use yet. Do not open, read, decode, copy, or process any real
patient video, frame, or metadata. Do not write code that touches those paths
until this rule is lifted here in writing.

**Dataset root is read-only.** `/data1_ycao/chua/datasets/C3VDv2` is never
written to, renamed, or deleted from. Discover sequences from `*.zip` archives
only; never treat an extracted directory as a data source, since those are
temporary user extractions. Extract only into `scratch/`, and only what is
needed. Every extraction must be reproducible from the archives.

---

## Frozen conventions

These were verified empirically (see `docs/conventions.md`,
`docs/oracle_check.md`, `docs/gpu_validation.md`). Do not re-derive or
second-guess them silently.

- **Depth**: uint16 TIFF, linear over 0 to 100 mm, camera-frame **Z-depth**, not
  radial distance. `depth_mm = raw / 65535 * 100`. Mask `raw == 0` (no hit) and
  `raw == 65535` (clamped).
- **Pose**: `pose.txt` lines are 16 comma-separated floats.
  `np.array(vals).reshape(4,4).T` is the camera-to-world matrix. A plain
  `reshape(4,4)` is wrong and puts the point cloud ~400 mm off.
- **Camera**: Scaramuzza omnidirectional model from `camera_intrinsics.txt`.
  Never substitute a pinhole model in GT-side computation. The stretch matrix is
  `[[c, e], [d, 1]]` (the CUDA renderer's column-major form), NOT the
  `[[c, d], [e, 1]]` used by the dataset's own Python example. There is no `a1`
  term. Back-projection scales the un-normalized model ray by Z; it is not
  `unit_ray * depth`.
- **Coverage GT**: `coverage_mesh.obj`, `vt` index 1 = observed, 2 = unobserved.
  The GT criterion is pure geometric visibility: any primary ray hit, any pixel,
  any frame, no distance or incidence-angle threshold. Hits beyond 100 mm still
  count; MAX_DEPTH only clamps the depth image encoding.
- **Dataset composition**: 169 registered + 15 deformation + 8 screening = 192.
  Assert 169 whenever enumerating registered sequences.
- **v2/v3** are a valid paired comparison ONLY for the 48 of 58 combos that
  share an identical mesh (verified by vertex-array hash in
  `results/mesh_identity.csv`). The other 10 combos have different meshes and
  must be excluded from any debris ablation. Poses are near-identical but not
  byte-identical: the robot repeated the trajectory, it did not replay a stored
  one. **v1 vs v2** differ in both trajectory and imaging settings and are NOT a
  controlled comparison.
- **Analysis unit**: the primary unit is the unobserved region, using the 957
  regions with equivalent diameter > 5 mm. Regions below 5 mm are appendix only.

---

## Frozen project decisions

Unlike the conventions above, these are choices, not dataset facts. They are
frozen the same way: recorded once, not silently revisited. Rationale and
measurements are in the cited documents.

- **Pipeline input format**: methods receive the ORIGINAL fisheye frames. No
  undistortion. Undistorting to the widest practical pinhole loses 5 to 11
  percentage points of observed surface and creates 1 to 4 spurious >5 mm
  coverage gaps per sequence. See `docs/pinhole_tradeoff.md` and
  `docs/conventions.md` section 6. Undistorted input exists only as a controlled
  ablation on 1 to 2 pipelines. Never switch a method to undistorted input to
  improve its numbers.
- **Intrinsics for methods that require them**: the same pinhole approximation
  for every method, f = 541.29, principal point at image center. No per-method
  tuning.
- **GT visibility evaluation always uses the omnidirectional model**, regardless
  of what a method consumed.
- **Mixed configurations use post-hoc substitution**: no surveyed pipeline
  supports conditioning on GT pose or GT depth, so the mixed conditions are
  produced at the fusion stage, with the method itself run unmodified. Use the
  term "post-hoc substitution" consistently; do not invent alternative names.
  See the deviation entry in `docs/success_criteria.md` section 6.

---

## Reporting

**Separate MEASURED from INTERPRETATION.** Put interpretations in a clearly
marked section. For each one, state what evidence would confirm or refute it.
Never present an unverified mechanism as a finding.

**Documentation beats inference.** When dataset structure is documented (README,
`docs/dataset_official_description.md`, the paper), quote it as the authority
and reconcile disk scans against it. If they disagree, report the discrepancy
explicitly instead of silently trusting the scan.

**Report independence, not just counts.** 169 sequences are backed by 103
distinct meshes and 113 trajectories. Every aggregate states how many
independent meshes and trajectories are behind it. Cluster statistics at the
mesh level.

**Report failures as results.** Sequences where a pipeline crashes, loses
tracking, or produces degenerate output are never dropped from an aggregate.
Report the count, the sequence names, and the failure mode.

**Quote sources with file and line.** When a claim comes from code or a README,
cite the path and line numbers so it can be checked.

**Flag uncertainty as UNKNOWN.** Do not guess an equation, a unit, or a
convention. Mark it UNKNOWN and say what would settle it.

---

## Workflow

- Long jobs run under `tmux`, non-blocking. Redirect both streams to the log
  file, `... 2>&1 | tee logs/<task>.log`; do not rely on the script's own
  logging calls alone, or a crash leaves no traceback. Do not run anything over
  ~10 minutes inside the session.
- Log progress periodically in long runs (elapsed, ETA) so a stall is visible.
- Shared GPU box. Default to CPU when the work is not neural. Before any GPU
  job, run `scripts/gpu_status.py`, log the full `nvidia-smi` output with a
  timestamp, pick the freest device, and set `CUDA_VISIBLE_DEVICES` to that
  single index explicitly. Never use more than one GPU without asking.
- On CUDA OOM or any CUDA error, log the timestamp, the full `nvidia-smi` output
  at failure time, this process's peak allocated memory, and the exception, then
  exit non-zero. Do not silently retry with a smaller batch.
- Commit after each finished step. Small, reviewable diffs.
- Append a short entry to `EXPERIMENTS.md` after each finished step: date, what
  was run, result, open questions.
- Ask before extracting more than one sequence, before downloading model weights,
  and before any run expected to take over an hour.
- Prefer plan mode for anything touching geometry, metrics, or data loading.

## Code style

- Code, comments, docstrings, file names, and documents in English.
- Every coordinate or unit conversion needs a unit test in `tests/conventions/`.
- Any performance rewrite of geometry code needs a test comparing the new
  implementation against the previous one on the same inputs.
- No silent fallbacks. If an input is missing or malformed, fail loudly.
- Deterministic seeds; record them.

## Layout

```
src/geometry/   camera model, pose handling, back-projection, coverage mesh
                parsing (coverage_mesh.py and pose.py live here, not
                src/gt/ -- historical: written before src/gt/ existed, and
                scripts/visibility_full.py plus other already-frozen code
                import from geometry.coverage_mesh/geometry.pose, so they
                were not relocated when src/gt/ was introduced)
src/gt/         depth loading, visibility raster                      (locked)
src/eval/       metrics, region matching, registration                (locked)
scripts/        runnable entry points
tests/          unit tests, tests/conventions/ for units and frames
docs/           conventions, inventory, results write-ups
results/        generated tables, meshes, renders
scratch/        extractions and scratch data, reproducible, safe to delete
logs/           tmux job logs
```