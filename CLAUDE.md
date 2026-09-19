# CLAUDE.md

Project: missed-region localization benchmark on C3VDv2.
Goal: evaluate how accurately depth + pose pipelines localize unobserved colon
surface, against the released GT coverage mesh.

---

## Hard rules

**Never edit `docs/success_criteria.md`.** It is pre-registration. If a
criterion looks wrong or unmeasurable, say so in your report and stop. Do not
reword, clarify, or extend it.

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
`docs/oracle_check.md`). Do not re-derive or second-guess them silently.

- **Depth**: uint16 TIFF, linear over 0 to 100 mm, camera-frame **Z-depth**, not
  radial distance. `depth_mm = raw / 65535 * 100`. Mask `raw == 0` (no hit) and
  `raw == 65535` (clamped).
- **Pose**: `pose.txt` lines are 16 comma-separated floats.
  `np.array(vals).reshape(4,4).T` is the camera-to-world matrix. A plain
  `reshape(4,4)` is wrong and puts the point cloud ~400 mm off.
- **Camera**: omnidirectional (Scaramuzza-style) model from
  `camera_intrinsics.txt`. Never substitute a pinhole model. Back-projection
  scales the un-normalized model ray by Z; it is not `unit_ray * depth`.
- **Coverage GT**: `coverage_mesh.obj`, `vt` index 1 = observed, 2 = unobserved.
  The GT criterion is pure geometric visibility: any primary ray hit, any pixel,
  any frame, no distance or incidence-angle threshold.
- **Dataset composition**: 169 registered + 15 deformation + 8 screening = 192.
  Assert 169 whenever enumerating registered sequences.
- **v2/v3** v2/v3 are a valid paired comparison ONLY for the 48 of 58 combos that share an
  identical mesh (verified by vertex-array hash in results/mesh_identity.csv).
  The other 10 combos have different meshes and must be excluded from any
  debris ablation. Poses are near-identical but not byte-identical: the robot
  repeated the trajectory, it did not replay a stored one. **v1 vs v2** differ in both
  trajectory and imaging settings and are NOT a controlled comparison.

---

## Reporting

**Separate MEASURED from INTERPRETATION.** Put interpretations in a clearly
marked section. For each one, state what evidence would confirm or refute it.
Never present an unverified mechanism as a finding.

**Documentation beats inference.** When dataset structure is documented (README,
`docs/dataset_official_description.md`, the paper), quote it as the authority
and reconcile disk scans against it. If they disagree, report the discrepancy
explicitly instead of silently trusting the scan.

**Report independence, not just counts.** Sequences within a (colon, segment)
share one mesh, and v2/v3 pairs share geometry. Every aggregate states how many
independent meshes and trajectories are behind it. Cluster statistics at the
mesh level.

**Quote sources with file and line.** When a claim comes from code or a README,
cite the path and line numbers so it can be checked.

**Flag uncertainty as UNKNOWN.** Do not guess an equation, a unit, or a
convention. Mark it UNKNOWN and say what would settle it.

---

## Workflow

- Long jobs run under `tmux`, logging to `logs/<task>.log`, non-blocking. Do not
  run anything over ~10 minutes inside the session.
- Shared GPU box: always set `CUDA_VISIBLE_DEVICES` explicitly. Check `nvidia-smi`
  before launching.
- Commit after each finished step. Small, reviewable diffs.
- Append a short entry to `EXPERIMENTS.md` after each finished step: date, what
  was run, result, open questions.
- Ask before extracting more than one sequence, before downloading model weights,
  and before any run expected to take over an hour.
- Prefer plan mode for anything touching geometry, metrics, or data loading.

## Code style

- Code, comments, docstrings, file names, and documents in English.
- Every coordinate or unit conversion needs a unit test in `tests/conventions/`.
- No silent fallbacks. If an input is missing or malformed, fail loudly.
- Deterministic seeds; record them.

## Layout

```
src/geometry/   camera model, pose handling, back-projection
src/gt/         GT loaders, coverage mesh parsing        (locked)
src/eval/       metrics, region matching, registration   (locked)
scripts/        runnable entry points
tests/          unit tests, tests/conventions/ for units and frames
docs/           conventions, inventory, results write-ups
results/        generated tables, meshes, renders
scratch/        extractions and scratch data, reproducible, safe to delete
logs/           tmux job logs
```