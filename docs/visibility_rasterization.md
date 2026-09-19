# Full 169-sequence GPU visibility rasterization

Script: `scripts/visibility_full.py`, run under tmux (session `visibility_full`), logged to `logs/visibility_full.log`. **Total wall time: 19.3 minutes** for all 169 sequences, full resolution (1350x1080, stride 1), one ray per pixel per frame, GPU (NVIDIA Warp).

## Run environment

| | |
|---|---|
| Warp version | 1.17.0 |
| GPU | NVIDIA RTX 6000 Ada Generation, index 7 (re-selected via `scripts/gpu_status.py` at launch: 32.1 GB free, 0% utilized — logged in full in `logs/visibility_full.log`) |
| Commit hash | `44d8a07523c73d64cdaeb7d8ca02b6b69c0927e7` |
| Kernel cache | Machine-level (`~/.cache/warp/1.17.0`). **Will recompile once on a new machine** (~1s observed one-time JIT cost for this kernel signature); every subsequent run on the same machine loads from cache. |

## A bug found and fixed during launch

The first launch attempt crashed (exit 1) after 10 sequences with no captured traceback (a process/logging mistake on my part — the tmux launch command didn't redirect stdout/stderr to the log file that time, only the script's own internal logging calls were captured, and the tmux pane closed before I could inspect it). Reproduced directly and found: `stream_poses_from_zip()` (written fresh for this run) didn't have the archive-path-resolution fix that `stream_coverage_mesh_from_zip()` already had for `c1_cecum_t1_v3.zip`'s one-off subfolder-wrapped packaging (`docs/coverage_stats.md`'s earlier finding). Fixed by having `pose.py` reuse `coverage_mesh.py`'s `_resolve_zip_member()` helper instead of assuming a fixed root-level path. Verified against 17 sequences spanning the fix point (including the previously-failing one) before relaunching the full run, this time with correct log capture.

## Outputs

- `results/visibility_full/metrics.csv` — 169 rows: `n_faces, n_frames, iou, false_observed, false_unobserved, n_gt_observed, n_pred_observed`, plus per-stage timing (`t_stream_s, t_bvh_s, t_cast_s`) and joined index columns (Segment, Colon, Debris, etc.).
- `results/visibility_full/observed_masks.npz` — one packed-bits (`np.packbits`, uint8) boolean array per sequence, keyed by Video Name (169 keys, 4.9 MB total).
- `results/visibility_full/run_environment.json` — the environment table above, machine-readable.

## IoU distribution (vs. released `coverage_mesh.obj`)

| min | 25% | median | 75% | max | mean |
|---|---|---|---|---|---|
| 0.9731 | 0.9976 | 0.9988 | 0.9992 | 0.9996 | 0.9976 |

Consistent with the single-sequence validation in `docs/gpu_validation.md` (0.99883 for `c1_cecum_t1_v1`) — median and 75th percentile land right on that number.

## 10 worst sequences by IoU

| Video Name | Segment | IoU | false_observed | false_unobserved | n_gt_observed |
|---|---|---|---|---|---|
| c2_transverse2_t3_v1 | transverse2 | 0.97307 | 66 | 4159 | 156,847 |
| c1_sigmoid1_t2_v2 | sigmoid1 | 0.98399 | 2273 | 2268 | 281,387 |
| c1_ascending_t4_v3 | ascending | 0.98402 | 147 | 8049 | 512,630 |
| c1_ascending_t4_v2 | ascending | 0.98403 | 158 | 8033 | 512,586 |
| c1_sigmoid1_t2_v3 | sigmoid1 | 0.98408 | 2250 | 2266 | 281,495 |
| c2_rectum_t4_v1 | rectum | 0.98460 | 546 | 3070 | 234,276 |
| c2_rectum_t3_v1 | rectum | 0.98770 | 174 | 2594 | 224,848 |
| c1_ascending_t3_v1 | ascending | 0.99211 | 203 | 4652 | 615,047 |
| c1_sigmoid1_t1_v1 | sigmoid1 | 0.99349 | 818 | 805 | 248,387 |
| c1_ascending_t2_v2 | ascending | 0.99421 | 137 | 3396 | 609,625 |

## Flagged: IoU < 0.99 (7 sequences, listed only, not diagnosed or fixed)

`c2_transverse2_t3_v1`, `c1_sigmoid1_t2_v2`, `c1_ascending_t4_v3`, `c1_ascending_t4_v2`, `c1_sigmoid1_t2_v3`, `c2_rectum_t4_v1`, `c2_rectum_t3_v1` — the same 7 as the "10 worst" table's top 7.

One observation for whoever inspects these next (not a diagnosis — flagging pattern only, per instructions not to fix anything here): in all 7, `false_unobserved` is 4-25x larger than `false_observed` (e.g. `c1_ascending_t4_v2`: 158 vs 8033), unlike the validated `c1_cecum_t1_v1` baseline where the two were roughly balanced (~371 vs ~375, consistent with pure tie-breaking noise per `docs/gpu_validation.md`). This imbalance — systematically under-counting observed faces relative to GT rather than symmetric noise — is a different signature and is worth investigating before trusting these 7 sequences' numbers downstream.

## Real end-to-end timing (169 sequences, 67,886 frames)

| Stage | Total | Per sequence (mean) |
|---|---|---|
| Stream archive + parse OBJ + pose.txt | 255.3s | 1.51s |
| Build GPU BVH | 0.7s | 0.004s |
| Ray cast (all frames) | 899.2s | 5.32s |
| **Total** | **1155.2s (19.3 min)** | **6.84s** |

Faster than the ~36 min pre-run estimate in `docs/gpu_validation.md` — that estimate used per-sequence costs measured before an optimization made mid-launch (see below), so it was already stale by the time of the real run.

**Optimization made during this task** (not present in the validated benchmark scripts): the initial `process_sequence` implementation reused the same CPU-side pose-transform code as the validation scripts (`transform_points()`, called twice per frame, plus a redundant re-normalization and a full per-pixel origin-tiling step) and measured **4.5 fps** — meaning the CPU-side ray math, not the GPU kernel, was the bottleneck by roughly 100x, and a full run would have taken over 2 hours. Replaced with a direct rotation-submatrix shortcut (`world_dir = cam_ray @ M[:3,:3]`, origin `= M[3,:3]`, no renormalization needed since ray-mesh intersection depends only on direction and `M[:3,:3]` is already very close to orthonormal), verified numerically equivalent to the original approach (direction match to 3.4e-14 after normalizing both, origin match exact), giving **64 fps** in isolation and ~41 fps end-to-end on the validation sequence — this is what made the real 19.3-minute run possible.
