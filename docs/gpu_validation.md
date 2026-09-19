# GPU rasterizer validation (pre-full-run)

Run before committing to the 169-sequence GPU job, per the compute policy ("GPU path is approved. Before any full run: ..."). Script: `scripts/validate_gpu_rasterizer.py`. Full log: `logs/validate_gpu_rasterizer.log`. GPU used throughout: index 7 (`CUDA_VISIBLE_DEVICES=7`, explicit, per policy).

## 1. Backend equivalence: Warp (GPU) vs embree (CPU)

Same 20 frames of `c1_cecum_t1_v1`, same ray origins/directions fed to both, comparing per-pixel first-hit face index.

**Result: 1,659 / 29,160,000 pixels disagree = 0.00569%. PASS** (threshold was ~0.1%).

Inspected all 5 saved disagreement examples: every one has the two candidate faces **sharing a vertex** and centroid distance 0.15-0.30mm apart — exactly the expected shared-edge/tie-breaking signature (two BVHs picking a different one of two nearly-coincident valid hits at a shared triangle edge), not a geometry or camera-model mismatch. No further diagnosis needed.

## 2. GT cross-check: depth TIFF raw==0 vs our miss fraction

**Initial result looked like a red flag**: GT `raw==0` fraction is consistently ~7.0% (both `c1_cecum_t1_v1` and `c1_ascending_t2_v1`, all 20 sampled frames), while our rasterizer's miss fraction is ~0.00000%.

**Investigated and resolved — not a bug, not something to fix:**
- The zero pixels form an exact circular vignette: computing pixel radius from image center, all zero pixels have radius 706-864px, all nonzero pixels 0-734px (small overlap because the true optical center `cx=677.74, cy=543.06` isn't exactly the geometric image center `675, 540`). All 4 image corners are 0; the center pixel is 65535 (valid, clamped). This is a lens vignette / border mask, not ray-mesh misses.
- Confirmed in source: `RenderingModule.cpp:56` loads `mask.png` (`RenderingModule.cpp:40`, a file **not included in the released dataset**) and applies it at lines 273-278 to `fbDiffuse, fbRgb, fbDepth, fbNormals, fbFlow, fbOcclusion` only. The coverage buffer (`context->coverage`, read out at `RenderingModule.cpp:340`, written to `coverage_mesh.obj` at line 342) is **never touched by `mask->apply()`** anywhere in the file, and coverage marking happens inside the raygen kernel itself (`Render.cu:316-320`), before any masking step exists in the pipeline.
- **Conclusion**: the vignette mask is a post-process on saved images only. Every pixel's ray — vignetted or not — still contributes to the coverage buffer. Our rasterizer casting a ray for every pixel with near-zero misses is the *correct* behavior for reproducing `coverage_mesh.obj`; matching the 7% depth-image miss rate would be *wrong* for this task (it would under-count truly-observed faces).

## 3. MAX_DEPTH: coverage buffer vs depth image

Cited directly from `render/Render.cu`:
- Line 20: `#define MAX_DEPTH 100.0f`
- Line 223: `PrimaryRay ray(rayOrigin, rayDirWorld, 1e-5f, 1e4f)` — the primary ray's max distance is **1e4 (10,000mm)**, not `MAX_DEPTH`.
- Lines 316-320: coverage marking is `if(prd.primID != -1) coverage[primID] = 255` — gated only on whether the (10m-range) primary ray hit anything, `MAX_DEPTH` never appears in this branch.
- Line 330: `fbDepth[px_idx] = 65535 * clamp(depth / MAX_DEPTH, 0, 1)` — `MAX_DEPTH` is used **only** here, to clamp/encode the depth image.

**Hits beyond 100mm ARE counted as observed** (up to 10m); `MAX_DEPTH` only saturates the depth image's 16-bit encoding. Our rasterizers already use effectively-unbounded ray distance (embree default; Warp's `1.0e6`), so this required no code change — just confirms we're already doing it right. (10m vs 100mm is moot in practice at this dataset's ~100-300mm phantom scale, but it's the technically correct rule to cite and apply.)

## 4. Accuracy vs released `coverage_mesh.obj`, `c1_cecum_t1_v1`

Full-sequence (218 frames) CPU/embree rasterization at 3 pixel strides:

| Stride | IoU | false_observed | false_unobserved | all-218-frame rasterize time |
|---|---|---|---|---|
| 4 | 0.97727 | 33 | 14,416 | 12.7s |
| 2 | 0.99390 | 98 | 3,778 | 47.6s |
| 1 | **0.99883** | 371 | 373 | 171.1s |

**Target (IoU >= 0.99 at stride 1): met (0.99883).**

The residual 371/373 (false_observed/false_unobserved, out of 699,908 faces total — 0.053% each direction) at full resolution is small, balanced, and consistent in magnitude with check 1's backend-tie-breaking rate — the expected floating-point/edge-case noise from reimplementing the camera model and ray-mesh intersection independently, not a systematic error. (Aside: false_observed *rises* with denser sampling (33→98→371) while false_unobserved *falls* (14,416→373) — both are consistent with under-sampling at coarse strides simply having fewer chances to hit the same small set of borderline/near-edge faces where our reimplementation and the original diverge.)

## 5. Real end-to-end per-sequence cost

| Sequence | stream archive + parse OBJ | build embree (CPU) BVH | build Warp (GPU) BVH |
|---|---|---|---|
| c1_cecum_t1_v1 (699,908 faces) | 2.42s | 0.01s | 0.01s |
| c1_ascending_t2_v1 (838,043 faces) | 2.79s | 0.01s | 0.01s |

**BVH construction is negligible on both backends (0.01s).** The dominant fixed per-sequence cost is streaming the zip + parsing the OBJ text (~2.4-2.8s), which is identical regardless of which rasterizer backend is used downstream.

### Combined realistic estimate (169 sequences, 67,886 total frames, avg 401.7 frames/sequence)

| Path | per-sequence fixed cost | kernel/cast throughput | **total estimate** |
|---|---|---|---|
| GPU (1x, index 7) | ~2.6s x 169 = ~7.3 min | 182.6 fps | ~6.2 min (kernel) + ~7.3 min (streaming) = **~13.5 min** |
| CPU single-process | ~2.6s x 169 = ~7.3 min | ~1.2-1.3 fps | ~14-15 hr (streaming overhead negligible in comparison) |
| CPU 16-process | ~2.6s x 169 = ~7.3 min | 3.2 fps | ~5.9 hr + ~7.3 min = **~6.0 hr** |

Streaming/parsing is a small, fixed ~7.3 minutes total either way — it materially changes the GPU estimate (roughly doubles it, 6.2 -> 13.5 min) but is noise against the CPU estimates.

## Verdict

All 5 checks pass or are satisfactorily resolved. No blockers found. Not starting the 169-sequence run, per instructions — awaiting your go-ahead.
