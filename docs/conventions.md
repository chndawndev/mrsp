# C3VDv2 Conventions — verbatim source audit

Sources: `/data1_ycao/chua/datasets/C3VDv2/README.md`, `/data1_ycao/chua/datasets/C3VDv2/camera_intrinsics.txt`, and (where the dataset text was silent) `DurrLab/C3VDv3` GitHub repo (cloned to `scratch/C3VDv3`, the rendering/capture pipeline that produced this dataset).

## 1. Depth encoding

**CONFIRMED** — README, "Methodological and File Level Information", item 2:
> "Depth Frame: depth/NNNN_depth.tiff represents the depth along the camera frame's Z-axis, clamped between 0 and 100 mm, and linearly scaled to be encoded as a 16-bit grayscale image."

This settles both sub-questions from the README text alone:
- **Encoding**: uint16, linear scale, range clamped to **[0, 100] mm**.
- **Z-depth vs. radial**: explicitly "**along the camera frame's Z-axis**" — this is camera-space Z-depth, *not* radial/ray distance.

**CONFIRMED (code cross-check)**, `C3VDv3/render/Render.cu`:
- Line 20: `#define MAX_DEPTH   100.0f` — matches the README's 100 mm clamp.
- Line 117: `float depth = glm::vec3(glm::vec4(hitPos-origin,1.0)*optixLaunchParams.t_curr).z;` (comment at line 37: `/* Z-depth of intersection point. */`) — the hit point is transformed into camera space and only the **Z component** is kept, confirming z-depth, not Euclidean/radial distance from the camera origin.
- Line 330: `fbDepth[px_idx] = (uint16_t)65535.f*(owl::clamp(depth / MAX_DEPTH,0.f,1.f));` — confirms the linear 16-bit scaling.

## 2. Camera model

**CONFIRMED (partial)** — `camera_intrinsics.txt`, full verbatim contents:
```
; omnidirectional camera intrinsics
width = 1350
height = 1080
cx = 677.739464094188
cy = 543.057997844875
a0 = 767.733695862103
a1 = 0.0
a2 = -0.000592506426558248
a3 = -2.69440266600040e-07
a4 = -2.16380341010063e-10
c = 0.9999
d = 1.10e-4
e = -1.83e-4
```
The only model wording anywhere in the dataset is the comment line `; omnidirectional camera intrinsics`. The README's own text (searched in full) **never uses the word "omnidirectional," "Scaramuzza," "OCamCalib," or gives any projection/unprojection equation** — grepped for `scaramuzza|ocamcalib|omnidirectional|fisheye|unified camera|projection|unproject` across `README.md` and `camera_intrinsics.txt`; the only hit is that one comment line in the intrinsics file.

Parameter-name pattern match: `cx, cy` (principal point) + `a0..a4` (polynomial coefficients of a Taylor expansion in the radius) + `c, d, e` (affine sensor-plane distortion of the ideal pixel grid) is the exact parameter set of **Scaramuzza's omnidirectional camera model** (OCamCalib / "A Toolbox for Easily Calibrating Omnidirectional Cameras," used e.g. via the `a0..a4` inverse polynomial for backprojecting a pixel to a 3D ray, and `c,d,e` for the affine transform between ideal and sensor pixel coordinates). This is a strong, unambiguous naming match, but **it is an inference from parameter names, not a dataset-stated fact** — the dataset gives no equation and never names the model.

**CONFIRMED (code)** — the model is Scaramuzza, stated in a source comment, and **both** directions are implemented in the renderer.

### 2.1 Model identification (no longer an inference)

`/data1_ycao/chua/projects/mrsp/scratch/C3VDv3/render/Intrinsics.h:14-19`:
```c
/*  Omnidirectional camera intrinsics from
    Scaramuzza, D., A. Martinelli, and R. Siegwart.
    "A Toolbox for Easy Calibrating Omnidirectional Cameras."
    Proceedings to IEEE International Conference on
    Intelligent Robots and Systems, (IROS). Beijing, China,
    October 7–15, 2006.*/
```
So the naming match in the paragraph above is **confirmed by the renderer's own source**, though still not by the dataset README.

### 2.2 How the 12 numbers are packed — `a1` IS DISCARDED

`render/Intrinsics.h:20-37`:
```c
struct Intrinsics
{
    glm::ivec2  size;
    glm::vec2   center;
    glm::vec4   polyCoeff;
    glm::mat2   stretchMat;

    Intrinsics( unsigned int w, unsigned int h,
                float cx, float cy,
                float a0, float a1, float a2, float a3, float a4,
                float c, float d, float e)
    {
        size        = glm::ivec2(w,h);
        center      = glm::vec2(cx,cy);
        polyCoeff   = glm::vec4(a0,a2,a3,a4);
        stretchMat  = glm::mat2(c,d,e,1.0);
    }
};
```
Critical details:
- **`a1` is parsed from the config and then thrown away.** `RenderingModule.cpp:356` (`a1 = parser.aConfig<float>("a1");`) and `RenderingModule.cpp:50` (`Intrinsics intrinsics(width, height, cx, cy, a0, a1, a2, a3, a4, c, d, e);`) pass it in, but the constructor never stores it. The renderer hard-codes the linear term to zero (see `0*rho` at `render/Render.cu:85`). Harmless here because `camera_intrinsics.txt` has `a1 = 0.0`, but a Python port must **not** include an `a1*rho` term.
- **Index mapping (easy to get wrong):** `polyCoeff.x = a0`, `polyCoeff.y = a2`, `polyCoeff.z = a3`, `polyCoeff.w = a4`. There is **no** `polyCoeff` slot for `a1`.
- **`stretchMat` is column-major.** GLM's 4-scalar `mat2` constructor is `value[0] = col_type(x0,y0); value[1] = col_type(x1,y1);` (verified in GLM's `glm/detail/type_mat2x2.inl:47-59`; the repo's own `external/glm/` is an uninitialized submodule, so this was checked against another local GLM checkout — the constructor is unchanged across GLM versions), so `glm::mat2(c,d,e,1.0)` is the linear map
  ```
  M = [ c  e ]        M * (u,v) = ( c*u + e*v ,  d*u + 1*v )
      [ d  1 ]
  ```
  i.e. the **transpose** of the `[c d; e 1]` matrix written in the OCamCalib/MATLAB literature.

### 2.3 Unprojection (pixel → camera-space ray) — CONFIRMED

`render/Render.cu:71-94`, verbatim:
```c
inline __device__
glm::vec3 pixel2Ray(const glm::vec2 px)
{ 
    /* Convert to screen space. */
    glm::vec2 uvp = px - optixLaunchParams.intrinsics.center;

    /* Distort with stretch matrix. */
    glm::vec2 uvpp = glm::inverse(optixLaunchParams.intrinsics.stretchMat)*uvp;

    /*  Back-project pixel points onto the unit sphere by finding the coordinates
        of the vectors emanating from the single-effective-viewpoint to the unit
        sphere so that X^2 + Y^2 + Z^2 = 1. */
    float rho = sqrt( pow(uvpp.x,2) + pow(uvpp.y,2) );
    float z = optixLaunchParams.intrinsics.polyCoeff.x + 
             (0*rho) + 
             (optixLaunchParams.intrinsics.polyCoeff.y * pow(rho,2)) + 
             (optixLaunchParams.intrinsics.polyCoeff.z * pow(rho,3)) + 
             (optixLaunchParams.intrinsics.polyCoeff.w * pow(rho,4));

    /* Combine ray components and normalize. */
    glm::vec3 ray = glm::normalize(glm::vec3(uvpp.x,uvpp.y,z));

    return ray;
}
```
Substituting the packing from §2.2, the equation actually implemented is:
```
uvp   = (px_x - cx, px_y - cy)
uvpp  = inv([[c, e], [d, 1]]) @ uvp          # note: [[c,e],[d,1]], NOT [[c,d],[e,1]]
rho   = sqrt(uvpp_x^2 + uvpp_y^2)
z     = a0 + 0*rho + a2*rho^2 + a3*rho^3 + a4*rho^4
ray   = normalize( (uvpp_x, uvpp_y, z) )     # camera space, +Z forward
```
Note `+z` (not `-z`): `a0 = 767.7 > 0` and the depth in §1 is `+Z`, so the optical axis is **+Z** and no sign flip is applied anywhere.

Call site (the OptiX raygen program), `render/Render.cu:201-223`:
```c
OPTIX_RAYGEN_PROGRAM(render)()
{
    /* Current pixel index. */
    const glm::vec2 px(optixGetLaunchIndex().x,optixGetLaunchIndex().y);
    ...
    /* Generate ray in camera space. */
    glm::vec3 rayDirLocal = pixel2Ray(px);

    /* Transform ray to world space. */
    glm::vec3 rayDirWorld = glm::normalize(glm::vec3(optixLaunchParams.t_curr*
                                           glm::vec4(rayDirLocal,0.0f)));

    glm::vec3 rayOrigin = glm::vec3(optixLaunchParams.t_curr[3]);
```

**Independent cross-check — the repo ships a Python reference implementation** of exactly this unprojection: `scratch/C3VDv3/utils/exampleDataLoader.py:4-47`, `generate_omnidirectional_ray_map()`. It agrees on everything (`ix - cx`, `iy - cy`, inverse stretch, `a0 + a2*rho^2 + a3*rho^3 + a4*rho^4`, normalize) with **one discrepancy**: line 27 builds
```python
stretchMat = np.array([[intrinsics["c"], intrinsics["d"]], [intrinsics["e"], 1.0]])
```
which is the **transpose** of what the CUDA renderer uses (§2.2). With `d = 1.10e-4`, `e = -1.83e-4` the two differ by `(e-d)*v ≈ 0.2 px` at the image edge — negligible for most uses, but the CUDA form is what generated the data.

### 2.4 Projection (camera-space 3D point → pixel) — CONFIRMED

`render/Render.cu:49-67`, verbatim:
```c
inline __device__
glm::vec2 forwardProjectVertex(const glm::vec3 v)
{
    float m = v.z / (pow(pow(v.x, 2.0f) + pow(v.y, 2.0f), 0.5f) + 1.0e-20f);

    double rho = solveQuartic((double)optixLaunchParams.intrinsics.polyCoeff.w,
                              (double)optixLaunchParams.intrinsics.polyCoeff.z,
                              (double)optixLaunchParams.intrinsics.polyCoeff.y,
                             -(double)m,
                              (double)optixLaunchParams.intrinsics.polyCoeff.x);

    glm::vec2 raw_uv;
    raw_uv.x = v.x / (pow(pow(v.x, 2.0f) + pow(v.y, 2.0f), 0.5f) + 1.0e-20f) * (float)rho;
    raw_uv.y = v.y / (pow(pow(v.x, 2.0f) + pow(v.y, 2.0f), 0.5f) + 1.0e-20f) * (float)rho;

    glm::vec2 ukr = optixLaunchParams.intrinsics.stretchMat*raw_uv + optixLaunchParams.intrinsics.center;

    return ukr;
}
```
The polynomial is inverted by **exact closed-form quartic root-finding, not Newton-Raphson.** `solveQuartic(a,b,c,d,e)` solves `a*x^4 + b*x^3 + c*x^2 + d*x + e == 0` (`render/Quartic.cuh:9-11`, adapted from `github.com/sidneycadot/quartic`) using complex Ferrari/resolvent arithmetic in `double`, then selects the root as follows (`render/Quartic.cuh:38-51`):
```c
    /* Find minimum real root greater than 0. */
    double solution = 1.0e50;
    if(root0.real() > 0.0 && (root0.imag() == 0.0))
        solution = min(solution,root0.real());
    ... /* same for root1, root2, root3 */
    if(solution == 1.0e50)
        solution = 0.0;
    return solution;
```
i.e. **the smallest strictly-positive purely-real root, falling back to `rho = 0.0` if none exists.** (`imag() == 0.0` is an exact float comparison, so numerically-tiny imaginary parts reject a root.)

Substituting the argument mapping, the equation actually implemented is:
```
r_xy  = sqrt(v_x^2 + v_y^2) + 1e-20
m     = v_z / r_xy
rho   = smallest real root > 0 of:  a4*rho^4 + a3*rho^3 + a2*rho^2 - m*rho + a0 = 0   (else 0)
raw_uv = (v_x / r_xy * rho,  v_y / r_xy * rho)
px    = [[c, e], [d, 1]] @ raw_uv + (cx, cy)
```
This is the exact algebraic inverse of §2.3 (`a0 + a2*rho^2 + a3*rho^3 + a4*rho^4 = m*rho`), so `project(unproject(px)) == px` up to root-selection and float precision. Note the projection uses `stretchMat` **forward** while unprojection uses `glm::inverse(stretchMat)` — consistent.

Used for optical flow, `render/Render.cu:257-270`:
```c
            glm::vec3 v_cam_prev = glm::vec3(glm::inverse(optixLaunchParams.t_prev)*glm::vec4(prd.hitPosPrev,1.0));
            
            /* Forward project the point to the sensor plane. */
            glm::vec2 px_prev = forwardProjectVertex(v_cam_prev);
```
and the resulting flow is `flow = px_prev - px` (`render/Render.cu:290`) — i.e. **backward** flow, in pixels, encoded as `(flow + 20)/40 * 65535` with `MAX_FLOW 20.0f` (`render/Render.cu:337-340`).

**Scope note:** `forwardProjectVertex` / `pixel2Ray` in `render/Render.cu` are the **only** projection/unprojection implementations in the C++/CUDA pipeline. Grepped `.cpp/.h/.cu/.cuh` repo-wide for `polyCoeff|stretchMat|\.center|cx|cy|a0..a4`: hits are confined to `render/Intrinsics.h`, `render/Render.cu`, `render/LaunchParams.h`, `render/RenderContext.*`, and the config plumbing in `RenderingModule.cpp` / `AlignmentModule.cpp` (which only read the scalars from `config.ini`). `tools/Handeye.cpp` is pure pose algebra (`A2B(A) = B_cal * inverse(X) * inverse(A_cal) * A * X`, `tools/Handeye.cpp:21-22`) with no camera model; `RegistrationModule.cpp` and `AlignmentModule.cpp` contain no pixel-projection code; `render/Mask.cu` is a pure per-pixel binary mask multiply with no intrinsics. **No Newton-Raphson / iterative inversion exists anywhere in the repo** (grepped `newton|raphson|invertPoly|root` — only `Quartic.cuh`'s closed-form roots).

### 2.5 Pixel indexing and axis conventions — CONFIRMED, no flips

- **Origin**: `cx, cy` are in **pixel units with the origin at the center of pixel (0,0)**, i.e. top-left corner convention with *integer* pixel indices (no `+0.5` half-pixel offset anywhere). The raygen feeds raw integer launch indices straight in — `render/Render.cu:204`:
  ```c
  const glm::vec2 px(optixGetLaunchIndex().x,optixGetLaunchIndex().y);
  ```
  and the Python reference does the same with `np.arange` — `utils/exampleDataLoader.py:19-23`:
  ```python
  ix, iy = np.meshgrid(np.arange(intrinsics['width']), np.arange(intrinsics['height']))
  uvp_x = ix - intrinsics['cx']
  uvp_y = iy - intrinsics['cy']
  ```
- **`px.x` is the column (x, width, 0..1349); `px.y` is the row (y, height, 0..1079).** The launch is `owlLaunch2D(rayGen, intrinsics.size.x, intrinsics.size.y, launchParams);` (`render/RenderContext.cpp:212`), with `size = glm::ivec2(w,h)`.
- **No row/column swap and no Y-flip** between the math and the frame buffer. `render/Render.cu:325`:
  ```c
  const uint32_t px_idx = px.x + px.y*optixLaunchParams.intrinsics.size.x;
  ```
  — standard row-major, stride = width, row index = `px.y` (**not** `height-1-px.y`). The same ordering is used on the host: `render/Mask.cu:22` builds the mask as `mask_host[y*width + x] = mask(x,y) > 0`, and `render/Mask.cu:44` reads `mask_dev[pixelX + width*pixelY]`.
- **Written to disk with no flip either**: `RenderingModule.cpp:300` writes `stbi_write_png(diffuseFilename.c_str(),width,height,1,diffuse_host,width*sizeof(uint8_t))` and `RenderingModule.cpp:306`/`:311` open TIFFs with `TinyTIFFWriter_open(...,width,height,...)` then `TinyTIFFWriter_writeImage(depthTiff, depth_host)` directly from the same buffer. So **buffer row 0 = image top row = launch index `y = 0`**; a Python `img[row, col]` maps directly to `px = (col, row)`.
- **Bounds test uses a half-open `[0, size)` interval** on the float pixel coords, `render/Render.cu:263-266`:
  ```c
            if(px_prev.x < optixLaunchParams.intrinsics.size.x &&
               px_prev.x >= 0 &&
               px_prev.y < optixLaunchParams.intrinsics.size.y &&
               px_prev.y >= 0)
  ```
- **Camera space**: right-handed with **+Z forward** (optical axis), +X = image `x`/column direction, +Y = image `y`/row direction (downward in the image). This follows from `pixel2Ray` returning `(uvpp.x, uvpp.y, z)` with `uvpp` derived from `(col - cx, row - cy)` and `z > 0`, and is consistent with §1's `+Z` depth.

**Remaining UNKNOWN**: the *dataset* still never states any of this — every equation above is confirmed from `DurrLab/C3VDv3` source, which is the renderer that produced C3VDv2, not from `README.md` or `camera_intrinsics.txt`. Also genuinely **UNKNOWN**: whether the shipped `camera_intrinsics.txt` was consumed through the CUDA `[[c,e],[d,1]]` form or the Python `[[c,d],[e,1]]` form for any given released derived product (§2.3).

## 3. `pose.txt` convention

**CONFIRMED** — README, item 7:
> "Camera Pose: pose.txt contains each frame's flattened homogeneous camera-to-world transformation matrix (row major order)."

So: **row-major**, **camera-to-world** (not world-to-camera), one flattened 4×4 homogeneous matrix per line (16 comma-separated values, verified directly against `scratch/c1_cecum_t1_v1/pose.txt`: `[R(3x3) 0 | ... ] , [tx,ty,tz,1]` pattern confirms row-major 4x4 with translation in the last row's first 3 entries, consistent with row-major flattening of `[R t; 0 1]`... i.e. reading 4 numbers at a time gives each row).

For `simulated_screening_videos`, README states separately (paragraph after item 8):
> "Each folder comprises RGB frames and a pose.txt file containing the camera poses in a frame-wise homogeneous format."
This does **not** repeat "row major" or "camera-to-world" explicitly for the screening videos — **UNKNOWN whether the same row-major/camera-to-world convention is guaranteed to hold there**, only inferred by consistency with the rest of the dataset.

**UNKNOWN** — units: the README never states pose units. Not stated as mm explicitly anywhere near item 7. (Circumstantial: `coverage_mesh.obj` vertex coordinates and `pose.txt` translation values are on the same numeric scale, e.g. mesh vertices like `-259.004 -7.22158 -389.873` vs. pose translation `-265.339 24.9915 -400.877` in the same file's frame 0 — consistent with millimeters, matching the depth clamp of "0 to 100 mm" and the calibration target's "10 mm" square size — but this is inference from co-located numbers, not a quoted statement.)

**UNKNOWN** — world frame definition: the README does not state what the world origin/axes are (e.g., first camera frame, a fixed rig/phantom frame, or the coverage-mesh model frame). I did not find this defined in the parts of `C3VDv3` I searched (`tools/PoseLog.cpp`, `tools/Handeye.cpp`, `render/Intrinsics.h` had no world/units/scale keyword hits). Not confirmed.

## 4. `coverage_mesh.obj` — "observed" criterion

**README text is insufficient** — item 8 only states *what* the encoding means, not *how* it's computed:
> "3D Model and Coverage Map: coverage_mesh.obj stores the ground truth triangulated mesh. Texture vertices store coverage values, where vt=1 indicates an observed face, and vt=2 indicates an unobserved face."
No threshold, no viewing-angle criterion, no algorithm — marked **UNKNOWN from the README alone**, per the task's fallback instruction I went to the `C3VDv3` rendering source.

**CONFIRMED (from code)** — `DurrLab/C3VDv3`, cloned to `scratch/C3VDv3`:

- `render/Render.cu`, lines 315–320 (inside the per-pixel ray-trace kernel):
```c
/* Coverage. */
if(optixLaunchParams.renderFlags & RenderFlags::COVERAGE)
{
    /* Mark face in coverage texture if intersected. */
    if(prd.primID != -1)
        optixLaunchParams.coverage[(uint32_t)prd.primID] = (uint8_t)255;
}
```
So the criterion is exactly: **a mesh face is "observed" if it is ever the closest ray-hit (primary/visible-surface hit, `primID != -1`) for *any* pixel, in *any* frame, of the video** — a pure binary visibility test with **no distance threshold and no viewing-angle/incidence-angle criterion**.

- Accumulation is across the *whole sequence*, not per-frame: the `coverage` GPU buffer is allocated and zeroed exactly once, in the `RenderContext` constructor — `render/RenderContext.cpp` lines 93–94:
```c
coverage    = owlDeviceBufferCreate(context,OWL_UINT,model->meshes[0]->index.size(),nullptr);
owlBufferClear(coverage);
```
— while `RenderingModule.cpp` calls `context->render()` once per frame inside the frame loop (line 270) and only reads the accumulated `coverage_host` buffer back out **after** that loop, at line 340 (`cudaMemcpy(coverage_host, owlBufferGetPointer(context->coverage,0), ...)`), immediately followed by `writeOBJ(...)` at line 342. So coverage is an OR-accumulation of hits over every frame of the sequence, not a single-frame snapshot.

- The `vt=1`/`vt=2` mapping itself: `RenderingModule.cpp` lines 537–548:
```c
/* Output vertex texture coords for coverage
        [0 0]: observed
...
/* If observed, use text coord 0, otherwise 1. */
int vt = (int)coverageTex[i] == 255 ? 1 : 2;
```
(1-indexed `vt` list position; `vt` index 1 → `vt 0.0 0.0` → observed; `vt` index 2 → `vt 1.0 1.0` → unobserved — matches what I found by direct inspection of `coverage_mesh.obj` in the previous inventory task.)

## 5. Score / flag definitions

**CONFIRMED** — README, items 9 and 10:
> "Qualitative Score: Subjective score from 1 to 3 indicating the quality of alignment between RGB frames and rendered views from the 3D model. 1 - Best Alignment, 2 - Good Alignment, 3 - Misalignment due to phantom manufacturing defects."

> "Quantitative Score: Comprises a combination of dice score (overlap) and chamfer distance (misalignment) on edges extracted from RGB frames and depth frames rendered from the 3D model. Refer to the manuscript for more details."
(The exact formula combining dice score and chamfer distance is **not given** — README explicitly defers to "the manuscript," which is outside the dataset files. UNKNOWN beyond this description.)

**CONFIRMED** — README, item 6:
> "Open End Visible: Colon segment phantoms have two open ends, one of which is used for camera insertion. This field indicates if the other end is visible in the video."

## Summary table

| Item | Status |
|---|---|
| 1. Depth: uint16, 0–100mm, camera-Z (not radial) | **CONFIRMED** (README + code) |
| 2. Camera model: Scaramuzza omnidirectional, exact forward+inverse equations | **CONFIRMED** via `C3VDv3` source (§2.1–2.5); model never named by the dataset itself, only by the renderer's source comment |
| 3a. Pose: row-major, camera-to-world | **CONFIRMED** for registered/deformation videos; not explicitly repeated for screening videos |
| 3b. Pose units | **UNKNOWN** (circumstantially mm) |
| 3c. World frame definition | **UNKNOWN** |
| 4. Coverage "observed" criterion | README **UNKNOWN**; **CONFIRMED via C3VDv3 code**: any primary ray hit (`primID != -1`), no distance/angle threshold, accumulated over the whole sequence |
| 5. Qualitative/Quantitative Score, Open End Visible | **CONFIRMED** (README); Quantitative Score's exact combination formula **UNKNOWN**, deferred to manuscript |

