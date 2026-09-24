# Results viewer, Stage 2: `c1_cecum_t1_v1`

A single self-contained HTML file showing the Stage 1 export
(`results/viewer/c1_cecum_t1_v1/`) in 3D next to its numbers. **The
viewer never computes a metric.** Every number in the panel is read
directly from `metrics.json` / `regions.json`. The one thing it computes
is the per-face color category, from the exported boolean arrays.

Source: `tools/viewer/` (`template.html`, `core.js`, `viewer.js`,
`viewer.css`). Build: `scripts/build_viewer.py` (reads the Stage 1 export
and the viewer source only; never writes to the export). Output:
`results/viewer/c1_cecum_t1_v1_viewer.html`. Browser check:
`scripts/check_viewer.py` (Playwright).

## Build and open

```
scratch/.venv/bin/python scripts/build_viewer.py
```

Then open `results/viewer/c1_cecum_t1_v1_viewer.html` by double-click, or
`file://` in any browser. No server, no build step at open time -- all
data is embedded (gzip + base64, decoded with the native
`DecompressionStream` API). Requires internet access at open time only
for three.js itself (loaded from a pinned CDN URL, see below) -- the
Stage 1 data is fully local.

## Size

**9.48 MB** (well under the 60 MB limit; no optimization was needed).
Breakdown of the embedded payloads (gzip+base64, from `manifest.json`'s
byte sizes): `vertices_f32` 4.70MB, `faces_i32` 4.38MB, `face-level u8/i32
arrays` ~0.3MB combined, `predicted_observed_packed` 0.26MB, the three
JSON files ~0.06MB combined. Float64 copies and `face_area_mm2` are
**not** embedded (verification-only in Stage 1; no viewer feature needs
them).

## three.js version

**r160 (0.160.0)**, loaded via an import map from
`https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js` and
`https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/controls/OrbitControls.js`.
Both URLs confirmed reachable (HTTP 200) from this server at build time.

## Color mapping (audit this against the metrics)

Single-config view, a disjoint partition of every face
(`tools/viewer/core.js::categorizeFaces`, condition `pu = ¬predicted_observed[config, tau]`):

| category | condition | color | hex |
|---|---|---|---|
| Correctly flagged unobserved | `¬gt_observed ∧ pu` | blue | `#0072B2` |
| False reassurance | `¬gt_observed ∧ ¬pu` | vermillion | `#D55E00` |
| False alarm | `gt_observed ∧ pu ∧ ¬ignore_set` | orange | `#E69F00` |
| Correctly observed | `gt_observed ∧ ¬pu ∧ ¬ignore_set` | neutral gray | `#9A9A9A` |
| Ignore set | `ignore_set` (regardless of prediction) | light gray + diagonal hatching | `#D9D9D9` |

Okabe-Ito colorblind-safe palette. `ignore_set ⊆ gt_observed` by
definition (`src/eval/ignore_set.py:50`), so the ignore-set category never
collides with "correctly flagged unobserved" or "false reassurance."

Diff mode (two configs, same tau; raw `predicted_unobserved`, no
ignore-set handling -- matches Stage 1's check C definition exactly, see
`docs/viewer_export.md`):

| category | condition | color | hex |
|---|---|---|---|
| Only A | `puA ∧ ¬puB` | orange | `#E69F00` |
| Only B | `puB ∧ ¬puA` | sky blue | `#56B4E9` |
| Both | `puA ∧ puB` | reddish purple | `#CC79A7` |
| Neither | `¬puA ∧ ¬puB` | light gray | `#D9D9D9` |

## Checks

### Category-count check (required)

`scripts/build_viewer.py` computes per-face category counts directly in
numpy from the Stage 1 export, for all 4 configs x 4 taus plus one diff
pair, to `results/viewer/category_counts_python.json` (not embedded in
the HTML). `scripts/check_viewer.py` then loads the **built** HTML in a
real headless Chromium (Playwright, SwiftShader software WebGL) and calls
`window.__viewerComputeCounts(config, tau)` -- the exact function the page
uses to color the mesh -- for the same 16 combinations, and diffs.

**Result: exact match, all 16 config x tau combinations.**
`oracle @ tau=0.25`: JS `[63814, 341, 1, 615658, 20094]`, Python
`[63814, 341, 1, 615658, 20094]` (order: correctly-flagged-unobserved,
false-reassurance, false-alarm, correctly-observed, ignore-set). Full
comparison in `results/viewer/check_viewer_report.json`.

### Headless browser check

Chromium 153 (Playwright 1.63.0, installed into `scratch/.venv`), headless,
`--use-gl=swiftshader`. Loaded `file://.../c1_cecum_t1_v1_viewer.html`,
waited for the ready signal, then exercised: config switch, tau switch,
headline-only toggle, diff mode (on, select A/B, off), the frame slider,
and a canvas click (confirmed the region-info panel populated).

**Result: 0 console errors, 0 page errors, 0 failed requests.** 4
benign SwiftShader driver warnings (`GPU stall due to ReadPixels`,
software-rasterizer performance chatter, not a correctness signal).
Screenshot: `results/viewer/check_viewer_screenshot.png`. Full console
log: `results/viewer/check_viewer_report.json`.

### What these checks do and don't cover

- The category-count check exercises the exact shipped decoding +
  categorization code (base64 -> gzip -> typed array -> bit-unpack ->
  categorize), since it calls the same function the page's own render
  path calls -- not a reimplementation.
- SwiftShader is a software GL rasterizer: it confirms the WebGL pipeline
  runs correctly (shaders compile, geometry renders, no GL errors) but
  says nothing about real-GPU frame rate.
- The UI exercise (clicking, toggling) confirms those code paths run
  without throwing, not that every visual detail is pixel-perfect --
  reviewed by screenshot, not by an automated visual diff.

## Known limitations

- **Internet required at open time** for the three.js CDN URLs. If this
  matters for an offline environment, three.js would need to be embedded
  too (not done here -- out of scope for this stage).
- **Browser support**: `DecompressionStream` requires Chrome/Edge 80+,
  Firefox 113+, or Safari 16.4+. No fallback is implemented.
- **Picking performance**: face selection uses three.js's default
  `Raycaster` against a 699,908-triangle non-indexed mesh (a linear BVH is
  not built) -- roughly 0.1-0.3s per click on this machine, not
  imperceptible but not a blocker for exploratory use.
- **Diff mode ignores the ignore set** (by design, matching Stage 1's
  check C definition exactly -- see `docs/viewer_export.md`). A face in
  the ignore set can appear colored "only A" / "only B" / "both" in diff
  mode even though the single-config view would gray it out.
- **Hatching** is a screen-space diagonal stripe pattern injected into
  `MeshBasicMaterial`'s fragment shader via `onBeforeCompile`, keyed to
  `gl_FragCoord`. It is not UV-based, so the stripe orientation is
  camera-relative, not surface-relative (moves as you orbit) -- a
  deliberate simplification, not a bug.
- **Below-headline GT regions** (d < 5mm) have no per-config coverage,
  detection, or localization data -- Stage 1 only exported that for
  headline (d >= 5mm) regions, per the original task spec. Clicking a
  below-headline-region face shows id/area/diameter only, with an explicit
  note, not a silently-missing value.
- Camera starts at a fixed offset from the mesh's bounding-sphere center;
  orbit/zoom (mouse drag / scroll) reaches any other view.
