# Open-end artifact check

Do the large unobserved regions found in `docs/coverage_stats.md` reflect genuine lumen-wall coverage gaps (e.g. behind folds/haustra), or are they mostly a grazing-angle artifact of the phantom's open ends? Script: `scripts/openend_artifact_check.py` (analysis) + `scripts/render_coverage_views.py` (reusable renderer). Full log: `logs/openend_artifact_check.log`. Data: `results/openend_check/component_distances.csv`. Images: `results/openend_check/*.png` (18 total: 6 sequences x {view1, view2, cutopen}).

## Sequences (6, spread across segments and Open End Visible)

| Video Name | Segment | Open End Visible |
|---|---|---|
| c2_transverse1_t2_v1 | transverse1 | yes |
| c1_ascending_t4_v2 | ascending | yes |
| c2_rectum_t1_v3 | rectum | yes |
| c2_transverse2_t3_v2 | transverse2 | no |
| c2_cecum_t3_v2 | cecum | no |
| c1_sigmoid1_t1_v3 | sigmoid1 | no |

## Method notes

- **Boundary edges**: edges belonging to exactly one face (mesh holes). Every sequence has **2 boundary loops** (the phantom's two open ends: insertion + far end) **except `c2_cecum_t3_v2`, which has only 1** — cecum is a blind pouch with a single opening, not a through-tube. This is a clean, independent confirmation of why cecum is the best-covered segment in `docs/coverage_stats.md` (median 9.7% unobserved): there's no "far end" to lose coverage near.
- **Distance metric**: Euclidean (not geodesic) distance from each face to the nearest boundary *vertex*, via a KD-tree — the "Euclidean" option explicitly permitted by the task's "geodesic-or-euclidean" spec. Reported both **pooled across both loops** (matching the task's literal min/median spec) and, as a refinement, **per-loop** (`nearest_loop_label`/`nearest_loop_centroid_dist_mm` in the CSV) — pooling both ends together can't distinguish "near the insertion end" from "near the far end," so the per-loop breakdown is needed to actually answer the open-end question precisely.
- **Rendering**: `scripts/render_coverage_views.py` computes PCA axes of the mesh, renders 2 exterior views (from the short and mid axes) and a "cut-open" view. The cut-open view keeps faces on one side of a *locally-centered* cutting surface (binned along the long axis, not a single global plane) because these colon segments are curved/bent, not straight tubes — a single global plane cut off nearly the whole mesh away from its middle on first attempt. Rendered with `pyrender` (EGL, GPU offscreen), materials forced double-sided (the mesh's face-normal winding doesn't consistently face outward, so the interior lumen surface exposed by the cut was rendering blank under default single-sided culling).

## Results

Rank-1 (largest) component, every sequence:

| Video Name | Open End Visible | # boundary loops | min dist | median dist | frac faces <5mm of boundary |
|---|---|---|---|---|---|
| c2_transverse1_t2_v1 | yes | 2 | 0.00mm | 33.7mm | 4.6% |
| c1_ascending_t4_v2 | yes | 2 | 0.00mm | 15.6mm | 16.1% |
| c2_rectum_t1_v3 | yes | 2 | 0.00mm | 15.2mm | 12.3% |
| c2_transverse2_t3_v2 | no | 2 | 0.00mm | 19.3mm | 15.8% |
| c2_cecum_t3_v2 | no | 1 | 0.00mm | 11.6mm | 23.3% |
| c1_sigmoid1_t1_v3 | no | 2 | 0.00mm | 21.3mm | 28.2% |

The largest component **always touches an open-end boundary exactly (min distance 0mm in all 6 sequences)** — but its bulk sits well inside the tube: median distance 11.6-33.7mm, and only **5-28% of its own faces** are within 5mm of any boundary. Visually (see images below) these are not thin rim strips; several are an entire unimaged longitudinal side of the tube, which happens to touch both open ends simply because it runs the tube's full length.

Ranks 2-5 (smaller components), pooled across all 6 sequences (24 components): **12/24 (50%) are genuinely isolated mid-wall** (0% of faces within 5mm of any boundary, several 15-35mm from the nearest open end — e.g. `c1_ascending_t4_v2` components 3-5, `c2_cecum_t3_v2` components 2-5, `c2_transverse2_t3_v2` components 2/4/5), **8/24 (33%) are small fragments hugging an open end** (>85% of faces within 5mm, e.g. `c1_sigmoid1_t1_v3` components 2-4), and 4/24 are mixed.

## Images

`c1_ascending_t4_v2` (ascending, Open End Visible=**yes**) -- rank-1 (red) anchored at the top open end, extending far down the tube; rank-2 (orange) is a separate patch near the *other* end:

![ascending view1](../results/openend_check/c1_ascending_t4_v2_view1.png)

Cut-open view of the same sequence -- the interior lumen wall is clean/gray in the middle (well-observed), confirming the large unobserved area is concentrated at the two ends, not spread through the tube's midsection:

![ascending cutopen](../results/openend_check/c1_ascending_t4_v2_cutopen.png)

`c1_sigmoid1_t1_v3` (sigmoid1, Open End Visible=**no**) -- here the rank-1 component (red) is an entire unimaged longitudinal strip running nearly the full length of the segment, not a rim artifact -- the camera's trajectory simply never circled around to this side:

![sigmoid1 view2](../results/openend_check/c1_sigmoid1_t1_v3_view2.png)

All 18 renders (6 sequences x view1/view2/cutopen) are in `results/openend_check/`.

## Answer

**Both, in a specific split by component rank:**

- **The single largest unobserved region per sequence is anchored at an open end but is not merely an edge artifact.** It always touches a boundary (min distance 0mm, every sequence) yet mostly consists of lumen wall well inside the tube (72-95% of its own area farther than 5mm from any edge). Visual inspection shows these are typically one whole side of the tube the camera trajectory never circled around to see -- a genuine coverage gap from the imaging path, not a close-to-the-rim grazing-angle effect. It happens to touch an open end because such a longitudinal strip necessarily reaches both ends of the segment.
- **The next-largest components (ranks 2-5) are a real mix**, and about half (12/24, 50%) are **genuinely on the lumen wall away from either open end** -- isolated patches 5.5-35mm from the nearest boundary, the population most consistent with occlusion behind folds/haustra. The other half are small fragments immediately next to an open end, plausibly true grazing-angle rim artifacts (or just the ragged remainder of the same large near-end component, separated by a sliver of marginally-observed faces).
- **Open End Visible (yes/no) does not obviously change this picture** in this 6-sequence sample -- if anything, the "no" group shows slightly *higher* frac-within-5mm for their rank-1 component (11.6-28.2% vs 4.6-16.1% for "yes"), the opposite of what a pure "visible far end gets under-imaged" story would predict. The cecum case (`c2_cecum_t3_v2`, only 1 boundary loop since it's a blind pouch, not `Open End Visible` capable of "yes") independently confirms the open-end mechanism: fewer open ends -> best coverage in the whole dataset.

**Bottom line**: don't treat "large unobserved region" as synonymous with "open-end artifact to exclude." A large fraction of it -- likely most of it by area, given the largest component's low near-boundary fraction -- is genuine lumen-wall coverage gap from the camera's trajectory, valuable/important ground truth for coverage-based evaluation, not a boundary effect to discount.
