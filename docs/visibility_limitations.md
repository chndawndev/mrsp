# Visibility rasterization: limitations (for paper use)

Our GPU visibility rasterizer (`scripts/visibility_full.py`) against the released GT `coverage_mesh.obj`, all 169 registered sequences: median face-level IoU **0.9988**, worst **0.9731** (`docs/visibility_rasterization.md`). 7 sequences fall below 0.99 and were diagnosed in full (`docs/visibility_outliers.md`); nothing was changed as a result — the rasterizer is validated and **frozen**.

## What the disagreements are

The 7 sub-0.99 sequences split into three regimes, not one:

- **Sub-pixel projected footprint** (`c1_sigmoid1_t2_v2`, `c1_sigmoid1_t2_v3`): disagreements are single/few-face specks, largest connected component 0.23-0.25 mm² (equivalent diameter ~0.5mm). Median projected footprint of the disagreeing faces is **~0.03 px²** — geometrically "observed" under the GT criterion (any ray hit, any frame, no size threshold — `docs/conventions.md`), but at roughly 1/30 of a pixel's projected area, far below any plausible clinical visibility threshold. A 2x2 supersampling test partially confirmed this mechanism (2.7-3.8% of these specific faces flip to "observed" under finer sampling, well above the ~0.006% backend-noise floor) but did not fully resolve it — these faces are directionally sub-pixel-limited, not fully explained. **Cross-reference**: this is exactly the kind of case the planned criteria-sensitivity analysis (distance/incidence-angle/minimum-pixel-footprint thresholds, proposed early in this project but not yet built or run) would be expected to exclude outright via a minimum-footprint criterion — worth using this diagnosis as a concrete test case once that analysis exists.
- **Unexplained boundary-contour speckle** (`c1_ascending_t4_v2`, `c1_ascending_t4_v3`): also single-face specks (largest component 0.18 mm²), but visually and statistically a different signature — a dense speckle running along the observed/unobserved boundary contour, with a *larger* (not smaller) projected footprint than correctly-observed faces, and no response to supersampling (0.0-0.23% recovery, indistinguishable from backend noise). Mechanism logged as **UNKNOWN**; a boundary-alignment discrepancy between our ray grid and the original renderer's is the leading unverified candidate.
- **Larger disagreement components, not sub-millimetre** (`c2_transverse2_t3_v1`, `c2_rectum_t4_v1`, `c2_rectum_t3_v1`): the largest disagreeing component per sequence is 50-86 mm², i.e. **7.96-10.46mm equivalent diameter** — the same size class as headline (>5mm) regions, not a sub-pixel artifact. `c2_rectum_t4_v1` in particular fits neither of the other two regimes on any measured axis (distance, incidence angle, footprint) and its cause is separately **UNKNOWN**.

## Impact on the primary (>5mm) analysis

**This needs a more careful statement than "cannot affect it."** Of the 7 flagged sequences:
- **4** (`c1_ascending_t4_v2/v3`, `c1_sigmoid1_t2_v2/v3`) disagree only in sub-millimetre single-face specks (largest component 0.18-0.25mm², equivalent diameter under 0.6mm) — these genuinely cannot register as a region under a >5mm filter, and the primary analysis is unaffected for these sequences.
- **3** (`c2_transverse2_t3_v1`, `c2_rectum_t4_v1`, `c2_rectum_t3_v1`) have their largest disagreeing component at 7.96-10.46mm equivalent diameter — **above** the 5mm primary-analysis threshold, not below it. If any of these three sequences is used in the primary analysis, this specific component should be checked directly (is it counted as a spurious region on one side of the GT/predicted comparison, and does it change that sequence's region recall) rather than assumed immaterial by size alone.

## Status

Visibility rasterization tooling is frozen as of this diagnosis (`scripts/visibility_full.py`, `scripts/render_coverage_views.py`, `src/geometry/`). No further changes pending; the 3 open UNKNOWNs above (ascending boundary speckle, `c2_rectum_t4_v1`, and full resolution of the sigmoid1 sub-pixel gap) are documented, not blocking, and out of scope for further work here.
