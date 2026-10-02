"""MASt3R-SLAM internal grid (512x400) -> 1350x1080 depth mapping
(scripts/pipeline_adapters.py::BilinearGridMap), docs/eval_protocol.md
"2026-09-29: Missing predictions, internal crops, confidence maps"."""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from pipeline_adapters import (  # noqa: E402
    ORIGINAL_H, ORIGINAL_W, BilinearGridMap, bilinear_reference, original_to_model,
)

# Stage 2's measured mapping (docs/pipelines/mast3r_slam.md, Stage 2 step 5)
M = {"original_w": 1350, "original_h": 1080, "model_grid_w": 512, "model_grid_h": 400,
     "scale_w": 2.63671875, "scale_h": 2.6341463414634148, "half_crop_w": 0.0, "half_crop_h": 5.0}


def test_known_pixel_mapping():
    # Pixel-center aligned: the continuous image centre (pixel index
    # 674.5, 539.5) maps to the centre of the resized 512x410 image
    # (index 255.5, 204.5), then 5 rows are cropped.
    px, py = original_to_model(np.array([674.5, 0.0]), np.array([539.5, 0.0]), M)
    assert abs(px[0] - 255.5) < 1e-9 and abs(py[0] - 199.5) < 1e-9
    assert abs(px[1] - (0.5 / M["scale_w"] - 0.5)) < 1e-12
    assert abs(py[1] - (0.5 / M["scale_h"] - 0.5 - 5.0)) < 1e-12
    # grid pixel centre (r + 0.5) <-> original coordinate (r + 0.5) * scale
    px, py = original_to_model(np.array([100.5 * M["scale_w"] - 0.5]), np.array([(50 + 5 + 0.5) * M["scale_h"] - 0.5]), M)
    assert abs(px[0] - 100.0) < 1e-9 and abs(py[0] - 50.0) < 1e-9


def test_mapping_matches_pil_resize_of_markers():
    """Replica of MASt3R-SLAM's resize (PIL LANCZOS to 512x410, crop rows
    5..405; mast3r_slam/mast3r_utils.py::resize_img) on Gaussian markers:
    the located centroids follow original_to_model and not the
    corner-aligned ox / scale. The same test through MASt3R-SLAM's own
    loader is scripts/mast3r_slam_grid_alignment_test.py."""
    from PIL import Image

    rng = np.random.default_rng(20261001)
    pts = np.stack([rng.uniform(80, ORIGINAL_W - 80, 12), rng.uniform(80, ORIGINAL_H - 80, 12)], axis=1)
    yy, xx = np.mgrid[0:ORIGINAL_H, 0:ORIGINAL_W].astype(np.float64)
    resized_w, resized_h = 512, 410
    assert abs(ORIGINAL_W / resized_w - M["scale_w"]) < 1e-12 and abs(ORIGINAL_H / resized_h - M["scale_h"]) < 1e-12
    for x, y in pts:  # one marker per image, so centroids cannot mix
        img = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 8.0**2))
        pil = Image.fromarray(np.uint8(np.round(img * 250))).resize((resized_w, resized_h), Image.LANCZOS)
        grid = np.asarray(pil.crop((0, 5, 512, 405))).astype(np.float64)
        assert grid.shape == (400, 512)
        gy, gx = np.mgrid[0:400, 0:512].astype(np.float64)
        found = np.array([(grid * gx).sum(), (grid * gy).sum()]) / grid.sum()
        px, py = original_to_model(np.array([x]), np.array([y]), M)
        assert abs(found[0] - px[0]) < 0.02 and abs(found[1] - py[0]) < 0.02
        corner = np.array([x / M["scale_w"], y / M["scale_h"] - 5.0])
        assert abs(found[0] - corner[0]) > 0.25 and abs(found[1] - corner[1]) > 0.25


def test_affine_field_reproduced_exactly():
    gm = BilinearGridMap(M)
    yy, xx = np.mgrid[0:400, 0:512].astype(np.float64)
    grid = 3.0 + 0.25 * xx - 0.5 * yy  # bilinear interpolation is exact on affine fields
    out = gm.sample(grid)
    X, Y = np.meshgrid(np.arange(ORIGINAL_W, dtype=np.float64), np.arange(ORIGINAL_H, dtype=np.float64))
    px, py = original_to_model(X.ravel(), Y.ravel(), M)
    expected = 3.0 + 0.25 * px - 0.5 * py
    a = gm.available
    assert np.allclose(out[a], expected[a], atol=1e-9, rtol=0)
    assert np.isnan(out[~a]).all()


def test_availability_is_exactly_the_grid_box():
    gm = BilinearGridMap(M)
    X, Y = np.meshgrid(np.arange(ORIGINAL_W, dtype=np.float64), np.arange(ORIGINAL_H, dtype=np.float64))
    px, py = original_to_model(X.ravel(), Y.ravel(), M)
    inside = (px >= 0) & (px <= 511) & (py >= 0) & (py <= 399)
    assert np.array_equal(gm.available, inside)
    a = gm.available.reshape(ORIGINAL_H, ORIGINAL_W)
    # top crop band: rows with py < 0 have no depth
    assert not a[0].any()
    # first and last original columns lie outside the first / last grid
    # pixel centre (px = -0.31 and 511.31): no depth, never extrapolated
    assert not a[:, 0].any() and not a[:, ORIGINAL_W - 1].any()
    assert a[540, 1] and a[540, ORIGINAL_W - 2]


def test_never_reads_outside_grid_and_matches_scalar_reference():
    gm = BilinearGridMap(M)
    rng = np.random.default_rng(0)
    grid = rng.uniform(1.0, 2.0, size=(400, 512))
    out = gm.sample(grid)
    assert (gm.x0 >= 0).all() and (gm.x0 + 1 <= 511).all()
    assert (gm.y0 >= 0).all() and (gm.y0 + 1 <= 399).all()
    idx = rng.choice(np.flatnonzero(gm.available), 200, replace=False)
    for p in idx:
        y, x = divmod(int(p), ORIGINAL_W)
        px, py = original_to_model(np.array([float(x)]), np.array([float(y)]), M)
        assert abs(out[p] - bilinear_reference(grid, px[0], py[0])) < 1e-12


def test_invalid_depth_cells_make_their_bilinear_neighbourhood_unavailable():
    """docs/eval_protocol.md 2026-10-03: non-finite or non-positive predicted
    depth is unavailable; interpolation never crosses into such a cell."""
    from pipeline_adapters import sample_valid_only

    gm = BilinearGridMap(M)
    rng = np.random.default_rng(1)
    grid = rng.uniform(1.0, 2.0, size=(400, 512))
    out0, av0, n0 = sample_valid_only(gm, grid)
    assert n0 == 0 and av0 is gm.available and np.array_equal(out0, gm.sample(grid), equal_nan=True)

    bad = grid.copy()
    bad[100, 200], bad[250, 30], bad[10, 500] = -0.5, np.nan, 0.0
    out, av, n = sample_valid_only(gm, bad)
    assert n == 3
    X, Y = np.meshgrid(np.arange(ORIGINAL_W, dtype=np.float64), np.arange(ORIGINAL_H, dtype=np.float64))
    px, py = original_to_model(X.ravel(), Y.ravel(), M)
    touched = np.zeros(px.shape, dtype=bool)
    for (r, c) in [(100, 200), (250, 30), (10, 500)]:
        touched |= (np.abs(px - c) < 1) & (np.abs(py - r) < 1)  # the cell is one of the four neighbours
    assert np.array_equal(av, gm.available & ~touched)
    assert np.isnan(out[~av]).all() and np.isfinite(out[av]).all() and (out[av] > 0).all()
    assert np.array_equal(out[av], out0[av])  # pixels away from the invalid cells are unchanged
