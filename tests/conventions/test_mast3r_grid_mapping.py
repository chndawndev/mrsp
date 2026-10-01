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
    px, py = original_to_model(np.array([0.0, 1350.0 / 2]), np.array([0.0, 1080.0 / 2]), M)
    assert px[0] == 0.0 and abs(py[0] - (-5.0)) < 1e-12
    assert abs(px[1] - 256.0) < 1e-9 and abs(py[1] - (540.0 / M["scale_h"] - 5.0)) < 1e-9


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
    # top crop band: rows with py < 0 have no depth
    assert not gm.available.reshape(ORIGINAL_H, ORIGINAL_W)[0].any()


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
