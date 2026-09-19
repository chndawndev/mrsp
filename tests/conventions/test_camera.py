"""Unit tests for the C3VDv2 Scaramuzza camera model (src/geometry/camera.py).

See docs/conventions.md section 2 for the source equations this is checked
against.
"""
from pathlib import Path

import numpy as np
import pytest

from geometry.camera import CameraIntrinsics, pinhole_project, project, unproject

INTRINSICS_PATH = Path("/data1_ycao/chua/datasets/C3VDv2/camera_intrinsics.txt")


@pytest.fixture(scope="module")
def intr() -> CameraIntrinsics:
    return CameraIntrinsics.from_file(INTRINSICS_PATH)


def _pixel_grid(intr: CameraIntrinsics, n: int = 15) -> np.ndarray:
    """n x n grid of pixel coords spanning the full image, including corners."""
    xs = np.linspace(0, intr.width - 1, n)
    ys = np.linspace(0, intr.height - 1, n)
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx.ravel(), gy.ravel()], axis=-1)


def test_intrinsics_loaded_correctly(intr: CameraIntrinsics):
    assert intr.width == 1350
    assert intr.height == 1080
    assert intr.a0 == pytest.approx(767.733695862103)
    # a1 must never surface anywhere in the loaded object -- it is discarded
    # by the renderer (docs/conventions.md sec 2.2).
    assert not hasattr(intr, "a1")


def test_round_trip_unproject_project(intr: CameraIntrinsics):
    """unproject(px) -> ray; scale to a fixed depth; project(point) -> px.

    Must recover the original pixel to well under 0.1 px, across the whole
    image including all four corners.
    """
    px = _pixel_grid(intr, n=15)  # includes (0,0) and (w-1,h-1) corners exactly
    ray = unproject(px, intr)  # unit ray, +Z forward

    # Place points at a fixed camera-frame Z-depth in the middle of the
    # dataset's clamped depth range (README: depth clamped to [0, 100] mm).
    depth = 50.0
    points = ray * (depth / ray[..., 2:3])

    px_reprojected = project(points, intr)
    err = np.linalg.norm(px_reprojected - px, axis=-1)

    assert err.max() < 0.1, f"max round-trip error {err.max():.4f}px exceeds 0.1px"


def test_round_trip_corners_explicitly(intr: CameraIntrinsics):
    """Explicit corner check (the grid test already includes these, but a
    dedicated assertion makes a corner-case regression obvious)."""
    corners = np.array(
        [
            [0, 0],
            [intr.width - 1, 0],
            [0, intr.height - 1],
            [intr.width - 1, intr.height - 1],
        ],
        dtype=np.float64,
    )
    ray = unproject(corners, intr)
    points = ray * (50.0 / ray[..., 2:3])
    reprojected = project(points, intr)
    err = np.linalg.norm(reprojected - corners, axis=-1)
    assert err.max() < 0.1, f"corner round-trip error {err.max():.4f}px"


def test_pinhole_disparity(intr: CameraIntrinsics, capsys):
    """Compare the true omnidirectional model against a pinhole approximation
    (f = a0) at a fixed depth, and report the max pixel disparity -- this is
    a report of distortion strength, not a pass/fail on a specific bound,
    but we do assert disparity is large (confirming real fisheye-like
    distortion is present, i.e. the pinhole approximation is NOT adequate)
    and that it's near-zero at the image center (where distortion vanishes).
    """
    px = _pixel_grid(intr, n=15)
    ray = unproject(px, intr)
    depth = 50.0
    points = ray * (depth / ray[..., 2:3])

    px_pinhole = pinhole_project(points, intr)
    disparity = np.linalg.norm(px_pinhole - px, axis=-1)

    center_px = np.array([[intr.cx, intr.cy]])
    center_ray = unproject(center_px, intr)
    center_point = center_ray * (depth / center_ray[..., 2:3])
    center_disparity = np.linalg.norm(
        pinhole_project(center_point, intr) - center_px, axis=-1
    )[0]

    print(
        f"\npinhole vs. omnidirectional disparity at depth={depth}mm: "
        f"max={disparity.max():.2f}px, mean={disparity.mean():.2f}px, "
        f"center={center_disparity:.4f}px"
    )

    assert center_disparity < 0.1, "pinhole and omni model must agree at the image center"
    assert disparity.max() > 20, (
        "expected strong fisheye-like distortion far from center; "
        f"got only {disparity.max():.2f}px max disparity"
    )
