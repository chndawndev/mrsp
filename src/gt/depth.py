"""GT depth TIFF loading, streamed directly from a sequence's zip archive
(no bulk extraction -- `CLAUDE.md`). `depth_mm = raw / 65535 * 100`
(linear, 0-100mm, camera-frame Z-depth); `raw==0` (no hit) and
`raw==65535` (clamped) are masked, not valid depth readings (`CLAUDE.md`'s
frozen depth convention).
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Iterator

import numpy as np
import tifffile


def depth_members(zf: zipfile.ZipFile) -> list[str]:
    """Sorted member names of a sequence's `depth/*_depth.tiff` files --
    sort order matches `pose.txt` line order (zero-padded frame index),
    the same assumption already used throughout this project's earlier
    oracle-gap scripts (e.g. `scratch/pipelines/oracle_gap_vignette_169.py`)."""
    return sorted(
        n
        for n in zf.namelist()
        if n.endswith("_depth.tiff") and ("/depth/" in n or n.startswith("depth/"))
    )


def stream_raw_depth_frames_from_zip(archive_path: str | Path) -> Iterator[np.ndarray]:
    """Yields raw uint16 `(H, W)` depth arrays, one per frame, in
    `pose.txt` frame order, streamed directly from the zip (no extraction)."""
    with zipfile.ZipFile(archive_path) as zf:
        members = depth_members(zf)
        for member in members:
            yield tifffile.imread(io.BytesIO(zf.read(member)))


def depth_to_mm(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """raw uint16 depth -> `(depth_mm float64, valid bool)`. `depth_mm` is
    only meaningful where `valid` is True (`raw != 0` and `raw != 65535`,
    `CLAUDE.md`'s frozen depth convention)."""
    depth_mm = raw.astype(np.float64) / 65535.0 * 100.0
    valid = (raw != 0) & (raw != 65535)
    return depth_mm, valid
