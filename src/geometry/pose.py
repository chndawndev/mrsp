"""pose.txt loading with both candidate flattening interpretations.

docs/conventions.md section 3 confirms (README quote) that pose.txt holds,
per frame, "each frame's flattened homogeneous camera-to-world
transformation matrix (row major order)" -- but does not by itself prove
which 4x4 layout the 16 numbers unflatten to. Empirically, a raw row-major
numpy reshape of one line gives a matrix with translation in the LAST ROW
(positions 12-14) and zeros in the last COLUMN of the first three rows --
not the textbook [R t; 0 0 0 1] layout with translation in the last column.

We therefore expose both interpretations named exactly as requested:
  (a) "raw"        -- A = reshape(16, (4,4)); world = A @ [cam; 1]
  (b) "transposed"  -- A.T @ [cam; 1]

and let the oracle check decide empirically which one is correct, rather
than assuming.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from .coverage_mesh import _resolve_zip_member


def _load_poses_from_lines(lines: list[str]) -> np.ndarray:
    values = np.array(
        [[float(x) for x in line.split(",")] for line in lines if line.strip()], dtype=np.float64
    )
    if values.shape[1] != 16:
        raise ValueError(f"expected 16 values per line, got {values.shape[1]}")
    return values.reshape(-1, 4, 4)


def stream_poses_from_zip(archive_path: str | Path, member: str = "pose.txt") -> np.ndarray:
    """Load pose.txt directly from a zip archive member via `unzip -p`,
    without extracting the archive to disk. Same layout/caveats as
    load_poses() -- see module docstring.

    Resolves the actual internal path first (like
    coverage_mesh.stream_coverage_mesh_from_zip) rather than assuming
    `member` sits at the archive root: c1_cecum_t1_v3.zip is a confirmed
    one-off exception that wraps its contents in a `c1_cecum_t1_v3/`
    subfolder, and a naive `unzip -p archive pose.txt` fails loudly on it
    (exit code 11) rather than silently returning wrong data -- this bit us
    once already for coverage_mesh.obj (docs/coverage_stats.md) and again
    here for pose.txt before this fix.
    """
    resolved_member = _resolve_zip_member(archive_path, member)
    result = subprocess.run(
        ["unzip", "-p", str(archive_path), resolved_member], capture_output=True, text=True, check=True
    )
    return _load_poses_from_lines(result.stdout.splitlines())


def load_poses(path: str | Path) -> np.ndarray:
    """Load pose.txt -> (N, 4, 4) array, each row-major-reshaped as given.

    This is interpretation (a) ("raw"): a straight row-major reshape of the
    16 comma-separated floats on each line. Does NOT assume this is a valid
    [R t; 0 0 0 1] transform -- see module docstring.
    """
    return _load_poses_from_lines(Path(path).read_text().splitlines())


def transform_points(points: np.ndarray, matrices: np.ndarray, interpretation: str) -> np.ndarray:
    """Apply a per-frame 4x4 camera-to-world matrix to camera-space points.

    points: (N, 3) camera-space points for ONE frame.
    matrices: (4, 4) or (N, 4, 4) raw ("a") matrices from load_poses().
    interpretation: "raw" (use matrices as-is) or "transposed" (use matrices.T).
    """
    if interpretation not in ("raw", "transposed"):
        raise ValueError(f"unknown interpretation {interpretation!r}")

    M = matrices if interpretation == "raw" else np.swapaxes(matrices, -1, -2)

    ones = np.ones((*points.shape[:-1], 1), dtype=np.float64)
    homog = np.concatenate([points, ones], axis=-1)  # (N, 4)

    if M.ndim == 2:
        world = homog @ M.T  # (N,4) @ (4,4) == (M @ p) per row
    else:
        # M: (N, 4, 4), homog: (N, 4) -> per-point matrix-vector product.
        world = np.einsum("nij,nj->ni", M, homog)

    return world[..., :3]
