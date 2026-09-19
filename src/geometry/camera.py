"""Scaramuzza omnidirectional camera model for C3VDv2.

Implements project()/unproject() to match, bit-for-bit in intent, the CUDA
renderer that generated the dataset: DurrLab/C3VDv3 render/Render.cu
(pixel2Ray, forwardProjectVertex) and render/Intrinsics.h. See
docs/conventions.md section 2 for verbatim source citations.

Key facts this implementation depends on (see docs/conventions.md §2.1-2.5):
  - a1 is parsed from camera_intrinsics.txt but discarded by the renderer;
    the polynomial is a0 + a2*rho^2 + a3*rho^3 + a4*rho^4 (no linear term).
  - The stretch matrix is built from (c, d, e) as the CUDA renderer's
    column-major glm::mat2(c, d, e, 1.0), which is [[c, e], [d, 1]] -- NOT
    [[c, d], [e, 1]] as the repo's own Python reference implementation
    (utils/exampleDataLoader.py) uses. The CUDA form is authoritative
    because it is what actually generated the released data.
  - Unprojection divides by the stretch matrix (its inverse); projection
    multiplies by it (forward).
  - Pixel indexing: px = (col, row), origin at the center of pixel (0, 0),
    no half-pixel offset, no axis flips. Camera space is right-handed with
    +Z forward (the optical axis) -- consistent with depth being "camera
    frame Z-axis" (docs/conventions.md §1).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    cx: float
    cy: float
    a0: float
    a2: float
    a3: float
    a4: float
    c: float
    d: float
    e: float

    @property
    def center(self) -> np.ndarray:
        return np.array([self.cx, self.cy], dtype=np.float64)

    @property
    def stretch_mat(self) -> np.ndarray:
        # render/Intrinsics.h:75 -- glm::mat2(c, d, e, 1.0) is column-major,
        # i.e. column0=(c,d), column1=(e,1) -> matrix [[c, e], [d, 1]].
        return np.array([[self.c, self.e], [self.d, 1.0]], dtype=np.float64)

    @classmethod
    def from_file(cls, path: str | Path) -> "CameraIntrinsics":
        """Parse a camera_intrinsics.txt (';'-comments, 'key = value' lines).

        a1 is read (if present) and ignored, matching the renderer
        (RenderingModule.cpp:356 parses it; Render.cu:85 hardcodes `0*rho`).
        """
        vals: dict[str, str] = {}
        for line in Path(path).read_text().splitlines():
            line = line.split(";", 1)[0].strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            vals[key.strip()] = value.strip()
        return cls(
            width=int(float(vals["width"])),
            height=int(float(vals["height"])),
            cx=float(vals["cx"]),
            cy=float(vals["cy"]),
            a0=float(vals["a0"]),
            a2=float(vals["a2"]),
            a3=float(vals["a3"]),
            a4=float(vals["a4"]),
            c=float(vals["c"]),
            d=float(vals["d"]),
            e=float(vals["e"]),
        )


def _unproject_unnormalized(px: np.ndarray, intr: CameraIntrinsics) -> np.ndarray:
    """pixel (..., 2) [col, row] -> un-normalized camera-space ray (..., 3).

    Matches render/Render.cu:71-94 `pixel2Ray` up to (but not including) the
    final `normalize()` call. The z-component here is the raw polynomial
    value, which is what backproject_depth() needs to rescale a ray onto a
    known camera-frame Z-depth.
    """
    px = np.asarray(px, dtype=np.float64)
    uvp = px - intr.center
    inv_stretch = np.linalg.inv(intr.stretch_mat)
    # uvpp = inv(stretch_mat) @ uvp, batched: (M @ x) == (x @ M.T) for row vectors x.
    uvpp = uvp @ inv_stretch.T
    rho = np.linalg.norm(uvpp, axis=-1)
    z = intr.a0 + intr.a2 * rho**2 + intr.a3 * rho**3 + intr.a4 * rho**4
    return np.concatenate([uvpp, z[..., None]], axis=-1)


def unproject(px: np.ndarray, intr: CameraIntrinsics) -> np.ndarray:
    """pixel (..., 2) [col, row] -> unit camera-space ray direction (..., 3).

    Matches render/Render.cu:71-94 `pixel2Ray` exactly, including the final
    `glm::normalize`.
    """
    ray = _unproject_unnormalized(px, intr)
    return ray / np.linalg.norm(ray, axis=-1, keepdims=True)


def backproject_depth(px: np.ndarray, depth: np.ndarray, intr: CameraIntrinsics) -> np.ndarray:
    """(pixel, camera-frame Z-depth) -> camera-space 3D point (..., 3).

    depth is literally the camera-frame Z coordinate of the surface point
    (docs/conventions.md §1: "depth along the camera frame's Z-axis"), so we
    scale the un-normalized model ray (whose z-component is the raw
    polynomial value from pixel2Ray, before its final normalize) so that its
    Z equals `depth`. This is NOT the same as `unproject(px) * depth`
    (which would use depth as a ray length, i.e. radial distance) --
    z-depth requires dividing by the ray's own z-component first.
    """
    ray = _unproject_unnormalized(px, intr)
    scale = np.asarray(depth, dtype=np.float64) / ray[..., 2]
    return ray * scale[..., None]


def _solve_rho(m: float, intr: CameraIntrinsics, imag_tol: float = 1e-6) -> float:
    """Smallest strictly-positive real root of a4*rho^4+a3*rho^3+a2*rho^2-m*rho+a0=0.

    Matches the root selection of render/Quartic.cuh (used by
    render/Render.cu:49-67 `forwardProjectVertex`): smallest positive
    purely-real root, else 0.0. We use numpy.roots (a companion-matrix
    eigenvalue solve) rather than the CUDA closed-form Ferrari solver;
    numpy's roots carry tiny nonzero imaginary parts for genuinely-real
    roots (the CUDA code gets an exact 0.0 imaginary part from its
    closed-form method), so we treat |imag| < imag_tol as real.
    """
    coeffs = [intr.a4, intr.a3, intr.a2, -m, intr.a0]
    roots = np.roots(coeffs)
    real_positive = [r.real for r in roots if abs(r.imag) < imag_tol and r.real > 0]
    return min(real_positive) if real_positive else 0.0


def project(v: np.ndarray, intr: CameraIntrinsics) -> np.ndarray:
    """camera-space point(s) (..., 3) -> pixel (..., 2) [col, row].

    Matches render/Render.cu:49-67 `forwardProjectVertex` exactly (the exact
    algebraic inverse of unproject's polynomial, via closed-form quartic
    root-finding -- see docs/conventions.md §2.4).
    """
    v = np.asarray(v, dtype=np.float64)
    single = v.ndim == 1
    v2 = np.atleast_2d(v)

    r_xy = np.sqrt(v2[..., 0] ** 2 + v2[..., 1] ** 2) + 1e-20
    m = v2[..., 2] / r_xy
    rho = np.array([_solve_rho(mi, intr) for mi in m])

    raw_u = v2[..., 0] / r_xy * rho
    raw_v = v2[..., 1] / r_xy * rho
    raw_uv = np.stack([raw_u, raw_v], axis=-1)

    px = raw_uv @ intr.stretch_mat.T + intr.center
    return px[0] if single else px


def pinhole_project(v: np.ndarray, intr: CameraIntrinsics) -> np.ndarray:
    """Pinhole approximation: px = f * (X/Z, Y/Z) + (cx, cy), f = a0.

    a0 is the Taylor-series value of the omnidirectional polynomial at
    rho=0, i.e. the model's effective focal length on-axis -- this is the
    natural pinhole comparison point, ignoring the omnidirectional distortion
    polynomial (a2, a3, a4) and the stretch matrix (c, d, e ~= identity).
    Used only to quantify how strong the fisheye-like distortion is; not
    part of the dataset's actual camera model.
    """
    v = np.asarray(v, dtype=np.float64)
    xy_over_z = v[..., :2] / v[..., 2:3]
    return intr.a0 * xy_over_z + intr.center
