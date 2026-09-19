"""Fast loader for C3VDv2 coverage_mesh.obj, preserving per-face vt (coverage) labels.

docs/conventions.md section 4 / the dataset inventory established that these
OBJ files use exactly 2 `vt` entries (index 1 = observed, index 2 =
unobserved) referenced per-face (same vt index for all 3 vertices of a
face), with no `vn` lines. A generic OBJ loader (e.g. trimesh's) may
reprocess/merge/reorder vertices and lose the face<->vt correspondence, so
this module parses the handful of relevant line types directly.

Also supports streaming a coverage_mesh.obj straight out of a zip archive
member (via `unzip -p`) without extracting the archive, for bulk dataset
scans -- see stream_coverage_mesh_from_zip().
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass
class CoverageMesh:
    vertices: np.ndarray  # (N, 3) float64
    faces: np.ndarray  # (M, 3) int64, 0-indexed into vertices
    face_observed: np.ndarray  # (M,) bool -- True if vt index == 1 (observed)


def parse_coverage_mesh_lines(lines: Iterable[str]) -> CoverageMesh:
    verts: list[list[float]] = []
    faces: list[list[int]] = []
    face_vt: list[int] = []

    for line in lines:
        if line.startswith("v "):
            parts = line.split()
            verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif line.startswith("f "):
            parts = line.split()[1:]
            idx = []
            vt = None
            for p in parts:
                fields = p.split("/")
                idx.append(int(fields[0]) - 1)
                if len(fields) > 1 and fields[1]:
                    vt = int(fields[1])
            faces.append(idx)
            face_vt.append(vt if vt is not None else 1)
        # vt lines themselves are not needed: only 2 fixed entries exist
        # (index 1 -> observed, index 2 -> unobserved), confirmed in
        # docs/data_inventory.md section 3.

    vertices = np.asarray(verts, dtype=np.float64)
    faces_arr = np.asarray(faces, dtype=np.int64)
    face_vt_arr = np.asarray(face_vt, dtype=np.int64)
    face_observed = face_vt_arr == 1

    return CoverageMesh(vertices=vertices, faces=faces_arr, face_observed=face_observed)


def load_coverage_mesh(path: str | Path) -> CoverageMesh:
    with open(path) as f:
        return parse_coverage_mesh_lines(f)


def _resolve_zip_member(archive_path: str | Path, basename: str) -> str:
    """Find the exact internal zip path for a file named `basename`.

    Most C3VDv2 registered_videos archives store coverage_mesh.obj at the
    archive root, but at least one (c1_cecum_t1_v3.zip, confirmed the only
    exception found) wraps everything in a `<video_name>/` subfolder
    instead. Rather than assume a fixed layout, list the archive and match
    on the basename.
    """
    result = subprocess.run(
        ["unzip", "-Z1", str(archive_path)], capture_output=True, text=True, check=True
    )
    candidates = [line for line in result.stdout.splitlines() if line.endswith(basename)]
    if not candidates:
        raise RuntimeError(f"no member named {basename!r} found in {archive_path}")
    if len(candidates) > 1:
        raise RuntimeError(f"ambiguous member {basename!r} in {archive_path}: {candidates}")
    return candidates[0]


def stream_coverage_mesh_from_zip(
    archive_path: str | Path, member: str = "coverage_mesh.obj"
) -> CoverageMesh:
    """Parse coverage_mesh.obj directly from a zip archive member, via
    `unzip -p`, without extracting the archive to disk."""
    resolved_member = _resolve_zip_member(archive_path, member)
    proc = subprocess.Popen(
        ["unzip", "-p", str(archive_path), resolved_member],
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1 << 20,
    )
    try:
        assert proc.stdout is not None
        mesh = parse_coverage_mesh_lines(proc.stdout)
    finally:
        proc.stdout.close() if proc.stdout else None
        ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"unzip -p {archive_path} {resolved_member} failed with code {ret}")
    return mesh
