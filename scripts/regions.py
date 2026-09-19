#!/usr/bin/env python
"""Task: shift the unit of analysis from sequences to unobserved regions
(connected components). One row per unobserved component across all 169
registered sequences, streamed from zip archives (reuses
scripts/coverage_stats.py's discover_sequences() for the same 169-sequence
guarantee).

Also hashes each sequence's mesh vertex array (exact byte hash) to determine
how many truly INDEPENDENT meshes underlie the dataset -- v2/v3 sequences
are expected to share geometry (README: v3 reuses v2's trajectory), but this
should not be assumed for v1 vs v2/v3, or across phantom textures, without
checking (see docs/regions.md for what was actually found).

Writes:
  - results/regions.csv       one row per unobserved connected component
  - results/mesh_identity.csv one row per sequence: its mesh's content hash
                               (for the independent-mesh-count analysis)
"""
from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from coverage_stats import discover_sequences  # noqa: E402
from geometry.coverage_mesh import stream_coverage_mesh_from_zip  # noqa: E402
from geometry.mesh_stats import (  # noqa: E402
    area_to_diameter,
    boundary_edges,
    connected_components_of_subset,
    face_adjacency,
    face_areas,
    face_boundary_distances,
)
from render_coverage_views import principal_axes  # noqa: E402


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def component_compactness(comp_vertices: np.ndarray, comp_area: float) -> float:
    """area / (2D bounding-box area of the component, in its own best-fit
    plane). Compactness proxy: near 1 for a compact patch that fills its
    bounding box, near 0 for a long thin sliver. Uses SVD to find the
    component's local best-fit plane (2 dominant axes) since a mesh patch is
    only locally ~planar, not globally aligned with any fixed mesh axis.
    """
    if len(comp_vertices) < 3:
        return float("nan")
    centered = comp_vertices - comp_vertices.mean(axis=0)
    try:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return float("nan")
    proj = centered @ vt[:2].T
    w = proj[:, 0].max() - proj[:, 0].min()
    h = proj[:, 1].max() - proj[:, 1].min()
    bbox_area = w * h
    if bbox_area < 1e-9:
        return float("nan")
    return float(comp_area / bbox_area)


def analyze_sequence(name: str, path: Path) -> tuple[list[dict], dict]:
    mesh = stream_coverage_mesh_from_zip(path)
    n_faces = len(mesh.faces)

    areas = face_areas(mesh.vertices, mesh.faces)
    unobserved_mask = ~mesh.face_observed
    unobserved_area_total = float(areas[unobserved_mask].sum())

    adjacency = face_adjacency(mesh.vertices, mesh.faces)
    components = connected_components_of_subset(n_faces, adjacency, unobserved_mask)

    b_edges = boundary_edges(mesh.faces)
    face_dist = face_boundary_distances(mesh.vertices, mesh.faces, b_edges)

    centroid, axes = principal_axes(mesh.vertices)
    long_axis = axes[0]
    mesh_axis_coord = (mesh.vertices - centroid) @ long_axis
    axis_min, axis_max = float(mesh_axis_coord.min()), float(mesh_axis_coord.max())
    axis_range = axis_max - axis_min if axis_max > axis_min else 1.0

    mesh_hash = hashlib.md5(mesh.vertices.tobytes()).hexdigest()[:16]

    rows = []
    for comp in components:
        comp_area = float(areas[comp].sum())
        comp_vert_idx = np.unique(mesh.faces[comp])
        comp_verts = mesh.vertices[comp_vert_idx]
        d = face_dist[comp]
        comp_centroid = comp_verts.mean(axis=0)
        axis_pos = ((comp_centroid - centroid) @ long_axis - axis_min) / axis_range

        rows.append(
            {
                "Video Name": name,
                "area_mm2": comp_area,
                "equiv_diam_mm": area_to_diameter(comp_area),
                "n_faces": len(comp),
                "min_boundary_dist_mm": float(d.min()),
                "median_boundary_dist_mm": float(np.median(d)),
                "fully_interior": bool(np.mean(d <= 5.0) == 0.0),
                "compactness": component_compactness(comp_verts, comp_area),
                "centroid_axis_position": float(np.clip(axis_pos, 0.0, 1.0)),
            }
        )

    seq_meta = {
        "Video Name": name,
        "mesh_hash": mesh_hash,
        "n_vertices": len(mesh.vertices),
        "n_faces": n_faces,
        "n_unobserved_components": len(components),
        "unobserved_area_total_mm2": unobserved_area_total,
    }
    return rows, seq_meta


def main():
    seqs = discover_sequences()
    log(f"discovered {len(seqs)} sequences")

    all_rows = []
    mesh_meta = []
    t_start = time.time()
    for i, (name, kind, path) in enumerate(seqs):
        assert kind == "zip"
        t0 = time.time()
        rows, seq_meta = analyze_sequence(name, path)
        all_rows.extend(rows)
        mesh_meta.append(seq_meta)
        dt = time.time() - t0
        if i % 5 == 0 or i == len(seqs) - 1:
            elapsed = time.time() - t_start
            log(
                f"[{i+1}/{len(seqs)}] {name}: {len(rows)} regions, "
                f"mesh_hash={seq_meta['mesh_hash']} ({dt:.1f}s this seq, {elapsed:.0f}s elapsed)"
            )

    regions_df = pd.DataFrame(all_rows)
    index_df = pd.read_csv("docs/release_v1.csv").rename(columns={"Video Number ": "Video Number"})
    id_cols = ["Video Name", "Colon", "Segment", "Phantom Number", "Video Number", "Debris"]
    regions_df = regions_df.merge(index_df[id_cols], on="Video Name", how="left")

    # reorder: identifying columns first
    other_cols = [c for c in regions_df.columns if c not in id_cols]
    regions_df = regions_df[id_cols + other_cols]

    Path("results").mkdir(exist_ok=True)
    regions_df.to_csv("results/regions.csv", index=False)
    log(f"wrote results/regions.csv ({len(regions_df)} rows)")

    mesh_df = pd.DataFrame(mesh_meta).merge(index_df[id_cols], on="Video Name", how="left")
    mesh_df.to_csv("results/mesh_identity.csv", index=False)
    log(f"wrote results/mesh_identity.csv ({len(mesh_df)} rows)")

    log("ALL DONE")


if __name__ == "__main__":
    main()
