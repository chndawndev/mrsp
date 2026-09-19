#!/usr/bin/env python
"""Open-end artifact check: for 6 sequences (3 Open End Visible=yes, 3=no,
spread across segments), find the 5 largest unobserved connected components
of coverage_mesh.obj, render them highlighted (2 exterior views + 1 cut-open
view via scripts/render_coverage_views.py), and measure each component's
distance to the mesh's boundary (open-end) edges.

Writes results/openend_check/{<seq>_view1,_view2,_cutopen}.png,
results/openend_check/component_distances.csv, and (separately, after
inspecting results) docs/openend_artifact_check.md.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from geometry.coverage_mesh import stream_coverage_mesh_from_zip  # noqa: E402
from geometry.mesh_stats import (  # noqa: E402
    boundary_edges,
    boundary_loops,
    connected_components_of_subset,
    face_adjacency,
    face_areas,
    face_boundary_distances,
)
from render_coverage_views import principal_axes, render_three_views  # noqa: E402

REGISTERED_DIR = Path("/data1_ycao/chua/datasets/C3VDv2/registered_videos")
OUT_DIR = Path("results/openend_check")

SEQUENCES = [
    # (video_name, segment_label, open_end_visible)
    ("c2_transverse1_t2_v1", "transverse1", "yes"),
    ("c1_ascending_t4_v2", "ascending", "yes"),
    ("c2_rectum_t1_v3", "rectum", "yes"),
    ("c2_transverse2_t3_v2", "transverse2", "no"),
    ("c2_cecum_t3_v2", "cecum", "no"),
    ("c1_sigmoid1_t1_v3", "sigmoid1", "no"),
]

RANK_COLORS = [
    (220, 0, 0, 255),  # 1st largest: red
    (255, 140, 0, 255),  # 2nd: orange
    (230, 190, 0, 255),  # 3rd: gold
    (30, 100, 220, 255),  # 4th: blue
    (150, 50, 200, 255),  # 5th: purple
]
BASE_COLOR = (190, 190, 190, 255)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def analyze_and_render(name: str, segment: str, open_end: str) -> list[dict]:
    archive = REGISTERED_DIR / f"{name}.zip"
    t0 = time.time()
    mesh = stream_coverage_mesh_from_zip(archive)
    log(f"{name}: loaded mesh {len(mesh.faces)} faces ({time.time()-t0:.1f}s)")

    areas = face_areas(mesh.vertices, mesh.faces)
    unobserved_mask = ~mesh.face_observed
    adjacency = face_adjacency(mesh.vertices, mesh.faces)
    components = connected_components_of_subset(len(mesh.faces), adjacency, unobserved_mask)
    log(f"{name}: {len(components)} unobserved components")

    comp_areas = [float(areas[c].sum()) for c in components]
    order = np.argsort(comp_areas)[::-1][:5]
    top5 = [components[i] for i in order]
    top5_areas = [comp_areas[i] for i in order]

    b_edges = boundary_edges(mesh.faces)
    face_dist = face_boundary_distances(mesh.vertices, mesh.faces, b_edges)
    log(f"{name}: {len(b_edges)} boundary edges, boundary distances computed")

    # Separate the two open-end boundary loops so we can tell "near which
    # end" apart, not just "near a boundary" -- colon segments have two open
    # ends (insertion + far end), and README-defined "Open End Visible"
    # refers only to the far one, so pooling both ends together (as the
    # task's literal min/median-distance spec does) can't distinguish them.
    loops = boundary_loops(b_edges)
    centroid, axes = principal_axes(mesh.vertices)
    long_axis = axes[0]
    loop_long_coord = [float(np.mean((mesh.vertices[loop] - centroid) @ long_axis)) for loop in loops]
    loop_order = np.argsort(loop_long_coord)  # loop_order[0] = "low" end, [-1] = "high" end
    loop_label = {int(loop_order[0]): "end_low", int(loop_order[-1]): "end_high"}
    for i in range(len(loops)):
        loop_label.setdefault(i, f"extra_hole_{i}")
    log(f"{name}: {len(loops)} boundary loop(s) -- expect 2 (the phantom's two open ends)")

    from scipy.spatial import cKDTree

    loop_trees = [cKDTree(mesh.vertices[loop]) for loop in loops]

    rows = []
    face_colors = np.tile(np.array([BASE_COLOR], dtype=np.uint8), (len(mesh.faces), 1))
    for rank, (comp, area) in enumerate(zip(top5, top5_areas), start=1):
        d = face_dist[comp]
        comp_verts = np.unique(mesh.faces[comp])
        comp_centroid = mesh.vertices[comp_verts].mean(axis=0)
        per_loop_dist = [float(tree.query(comp_centroid)[0]) for tree in loop_trees]
        nearest_loop = int(np.argmin(per_loop_dist))
        row = {
            "Video Name": name,
            "Segment": segment,
            "Open End Visible": open_end,
            "component_rank": rank,
            "n_faces": len(comp),
            "area_mm2": area,
            "equiv_diam_mm": float(2.0 * np.sqrt(area / np.pi)),
            "min_boundary_dist_mm": float(d.min()),
            "median_boundary_dist_mm": float(np.median(d)),
            "frac_faces_within_5mm_of_boundary": float(np.mean(d <= 5.0)),
            "n_boundary_loops": len(loops),
            "nearest_loop_label": loop_label[nearest_loop],
            "nearest_loop_centroid_dist_mm": per_loop_dist[nearest_loop],
        }
        rows.append(row)
        face_colors[comp] = RANK_COLORS[rank - 1]
        log(
            f"  {name} comp#{rank}: area={area:.1f}mm2 diam={row['equiv_diam_mm']:.1f}mm "
            f"min_dist={row['min_boundary_dist_mm']:.2f}mm median_dist={row['median_boundary_dist_mm']:.2f}mm "
            f"frac<5mm={row['frac_faces_within_5mm_of_boundary']:.2f}"
        )

    out_prefix = OUT_DIR / name
    paths = render_three_views(mesh.vertices, mesh.faces, face_colors, out_prefix)
    log(f"{name}: rendered {list(paths.keys())}")

    return rows


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for name, segment, open_end in SEQUENCES:
        rows = analyze_and_render(name, segment, open_end)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    out_csv = OUT_DIR / "component_distances.csv"
    df.to_csv(out_csv, index=False)
    log(f"wrote {out_csv}")
    log("ALL DONE")


if __name__ == "__main__":
    main()
