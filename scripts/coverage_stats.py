#!/usr/bin/env python
"""Task B Part 1: GT coverage_mesh.obj stats for every registered_videos
sequence (c1/c2 only), streamed out of each zip archive without extraction.

For each sequence: face count/area, observed vs. unobserved (by count and by
area), connected components of unobserved faces (count, area of each,
top-5 largest, count exceeding 5mm/10mm equivalent diameter), and -- for
every unobserved face's Euclidean distance to the nearest mesh boundary
(open-end) vertex -- how much of the unobserved area sits near a boundary
vs. is fully interior (see docs/openend_artifact_check.md for the 6-sequence
qualitative investigation this generalizes to all 171 sequences).

Writes:
  - results/coverage_stats.csv     one row per sequence, joined with
                                    docs/release_v1.csv on "Video Name"
  - results/coverage_components.json  full per-sequence unobserved-component
                                    area lists (heavier detail than the CSV)

See docs/coverage_stats.md for the interpreted summary written after this
script runs, and src/geometry/{coverage_mesh,mesh_stats}.py for the
geometry code this reuses.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from geometry.coverage_mesh import stream_coverage_mesh_from_zip  # noqa: E402
from geometry.mesh_stats import (  # noqa: E402
    area_to_diameter,
    boundary_edges,
    connected_components_of_subset,
    face_adjacency,
    face_areas,
    face_boundary_distances,
)

DATASET_ROOT = Path("/data1_ycao/chua/datasets/C3VDv2")
REGISTERED_DIR = DATASET_ROOT / "registered_videos"
DIAMETER_THRESHOLDS_MM = (5.0, 10.0)
BOUNDARY_DIST_THRESHOLDS_MM = (5.0, 10.0)
LARGE_COMPONENT_DIAM_MM = 10.0


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


EXPECTED_N_SEQUENCES = 169


def discover_sequences() -> list[tuple[str, str, Path]]:
    """Returns list of (video_name, kind, path); kind is always 'zip'.

    Sequences are discovered ONLY from *.zip archives under registered_videos
    -- never from extracted directories, which can be incidental/temporary
    (e.g. c1_transverse1_t1_v1/ and c2_transverse1_t1_v1/ were previously
    present alongside their matching .zip archives and have since been
    removed; they were never a reliable data source). Deduplicated by
    sequence name and asserted against the known-good count of 169 c1/c2
    registered sequences.
    """
    seen: dict[str, Path] = {}
    for zpath in sorted(REGISTERED_DIR.glob("*.zip")):
        name = zpath.stem
        if name.startswith("c0_"):
            continue
        seen[name] = zpath  # dedupe by name; last write wins (paths are equivalent)

    seqs = [(name, "zip", path) for name, path in sorted(seen.items())]
    assert len(seqs) == EXPECTED_N_SEQUENCES, (
        f"expected {EXPECTED_N_SEQUENCES} c1/c2 registered_videos sequences, found {len(seqs)}"
    )
    return seqs


def analyze_sequence(name: str, path: Path) -> tuple[dict, list[float]]:
    mesh = stream_coverage_mesh_from_zip(path)

    areas = face_areas(mesh.vertices, mesh.faces)
    total_area = float(areas.sum())
    observed_mask = mesh.face_observed
    unobserved_mask = ~observed_mask

    n_faces = len(mesh.faces)
    n_observed = int(observed_mask.sum())
    n_unobserved = int(unobserved_mask.sum())

    observed_area = float(areas[observed_mask].sum())
    unobserved_area = float(areas[unobserved_mask].sum())

    adjacency = face_adjacency(mesh.vertices, mesh.faces)
    components = connected_components_of_subset(n_faces, adjacency, unobserved_mask)

    b_edges = boundary_edges(mesh.faces)
    face_dist = face_boundary_distances(mesh.vertices, mesh.faces, b_edges)

    # Per-component stats computed once, reused for both the top5/diameter
    # counts (existing) and the new boundary-distance breakdown.
    comp_stats = []
    for comp in components:
        comp_area_arr = areas[comp]
        comp_area = float(comp_area_arr.sum())
        comp_dist = face_dist[comp]
        comp_stats.append(
            {
                "area": comp_area,
                "diam": area_to_diameter(comp_area),
                "area_within": {
                    thr: float(comp_area_arr[comp_dist <= thr].sum()) for thr in BOUNDARY_DIST_THRESHOLDS_MM
                },
                "frac_faces_within_5mm": float(np.mean(comp_dist <= 5.0)) if len(comp_dist) else 0.0,
            }
        )
    comp_stats.sort(key=lambda c: c["area"], reverse=True)

    comp_areas = [c["area"] for c in comp_stats]
    n_components = len(comp_areas)
    top5 = (comp_areas + [0.0] * 5)[:5]

    diam_counts = {}
    for thr in DIAMETER_THRESHOLDS_MM:
        area_thr = np.pi * (thr / 2.0) ** 2
        diam_counts[thr] = int(sum(1 for a in comp_areas if a >= area_thr))

    # Boundary-distance breakdown of unobserved area (all components).
    unobs_area_within = {
        thr: sum(c["area_within"][thr] for c in comp_stats) for thr in BOUNDARY_DIST_THRESHOLDS_MM
    }

    # Same, restricted to "large" components (equivalent diameter > 10mm).
    large_comps = [c for c in comp_stats if c["diam"] > LARGE_COMPONENT_DIAM_MM]
    large_total_area = sum(c["area"] for c in large_comps)
    large_area_within = {
        thr: sum(c["area_within"][thr] for c in large_comps) for thr in BOUNDARY_DIST_THRESHOLDS_MM
    }

    # Fully-interior components: 0% of their faces within 5mm of any boundary.
    fully_interior = [c for c in comp_stats if c["frac_faces_within_5mm"] == 0.0]

    row = {
        "Video Name": name,
        "n_faces": n_faces,
        "total_area_mm2": total_area,
        "n_observed_faces": n_observed,
        "n_unobserved_faces": n_unobserved,
        "observed_area_mm2": observed_area,
        "unobserved_area_mm2": unobserved_area,
        "unobserved_frac_by_count": n_unobserved / n_faces if n_faces else float("nan"),
        "unobserved_frac_by_area": unobserved_area / total_area if total_area else float("nan"),
        "n_unobserved_components": n_components,
        "comp_area_top1_mm2": top5[0],
        "comp_area_top2_mm2": top5[1],
        "comp_area_top3_mm2": top5[2],
        "comp_area_top4_mm2": top5[3],
        "comp_area_top5_mm2": top5[4],
        "n_components_gt5mm_diam": diam_counts[5.0],
        "n_components_gt10mm_diam": diam_counts[10.0],
        "largest_component_equiv_diam_mm": area_to_diameter(comp_areas[0]) if comp_areas else 0.0,
        "n_boundary_edges": len(b_edges),
        "unobs_area_within_5mm_of_boundary_mm2": unobs_area_within[5.0],
        "unobs_area_within_10mm_of_boundary_mm2": unobs_area_within[10.0],
        "unobs_area_within_5mm_of_boundary_frac": (
            unobs_area_within[5.0] / unobserved_area if unobserved_area else float("nan")
        ),
        "unobs_area_within_10mm_of_boundary_frac": (
            unobs_area_within[10.0] / unobserved_area if unobserved_area else float("nan")
        ),
        "large_components_total_area_mm2": large_total_area,
        "large_unobs_area_within_5mm_of_boundary_frac": (
            large_area_within[5.0] / large_total_area if large_total_area else float("nan")
        ),
        "large_unobs_area_within_10mm_of_boundary_frac": (
            large_area_within[10.0] / large_total_area if large_total_area else float("nan")
        ),
        "n_fully_interior_components": len(fully_interior),
        "fully_interior_area_mm2": sum(c["area"] for c in fully_interior),
    }
    return row, comp_areas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-csv", default="docs/release_v1.csv")
    ap.add_argument("--out-csv", default="results/coverage_stats.csv")
    ap.add_argument("--components-json", default="results/coverage_components.json")
    ap.add_argument("--limit", type=int, default=None, help="process only first N sequences (debug)")
    args = ap.parse_args()

    seqs = discover_sequences()
    log(f"discovered {len(seqs)} c1/c2 registered_videos sequences")
    if args.limit:
        seqs = seqs[: args.limit]
        log(f"--limit set: processing only {len(seqs)}")

    rows = []
    components_by_seq = {}
    t_start = time.time()
    for i, (name, kind, path) in enumerate(seqs):
        assert kind == "zip"
        t0 = time.time()
        try:
            row, comp_areas = analyze_sequence(name, path)
        except Exception as exc:
            log(f"ERROR on {name}: {exc}")
            raise
        rows.append(row)
        components_by_seq[name] = comp_areas
        dt = time.time() - t0
        if i % 5 == 0 or i == len(seqs) - 1:
            elapsed = time.time() - t_start
            log(
                f"[{i+1}/{len(seqs)}] {name}: {row['n_faces']} faces, "
                f"unobs_frac_area={row['unobserved_frac_by_area']:.4f}, "
                f"{row['n_unobserved_components']} components ({dt:.1f}s this seq, "
                f"{elapsed:.0f}s elapsed)"
            )

    df = pd.DataFrame(rows)

    index_df = pd.read_csv(args.index_csv)
    merged = df.merge(index_df, on="Video Name", how="left", indicator=True)

    n_matched = int((merged["_merge"] == "both").sum())
    n_unmatched = int((merged["_merge"] == "left_only").sum())
    log(f"join with {args.index_csv}: {n_matched} matched, {n_unmatched} unmatched (no index row)")
    if n_unmatched:
        log(f"unmatched Video Names: {merged.loc[merged['_merge']=='left_only','Video Name'].tolist()}")
    merged = merged.drop(columns=["_merge"])

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out_csv, index=False)
    log(f"wrote {args.out_csv} ({len(merged)} rows)")

    Path(args.components_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.components_json).write_text(json.dumps(components_by_seq))
    log(f"wrote {args.components_json}")

    log("ALL DONE")


if __name__ == "__main__":
    main()
