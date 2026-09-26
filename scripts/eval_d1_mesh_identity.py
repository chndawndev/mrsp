#!/usr/bin/env python
"""D1 Stage B pre-flight 4: mesh geometry identity, descriptive only --
changes no analysis. For every pair of distinct mesh hashes within each
(Colon, Segment) group, and within each (Colon, Segment, Phantom Number)
group, rigidly align the two coverage meshes and report symmetric
point-to-surface distance. Does not conclude which grouping level is
"independent" -- reports distributions only.

CPU-only (no GPU needed), read-only against the dataset zips.

Method: PCA-plus-centroid initial alignment (tried under all 4
determinant-preserving sign combinations, best picked by a quick
subsampled nearest-neighbor cost, since eigenvector sign/axis-order
ambiguity can trap ICP in a mirrored local minimum), refined with
trimesh.registration.icp on a subsample (both point sets are large --
up to ~420k vertices -- so ICP itself runs on a bounded random subsample;
the final reported distances also use a bounded subsample per mesh,
documented as a Monte Carlo estimate, not an exhaustive comparison).
"""
from __future__ import annotations

import argparse
import itertools
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import trimesh

REPO = Path("/data1_ycao/chua/projects/mrsp")
sys.path.insert(0, str(REPO / "src"))

from geometry.coverage_mesh import stream_coverage_mesh_from_zip  # noqa: E402

DATASET_ROOT = Path("/data1_ycao/chua/datasets/C3VDv2/registered_videos")
MESH_IDENTITY_CSV = REPO / "results/mesh_identity.csv"
OUT_CSV = REPO / "results/d1/mesh_identity_check.csv"
ICP_SUBSAMPLE = 20000
DIST_SUBSAMPLE = 20000
SEED = 20260925


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}", flush=True)


def pca_axes(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centroid = points.mean(axis=0)
    centered = points - centroid
    cov = (centered.T @ centered) / len(points)
    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending eigenvalue order
    order = np.argsort(eigvals)[::-1]
    axes = eigvecs[:, order]  # columns = principal axes, descending variance
    return centroid, axes


def best_pca_initial(src: np.ndarray, dst: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """4x4 transform src->dst via centroid+PCA-axis alignment, trying every
    determinant-preserving sign flip of the axes and keeping the one with
    lowest subsampled nearest-neighbor cost (guards against the mirrored
    local minimum ICP alone can't escape)."""
    from scipy.spatial import cKDTree

    c_src, axes_src = pca_axes(src)
    c_dst, axes_dst = pca_axes(dst)

    n = min(len(src), 3000)
    idx = rng.choice(len(src), n, replace=False)
    sample = src[idx]
    tree = cKDTree(dst[rng.choice(len(dst), min(len(dst), 5000), replace=False)])

    best_cost, best_T = np.inf, None
    for signs in itertools.product([1, -1], repeat=2):
        S = np.diag([signs[0], signs[1], signs[0] * signs[1]])  # keep det=+1
        R = axes_dst @ S @ axes_src.T
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = c_dst - R @ c_src
        transformed = (R @ sample.T).T + T[:3, 3]
        dists, _ = tree.query(transformed)
        cost = float(np.mean(dists ** 2))
        if cost < best_cost:
            best_cost, best_T = cost, T
    return best_T


def symmetric_distance(mesh_a: trimesh.Trimesh, mesh_b: trimesh.Trimesh, rng: np.random.Generator) -> dict:
    from trimesh.proximity import ProximityQuery

    pq_b = ProximityQuery(mesh_b)
    pq_a = ProximityQuery(mesh_a)
    idx_a = rng.choice(len(mesh_a.vertices), min(len(mesh_a.vertices), DIST_SUBSAMPLE), replace=False)
    idx_b = rng.choice(len(mesh_b.vertices), min(len(mesh_b.vertices), DIST_SUBSAMPLE), replace=False)
    _, d_a_to_b, _ = pq_b.on_surface(mesh_a.vertices[idx_a])
    _, d_b_to_a, _ = pq_a.on_surface(mesh_b.vertices[idx_b])
    pooled = np.concatenate([d_a_to_b, d_b_to_a])
    return {
        "mean_mm": float(pooled.mean()), "p95_mm": float(np.percentile(pooled, 95)),
        "a_to_b_mean_mm": float(d_a_to_b.mean()), "b_to_a_mean_mm": float(d_b_to_a.mean()),
        "n_sampled_a": len(idx_a), "n_sampled_b": len(idx_b),
    }


def compare_pair(name_a: str, name_b: str, rng: np.random.Generator) -> dict:
    t0 = time.time()
    mesh_a = stream_coverage_mesh_from_zip(DATASET_ROOT / f"{name_a}.zip")
    mesh_b = stream_coverage_mesh_from_zip(DATASET_ROOT / f"{name_b}.zip")

    n_sub_a = min(len(mesh_a.vertices), ICP_SUBSAMPLE)
    n_sub_b = min(len(mesh_b.vertices), ICP_SUBSAMPLE)
    sub_a = mesh_a.vertices[rng.choice(len(mesh_a.vertices), n_sub_a, replace=False)]
    sub_b = mesh_b.vertices[rng.choice(len(mesh_b.vertices), n_sub_b, replace=False)]

    initial = best_pca_initial(sub_a, sub_b, rng)
    matrix, _, icp_cost = trimesh.registration.icp(sub_a, sub_b, initial=initial, max_iterations=50)

    tri_a = trimesh.Trimesh(vertices=mesh_a.vertices, faces=mesh_a.faces, process=False)
    tri_a.apply_transform(matrix)
    tri_b = trimesh.Trimesh(vertices=mesh_b.vertices, faces=mesh_b.faces, process=False)

    dist = symmetric_distance(tri_a, tri_b, rng)
    elapsed = time.time() - t0
    return {
        "sequence_a": name_a, "sequence_b": name_b,
        "n_vertices_a": len(mesh_a.vertices), "n_vertices_b": len(mesh_b.vertices),
        "n_faces_a": len(mesh_a.faces), "n_faces_b": len(mesh_b.faces),
        "icp_cost": float(icp_cost),
        **dist,
        "elapsed_s": elapsed,
    }


def _compare_task(args) -> dict:
    level, group_key, h_a, h_b, seq_a, seq_b, task_seed = args
    rng = np.random.default_rng(task_seed)
    row = compare_pair(seq_a, seq_b, rng)
    row.update({"level": level, "group": str(group_key), "mesh_hash_a": h_a, "mesh_hash_b": h_b})
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=12,
                         help="CPU-only, independent pairs -- parallel across pairs. "
                              "Shared 48-core box; 12 is moderate, not the full count.")
    args = parser.parse_args()

    mesh_df = pd.read_csv(MESH_IDENTITY_CSV)
    root_rng = np.random.default_rng(SEED)

    # one representative sequence per mesh_hash
    rep = mesh_df.drop_duplicates("mesh_hash").set_index("mesh_hash")["Video Name"].to_dict()

    tasks = []
    for level, cols in [("colon_segment", ["Colon", "Segment"]),
                         ("colon_segment_phantom", ["Colon", "Segment", "Phantom Number"])]:
        groups = mesh_df.groupby(cols)["mesh_hash"].unique()
        for group_key, hashes in groups.items():
            for h_a, h_b in itertools.combinations(sorted(hashes), 2):
                task_seed = int(root_rng.integers(0, 2**31 - 1))
                tasks.append((level, group_key, h_a, h_b, rep[h_a], rep[h_b], task_seed))
    log(f"{len(tasks)} total pairs across both levels, {args.workers} workers")

    rows = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_compare_task, t): t for t in tasks}
        n_done = 0
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            n_done += 1
            if n_done % 10 == 0 or n_done == len(tasks):
                elapsed = time.time() - t0
                eta = elapsed / n_done * (len(tasks) - n_done)
                log(f"{n_done}/{len(tasks)}: {row['sequence_a']} vs {row['sequence_b']} "
                    f"mean={row['mean_mm']:.3f}mm elapsed={elapsed:.1f}s ETA={eta:.1f}s")

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    log(f"wrote {OUT_CSV} ({len(df)} rows)")
    for level in df["level"].unique():
        sub = df[df["level"] == level]
        log(f"{level}: mean_mm median={sub['mean_mm'].median():.3f} "
            f"p95_mm median={sub['p95_mm'].median():.3f} n_pairs={len(sub)}")
    log("ALL DONE")


if __name__ == "__main__":
    main()
