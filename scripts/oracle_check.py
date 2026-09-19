#!/usr/bin/env python
"""Oracle check: does depth + pose.txt reconstruct 3D points matching coverage_mesh.obj?

For sequence c1_cecum_t1_v1: backproject every (masked) depth pixel of every
frame to a 3D point using src/geometry/camera.py, transform to world space
via pose.txt under BOTH candidate flattening interpretations
(src/geometry/pose.py), fuse all frames' points, and compare against
coverage_mesh.obj:
  - point-to-mesh-surface distance distribution (median/mean/p95/frac<1mm)
  - per-face "hit" set (nearest-surface face of every fused point) vs. the
    vt-based observed/unobserved set baked into the mesh -- IoU, false-
    observed, false-unobserved

Outputs, per interpretation, to results/oracle/:
  - fused_points_<interp>.ply
  - colored_mesh_<interp>.ply (green=agree, red=false-observed [we say
    observed, mesh says unobserved], blue=false-unobserved)
  - metrics_<interp>.json

See docs/conventions.md sections 1-4 for the source equations/criteria this
script implements, and docs/oracle_check.md for the interpreted PASS/FAIL
verdict written after this script is run.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import tifffile
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from geometry.camera import CameraIntrinsics, backproject_depth  # noqa: E402
from geometry.coverage_mesh import load_coverage_mesh  # noqa: E402
from geometry.pose import load_poses, transform_points  # noqa: E402

DEPTH_MAX_MM = 100.0
DEPTH_UINT16_MAX = 65535


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_depth_mm(path: Path) -> np.ndarray:
    raw = tifffile.imread(path)
    return raw


def fuse_points(
    sequence_dir: Path,
    intr: CameraIntrinsics,
    poses: np.ndarray,
    pixel_stride: int,
    frame_stride: int,
) -> dict[str, np.ndarray]:
    """Backproject + transform every (masked) sampled pixel of every sampled
    frame, under both pose interpretations. Returns dict with 'raw' and
    'transposed' -> (N, 3) fused world-space point arrays.
    """
    depth_dir = sequence_dir / "depth"
    depth_files = sorted(depth_dir.glob("*_depth.tiff"))
    n_frames = len(depth_files)
    log(f"found {n_frames} depth frames; using every {frame_stride} (pixel stride {pixel_stride})")

    rows = np.arange(0, intr.height, pixel_stride)
    cols = np.arange(0, intr.width, pixel_stride)
    grid_cols, grid_rows = np.meshgrid(cols, rows)  # (R, C)
    px_grid = np.stack([grid_cols.ravel(), grid_rows.ravel()], axis=-1).astype(np.float64)

    pts_raw: list[np.ndarray] = []
    pts_transposed: list[np.ndarray] = []
    total_valid = 0

    for i in range(0, n_frames, frame_stride):
        depth_path = depth_files[i]
        frame_idx = int(depth_path.name.split("_")[0])
        raw = load_depth_mm(depth_path)
        sub = raw[np.ix_(rows, cols)].ravel()

        valid = (sub != 0) & (sub != DEPTH_UINT16_MAX)
        if not np.any(valid):
            continue

        depth_mm = sub[valid].astype(np.float64) / DEPTH_UINT16_MAX * DEPTH_MAX_MM
        px_valid = px_grid[valid]

        cam_pts = backproject_depth(px_valid, depth_mm, intr)

        if frame_idx >= len(poses):
            log(f"WARNING: no pose for frame {frame_idx}, skipping")
            continue
        M = poses[frame_idx]

        pts_raw.append(transform_points(cam_pts, M, "raw"))
        pts_transposed.append(transform_points(cam_pts, M, "transposed"))
        total_valid += int(valid.sum())

        if (i // frame_stride) % 20 == 0:
            log(f"  frame {frame_idx} ({i+1}/{n_frames}): {int(valid.sum())} valid pts, running total {total_valid}")

    log(f"done fusing: {total_valid} total valid backprojected points")
    return {
        "raw": np.concatenate(pts_raw, axis=0),
        "transposed": np.concatenate(pts_transposed, axis=0),
    }


def evaluate_interpretation(
    name: str,
    points: np.ndarray,
    mesh: trimesh.Trimesh,
    face_observed_gt: np.ndarray,
    out_dir: Path,
    max_query_points: int | None = None,
) -> dict:
    full_n = len(points)
    if max_query_points is not None and full_n > max_query_points:
        # trimesh's non-embree nearest-surface query degrades badly (100s-1000x
        # slower) when query points are far outside the mesh's bounding volume,
        # since AABB pruning barely helps -- observed directly in a smoke test:
        # 2107 far-away ("raw" interpretation) points took 641s vs. 0.2s for
        # 2107 on-mesh ("transposed") points. We subsample far/likely-wrong
        # interpretations to keep runtime bounded; the correct interpretation
        # should always be run at full resolution.
        rng = np.random.default_rng(0)
        idx = rng.choice(full_n, size=max_query_points, replace=False)
        points = points[idx]
        log(f"[{name}] subsampled {full_n} -> {max_query_points} points for query "
            f"(nearest-surface query is pathologically slow for far-off-mesh points)")

    log(f"[{name}] querying nearest surface for {len(points)} points ...")
    t0 = time.time()
    closest, distances, face_idx = mesh.nearest.on_surface(points)
    log(f"[{name}] nearest-surface query done in {time.time()-t0:.1f}s")

    frac_under_1mm = float(np.mean(distances < 1.0))
    metrics = {
        "n_points_fused_total": int(full_n),
        "n_points_queried": int(len(points)),
        "distance_median_mm": float(np.median(distances)),
        "distance_mean_mm": float(np.mean(distances)),
        "distance_p95_mm": float(np.percentile(distances, 95)),
        "distance_frac_under_1mm": frac_under_1mm,
    }

    n_faces = len(mesh.faces)
    face_pred_observed = np.zeros(n_faces, dtype=bool)
    face_pred_observed[np.unique(face_idx)] = True

    gt = face_observed_gt
    pred = face_pred_observed
    intersection = np.sum(pred & gt)
    union = np.sum(pred | gt)
    iou = float(intersection / union) if union > 0 else float("nan")
    false_observed = int(np.sum(pred & ~gt))  # we predict observed, mesh says unobserved
    false_unobserved = int(np.sum(~pred & gt))  # mesh says observed, we never hit it

    metrics.update(
        {
            "n_faces": int(n_faces),
            "n_faces_gt_observed": int(gt.sum()),
            "n_faces_pred_observed": int(pred.sum()),
            "face_iou": iou,
            "false_observed_faces": false_observed,
            "false_unobserved_faces": false_unobserved,
        }
    )

    log(f"[{name}] distance median={metrics['distance_median_mm']:.4f}mm "
        f"p95={metrics['distance_p95_mm']:.4f}mm frac<1mm={frac_under_1mm:.4f} "
        f"IoU={iou:.4f}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Fused points PLY.
    pc = trimesh.points.PointCloud(points)
    pc.export(out_dir / f"fused_points_{name}.ply")

    # Colored mesh: green=agree, red=false-observed, blue=false-unobserved,
    # gray=agree-unobserved (both say unobserved).
    colors = np.tile(np.array([[180, 180, 180, 255]], dtype=np.uint8), (n_faces, 1))
    agree_observed = pred & gt
    colors[agree_observed] = [0, 200, 0, 255]
    colors[pred & ~gt] = [220, 0, 0, 255]
    colors[~pred & gt] = [0, 0, 220, 255]
    colored = mesh.copy()
    colored.visual.face_colors = colors
    colored.export(out_dir / f"colored_mesh_{name}.ply")

    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence-dir", default="scratch/c1_cecum_t1_v1")
    ap.add_argument("--intrinsics", default="/data1_ycao/chua/datasets/C3VDv2/camera_intrinsics.txt")
    ap.add_argument("--pixel-stride", type=int, default=8)
    ap.add_argument("--frame-stride", type=int, default=1)
    ap.add_argument("--out-dir", default="results/oracle")
    ap.add_argument("--metrics-out", default="results/oracle/metrics.json")
    ap.add_argument(
        "--raw-max-points",
        type=int,
        default=20000,
        help="cap on query points for the 'raw' (likely-wrong) interpretation; "
        "see evaluate_interpretation() docstring/comment for why this matters",
    )
    ap.add_argument(
        "--skip-raw",
        action="store_true",
        help="skip the 'raw' interpretation entirely (once already established wrong)",
    )
    args = ap.parse_args()

    sequence_dir = Path(args.sequence_dir)
    out_dir = Path(args.out_dir)

    log("loading intrinsics, poses, mesh ...")
    intr = CameraIntrinsics.from_file(args.intrinsics)
    poses = load_poses(sequence_dir / "pose.txt")
    mesh_data = load_coverage_mesh(sequence_dir / "coverage_mesh.obj")
    log(f"mesh: {len(mesh_data.vertices)} verts, {len(mesh_data.faces)} faces, "
        f"{mesh_data.face_observed.mean():.4f} observed fraction")

    mesh = trimesh.Trimesh(vertices=mesh_data.vertices, faces=mesh_data.faces, process=False)
    assert len(mesh.faces) == len(mesh_data.faces), "trimesh reprocessed faces despite process=False"

    fused = fuse_points(sequence_dir, intr, poses, args.pixel_stride, args.frame_stride)

    interpretations = ("transposed",) if args.skip_raw else ("raw", "transposed")
    all_metrics = {}
    for name in interpretations:
        cap = args.raw_max_points if name == "raw" else None
        all_metrics[name] = evaluate_interpretation(
            name, fused[name], mesh, mesh_data.face_observed, out_dir, max_query_points=cap
        )

    winner = min(all_metrics, key=lambda k: all_metrics[k]["distance_median_mm"])
    all_metrics["_verdict"] = {
        "winner": winner,
        "reasoning": "interpretation with lower median point-to-mesh distance",
    }

    Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics_out).write_text(json.dumps(all_metrics, indent=2))
    log(f"wrote metrics to {args.metrics_out}")
    log(f"VERDICT: interpretation '{winner}' fits the mesh better")
    log("ALL DONE")


if __name__ == "__main__":
    main()
