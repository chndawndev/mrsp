#!/usr/bin/env python
"""Diagnose the false_unobserved faces in the 9 flagged/worst sequences from
scripts/visibility_full.py (see docs/visibility_rasterization.md). Reuses the
already-saved results/visibility_full/observed_masks.npz -- does NOT rerun
GPU rasterization.

For each sequence:
  1. Connected components of the false_unobserved face set (count, largest area).
  2. Rim-adjacency: fraction of false_unobserved faces with >=1 face-adjacent
     neighbor in the GT-unobserved set (thin rim) vs fully interior to the
     GT-observed region (island).
  3. Per-face geometry over all frames: min camera distance, min incidence
     angle (cheap, pure world-space vector math, every frame); max projected
     pixel footprint (expensive -- needs the camera model's quartic-root
     projection -- evaluated only at each face's own best candidate frame,
     not every frame; see module docstring in main() for the exact proxy).
     Compared against a random control sample of correctly-observed faces.
  4. Frame-level: histogram of each flagged face's best (closest-distance)
     frame index; pose rotation-angle and min-scene-distance per frame,
     to check whether flagged faces' best frames cluster somewhere unusual.
  5. (sigmoid1 sequences only) mesh degeneracy: near-zero-area faces,
     duplicate/near-duplicate vertices, duplicate/near-duplicate faces.

Writes results/visibility_outliers/<name>_diagnosis.json (raw numbers) and
results/visibility_outliers/views/<name>_{view1,view2,cutopen}.png (magenta
false_unobserved highlight). docs/visibility_outliers.md is written
separately after inspecting this script's output.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from geometry.camera import CameraIntrinsics, project  # noqa: E402
from geometry.coverage_mesh import stream_coverage_mesh_from_zip  # noqa: E402
from geometry.mesh_stats import (  # noqa: E402
    area_to_diameter,
    connected_components_of_subset,
    face_adjacency,
    face_areas,
)
from geometry.pose import stream_poses_from_zip  # noqa: E402
from render_coverage_views import render_three_views  # noqa: E402

REGISTERED_DIR = Path("/data1_ycao/chua/datasets/C3VDv2/registered_videos")
INTRINSICS_PATH = "/data1_ycao/chua/datasets/C3VDv2/camera_intrinsics.txt"
OUT_DIR = Path("results/visibility_outliers")
RNG = np.random.default_rng(0)

PATTERN_A = ["c1_ascending_t4_v2", "c1_ascending_t4_v3", "c1_ascending_t3_v1",
             "c2_transverse2_t3_v1", "c2_rectum_t4_v1", "c2_rectum_t3_v1"]
PATTERN_B = ["c1_sigmoid1_t2_v2", "c1_sigmoid1_t2_v3", "c1_sigmoid1_t1_v1"]
ALL_SEQUENCES = PATTERN_A + PATTERN_B


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return n / norm


def compute_face_frame_stats(
    face_idx: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    poses: np.ndarray,
) -> dict:
    """For the given subset of face indices, compute over ALL frames:
    min camera distance, min incidence angle (degrees), and the frame index
    at which distance is minimized (used as the footprint-evaluation frame).
    Pure world-space vector math, fully vectorized per frame -- no camera
    projection here (that's the expensive part, done separately only at the
    best frame).
    """
    centroids = vertices[faces[face_idx]].mean(axis=1)  # (F, 3)
    face_normals_sub = normals[face_idx]  # (F, 3)
    F = len(face_idx)
    min_dist = np.full(F, np.inf)
    min_incidence = np.full(F, np.inf)
    best_frame = np.zeros(F, dtype=np.int64)

    for fi, M in enumerate(poses):
        origin = M[3, :3]
        view_vec = centroids - origin  # (F, 3)
        dist = np.linalg.norm(view_vec, axis=1)
        view_dir = view_vec / np.maximum(dist, 1e-9)[:, None]
        cos_inc = np.abs(np.sum(view_dir * face_normals_sub, axis=1))
        cos_inc = np.clip(cos_inc, 0.0, 1.0)
        incidence_deg = np.degrees(np.arccos(cos_inc))  # 0=head-on, 90=grazing

        better = dist < min_dist
        min_dist = np.where(better, dist, min_dist)
        best_frame = np.where(better, fi, best_frame)
        min_incidence = np.minimum(min_incidence, incidence_deg)

    return {"min_dist": min_dist, "min_incidence_deg": min_incidence, "best_frame": best_frame}


def compute_footprint_at_frame(
    face_idx: np.ndarray,
    frame_idx: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    poses: np.ndarray,
    intr: CameraIntrinsics,
) -> np.ndarray:
    """Projected pixel-space triangle area for each face, evaluated at its
    own specified frame (typically that face's closest-distance frame -- a
    proxy for "best chance to be seen big", not a search over all frames).
    """
    footprints = np.full(len(face_idx), np.nan)
    order = np.argsort(frame_idx)
    face_idx_sorted = face_idx[order]
    frame_idx_sorted = frame_idx[order]

    i = 0
    n = len(face_idx_sorted)
    while i < n:
        j = i
        while j < n and frame_idx_sorted[j] == frame_idx_sorted[i]:
            j += 1
        group_faces = face_idx_sorted[i:j]
        M = poses[frame_idx_sorted[i]]
        origin = M[3, :3]
        R = M[:3, :3]
        R_inv_approx = R  # near-orthonormal; consistent with rest of pipeline

        tri_world = vertices[faces[group_faces]]  # (G, 3, 3)
        tri_cam = (tri_world - origin) @ R_inv_approx.T  # (G, 3, 3)
        tri_cam_flat = tri_cam.reshape(-1, 3)
        px = project(tri_cam_flat, intr).reshape(-1, 3, 2)

        a = px[:, 1] - px[:, 0]
        b = px[:, 2] - px[:, 0]
        area = 0.5 * np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])

        footprints[order[i:j]] = area
        i = j

    return footprints


def check_mesh_degeneracy(vertices: np.ndarray, faces: np.ndarray, areas: np.ndarray) -> dict:
    near_zero_thresh = 1e-6
    n_near_zero = int(np.sum(areas < near_zero_thresh))

    rounded = np.round(vertices, decimals=4)
    _, inverse, counts = np.unique(rounded, axis=0, return_inverse=True, return_counts=True)
    n_duplicate_vertex_groups = int(np.sum(counts > 1))
    n_duplicate_vertices = int(np.sum(counts[counts > 1]))

    face_verts_sorted = np.sort(faces, axis=1)
    _, face_counts = np.unique(face_verts_sorted, axis=0, return_counts=True)
    n_duplicate_faces = int(np.sum(face_counts > 1))

    return {
        "n_faces": len(faces),
        "n_near_zero_area_faces": n_near_zero,
        "near_zero_area_threshold_mm2": near_zero_thresh,
        "n_duplicate_vertex_groups": n_duplicate_vertex_groups,
        "n_duplicate_vertices_total": n_duplicate_vertices,
        "n_duplicate_faces_by_vertex_set": n_duplicate_faces,
    }


def diagnose_sequence(name: str, masks: dict) -> dict:
    log(f"=== {name} ===")
    archive = REGISTERED_DIR / f"{name}.zip"
    mesh_data = stream_coverage_mesh_from_zip(archive)
    poses = stream_poses_from_zip(archive)
    n_faces = len(mesh_data.faces)

    gt_observed = mesh_data.face_observed
    packed = masks[name]
    pred_observed = np.unpackbits(packed)[:n_faces].astype(bool)

    false_unobserved_mask = gt_observed & ~pred_observed
    false_observed_mask = ~gt_observed & pred_observed
    fu_idx = np.where(false_unobserved_mask)[0]
    fo_idx = np.where(false_observed_mask)[0]
    log(f"  n_faces={n_faces} false_unobserved={len(fu_idx)} false_observed={len(fo_idx)}")

    areas = face_areas(mesh_data.vertices, mesh_data.faces)
    adjacency = face_adjacency(mesh_data.vertices, mesh_data.faces)

    # 1. Connected components of false_unobserved.
    fu_components = connected_components_of_subset(n_faces, adjacency, false_unobserved_mask)
    fu_comp_areas = sorted((float(areas[c].sum()) for c in fu_components), reverse=True)
    log(f"  false_unobserved: {len(fu_components)} components, "
        f"largest area={fu_comp_areas[0] if fu_comp_areas else 0:.2f}mm2 "
        f"(equiv diam {area_to_diameter(fu_comp_areas[0]) if fu_comp_areas else 0:.2f}mm)")

    # 2. Rim-adjacency: does a false_unobserved face touch the GT-unobserved set?
    adj_a, adj_b = adjacency[:, 0], adjacency[:, 1]
    touches_gt_unobserved = np.zeros(n_faces, dtype=bool)
    gt_unobserved_mask = ~gt_observed
    mask_a_unobs = gt_unobserved_mask[adj_b]
    mask_b_unobs = gt_unobserved_mask[adj_a]
    np.logical_or.at(touches_gt_unobserved, adj_a, mask_a_unobs)
    np.logical_or.at(touches_gt_unobserved, adj_b, mask_b_unobs)
    fu_rim_adjacent = touches_gt_unobserved[fu_idx]
    frac_rim_adjacent = float(fu_rim_adjacent.mean()) if len(fu_idx) else float("nan")
    log(f"  rim-adjacent fraction of false_unobserved: {frac_rim_adjacent:.4f}")

    # 3. Per-face geometry: false_unobserved vs a control sample of correctly-observed faces.
    normals = face_normals(mesh_data.vertices, mesh_data.faces)
    correctly_observed_idx = np.where(gt_observed & pred_observed)[0]
    control_size = min(2000, len(correctly_observed_idx))
    control_idx = RNG.choice(correctly_observed_idx, size=control_size, replace=False)

    t0 = time.time()
    fu_stats = compute_face_frame_stats(fu_idx, mesh_data.vertices, mesh_data.faces, normals, poses)
    control_stats = compute_face_frame_stats(control_idx, mesh_data.vertices, mesh_data.faces, normals, poses)
    log(f"  distance/incidence pass: {time.time()-t0:.1f}s for {len(fu_idx)+len(control_idx)} faces x {len(poses)} frames")

    intr = CameraIntrinsics.from_file(INTRINSICS_PATH)
    t0 = time.time()
    fu_footprint = compute_footprint_at_frame(fu_idx, fu_stats["best_frame"], mesh_data.vertices, mesh_data.faces, poses, intr)
    control_footprint = compute_footprint_at_frame(control_idx, control_stats["best_frame"], mesh_data.vertices, mesh_data.faces, poses, intr)
    log(f"  footprint pass: {time.time()-t0:.1f}s")

    # 4. Frame-level clustering of best frames + pose motion diagnostics.
    best_frame_hist = np.bincount(fu_stats["best_frame"], minlength=len(poses)).tolist()
    rotations_deg = []
    for i in range(1, len(poses)):
        R_prev, R_curr = poses[i - 1][:3, :3], poses[i][:3, :3]
        R_rel = R_prev.T @ R_curr
        cos_ang = np.clip((np.trace(R_rel) - 1) / 2, -1, 1)
        rotations_deg.append(float(np.degrees(np.arccos(cos_ang))))

    result = {
        "name": name,
        "n_faces": n_faces,
        "n_frames": len(poses),
        "n_false_unobserved": len(fu_idx),
        "n_false_observed": len(fo_idx),
        "fu_n_components": len(fu_components),
        "fu_component_areas_top5": fu_comp_areas[:5] + [0.0] * max(0, 5 - len(fu_comp_areas)),
        "fu_rim_adjacent_fraction": frac_rim_adjacent,
        "fu_min_dist": fu_stats["min_dist"].tolist(),
        "fu_min_incidence_deg": fu_stats["min_incidence_deg"].tolist(),
        "fu_max_footprint_px2": fu_footprint.tolist(),
        "fu_best_frame": fu_stats["best_frame"].tolist(),
        "control_min_dist": control_stats["min_dist"].tolist(),
        "control_min_incidence_deg": control_stats["min_incidence_deg"].tolist(),
        "control_max_footprint_px2": control_footprint.tolist(),
        "best_frame_histogram": best_frame_hist,
        "pose_rotation_deg_between_consecutive_frames": rotations_deg,
    }

    if name in PATTERN_B:
        result["mesh_degeneracy"] = check_mesh_degeneracy(mesh_data.vertices, mesh_data.faces, areas)
        log(f"  degeneracy: {result['mesh_degeneracy']}")

    # Render: GT-observed light gray, GT-unobserved dark gray, false_unobserved magenta.
    face_colors = np.tile(np.array([[210, 210, 210, 255]], dtype=np.uint8), (n_faces, 1))
    face_colors[~gt_observed] = [90, 90, 90, 255]
    face_colors[false_unobserved_mask] = [230, 0, 230, 255]
    views_dir = OUT_DIR / "views"
    render_three_views(mesh_data.vertices, mesh_data.faces, face_colors, views_dir / name)
    log(f"  rendered views to {views_dir}")

    return result


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "views").mkdir(parents=True, exist_ok=True)
    masks = dict(np.load("results/visibility_full/observed_masks.npz"))

    for name in ALL_SEQUENCES:
        result = diagnose_sequence(name, masks)
        out_path = OUT_DIR / f"{name}_diagnosis.json"
        out_path.write_text(json.dumps(result))
        log(f"  wrote {out_path}")

    log("ALL DONE")


if __name__ == "__main__":
    main()
