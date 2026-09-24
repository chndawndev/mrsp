"""Part 1 steps 2-4, resumed with the corrected model: vignette = 135-
sequence majority mask (102,049px); open-end/mold escapes are ray misses
handled by protocol section 3, not a gap; only vignette and the 100mm
clamp create an oracle gap. For each of the 3 original sequences, full
density, GT pose: track per-face whether it was ever hit by an evaluable
ray (valid mask, <=100mm), ever hit via a vignette-region ray, ever hit
via a valid-mask ray beyond 100mm. See docs/eval_protocol_oracle_gap.md.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import trimesh

REPO = Path("/data1_ycao/chua/projects/mrsp")
sys.path.insert(0, str(REPO / "src"))
from geometry.camera import CameraIntrinsics, unproject  # noqa: E402
from geometry.coverage_mesh import load_coverage_mesh  # noqa: E402
from geometry.pose import load_poses, transform_points  # noqa: E402
from geometry.mesh_stats import (  # noqa: E402
    face_areas, face_adjacency, connected_components_of_subset, area_to_diameter,
)

SEQUENCES = ["c1_cecum_t1_v1", "c1_ascending_t3_v1", "c2_rectum_t1_v1"]
INTRINSICS_PATH = "/data1_ycao/chua/datasets/C3VDv2/camera_intrinsics.txt"
OUT_DIR = REPO / "results/pipelines/oracle_gap_part1"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MIN_DIAMETER_MM = 5.0


def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


intr = CameraIntrinsics.from_file(INTRINSICS_PATH)
W, H = intr.width, intr.height
n_pixels = W * H

vignette_packed = np.load(REPO / "results/pipelines/oracle_gap_vignette/c1_cecum_t1_v1_vignette_mask.npy")
vignette_mask = np.unpackbits(vignette_packed)[:n_pixels].reshape(H, W).astype(bool)
vignette_flat = vignette_mask.ravel()
log(f"vignette mask: {vignette_flat.sum()} px ({vignette_flat.mean()*100:.4f}%)")

cols, rows = np.meshgrid(np.arange(W), np.arange(H))
px_grid = np.stack([cols.ravel(), rows.ravel()], axis=-1).astype(np.float64)
cam_rays = unproject(px_grid, intr)

results = {}

for seq in SEQUENCES:
    seq_dir = REPO / "scratch" / seq
    mesh_data = load_coverage_mesh(seq_dir / "coverage_mesh.obj")
    mesh = trimesh.Trimesh(vertices=mesh_data.vertices, faces=mesh_data.faces, process=False)
    n_faces = len(mesh_data.faces)
    gt_poses = load_poses(seq_dir / "pose.txt")
    n_frames = len(sorted((seq_dir / "rgb").glob("*.png")))
    log(f"\n=== {seq}: {n_frames} frames, {n_faces} faces, "
        f"{mesh_data.face_observed.sum()} GT-observed ===")

    ever_evaluable = np.zeros(n_faces, dtype=bool)
    ever_vignette_hit = np.zeros(n_faces, dtype=bool)
    ever_beyond100_hit = np.zeros(n_faces, dtype=bool)

    t0 = time.time()
    for frame_idx in range(n_frames):
        M = gt_poses[frame_idx]
        origin = transform_points(np.zeros((1, 3)), M, "transposed")[0]
        endpoints = transform_points(cam_rays, M, "transposed")
        directions = endpoints - origin
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        origins = np.broadcast_to(origin, directions.shape)

        locations, index_ray, index_tri = mesh.ray.intersects_location(
            origins, directions, multiple_hits=False
        )
        if len(index_ray) == 0:
            continue

        # camera-frame Z-depth of each hit: rotate (hit - origin) back by R^T
        R = M[:3, :3].T  # "transposed" convention: world = M.T @ [cam;1], so R_c2w = M[:3,:3].T
        rel = locations - origin
        depth_cam_z = rel @ R  # (N,3) @ (3,3) -> R^T applied per-row equivalent to R.T@rel_i
        depth_cam_z = depth_cam_z[:, 2]

        valid = ~vignette_flat[index_ray]
        evaluable = valid & (depth_cam_z <= 100.0)
        vign_hit = ~valid
        beyond100 = valid & (depth_cam_z > 100.0)

        np.logical_or.at(ever_evaluable, index_tri[evaluable], True)
        np.logical_or.at(ever_vignette_hit, index_tri[vign_hit], True)
        np.logical_or.at(ever_beyond100_hit, index_tri[beyond100], True)

        if frame_idx % 50 == 0 or frame_idx == n_frames - 1:
            log(f"  frame {frame_idx}/{n_frames-1}, elapsed {time.time()-t0:.0f}s")

    log(f"{seq}: ray-casting done in {time.time()-t0:.0f}s")

    gt_observed = mesh_data.face_observed
    gap = gt_observed & ~ever_evaluable
    cat_vignette_only = gap & ever_vignette_hit
    cat_beyond100_only = gap & ever_beyond100_hit
    union = cat_vignette_only | cat_beyond100_only
    residual = gap & ~union

    n_gt_obs = int(gt_observed.sum())
    intersection_iou = int((gt_observed & ever_evaluable).sum())
    union_iou = int((gt_observed | ever_evaluable).sum())
    max_face_iou = intersection_iou / union_iou if union_iou > 0 else float("nan")

    log(f"  gap (GT-observed, never evaluable): {int(gap.sum())} / {n_gt_obs} "
        f"({gap.sum()/n_gt_obs*100:.4f}%)")
    log(f"    -- vignette-only-reachable: {int(cat_vignette_only.sum())} "
        f"({cat_vignette_only.sum()/n_gt_obs*100:.4f}%)")
    log(f"    -- beyond-100mm-only-reachable: {int(cat_beyond100_only.sum())} "
        f"({cat_beyond100_only.sum()/n_gt_obs*100:.4f}%)")
    log(f"    -- union: {int(union.sum())} ({union.sum()/n_gt_obs*100:.4f}%)")
    log(f"    -- residual (never hit evaluable/vignette/beyond100 at all): {int(residual.sum())}")
    log(f"  MAX FACE IoU (oracle ceiling vs released GT): {max_face_iou:.6f}")

    # ---- region-level impact ----
    log("  computing region-level impact ...")
    adjacency = face_adjacency(mesh_data.vertices, mesh_data.faces)
    areas = face_areas(mesh_data.vertices, mesh_data.faces)

    orig_unobserved = ~gt_observed
    orig_components = connected_components_of_subset(n_faces, adjacency, orig_unobserved)
    orig_regions_5mm = [c for c in orig_components if area_to_diameter(areas[c].sum()) >= MIN_DIAMETER_MM]

    modified_unobserved = orig_unobserved | gap
    mod_components = connected_components_of_subset(n_faces, adjacency, modified_unobserved)
    mod_regions_5mm = [c for c in mod_components if area_to_diameter(areas[c].sum()) >= MIN_DIAMETER_MM]

    # map each original >5mm region to the modified region(s) it overlaps
    face_to_mod_region = -np.ones(n_faces, dtype=np.int64)
    for i, comp in enumerate(mod_components):
        face_to_mod_region[comp] = i

    n_changed = 0
    n_merged = 0
    for comp in orig_regions_5mm:
        mod_ids = set(face_to_mod_region[comp].tolist())
        mod_ids.discard(-1)
        if len(mod_ids) != 1:
            n_changed += 1  # split across modified regions (shouldn't happen, adding faces can't split)
            continue
        mod_id = next(iter(mod_ids))
        mod_comp = mod_components[mod_id]
        if len(mod_comp) != len(comp):
            n_changed += 1
            # did it merge with faces belonging to a DIFFERENT original region?
            gained_faces = set(mod_comp.tolist()) - set(comp.tolist())
            gained_beyond_gap = gained_faces - set(np.where(gap)[0].tolist())
            if gained_beyond_gap:
                n_merged += 1

    log(f"  original >5mm GT-unobserved regions: {len(orig_regions_5mm)}")
    log(f"  modified (gap reclassified unobserved) >5mm regions: {len(mod_regions_5mm)}")
    log(f"  original regions whose face-set changed: {n_changed} / {len(orig_regions_5mm)}")
    log(f"  of those, regions that merged with other original regions: {n_merged}")

    results[seq] = {
        "n_frames": n_frames, "n_faces": n_faces, "n_gt_observed": n_gt_obs,
        "gap_count": int(gap.sum()), "gap_frac": float(gap.sum() / n_gt_obs),
        "vignette_only_count": int(cat_vignette_only.sum()),
        "beyond100_only_count": int(cat_beyond100_only.sum()),
        "union_count": int(union.sum()), "residual_count": int(residual.sum()),
        "max_face_iou": float(max_face_iou),
        "n_orig_regions_5mm": len(orig_regions_5mm), "n_mod_regions_5mm": len(mod_regions_5mm),
        "n_regions_changed": n_changed, "n_regions_merged": n_merged,
    }

with open(OUT_DIR / "summary.json", "w") as f:
    json.dump(results, f, indent=2)
log(f"\nwrote {OUT_DIR / 'summary.json'}")
log("ALL DONE")
