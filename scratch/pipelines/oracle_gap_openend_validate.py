"""Part 1 (corrected) step 2: validate that open-end escapes (raw==0 OUTSIDE
the vignette mask) agree with our GPU/ray-cast rasterizer's own no-hit
pixels, using GT pose. If they agree, open-end escapes need no special
handling -- section 3 of the protocol already discards+counts ray misses,
and the released GT gets no face from them either, so they aren't an
oracle gap. See docs/eval_protocol_oracle_gap.md.
"""
import sys
import time
from pathlib import Path

import numpy as np
import tifffile
import trimesh

REPO = Path("/data1_ycao/chua/projects/mrsp")
sys.path.insert(0, str(REPO / "src"))
from geometry.camera import CameraIntrinsics, unproject  # noqa: E402
from geometry.coverage_mesh import load_coverage_mesh  # noqa: E402
from geometry.pose import load_poses, transform_points  # noqa: E402

INTRINSICS_PATH = "/data1_ycao/chua/datasets/C3VDv2/camera_intrinsics.txt"
import os
SEQUENCES = os.environ.get("OG_SEQUENCES", "c1_ascending_t3_v1,c2_rectum_t1_v1").split(",")
VIGNETTE_MASK_NPY = REPO / "results/pipelines/oracle_gap_vignette/c1_cecum_t1_v1_vignette_mask.npy"


def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


intr = CameraIntrinsics.from_file(INTRINSICS_PATH)
W, H = intr.width, intr.height
n_pixels = W * H

vignette_packed = np.load(VIGNETTE_MASK_NPY)
vignette_mask = np.unpackbits(vignette_packed)[:n_pixels].reshape(H, W).astype(bool)
log(f"loaded vignette mask: {vignette_mask.sum()} px ({vignette_mask.mean()*100:.4f}%)")

cols, rows = np.meshgrid(np.arange(W), np.arange(H))
px_grid = np.stack([cols.ravel(), rows.ravel()], axis=-1).astype(np.float64)
cam_rays = unproject(px_grid, intr)  # (n_pixels, 3), row-major raveled (matches raw.ravel() order)

overall_stats = {}

for seq in SEQUENCES:
    seq_dir = REPO / "scratch" / seq
    mesh_data = load_coverage_mesh(seq_dir / "coverage_mesh.obj")
    mesh = trimesh.Trimesh(vertices=mesh_data.vertices, faces=mesh_data.faces, process=False)
    gt_poses = load_poses(seq_dir / "pose.txt")
    depth_files = sorted((seq_dir / "depth").glob("*_depth.tiff"))
    log(f"\n=== {seq}: {len(depth_files)} frames, mesh {len(mesh_data.faces)} faces ===")

    total_escape = 0
    total_nohit = 0
    total_agree = 0  # both escape and nohit
    total_escape_only = 0
    total_nohit_only = 0
    per_frame = []

    t0 = time.time()
    for fpath in depth_files:
        frame_idx = int(fpath.name.split("_")[0])
        raw = tifffile.imread(fpath)
        zero = (raw == 0)
        escape = zero & ~vignette_mask  # raw==0 outside vignette

        M = gt_poses[frame_idx]
        origin = transform_points(np.zeros((1, 3)), M, "transposed")[0]
        endpoints = transform_points(cam_rays, M, "transposed")
        directions = endpoints - origin
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        origins = np.broadcast_to(origin, directions.shape)

        _, index_ray, _ = mesh.ray.intersects_location(origins, directions, multiple_hits=False)
        hit_mask_flat = np.zeros(n_pixels, dtype=bool)
        hit_mask_flat[index_ray] = True
        # Restrict to OUTSIDE the vignette on both sides of the comparison: inside the
        # vignette, GT's raw==0 is forced by the mask.png post-process regardless of the
        # true underlying ray result (docs/gpu_validation.md), so it carries no information
        # there and a rasterizer miss inside that region is not a real escape disagreement.
        nohit = (~hit_mask_flat).reshape(H, W) & ~vignette_mask

        escape_flat = escape.ravel()
        nohit_flat = nohit.ravel()
        agree = escape_flat & nohit_flat
        escape_only = escape_flat & ~nohit_flat
        nohit_only = nohit_flat & ~escape_flat
        union = escape_flat | nohit_flat
        iou = agree.sum() / union.sum() if union.sum() > 0 else float("nan")

        per_frame.append({
            "frame": frame_idx, "n_escape": int(escape_flat.sum()), "n_nohit": int(nohit_flat.sum()),
            "n_agree": int(agree.sum()), "n_escape_only": int(escape_only.sum()),
            "n_nohit_only": int(nohit_only.sum()), "iou": float(iou),
            "escape_frac_of_frame": float(escape_flat.sum() / n_pixels),
        })
        total_escape += escape_flat.sum()
        total_nohit += nohit_flat.sum()
        total_agree += agree.sum()
        total_escape_only += escape_only.sum()
        total_nohit_only += nohit_only.sum()

    elapsed = time.time() - t0
    overall_union = total_escape + total_nohit - total_agree
    overall_iou = total_agree / overall_union if overall_union > 0 else float("nan")
    log(f"{seq} done in {elapsed:.0f}s ({elapsed/len(depth_files):.2f}s/frame)")
    log(f"  totals: escape={total_escape}, nohit={total_nohit}, agree={total_agree}, "
        f"escape_only={total_escape_only}, nohit_only={total_nohit_only}, overall IoU={overall_iou:.6f}")
    peak = max(per_frame, key=lambda p: p["n_escape"])
    log(f"  peak escape frame: {peak['frame']}, {peak['n_escape']} px "
        f"({peak['escape_frac_of_frame']*100:.2f}% of frame)")

    overall_stats[seq] = {
        "n_frames": len(depth_files), "total_escape": int(total_escape), "total_nohit": int(total_nohit),
        "total_agree": int(total_agree), "total_escape_only": int(total_escape_only),
        "total_nohit_only": int(total_nohit_only), "overall_iou": float(overall_iou),
        "per_frame": per_frame,
    }

import json
out_path = REPO / "results/pipelines/oracle_gap_openend_validation.json"
with open(out_path, "w") as f:
    json.dump(overall_stats, f, indent=2)
log(f"\nwrote {out_path}")
log("ALL DONE")
