"""Item 1, tightened: null baseline, corrected (per-piece-local-union)
indexing, and the decisive predicted_GT IoU test for the mold-index-
collision hypothesis. 5 sequences: 3 Group A1 (ascending), 1 Group A2
(c2_rectum_t4_v1, generalization test), 1 control (c1_cecum_t1_v1, no
open end). See docs/eval_protocol_oracle_gap.md.
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

INTRINSICS_PATH = "/data1_ycao/chua/datasets/C3VDv2/camera_intrinsics.txt"
OUT_DIR = REPO / "results/pipelines/oracle_gap_item1_tightened"
OUT_DIR.mkdir(parents=True, exist_ok=True)
N_NULL_DRAWS = 1000
RNG_SEED = 0

SEQUENCES = {
    "c1_ascending_t4_v2": ("c1_ascending", "A1"),
    "c1_ascending_t4_v3": ("c1_ascending", "A1"),
    "c1_ascending_t3_v1": ("c1_ascending", "A1"),
    "c2_rectum_t4_v1": ("c2_rectum", "A2"),
    "c1_cecum_t1_v1": ("c1_cecum", "control"),
}


def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


intr = CameraIntrinsics.from_file(INTRINSICS_PATH)
W, H = intr.width, intr.height
cols, rows = np.meshgrid(np.arange(W), np.arange(H))
px_grid = np.stack([cols.ravel(), rows.ravel()], axis=-1).astype(np.float64)
cam_rays = unproject(px_grid, intr)

results = {}
rng = np.random.default_rng(RNG_SEED)

for seq, (model_name, group) in SEQUENCES.items():
    log(f"\n=== {seq} (3D model: {model_name}, group {group}) ===")
    seq_dir = REPO / "scratch" / seq
    mesh_data = load_coverage_mesh(seq_dir / "coverage_mesh.obj")
    lumen_mesh = trimesh.Trimesh(vertices=mesh_data.vertices, faces=mesh_data.faces, process=False)
    n_lumen_faces = len(mesh_data.faces)
    gt_observed = mesh_data.face_observed
    poses = load_poses(seq_dir / "pose.txt")
    n_frames = len(poses)

    t_reg = time.time()
    raw_lumen = trimesh.load(REPO / f"scratch/3D_models/{model_name}_lumen/{model_name}.obj", process=False)
    transform, cost = trimesh.registration.mesh_other(raw_lumen, lumen_mesh, samples=2000, scale=False, seed=0)
    log(f"  registration RMS: {cost**0.5:.4f}mm ({time.time()-t_reg:.0f}s)")

    mold_dir = REPO / f"scratch/3D_models/{model_name}_mold/{model_name}"
    mold_pieces = {}
    for piece_file in sorted(mold_dir.glob("*.stl")):
        piece_name = piece_file.stem
        m = trimesh.load(piece_file, process=False)
        m.apply_transform(transform)
        mold_pieces[piece_name] = m
    log(f"  mold pieces: {[(k, len(v.faces)) for k, v in mold_pieces.items()]}")

    ever_hit_lumen = np.zeros(n_lumen_faces, dtype=bool)
    ever_hit_mold = {k: np.zeros(len(v.faces), dtype=bool) for k, v in mold_pieces.items()}

    t0 = time.time()
    for frame_idx in range(n_frames):
        M = poses[frame_idx]
        origin = transform_points(np.zeros((1, 3)), M, "transposed")[0]
        endpoints = transform_points(cam_rays, M, "transposed")
        directions = endpoints - origin
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        origins = np.broadcast_to(origin, directions.shape)

        _, _, index_tri = lumen_mesh.ray.intersects_location(origins, directions, multiple_hits=False)
        if len(index_tri) > 0:
            np.logical_or.at(ever_hit_lumen, index_tri, True)
        for piece_name, piece_mesh in mold_pieces.items():
            _, _, p_idx = piece_mesh.ray.intersects_location(origins, directions, multiple_hits=False)
            if len(p_idx) > 0:
                np.logical_or.at(ever_hit_mold[piece_name], p_idx, True)

        if frame_idx % 100 == 0 or frame_idx == n_frames - 1:
            log(f"    frame {frame_idx}/{n_frames-1}, elapsed {time.time()-t0:.0f}s")
    log(f"  ray-casting done in {time.time()-t0:.0f}s")

    false_unobserved = gt_observed & ~ever_hit_lumen
    fu_idx = np.where(false_unobserved)[0]
    n_fu = len(fu_idx)
    log(f"  false_unobserved: {n_fu} / GT-observed={int(gt_observed.sum())}")

    # ---- (b) UNION of each piece's LOCAL 0-based hit indices ----
    mold_union = np.zeros(n_lumen_faces, dtype=bool)
    for piece_name, hit_arr in ever_hit_mold.items():
        k = min(len(hit_arr), n_lumen_faces)
        mold_union[:k] |= hit_arr[:k]

    overlap_per_piece = {}
    for piece_name, hit_arr in ever_hit_mold.items():
        k = len(hit_arr)
        comparable = fu_idx[fu_idx < k]
        hit_set = set(np.where(hit_arr)[0].tolist())
        inter = len(hit_set & set(comparable.tolist()))
        overlap_per_piece[piece_name] = {
            "n_comparable": len(comparable), "n_overlap": inter,
            "frac_explained": inter / len(comparable) if len(comparable) else float("nan"),
        }

    union_overlap = int((mold_union[fu_idx]).sum())
    union_frac_explained = union_overlap / n_fu if n_fu else float("nan")
    log(f"  (b) overlap per piece: {overlap_per_piece}")
    log(f"  (b) UNION overlap: {union_overlap}/{n_fu} = {union_frac_explained*100:.2f}%")

    # ---- (a) null baseline: 1000 random draws, same size as fu_idx ----
    def null_baseline(pool_idx, n_draws=N_NULL_DRAWS):
        fracs = np.zeros(n_draws)
        pool_idx = np.asarray(pool_idx)
        for d in range(n_draws):
            draw = rng.choice(pool_idx, size=n_fu, replace=False) if n_fu <= len(pool_idx) else pool_idx
            fracs[d] = mold_union[draw].mean()
        return fracs

    null_all = null_baseline(np.arange(n_lumen_faces)) if n_fu > 0 else np.array([])
    null_gtobs = null_baseline(np.where(gt_observed)[0]) if n_fu > 0 else np.array([])
    log(f"  (a) null baseline (all lumen faces): mean={null_all.mean()*100:.3f}% "
        f"std={null_all.std()*100:.3f}% p95={np.percentile(null_all,95)*100:.3f}%"
        if n_fu > 0 else "  (a) skipped (n_fu=0)")
    log(f"  (a) null baseline (GT-observed only): mean={null_gtobs.mean()*100:.3f}% "
        f"std={null_gtobs.std()*100:.3f}% p95={np.percentile(null_gtobs,95)*100:.3f}%"
        if n_fu > 0 else "")
    log(f"  (a) observed union overlap {union_frac_explained*100:.2f}% vs. null "
        f"(GT-observed pool) mean {null_gtobs.mean()*100:.3f}% "
        f"-> z-like ratio {union_frac_explained/max(null_gtobs.mean(),1e-9):.1f}x"
        if n_fu > 0 else "")

    # ---- (c) decisive test: predicted_GT = ever_hit_lumen | mold_union ----
    predicted_gt_with_mold = ever_hit_lumen | mold_union
    predicted_gt_without_mold = ever_hit_lumen

    def iou(a, b):
        inter = (a & b).sum()
        union = (a | b).sum()
        return inter / union if union else float("nan")

    iou_without = iou(predicted_gt_without_mold, gt_observed)
    iou_with = iou(predicted_gt_with_mold, gt_observed)
    log(f"  (c) IoU vs released GT: WITHOUT mold-collision term = {iou_without:.6f}, "
        f"WITH = {iou_with:.6f}")

    results[seq] = {
        "group": group, "n_frames": n_frames, "n_lumen_faces": n_lumen_faces,
        "n_gt_observed": int(gt_observed.sum()), "n_false_unobserved": n_fu,
        "registration_rms_mm": float(cost**0.5),
        "overlap_per_piece": overlap_per_piece,
        "union_overlap": union_overlap, "union_frac_explained": union_frac_explained,
        "null_all_mean": float(null_all.mean()) if n_fu > 0 else None,
        "null_all_p95": float(np.percentile(null_all, 95)) if n_fu > 0 else None,
        "null_gtobs_mean": float(null_gtobs.mean()) if n_fu > 0 else None,
        "null_gtobs_p95": float(np.percentile(null_gtobs, 95)) if n_fu > 0 else None,
        "iou_without_mold_term": float(iou_without), "iou_with_mold_term": float(iou_with),
    }
    np.savez(OUT_DIR / f"{seq}_arrays.npz", ever_hit_lumen=ever_hit_lumen,
             gt_observed=gt_observed, mold_union=mold_union, fu_idx=fu_idx,
             **{f"mold_{k}": v for k, v in ever_hit_mold.items()})

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(results, f, indent=2)

log(f"\nwrote {OUT_DIR / 'summary.json'}")
log("ALL DONE")
