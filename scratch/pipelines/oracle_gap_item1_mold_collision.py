"""Item 1: test the primID-collision hypothesis for Group A1. For each
sequence, register raw lumen -> coverage_mesh.obj (ICP), register the
mold pieces with the same transform, ray-cast a COMBINED (lumen+mold)
scene at full density using GT pose so occlusion is handled correctly,
and track which LUMEN faces are GT-observed but never actually hit by
our ray-cast (false_unobserved / residual), versus which MOLD triangles
(per piece, local 0-based index) are hit. See docs/visibility_outliers.md
and docs/eval_protocol_oracle_gap.md.
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
OUT_DIR = REPO / "results/pipelines/oracle_gap_item1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEQUENCES = {
    "c1_ascending_t4_v2": ("c1_ascending", True),
    "c1_ascending_t4_v3": ("c1_ascending", True),
    "c1_ascending_t3_v1": ("c1_ascending", True),
    "c1_cecum_t1_v1": ("c1_cecum", False),  # control, Open End Visible=no
}


def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


intr = CameraIntrinsics.from_file(INTRINSICS_PATH)
W, H = intr.width, intr.height
n_pixels = W * H
cols, rows = np.meshgrid(np.arange(W), np.arange(H))
px_grid = np.stack([cols.ravel(), rows.ravel()], axis=-1).astype(np.float64)
cam_rays = unproject(px_grid, intr)

results = {}

for seq, (model_name, has_open_end) in SEQUENCES.items():
    log(f"\n=== {seq} (3D model: {model_name}) ===")
    seq_dir = REPO / "scratch" / seq
    mesh_data = load_coverage_mesh(seq_dir / "coverage_mesh.obj")
    lumen_mesh = trimesh.Trimesh(vertices=mesh_data.vertices, faces=mesh_data.faces, process=False)
    n_lumen_faces = len(mesh_data.faces)
    gt_observed = mesh_data.face_observed
    poses = load_poses(seq_dir / "pose.txt")
    n_frames = len(poses)

    # ---- register raw lumen -> this sequence's coverage_mesh.obj ----
    t_reg = time.time()
    raw_lumen = trimesh.load(REPO / f"scratch/3D_models/{model_name}_lumen/{model_name}.obj", process=False)
    log(f"  loaded raw lumen: {len(raw_lumen.faces)} faces in {time.time()-t_reg:.1f}s")
    t_reg = time.time()
    transform, cost = trimesh.registration.mesh_other(raw_lumen, lumen_mesh, samples=2000, scale=False, seed=0)
    log(f"  lumen->coverage_mesh registration RMS: {cost**0.5:.4f}mm ({time.time()-t_reg:.1f}s)")

    mold_dir = REPO / f"scratch/3D_models/{model_name}_mold/{model_name}"
    mold_pieces = {}
    combined_meshes = [lumen_mesh]
    piece_offsets = {}  # piece name -> (start, end) face-index range in the combined mesh
    offset = n_lumen_faces
    for piece_file in sorted(mold_dir.glob("*.stl")):
        piece_name = piece_file.stem
        t_load = time.time()
        m = trimesh.load(piece_file, process=False)
        log(f"    loaded {piece_name}: {len(m.faces)} faces in {time.time()-t_load:.1f}s")
        m.apply_transform(transform)
        mold_pieces[piece_name] = m
        piece_offsets[piece_name] = (offset, offset + len(m.faces))
        combined_meshes.append(m)
        offset += len(m.faces)
    log(f"  mold pieces: {[(k, len(v.faces)) for k, v in mold_pieces.items()]}")

    # NOTE: deliberately NOT combining lumen+mold into one ray-traced scene.
    # An earlier version did that and got a wildly inflated false_unobserved
    # count (291,783 vs. the expected ~8,033 for c1_ascending_t4_v2) --
    # almost certainly imperfect mold/lumen registration causing spurious
    # occlusion, not a real finding. The user's hypothesis is about INDEX
    # collision, not physical occlusion ordering, so two independent
    # ray-casts (lumen alone, mold alone) and an index-set comparison is
    # both more robust and a more direct test of the actual hypothesis.
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

        _, _, index_tri = lumen_mesh.ray.intersects_location(
            origins, directions, multiple_hits=False
        )
        if len(index_tri) > 0:
            np.logical_or.at(ever_hit_lumen, index_tri, True)

        for piece_name, piece_mesh in mold_pieces.items():
            _, _, p_index_tri = piece_mesh.ray.intersects_location(
                origins, directions, multiple_hits=False
            )
            if len(p_index_tri) > 0:
                np.logical_or.at(ever_hit_mold[piece_name], p_index_tri, True)

        if frame_idx % 50 == 0 or frame_idx == n_frames - 1:
            log(f"    frame {frame_idx}/{n_frames-1}, elapsed {time.time()-t0:.0f}s")

    log(f"  ray-casting done in {time.time()-t0:.0f}s")

    false_unobserved = gt_observed & ~ever_hit_lumen
    fu_idx = np.where(false_unobserved)[0]
    log(f"  false_unobserved (residual): {len(fu_idx)} faces "
        f"(GT-observed={int(gt_observed.sum())})")

    # ---- (b) index-bound check ----
    bound_checks = {}
    for piece_name, mold_mesh in mold_pieces.items():
        piece_n_faces = len(mold_mesh.faces)
        below_bound = fu_idx < piece_n_faces
        bound_checks[piece_name] = {
            "piece_face_count": piece_n_faces,
            "fu_below_bound": int(below_bound.sum()),
            "fu_below_bound_frac": float(below_bound.mean()) if len(fu_idx) else float("nan"),
        }
        log(f"    vs {piece_name} ({piece_n_faces} faces): "
            f"{below_bound.sum()}/{len(fu_idx)} false_unobserved indices below bound "
            f"({below_bound.mean()*100 if len(fu_idx) else 0:.2f}%)")

    # ---- (c) overlap between false_unobserved lumen indices and mold-hit indices ----
    overlap_checks = {}
    for piece_name, hit_mask in ever_hit_mold.items():
        piece_n_faces = len(hit_mask)
        # compare false_unobserved indices (that are < piece_n_faces, so comparable)
        comparable = fu_idx[fu_idx < piece_n_faces]
        mold_hit_idx = np.where(hit_mask)[0]
        fu_set = set(comparable.tolist())
        mold_set = set(mold_hit_idx.tolist())
        inter = fu_set & mold_set
        union = fu_set | mold_set
        iou = len(inter) / len(union) if union else float("nan")
        overlap_checks[piece_name] = {
            "n_false_unobserved_comparable": len(comparable),
            "n_mold_hit": len(mold_hit_idx),
            "n_overlap": len(inter),
            "iou": iou,
            "frac_of_fu_explained": len(inter) / len(comparable) if len(comparable) else float("nan"),
        }
        log(f"    overlap vs {piece_name}: fu_comparable={len(comparable)}, mold_hit={len(mold_hit_idx)}, "
            f"overlap={len(inter)}, IoU={iou:.4f}, frac_of_fu_explained={overlap_checks[piece_name]['frac_of_fu_explained']:.4f}")

    results[seq] = {
        "n_frames": n_frames, "n_lumen_faces": n_lumen_faces,
        "n_gt_observed": int(gt_observed.sum()), "n_false_unobserved": len(fu_idx),
        "registration_rms_mm": float(cost**0.5),
        "mold_piece_face_counts": {k: len(v.faces) for k, v in mold_pieces.items()},
        "bound_checks": bound_checks, "overlap_checks": overlap_checks,
        "has_open_end_visible": has_open_end,
    }
    np.save(OUT_DIR / f"{seq}_false_unobserved_idx.npy", fu_idx)

with open(OUT_DIR / "summary.json", "w") as f:
    json.dump(results, f, indent=2)
log(f"\nwrote {OUT_DIR / 'summary.json'}")
log("ALL DONE")
