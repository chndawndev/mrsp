"""Geometric consistency test for EndoDAC's independently-fit depth and pose
scale factors, replacing the flawed unit-incommensurable "3.15x" comparison
in docs/pipelines/endodac_scale.md section 3.

Four configurations, 20 frames spread across c1_cecum_t1_v1, backprojected
via the project's own validated camera model (src/geometry/camera.py) and
compared against coverage_mesh.obj via the same methodology as
scripts/oracle_check.py (trimesh nearest-surface query):

  A. oracle:            GT depth (mm)         + GT pose (world frame)
  B. fully predicted:    pred depth (scaled)   + pred pose (world-anchored)
  C. pred depth only:    pred depth (scaled)   + GT pose (world frame)
  D. pred pose only:     GT depth (mm)         + pred pose (world-anchored)

Predicted poses are only ever available in an arbitrary reference frame
(chain starts at identity, unrelated to the mesh's world coordinates), so
"pred pose (world-anchored)" applies the FULL Umeyama similarity transform
(rotation R_u, translation t_u, scale s_pose) fitted over the whole
218-frame trajectory -- not just the bare scalar s_pose -- to bring
predicted camera poses into the same world frame as the mesh. This is a
necessary bookkeeping step, not part of what's under test; the scale
factor s_pose is exactly docs/pipelines/endodac_scale.md's already-fitted
value, unchanged.

Run under the project's own venv (scratch/.venv), NOT the endodac conda
env -- this uses src/geometry and trimesh, project code, not the pipeline
under test.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import tifffile
import trimesh

REPO = Path("/data1_ycao/chua/projects/mrsp")
sys.path.insert(0, str(REPO / "src"))
from geometry.camera import CameraIntrinsics, backproject_depth  # noqa: E402
from geometry.coverage_mesh import load_coverage_mesh  # noqa: E402
from geometry.pose import load_poses, transform_points  # noqa: E402

SEQ_DIR = REPO / "scratch" / "c1_cecum_t1_v1"
PRED_DIR = REPO / "results" / "pipelines" / "endodac_full_run" / "c1_cecum_t1_v1"
INTRINSICS_PATH = "/data1_ycao/chua/datasets/C3VDv2/camera_intrinsics.txt"
OUT_DIR = REPO / "results" / "pipelines" / "endodac_fusion_check"
N_TOTAL_FRAMES = 218
N_TEST_FRAMES = 20
PIXEL_STRIDE = 8  # matches scripts/oracle_check.py's main run, for a comparable oracle number
MAX_QUERY_POINTS = 20000  # matches oracle_check.py's precedent cap
DEPTH_MAX_MM = 100.0
DEPTH_UINT16_MAX = 65535


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def umeyama(src: np.ndarray, dst: np.ndarray):
    N = src.shape[0]
    mu_src, mu_dst = src.mean(axis=0), dst.mean(axis=0)
    src_c, dst_c = src - mu_src, dst - mu_dst
    var_src = (src_c ** 2).sum() / N
    Sigma = (dst_c.T @ src_c) / N
    U, D, Vt = np.linalg.svd(Sigma)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    c = np.trace(np.diag(D) @ S) / var_src
    t = mu_dst - c * R @ mu_src
    return R, t, c


def load_gt_depth_mm(frame_idx: int) -> np.ndarray:
    raw = tifffile.imread(SEQ_DIR / "depth" / f"{frame_idx:04d}_depth.tiff")
    return raw  # keep raw uint16 for exact masking; scale inline where used


def main():
    log("loading intrinsics, GT poses, predicted poses, mesh ...")
    intr = CameraIntrinsics.from_file(INTRINSICS_PATH)
    gt_poses = load_poses(SEQ_DIR / "pose.txt")  # (218,4,4) raw pose.txt layout
    pred_poses = np.load(PRED_DIR / "poses_pred.npy")  # (218,4,4) already camera-to-world, arbitrary frame

    with open(REPO / "results/pipelines/endodac_scale_analysis.json") as f:
        scale_info = json.load(f)
    depth_scale = scale_info["depth_scale"]["median"]
    log(f"using depth_scale (global median) = {depth_scale:.4f}")

    # Recompute the Umeyama fit fresh (same 218-frame positions as endodac_scale_analysis.py)
    # so R_u, t_u are available here (only the scalar was persisted there).
    gt_positions_full = transform_points(
        np.zeros((N_TOTAL_FRAMES, 3)), gt_poses, "transposed"
    )  # camera centers = translate the origin by each pose
    pred_positions_full = pred_poses[:, :3, 3]
    R_u, t_u, s_pose = umeyama(pred_positions_full, gt_positions_full)
    log(f"Umeyama fit: s_pose={s_pose:.4f} (cross-check vs saved {scale_info['pose_scale']['s_pose']:.4f})")
    assert abs(s_pose - scale_info["pose_scale"]["s_pose"]) < 1e-3, "Umeyama scale mismatch vs saved value"

    def world_pose_pred(i: int) -> tuple[np.ndarray, np.ndarray]:
        """World-anchored (R, T) for predicted pose at frame i: rotation composed
        with R_u, translation scaled by s_pose and re-based through R_u, t_u."""
        R_i = pred_poses[i, :3, :3]
        t_i = pred_poses[i, :3, 3]
        R_world = R_u @ R_i
        T_world = s_pose * (R_u @ t_i) + t_u
        return R_world, T_world

    mesh_data = load_coverage_mesh(SEQ_DIR / "coverage_mesh.obj")
    mesh = trimesh.Trimesh(vertices=mesh_data.vertices, faces=mesh_data.faces, process=False)
    log(f"mesh: {len(mesh_data.vertices)} verts, {len(mesh_data.faces)} faces")

    test_frames = sorted(set(np.linspace(0, N_TOTAL_FRAMES - 1, N_TEST_FRAMES).astype(int).tolist()))
    log(f"test frames ({len(test_frames)}): {test_frames}")

    rows = np.arange(0, intr.height, PIXEL_STRIDE)
    cols = np.arange(0, intr.width, PIXEL_STRIDE)
    grid_cols, grid_rows = np.meshgrid(cols, rows)
    px_grid = np.stack([grid_cols.ravel(), grid_rows.ravel()], axis=-1).astype(np.float64)

    configs = {"A_oracle": [], "B_fully_predicted": [], "C_pred_depth_only": [], "D_pred_pose_only": []}

    for frame_idx in test_frames:
        raw = load_gt_depth_mm(frame_idx)
        sub_raw = raw[np.ix_(rows, cols)].ravel()
        valid = (sub_raw != 0) & (sub_raw != DEPTH_UINT16_MAX)
        px_valid = px_grid[valid]
        gt_depth_mm = sub_raw[valid].astype(np.float64) / DEPTH_UINT16_MAX * DEPTH_MAX_MM

        pred_depth_full = np.load(PRED_DIR / "depth" / f"{frame_idx:04d}_pred_depth.npy")
        pred_depth_sub = pred_depth_full[np.ix_(rows, cols)].ravel()[valid]  # SAME valid mask (GT-valid pixels)
        pred_depth_mm = pred_depth_sub.astype(np.float64) * depth_scale

        cam_pts_gt_depth = backproject_depth(px_valid, gt_depth_mm, intr)
        cam_pts_pred_depth = backproject_depth(px_valid, pred_depth_mm, intr)

        gt_M = gt_poses[frame_idx]
        world_pts_A = transform_points(cam_pts_gt_depth, gt_M, "transposed")
        world_pts_C = transform_points(cam_pts_pred_depth, gt_M, "transposed")

        R_world, T_world = world_pose_pred(frame_idx)
        world_pts_B = (R_world @ cam_pts_pred_depth.T).T + T_world
        world_pts_D = (R_world @ cam_pts_gt_depth.T).T + T_world

        configs["A_oracle"].append(world_pts_A)
        configs["B_fully_predicted"].append(world_pts_B)
        configs["C_pred_depth_only"].append(world_pts_C)
        configs["D_pred_pose_only"].append(world_pts_D)

        log(f"  frame {frame_idx}: {valid.sum()} valid pts")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    rng = np.random.default_rng(0)
    for name, chunks in configs.items():
        points = np.concatenate(chunks, axis=0)
        full_n = len(points)
        query_points = points
        if full_n > MAX_QUERY_POINTS:
            idx = rng.choice(full_n, size=MAX_QUERY_POINTS, replace=False)
            query_points = points[idx]
        log(f"[{name}] querying nearest surface for {len(query_points)}/{full_n} points ...")
        t0 = time.time()
        closest, distances, face_idx = mesh.nearest.on_surface(query_points)
        elapsed = time.time() - t0
        log(f"[{name}] done in {elapsed:.1f}s: median={np.median(distances):.4f}mm "
            f"p95={np.percentile(distances,95):.4f}mm")
        results[name] = {
            "n_points_fused": int(full_n),
            "n_points_queried": int(len(query_points)),
            "distance_median_mm": float(np.median(distances)),
            "distance_mean_mm": float(np.mean(distances)),
            "distance_p95_mm": float(np.percentile(distances, 95)),
            "distance_max_mm": float(np.max(distances)),
            "query_seconds": elapsed,
        }
        pc = trimesh.points.PointCloud(query_points)
        pc.export(OUT_DIR / f"fused_points_{name}.ply")

    results["_meta"] = {
        "test_frames": test_frames, "pixel_stride": PIXEL_STRIDE,
        "depth_scale_used": depth_scale, "s_pose_used": s_pose,
        "max_query_points": MAX_QUERY_POINTS,
    }
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    log(f"wrote {OUT_DIR / 'metrics.json'}")
    log("ALL DONE")


if __name__ == "__main__":
    main()
