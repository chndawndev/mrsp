"""Measure scale consistency between EndoDAC's predicted depth and predicted
pose on c1_cecum_t1_v1, before writing any evaluation code. Uses the
outputs already produced by endodac_full_sequence_inference.py (baseline
input, all 218 frames) -- no new inference run. See
docs/pipelines/endodac_scale.md.
"""
import json

import numpy as np
from PIL import Image

SEQ_DIR = "/data1_ycao/chua/projects/mrsp/scratch/c1_cecum_t1_v1"
PRED_DIR = "/data1_ycao/chua/projects/mrsp/results/pipelines/endodac_full_run/c1_cecum_t1_v1"
N_FRAMES = 218


def load_gt_depth_mm(frame_idx):
    path = f"{SEQ_DIR}/depth/{frame_idx:04d}_depth.tiff"
    gt = np.array(Image.open(path))
    if gt.ndim == 3:
        gt = gt[:, :, 0]
    return gt.astype(np.float32) * (100.0 / 65535.0)


def load_pred_depth(frame_idx):
    return np.load(f"{PRED_DIR}/depth/{frame_idx:04d}_pred_depth.npy")


def umeyama(src, dst):
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
    aligned = (c * (R @ src.T).T) + t
    return R, t, c, aligned


# ---- 1. Depth scale: per-frame median(GT)/median(pred) over GT-valid pixels ----
print("=== 1. Depth scale, per frame ===")
depth_ratios = []
for i in range(N_FRAMES):
    gt_mm = load_gt_depth_mm(i)
    pred = load_pred_depth(i)
    valid = (gt_mm > 0) & (gt_mm < 100)
    ratio = np.median(gt_mm[valid]) / np.median(pred[valid])
    depth_ratios.append(ratio)
depth_ratios = np.array(depth_ratios)

d_median = np.median(depth_ratios)
d_q1, d_q3 = np.percentile(depth_ratios, [25, 75])
d_iqr = d_q3 - d_q1
d_min, d_max = depth_ratios.min(), depth_ratios.max()
d_std = depth_ratios.std()
print(f"n={N_FRAMES}, median={d_median:.4f}, IQR=[{d_q1:.4f}, {d_q3:.4f}] (width {d_iqr:.4f}), "
      f"min={d_min:.4f}, max={d_max:.4f}, std={d_std:.4f}")
print(f"relative IQR (IQR/median): {d_iqr/d_median:.4f}")
print(f"relative range ((max-min)/median): {(d_max-d_min)/d_median:.4f}")

# ---- 2. Pose scale: Umeyama-with-scale over the full 218-frame trajectory ----
print("\n=== 2. Pose scale, full trajectory ===")
with open(f"{SEQ_DIR}/pose.txt") as f:
    lines = f.readlines()
gt_positions = np.array([
    (np.array([float(x) for x in line.strip().split(",")]).reshape(4, 4).T)[:3, 3]
    for line in lines[:N_FRAMES]
])
pred_poses = np.load(f"{PRED_DIR}/poses_pred.npy")
pred_positions = pred_poses[:, :3, 3]

R, t, s_pose, aligned_pred = umeyama(pred_positions, gt_positions)
ate = np.sqrt(np.mean(np.sum((aligned_pred - gt_positions) ** 2, axis=1)))
print(f"Umeyama scale s_pose = {s_pose:.6f}")
print(f"post-alignment ATE (RMSE) = {ate:.4f} mm")

# ---- 3. Compare depth scale vs pose scale ----
print("\n=== 3. Depth scale vs pose scale comparison ===")
ratio_pose_over_depth = s_pose / d_median
pct_diff = (ratio_pose_over_depth - 1) * 100
print(f"pose scale s_pose = {s_pose:.4f}")
print(f"median depth scale = {d_median:.4f}")
print(f"ratio (pose/depth) = {ratio_pose_over_depth:.4f}  ({pct_diff:+.1f}%)")
print(f"differ by more than 20%? {'YES' if abs(pct_diff) > 20 else 'no'}")

# ---- 4. Trajectory drift ----
print("\n=== 4. Trajectory drift ===")
gt_step_lens = np.linalg.norm(np.diff(gt_positions, axis=0), axis=1)
gt_path_length = gt_step_lens.sum()
pred_step_lens_raw = np.linalg.norm(np.diff(pred_positions, axis=0), axis=1)
pred_path_length_raw = pred_step_lens_raw.sum()
pred_path_length_scaled = pred_path_length_raw * s_pose
aligned_step_lens = np.linalg.norm(np.diff(aligned_pred, axis=0), axis=1)
aligned_path_length = aligned_step_lens.sum()  # cross-check, should equal pred_path_length_scaled

endpoint_err = np.linalg.norm(aligned_pred[-1] - gt_positions[-1])
endpoint_err_frac_of_gt_path = endpoint_err / gt_path_length

print(f"GT path length: {gt_path_length:.2f} mm")
print(f"predicted path length (raw network units): {pred_path_length_raw:.6f}")
print(f"predicted path length (scaled by s_pose): {pred_path_length_scaled:.2f} mm")
print(f"predicted path length (aligned trajectory, cross-check): {aligned_path_length:.2f} mm")
print(f"path length ratio (pred_scaled / GT): {pred_path_length_scaled/gt_path_length:.4f}")
print(f"endpoint error (aligned pred vs GT, frame 217): {endpoint_err:.2f} mm")
print(f"endpoint error as fraction of GT path length: {endpoint_err_frac_of_gt_path:.4f} "
      f"({endpoint_err_frac_of_gt_path*100:.2f}%)")

results = {
    "depth_scale": {
        "n_frames": N_FRAMES, "median": float(d_median), "q1": float(d_q1), "q3": float(d_q3),
        "iqr": float(d_iqr), "min": float(d_min), "max": float(d_max), "std": float(d_std),
        "relative_iqr": float(d_iqr / d_median), "relative_range": float((d_max - d_min) / d_median),
        "per_frame_ratios": depth_ratios.tolist(),
    },
    "pose_scale": {"s_pose": float(s_pose), "ate_mm": float(ate)},
    "comparison": {"ratio_pose_over_depth": float(ratio_pose_over_depth), "pct_diff": float(pct_diff)},
    "trajectory_drift": {
        "gt_path_length_mm": float(gt_path_length),
        "pred_path_length_scaled_mm": float(pred_path_length_scaled),
        "path_length_ratio": float(pred_path_length_scaled / gt_path_length),
        "endpoint_error_mm": float(endpoint_err),
        "endpoint_error_frac_of_gt_path": float(endpoint_err_frac_of_gt_path),
    },
}
with open("/data1_ycao/chua/projects/mrsp/results/pipelines/endodac_scale_analysis.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nWrote results/pipelines/endodac_scale_analysis.json")
