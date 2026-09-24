"""Empirically validate EndoDAC's predicted relative-pose convention against
our frozen pose.txt camera-to-world convention (reshape(4,4).T). Runs pose
inference on 20 consecutive frames of c1_cecum_t1_v1, chains relative
transforms into a trajectory under two hypotheses, Umeyama-aligns each to
GT, and reports ATE. Baseline input (no crop, no inpaint). See
docs/pipelines/endodac.md.
"""
import os
import sys

import numpy as np
import torch
from PIL import Image

REPO = "/data1_ycao/chua/projects/mrsp/scratch/pipelines/EndoDAC"
sys.path.insert(0, REPO)
import models.encoders as encoders  # noqa: E402
import models.decoders as decoders  # noqa: E402
from utils.layers import transformation_from_parameters  # noqa: E402

CKPT_DIR = os.path.join(REPO, "checkpoints", "endodac")
SEQ_DIR = "/data1_ycao/chua/projects/mrsp/scratch/c1_cecum_t1_v1"
START_FRAME = int(os.environ.get("START_FRAME", "0"))
N_FRAMES = 20
FEED_W, FEED_H = 320, 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- build pose network, exactly matching evaluate_pose.py's construction ----
pose_encoder = encoders.ResnetEncoder(18, False, 2)
pose_encoder.load_state_dict(torch.load(os.path.join(CKPT_DIR, "pose_encoder.pth"), map_location="cpu"))
pose_decoder = decoders.PoseDecoder(pose_encoder.num_ch_enc, 1, 2)
pose_decoder.load_state_dict(torch.load(os.path.join(CKPT_DIR, "pose.pth"), map_location="cpu"))
pose_encoder.to(device).eval()
pose_decoder.to(device).eval()


print(f"START_FRAME={START_FRAME}, N_FRAMES={N_FRAMES}")


def load_tensor(frame_idx):
    path = os.path.join(SEQ_DIR, "rgb", f"{frame_idx:04d}.png")
    im = Image.open(path).convert("RGB").resize((FEED_W, FEED_H), Image.LANCZOS)
    return torch.from_numpy(np.array(im)).permute(2, 0, 1).float().div(255.0).unsqueeze(0)


# ---- run pose inference on 19 consecutive pairs (i, i+1) for i in 0..18 ----
T_list = []  # T_list[i] = transform predicted for pair (frame i, frame i+1)
with torch.no_grad():
    for i in range(START_FRAME, START_FRAME + N_FRAMES - 1):
        t_i = load_tensor(i).to(device)
        t_ip1 = load_tensor(i + 1).to(device)
        # exactly matches evaluate_pose.py: cat([color_1, color_0], dim=1)
        all_color_aug = torch.cat([t_ip1, t_i], 1)
        features = [pose_encoder(all_color_aug)]
        axisangle, translation, _ = pose_decoder(features)
        T = transformation_from_parameters(axisangle[:, 0], translation[:, 0]).cpu().numpy()[0]
        T_list.append(T)
        print(f"pair ({i},{i+1}): translation norm (network units) = {np.linalg.norm(T[:3,3]):.6f}")

# ---- GT trajectory: pose.txt lines 0..19, our frozen reshape(4,4).T convention ----
with open(os.path.join(SEQ_DIR, "pose.txt")) as f:
    lines = f.readlines()
gt_poses = []
for i in range(START_FRAME, START_FRAME + N_FRAMES):
    vals = [float(x) for x in lines[i].strip().split(",")]
    M = np.array(vals).reshape(4, 4).T
    gt_poses.append(M)
gt_positions = np.array([M[:3, 3] for M in gt_poses])  # (20, 3)
gt_step_lens = np.linalg.norm(np.diff(gt_positions, axis=0), axis=1)
print(f"GT window total path length: {gt_step_lens.sum():.4f} mm, "
      f"end-to-end: {np.linalg.norm(gt_positions[-1]-gt_positions[0]):.4f} mm, "
      f"per-step: {gt_step_lens}")


def chain(T_list, invert_each):
    poses = [np.eye(4)]
    for T in T_list:
        Tuse = np.linalg.inv(T) if invert_each else T
        poses.append(poses[-1] @ Tuse)
    return poses


def umeyama(src, dst):
    """Umeyama (1991) closed-form similarity alignment (rotation, translation,
    scale) mapping src -> dst. src, dst: (N,3). Returns (R, t, c, aligned_src)."""
    N = src.shape[0]
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
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


def ate(aligned, dst):
    return np.sqrt(np.mean(np.sum((aligned - dst) ** 2, axis=1)))


print("\n=== Hypothesis A: chain T_i as predicted (right-multiply, as in evaluate_pose.py's dump_xyz) ===")
poses_A = chain(T_list, invert_each=False)
pos_A = np.array([M[:3, 3] for M in poses_A])
R_A, t_A, c_A, aligned_A = umeyama(pos_A, gt_positions)
ate_A = ate(aligned_A, gt_positions)
print(f"Umeyama scale c = {c_A:.6f}")
print(f"ATE (RMSE after Umeyama alignment) = {ate_A:.6f} mm")

print("\n=== Hypothesis B: chain inv(T_i) (right-multiply the inverse) ===")
poses_B = chain(T_list, invert_each=True)
pos_B = np.array([M[:3, 3] for M in poses_B])
R_B, t_B, c_B, aligned_B = umeyama(pos_B, gt_positions)
ate_B = ate(aligned_B, gt_positions)
print(f"Umeyama scale c = {c_B:.6f}")
print(f"ATE (RMSE after Umeyama alignment) = {ate_B:.6f} mm")

print(f"\n=== conclusion ===")
print(f"Hypothesis A ATE: {ate_A:.4f} mm")
print(f"Hypothesis B ATE: {ate_B:.4f} mm")
winner = "A (chain as predicted)" if ate_A < ate_B else "B (chain the inverse)"
ratio = max(ate_A, ate_B) / min(ate_A, ate_B)
print(f"Winner: {winner}, ratio of ATEs = {ratio:.2f}x")

OUT = "/data1_ycao/chua/projects/mrsp/results/pipelines"
np.save(f"{OUT}/endodac_pose_validation_f{START_FRAME}_gt_positions.npy", gt_positions)
np.save(f"{OUT}/endodac_pose_validation_f{START_FRAME}_pos_A.npy", pos_A)
np.save(f"{OUT}/endodac_pose_validation_f{START_FRAME}_pos_B.npy", pos_B)
np.save(f"{OUT}/endodac_pose_validation_f{START_FRAME}_aligned_A.npy", aligned_A)
np.save(f"{OUT}/endodac_pose_validation_f{START_FRAME}_aligned_B.npy", aligned_B)
