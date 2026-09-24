"""Supplementary to endodac_pose_validation.py: the global chain+Umeyama+ATE
test came back inconclusive (near-identical ATE for both chaining
hypotheses, on two different windows). This does a PER-STEP comparison
instead -- no global alignment flexibility to absorb a systematic error --
comparing predicted relative rotation/translation-direction directly
against the GT relative transform for each consecutive pair, under both
hypotheses. Baseline input (no crop, no inpaint).
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
START_FRAME = int(os.environ.get("START_FRAME", "139"))
N_FRAMES = 20
FEED_W, FEED_H = 320, 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pose_encoder = encoders.ResnetEncoder(18, False, 2)
pose_encoder.load_state_dict(torch.load(os.path.join(CKPT_DIR, "pose_encoder.pth"), map_location="cpu"))
pose_decoder = decoders.PoseDecoder(pose_encoder.num_ch_enc, 1, 2)
pose_decoder.load_state_dict(torch.load(os.path.join(CKPT_DIR, "pose.pth"), map_location="cpu"))
pose_encoder.to(device).eval()
pose_decoder.to(device).eval()


def load_tensor(frame_idx):
    path = os.path.join(SEQ_DIR, "rgb", f"{frame_idx:04d}.png")
    im = Image.open(path).convert("RGB").resize((FEED_W, FEED_H), Image.LANCZOS)
    return torch.from_numpy(np.array(im)).permute(2, 0, 1).float().div(255.0).unsqueeze(0)


with open(os.path.join(SEQ_DIR, "pose.txt")) as f:
    lines = f.readlines()


def gt_c2w(i):
    vals = [float(x) for x in lines[i].strip().split(",")]
    return np.array(vals).reshape(4, 4).T


def rot_angle_deg(R):
    """Angle of rotation matrix R, in degrees."""
    c = (np.trace(R) - 1) / 2
    c = np.clip(c, -1, 1)
    return np.degrees(np.arccos(c))


rot_err_A, rot_err_B = [], []
cos_sim_A, cos_sim_B = [], []

with torch.no_grad():
    for i in range(START_FRAME, START_FRAME + N_FRAMES - 1):
        t_i = load_tensor(i).to(device)
        t_ip1 = load_tensor(i + 1).to(device)
        all_color_aug = torch.cat([t_ip1, t_i], 1)
        features = [pose_encoder(all_color_aug)]
        axisangle, translation, _ = pose_decoder(features)
        T = transformation_from_parameters(axisangle[:, 0], translation[:, 0]).cpu().numpy()[0]
        T_inv = np.linalg.inv(T)

        C2W_i = gt_c2w(i)
        C2W_ip1 = gt_c2w(i + 1)
        G = np.linalg.inv(C2W_i) @ C2W_ip1  # GT relative transform, hypothesis-A-style labeling

        R_gt, t_gt = G[:3, :3], G[:3, 3]
        R_A, t_A = T[:3, :3], T[:3, 3]
        R_B, t_B = T_inv[:3, :3], T_inv[:3, 3]

        rot_err_A.append(rot_angle_deg(R_gt.T @ R_A))
        rot_err_B.append(rot_angle_deg(R_gt.T @ R_B))

        t_gt_n = t_gt / (np.linalg.norm(t_gt) + 1e-12)
        t_A_n = t_A / (np.linalg.norm(t_A) + 1e-12)
        t_B_n = t_B / (np.linalg.norm(t_B) + 1e-12)
        cos_sim_A.append(float(t_gt_n @ t_A_n))
        cos_sim_B.append(float(t_gt_n @ t_B_n))

        print(f"pair ({i},{i+1}): rot_err_A={rot_err_A[-1]:6.2f} deg, rot_err_B={rot_err_B[-1]:6.2f} deg, "
              f"cos_sim_A={cos_sim_A[-1]:+.4f}, cos_sim_B={cos_sim_B[-1]:+.4f}")

print(f"\n=== summary, window frames {START_FRAME}-{START_FRAME+N_FRAMES-1} ===")
print(f"Hypothesis A (T as predicted): mean rot err = {np.mean(rot_err_A):.2f} deg, "
      f"mean translation-direction cosine sim = {np.mean(cos_sim_A):+.4f}")
print(f"Hypothesis B (inv(T)):         mean rot err = {np.mean(rot_err_B):.2f} deg, "
      f"mean translation-direction cosine sim = {np.mean(cos_sim_B):+.4f}")
