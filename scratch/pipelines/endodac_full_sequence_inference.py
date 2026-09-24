"""Depth + pose + intrinsics inference on the full c1_cecum_t1_v1 (218
frames), baseline input (no crop, no inpaint) per instructions. One
sequence only -- not a full-corpus run. Saves output in a layout intended
for later evaluation code to consume; see the generated MANIFEST.md for
the exact format. See docs/pipelines/endodac.md section 8.
"""
import os
import sys
import time
import json

import numpy as np
import torch
from PIL import Image

REPO = "/data1_ycao/chua/projects/mrsp/scratch/pipelines/EndoDAC"
sys.path.insert(0, REPO)
import models.endodac as endodac  # noqa: E402
import models.encoders as encoders  # noqa: E402
import models.decoders as decoders  # noqa: E402
from utils.layers import disp_to_depth, transformation_from_parameters  # noqa: E402

CKPT_DIR = os.path.join(REPO, "checkpoints", "endodac")
PRETRAINED_DIR = os.path.join(REPO, "pretrained_model")
SEQ_DIR = "/data1_ycao/chua/projects/mrsp/scratch/c1_cecum_t1_v1"
OUT_DIR = "/data1_ycao/chua/projects/mrsp/results/pipelines/endodac_full_run/c1_cecum_t1_v1"
N_FRAMES = 218
FEED_W, FEED_H = 320, 256
MIN_DEPTH, MAX_DEPTH = 0.1, 150.0  # disp_to_depth params, network's native (scale-ambiguous) units

os.makedirs(os.path.join(OUT_DIR, "depth"), exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device={device}, N_FRAMES={N_FRAMES}")

# ---- depth model ----
depther_dict = torch.load(os.path.join(CKPT_DIR, "depth_model.pth"), map_location="cpu")
depther = endodac.endodac(
    backbone_size="base", r=4, lora_type="dvlora", image_shape=(224, 280),
    pretrained_path=PRETRAINED_DIR, residual_block_indexes=[2, 5, 8, 11],
    include_cls_token=True,
)
model_dict = depther.state_dict()
depther.load_state_dict({k: v for k, v in depther_dict.items() if k in model_dict})
depther.to(device).eval()

# ---- pose + intrinsics models ----
pose_encoder = encoders.ResnetEncoder(18, False, 2)
pose_encoder.load_state_dict(torch.load(os.path.join(CKPT_DIR, "pose_encoder.pth"), map_location="cpu"))
pose_decoder = decoders.PoseDecoder(pose_encoder.num_ch_enc, 1, 2)
pose_decoder.load_state_dict(torch.load(os.path.join(CKPT_DIR, "pose.pth"), map_location="cpu"))
intrinsics_decoder = decoders.IntrinsicsHead(pose_encoder.num_ch_enc)
intrinsics_decoder.load_state_dict(torch.load(os.path.join(CKPT_DIR, "intrinsics_head.pth"), map_location="cpu"))
pose_encoder.to(device).eval()
pose_decoder.to(device).eval()
intrinsics_decoder.to(device).eval()


def load_tensor(frame_idx):
    path = os.path.join(SEQ_DIR, "rgb", f"{frame_idx:04d}.png")
    im = Image.open(path).convert("RGB")
    ow, oh = im.size
    resized = im.resize((FEED_W, FEED_H), Image.LANCZOS)
    t = torch.from_numpy(np.array(resized)).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
    return t, ow, oh


# ---- depth pass, all 218 frames ----
print("\n=== depth inference ===")
t0 = time.time()
with torch.no_grad():
    for i in range(N_FRAMES):
        tensor, ow, oh = load_tensor(i)
        tensor = tensor.to(device)
        outputs = depther(tensor)
        disp = outputs[("disp", 0)]
        _, depth = disp_to_depth(disp, MIN_DEPTH, MAX_DEPTH)
        depth_up = torch.nn.functional.interpolate(depth, (oh, ow), mode="bilinear", align_corners=False)
        depth_np = depth_up.squeeze().cpu().numpy().astype(np.float32)
        np.save(os.path.join(OUT_DIR, "depth", f"{i:04d}_pred_depth.npy"), depth_np)
        if i % 50 == 0 or i == N_FRAMES - 1:
            print(f"  frame {i}/{N_FRAMES-1}, elapsed {time.time()-t0:.1f}s")
depth_elapsed = time.time() - t0
print(f"depth inference done: {depth_elapsed:.2f}s total, {depth_elapsed/N_FRAMES*1000:.2f}ms/frame")

# ---- pose + intrinsics pass, 217 consecutive pairs ----
print("\n=== pose + intrinsics inference ===")
t0 = time.time()
relative_T = []       # raw network output, per pair, BEFORE the Check-7 inversion
intrinsics_list = []  # per pair, feed-resolution pixel units (320x256)
with torch.no_grad():
    for i in range(N_FRAMES - 1):
        t_i, _, _ = load_tensor(i)
        t_ip1, _, _ = load_tensor(i + 1)
        t_i, t_ip1 = t_i.to(device), t_ip1.to(device)
        all_color_aug = torch.cat([t_ip1, t_i], 1)
        features = [pose_encoder(all_color_aug)]
        axisangle, translation, intermediate_feature = pose_decoder(features)
        T = transformation_from_parameters(axisangle[:, 0], translation[:, 0]).cpu().numpy()[0]
        relative_T.append(T)

        K = intrinsics_decoder(intermediate_feature, FEED_W, FEED_H)[:, :3, :3].cpu().numpy()[0]
        intrinsics_list.append(K)
        if i % 50 == 0 or i == N_FRAMES - 2:
            print(f"  pair ({i},{i+1}), elapsed {time.time()-t0:.1f}s")
pose_elapsed = time.time() - t0
print(f"pose+intrinsics inference done: {pose_elapsed:.2f}s total, "
      f"{pose_elapsed/(N_FRAMES-1)*1000:.2f}ms/pair")

# ---- chain into absolute camera-to-world poses, INVERTING each T per Check 7's finding ----
poses = [np.eye(4, dtype=np.float32)]
for T in relative_T:
    T_inv = np.linalg.inv(T)
    poses.append((poses[-1] @ T_inv).astype(np.float32))
poses = np.stack(poses)  # (218, 4, 4)

np.save(os.path.join(OUT_DIR, "poses_pred.npy"), poses)
with open(os.path.join(OUT_DIR, "poses_pred.txt"), "w") as f:
    for M in poses:
        vals = M.T.flatten()  # inverse of the project's own reshape(4,4).T loading convention
        f.write(",".join(f"{v:.6f}" for v in vals) + "\n")

intrinsics_arr = np.stack(intrinsics_list).astype(np.float32)  # (217, 3, 3), feed-res (320x256) pixel units
np.save(os.path.join(OUT_DIR, "intrinsics_pred_per_pair.npy"), intrinsics_arr)
fx = intrinsics_arr[:, 0, 0]
fy = intrinsics_arr[:, 1, 1]
cx = intrinsics_arr[:, 0, 2]
cy = intrinsics_arr[:, 1, 2]
intrinsics_summary = {
    "feed_resolution_wh": [FEED_W, FEED_H],
    "fx_mean": float(fx.mean()), "fx_std": float(fx.std()),
    "fy_mean": float(fy.mean()), "fy_std": float(fy.std()),
    "cx_mean": float(cx.mean()), "cx_std": float(cx.std()),
    "cy_mean": float(cy.mean()), "cy_std": float(cy.std()),
    "n_pairs": len(intrinsics_list),
}
with open(os.path.join(OUT_DIR, "intrinsics_pred_summary.json"), "w") as f:
    json.dump(intrinsics_summary, f, indent=2)
print("\nintrinsics summary:", json.dumps(intrinsics_summary, indent=2))

manifest = {
    "sequence": "c1_cecum_t1_v1",
    "n_frames": N_FRAMES,
    "input_variant": "baseline (Path A: full fisheye frame, resize only, no crop, no inpaint)",
    "feed_resolution_wh": [FEED_W, FEED_H],
    "depth": {
        "path": "depth/{frame:04d}_pred_depth.npy",
        "dtype": "float32", "shape": "[H, W] at native frame resolution (1350x1080)",
        "units": "network-native, scale-ambiguous (monodepth2 disp_to_depth convention, "
                 "min_depth=0.1 max_depth=150 -- NOT millimeters, NOT comparable to GT depth "
                 "without a scale-recovery step; see docs/pipelines/endodac.md section 3)",
        "recover_raw_disparity": "disp = (1/depth - 1/MAX_DEPTH) / (1/MIN_DEPTH - 1/MAX_DEPTH), "
                                  "MIN_DEPTH=0.1, MAX_DEPTH=150",
    },
    "pose": {
        "path_npy": "poses_pred.npy", "path_txt": "poses_pred.txt",
        "dtype": "float32", "shape": "[218, 4, 4]",
        "convention": "camera-to-world, frame 0 = identity (arbitrary origin, no absolute GT anchor). "
                       "poses_pred.txt uses the SAME 16-comma-separated-float-per-line, "
                       "reshape(4,4).T-to-reload format as this project's own pose.txt.",
        "scale": "network-native, scale-ambiguous (monocular self-supervised); NOT metric mm, "
                 "would need a scale-recovery step (e.g. Umeyama with scale against a reference) "
                 "before any metric comparison to GT.",
        "known_issue": "transformation_from_parameters' raw per-pair output was INVERTED before "
                        "chaining, per the empirical convention check in "
                        "docs/pipelines/endodac.md section 7 -- do not chain the raw output directly.",
    },
    "intrinsics": {
        "path_per_pair": "intrinsics_pred_per_pair.npy", "path_summary": "intrinsics_pred_summary.json",
        "dtype": "float32", "shape": "[217, 3, 3]",
        "units": "pixel units, but at the FEED resolution (320x256), NOT the native frame "
                 "resolution (1350x1080) -- rescale fx,fy,cx,cy by (1350/320, 1080/256) respectively "
                 "before using with full-resolution depth/frames.",
        "note": "predicted per consecutive FRAME PAIR (217 values for 218 frames), not per single "
                "frame -- there is no single canonical per-frame intrinsics estimate from this "
                "checkpoint. Use intrinsics_pred_summary.json's mean/std for a single-value estimate, "
                "or the per-pair array if frame-varying intrinsics matter.",
        "camera_model_caveat": "this is a plain pinhole 3x3 matrix; NOT compatible with this "
                                "project's Scaramuzza omnidirectional GT camera model without "
                                "treating it as a rough pinhole approximation only.",
    },
    "timing": {"depth_seconds_total": depth_elapsed, "pose_intrinsics_seconds_total": pose_elapsed},
}
with open(os.path.join(OUT_DIR, "MANIFEST.json"), "w") as f:
    json.dump(manifest, f, indent=2)
print(f"\nWrote MANIFEST.json and all outputs to {OUT_DIR}")
