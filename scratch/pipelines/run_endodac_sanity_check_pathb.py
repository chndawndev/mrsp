"""Diagnostic follow-up to run_endodac_sanity_check.py: same 3 frames, but
using "Path B" preprocessing (the repo's own C3VDDataset crop box
(200,180,1150,900) applied before resize) instead of "Path A" (direct
resize, no crop). Purpose: check whether Path A's implausible-looking
output (no lumen/fold structure) is a preprocessing domain-mismatch
artifact. Diagnostic only -- does not change any frozen preprocessing
decision, does not touch src/eval or src/gt. See docs/pipelines/endodac.md.
"""
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

REPO = "/data1_ycao/chua/projects/mrsp/scratch/pipelines/EndoDAC"
sys.path.insert(0, REPO)

import models.endodac as endodac  # noqa: E402
from utils.layers import disp_to_depth  # noqa: E402

CKPT_DIR = os.path.join(REPO, "checkpoints", "endodac")
PRETRAINED_DIR = os.path.join(REPO, "pretrained_model")
SEQ_DIR = "/data1_ycao/chua/projects/mrsp/scratch/c1_cecum_t1_v1"
OUT_DIR = "/data1_ycao/chua/projects/mrsp/results/pipelines/endodac_sanity_check"
FRAMES = ["0000", "0109", "0217"]
CROP_BOX = (200, 180, 1150, 900)  # datasets/c3vd_dataset.py:99

os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

depther_dict = torch.load(os.path.join(CKPT_DIR, "depth_model.pth"), map_location="cpu")
feed_height = depther_dict["height"]
feed_width = depther_dict["width"]

depther = endodac.endodac(
    backbone_size="base", r=4, lora_type="dvlora", image_shape=(224, 280),
    pretrained_path=PRETRAINED_DIR, residual_block_indexes=[2, 5, 8, 11],
    include_cls_token=True,
)
model_dict = depther.state_dict()
depther.load_state_dict({k: v for k, v in depther_dict.items() if k in model_dict})
depther.to(device)
depther.eval()

def rescale_depth_mm_to_u8(depth_mm, vmax=100.0):
    d = np.clip(depth_mm, 0, vmax) / vmax * 255.0
    return d.astype(np.uint8)

with torch.no_grad():
    for frame in FRAMES:
        rgb_path = os.path.join(SEQ_DIR, "rgb", f"{frame}.png")
        gt_path = os.path.join(SEQ_DIR, "depth", f"{frame}_depth.tiff")

        input_image = Image.open(rgb_path).convert("RGB")
        cropped = input_image.crop(CROP_BOX)
        crop_w, crop_h = cropped.size
        resized = cropped.resize((feed_width, feed_height), Image.LANCZOS)
        tensor = torch.from_numpy(np.array(resized)).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
        tensor = tensor.to(device)

        outputs = depther(tensor)
        disp = outputs[("disp", 0)]
        _, pred_depth = disp_to_depth(disp, 0.1, 150)
        pred_depth = torch.nn.functional.interpolate(
            pred_depth, (crop_h, crop_w), mode="bilinear", align_corners=False
        )
        pred_depth_np = pred_depth.squeeze().cpu().numpy()

        gt = np.array(Image.open(gt_path))
        if gt.ndim == 3:
            gt = gt[:, :, 0]
        gt_mm = gt.astype(np.float32) * (100.0 / 65535.0)
        gt_mm_cropped = gt_mm[CROP_BOX[1]:CROP_BOX[3], CROP_BOX[0]:CROP_BOX[2]]

        valid = (gt_mm_cropped > 0) & (gt_mm_cropped < 100)
        ratio = np.median(gt_mm_cropped[valid]) / np.median(pred_depth_np[valid])
        pred_scaled_mm = pred_depth_np * ratio

        Image.fromarray(rescale_depth_mm_to_u8(pred_scaled_mm)).save(
            os.path.join(OUT_DIR, f"{frame}_pred_depth_pathB_medianscaled.png")
        )
        Image.fromarray(rescale_depth_mm_to_u8(gt_mm_cropped)).save(
            os.path.join(OUT_DIR, f"{frame}_gt_depth_pathB_cropped.png")
        )
        print(f"frame {frame}: Path B scale ratio {ratio:.4f}")
