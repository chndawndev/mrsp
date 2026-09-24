"""One-off sanity check: run EndoDAC depth inference on 3 frames of
c1_cecum_t1_v1 and save predicted depth next to GT depth as PNGs, for
visual (not metric) comparison. See docs/pipelines/endodac.md section 5.

Preprocessing follows "Path A" documented there: resize the original
fisheye frame directly to the checkpoint's feed size, no crop, no
undistortion (frozen project decision, CLAUDE.md).
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

os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

depther_dict = torch.load(os.path.join(CKPT_DIR, "depth_model.pth"), map_location="cpu")
feed_height = depther_dict["height"]
feed_width = depther_dict["width"]
print(f"feed size from checkpoint: {feed_width}x{feed_height}")

depther = endodac.endodac(
    backbone_size="base",
    r=4,
    lora_type="dvlora",
    image_shape=(224, 280),
    pretrained_path=PRETRAINED_DIR,
    residual_block_indexes=[2, 5, 8, 11],
    include_cls_token=True,
)
model_dict = depther.state_dict()
depther.load_state_dict({k: v for k, v in depther_dict.items() if k in model_dict})
depther.to(device)
depther.eval()

def rescale_depth_mm_to_u8(depth_mm, vmax=100.0):
    d = np.clip(depth_mm, 0, vmax) / vmax * 255.0
    return d.astype(np.uint8)

times = []
with torch.no_grad():
    for frame in FRAMES:
        rgb_path = os.path.join(SEQ_DIR, "rgb", f"{frame}.png")
        gt_path = os.path.join(SEQ_DIR, "depth", f"{frame}_depth.tiff")

        input_image = Image.open(rgb_path).convert("RGB")
        original_width, original_height = input_image.size
        resized = input_image.resize((feed_width, feed_height), Image.LANCZOS)
        tensor = torch.from_numpy(np.array(resized)).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
        tensor = tensor.to(device)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        outputs = depther(tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.time() - t0)

        disp = outputs[("disp", 0)]
        _, pred_depth = disp_to_depth(disp, 0.1, 150)
        pred_depth = torch.nn.functional.interpolate(
            pred_depth, (original_height, original_width), mode="bilinear", align_corners=False
        )
        pred_depth_np = pred_depth.squeeze().cpu().numpy()

        gt = np.array(Image.open(gt_path))
        if gt.ndim == 3:
            gt = gt[:, :, 0]
        gt_mm = gt.astype(np.float32) * (100.0 / 65535.0)

        # Median-scale predicted (unitless, monodepth2 convention) depth
        # against GT for a visually comparable range only -- NOT a metric,
        # just so both PNGs use a comparable colormap scale for eyeballing.
        valid = (gt_mm > 0) & (gt_mm < 100)
        ratio = np.median(gt_mm[valid]) / np.median(pred_depth_np[valid])
        pred_scaled_mm = pred_depth_np * ratio

        Image.fromarray(rescale_depth_mm_to_u8(pred_scaled_mm)).save(
            os.path.join(OUT_DIR, f"{frame}_pred_depth_medianscaled.png")
        )
        Image.fromarray(rescale_depth_mm_to_u8(gt_mm)).save(
            os.path.join(OUT_DIR, f"{frame}_gt_depth.png")
        )
        np.save(os.path.join(OUT_DIR, f"{frame}_pred_depth_raw.npy"), pred_depth_np)
        print(f"frame {frame}: inference {times[-1]*1000:.1f} ms, median-scale ratio {ratio:.4f}")

print(f"mean inference time over {len(times)} frames (after warmup frame 1): "
      f"{np.mean(times[1:])*1000:.2f} ms" if len(times) > 1 else "")
print(f"all {len(times)} frames: {np.mean(times)*1000:.2f} ms mean, {times}")
