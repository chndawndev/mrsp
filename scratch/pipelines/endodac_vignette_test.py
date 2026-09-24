"""Test the vignette hypothesis: does the hard black vignette border (a
pattern the SCARED-trained backbone never saw) explain EndoDAC's flat,
lumen-free depth prediction on C3VDv2? Three input variants on the same 3
sanity-check frames, no new downloads. See docs/pipelines/endodac.md.

Vignette geometry from docs/gpu_validation.md: optical center
cx=677.74, cy=543.06 (native 1350x1080); all-valid radius < 706px,
mixed-valid 706-864px (mask isn't a perfect circle), all-invalid > 864px.
"""
import os
import sys
import math

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from scipy.stats import pearsonr, spearmanr

REPO = "/data1_ycao/chua/projects/mrsp/scratch/pipelines/EndoDAC"
sys.path.insert(0, REPO)
import models.endodac as endodac  # noqa: E402
from utils.layers import disp_to_depth  # noqa: E402

CKPT_DIR = os.path.join(REPO, "checkpoints", "endodac")
PRETRAINED_DIR = os.path.join(REPO, "pretrained_model")
SEQ_DIR = "/data1_ycao/chua/projects/mrsp/scratch/c1_cecum_t1_v1"
OUT_DIR = "/data1_ycao/chua/projects/mrsp/results/pipelines/endodac_vignette_test"
FRAMES = ["0000", "0109", "0217"]

CX, CY = 677.74, 543.06
R_SAFE = 706.0  # radius below which every pixel is confirmed valid (gpu_validation.md)
HALF_SIDE = R_SAFE / math.sqrt(2)  # largest axis-aligned square fully inside r=706 circle

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


def load_gt_mm(path):
    gt = np.array(Image.open(path))
    if gt.ndim == 3:
        gt = gt[:, :, 0]
    return gt.astype(np.float32) * (100.0 / 65535.0)


def inpaint_vignette(rgb_arr):
    """Nearest-valid-pixel fill of pure-black pixels."""
    black_mask = (rgb_arr.sum(axis=2) == 0)
    filled = rgb_arr.copy()
    if black_mask.any():
        idx = ndimage.distance_transform_edt(black_mask, return_distances=False, return_indices=True)
        for c in range(3):
            filled[:, :, c] = rgb_arr[:, :, c][tuple(idx)]
    return filled, black_mask


def run_variant(rgb_pil, name, gt_mm_crop):
    """Run inference on a PIL image, return (pred_disp_upsampled, pred_depth_mm_scaled, correlations)."""
    ow, oh = rgb_pil.size
    resized = rgb_pil.resize((feed_width, feed_height), Image.LANCZOS)
    tensor = torch.from_numpy(np.array(resized)).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = depther(tensor)
        disp = outputs[("disp", 0)]
        disp_up = torch.nn.functional.interpolate(disp, (oh, ow), mode="bilinear", align_corners=False)
        disp_np = disp_up.squeeze().cpu().numpy()
        _, depth = disp_to_depth(disp, 0.1, 150)
        depth_up = torch.nn.functional.interpolate(depth, (oh, ow), mode="bilinear", align_corners=False)
        depth_np = depth_up.squeeze().cpu().numpy()

    valid = (gt_mm_crop > 0) & (gt_mm_crop < 100)
    n_valid = int(valid.sum())
    pearson_r, pearson_p = pearsonr(disp_np[valid], gt_mm_crop[valid])
    spearman_r, spearman_p = spearmanr(disp_np[valid], gt_mm_crop[valid])

    ratio = np.median(gt_mm_crop[valid]) / np.median(depth_np[valid])
    pred_scaled_mm = depth_np * ratio
    vis = (np.clip(pred_scaled_mm, 0, 100) / 100.0 * 255.0).astype(np.uint8)

    return {
        "name": name, "n_valid": n_valid,
        "pearson_r": pearson_r, "pearson_p": pearson_p,
        "spearman_r": spearman_r, "spearman_p": spearman_p,
        "disp_std_over_mean": disp_np[valid].std() / (abs(disp_np[valid].mean()) + 1e-12),
        "vis": vis,
    }


results_summary = []

for frame in FRAMES:
    rgb_path = os.path.join(SEQ_DIR, "rgb", f"{frame}.png")
    gt_path = os.path.join(SEQ_DIR, "depth", f"{frame}_depth.tiff")
    rgb_full = Image.open(rgb_path).convert("RGB")
    rgb_arr = np.array(rgb_full)
    gt_mm_full = load_gt_mm(gt_path)

    print(f"\n=== frame {frame} ===")

    # (a) baseline -- as-is, full frame
    res_a = run_variant(rgb_full, "a_baseline", gt_mm_full)
    Image.fromarray(res_a["vis"]).save(os.path.join(OUT_DIR, f"{frame}_a_baseline_pred.png"))

    # (b) vignette inpainted -- nearest-valid-pixel fill, same full frame size
    inpainted_arr, black_mask = inpaint_vignette(rgb_arr)
    black_frac = black_mask.mean()
    rgb_inpainted = Image.fromarray(inpainted_arr)
    res_b = run_variant(rgb_inpainted, "b_inpainted", gt_mm_full)
    Image.fromarray(res_b["vis"]).save(os.path.join(OUT_DIR, f"{frame}_b_inpainted_pred.png"))
    Image.fromarray(inpainted_arr).save(os.path.join(OUT_DIR, f"{frame}_b_inpainted_input.png"))

    # (c) tight circular crop -- largest axis-aligned square fully inside r=706 circle
    left = int(round(CX - HALF_SIDE))
    right = int(round(CX + HALF_SIDE))
    upper = int(round(CY - HALF_SIDE))
    lower = int(round(CY + HALF_SIDE))
    rgb_crop = rgb_full.crop((left, upper, right, lower))
    gt_mm_crop = gt_mm_full[upper:lower, left:right]
    black_frac_crop = (np.array(rgb_crop).sum(axis=2) == 0).mean()
    res_c = run_variant(rgb_crop, "c_tight_crop", gt_mm_crop)
    Image.fromarray(res_c["vis"]).save(os.path.join(OUT_DIR, f"{frame}_c_tightcrop_pred.png"))
    Image.fromarray(np.array(rgb_crop)).save(os.path.join(OUT_DIR, f"{frame}_c_tightcrop_input.png"))
    gt_vis_crop = (np.clip(gt_mm_crop, 0, 100) / 100.0 * 255.0).astype(np.uint8)
    Image.fromarray(gt_vis_crop).save(os.path.join(OUT_DIR, f"{frame}_c_tightcrop_gt.png"))

    print(f"  black-pixel fraction: full frame {black_mask.mean():.4f}, crop box {black_frac_crop:.6f} "
          f"(box: left={left} upper={upper} right={right} lower={lower}, size {right-left}x{lower-upper})")

    for res in (res_a, res_b, res_c):
        print(f"  [{res['name']}] n_valid={res['n_valid']}, "
              f"Pearson r={res['pearson_r']:.4f} (p={res['pearson_p']:.2e}), "
              f"Spearman rho={res['spearman_r']:.4f} (p={res['spearman_p']:.2e}), "
              f"disp std/|mean|={res['disp_std_over_mean']:.4f}")
        results_summary.append({"frame": frame, **{k: v for k, v in res.items() if k != "vis"}})

print("\n=== summary across frames ===")
for name in ("a_baseline", "b_inpainted", "c_tight_crop"):
    rs = [r["pearson_r"] for r in results_summary if r["name"] == name]
    ss = [r["spearman_r"] for r in results_summary if r["name"] == name]
    print(f"{name}: mean Pearson r = {np.mean(rs):.4f}, mean Spearman rho = {np.mean(ss):.4f}")
