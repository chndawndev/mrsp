"""MEASURED-only diagnostic: inspect raw network output (pre disp_to_depth),
input tensor statistics, and verify the endodac checkpoint's LoRA/head
weights actually got applied (vs. silently staying at backbone-only init).
Does not modify any repo model code. See docs/pipelines/endodac.md.
"""
import os
import sys

import numpy as np
import torch

REPO = "/data1_ycao/chua/projects/mrsp/scratch/pipelines/EndoDAC"
sys.path.insert(0, REPO)

import models.endodac as endodac  # noqa: E402
from utils.layers import disp_to_depth  # noqa: E402
from PIL import Image  # noqa: E402

CKPT_DIR = os.path.join(REPO, "checkpoints", "endodac")
PRETRAINED_DIR = os.path.join(REPO, "pretrained_model")
SEQ_DIR = "/data1_ycao/chua/projects/mrsp/scratch/c1_cecum_t1_v1"
FRAMES = ["0000", "0109", "0217"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"numpy {np.__version__}, torch {torch.__version__}, device {device}")

depther_dict = torch.load(os.path.join(CKPT_DIR, "depth_model.pth"), map_location="cpu")
feed_height = depther_dict["height"]
feed_width = depther_dict["width"]
print(f"checkpoint feed size: {feed_width}x{feed_height}")

# ---- Build model A: backbone-only (pretrained_path load, no depth_model.pth) ----
model_backbone_only = endodac.endodac(
    backbone_size="base", r=4, lora_type="dvlora", image_shape=(224, 280),
    pretrained_path=PRETRAINED_DIR, residual_block_indexes=[2, 5, 8, 11],
    include_cls_token=True,
)
backbone_only_state = {k: v.clone().cpu() for k, v in model_backbone_only.state_dict().items()}

# ---- Build model B: backbone + depth_model.pth (what the sanity script actually used) ----
model_full = endodac.endodac(
    backbone_size="base", r=4, lora_type="dvlora", image_shape=(224, 280),
    pretrained_path=PRETRAINED_DIR, residual_block_indexes=[2, 5, 8, 11],
    include_cls_token=True,
)
model_dict = model_full.state_dict()
filtered = {k: v for k, v in depther_dict.items() if k in model_dict}
print(f"\n=== weight-loading key audit ===")
print(f"depth_model.pth keys total: {len(depther_dict)} (includes non-tensor 'height'/'width' entries)")
tensor_keys_in_ckpt = {k for k, v in depther_dict.items() if isinstance(v, torch.Tensor)}
print(f"depth_model.pth tensor keys: {len(tensor_keys_in_ckpt)}")
print(f"model_full state_dict keys: {len(model_dict)}")
print(f"intersection (keys actually applied): {len(filtered)}")
missing_from_ckpt = set(model_dict.keys()) - set(filtered.keys())
extra_in_ckpt = tensor_keys_in_ckpt - set(model_dict.keys())
print(f"model keys with NO matching checkpoint entry (stay at backbone/random init): {len(missing_from_ckpt)}")
print(f"checkpoint tensor keys with no matching model key (silently dropped): {len(extra_in_ckpt)}")
if missing_from_ckpt:
    print("  sample unmatched model keys:", list(sorted(missing_from_ckpt))[:8])
if extra_in_ckpt:
    print("  sample dropped checkpoint keys:", list(sorted(extra_in_ckpt))[:8])

model_full.load_state_dict(filtered)
model_full.to(device)
model_full.eval()
model_backbone_only.to(device)
model_backbone_only.eval()

# ---- Compare specific LoRA / head parameter tensors: backbone-only init vs loaded checkpoint ----
print(f"\n=== LoRA / head parameter comparison (backbone-only init vs depth_model.pth-loaded) ===")
candidate_names = [n for n in model_dict.keys() if "lora_A" in n or "lora_B" in n or "lora_U" in n or "lora_V" in n]
print(f"total LoRA parameter tensors in model: {len(candidate_names)}")
sample_names = candidate_names[:3] + [n for n in candidate_names if "lora_B" in n][:3]
sample_names = list(dict.fromkeys(sample_names))  # dedupe, preserve order
full_state = {k: v.cpu() for k, v in model_full.state_dict().items()}
for name in sample_names:
    a = backbone_only_state[name]
    b = full_state[name]
    identical = torch.equal(a, b)
    print(f"  {name}: shape {tuple(a.shape)}, identical={identical}, "
          f"backbone-only stats (min={a.min():.6f} max={a.max():.6f} mean={a.mean():.6f} std={a.std():.6f}), "
          f"loaded stats (min={b.min():.6f} max={b.max():.6f} mean={b.mean():.6f} std={b.std():.6f})")

# also check a depth_head parameter (not part of DINOv2 backbone at all)
head_names = [n for n in model_dict.keys() if n.startswith("depth_head.") and "weight" in n][:3]
for name in head_names:
    a = backbone_only_state[name]
    b = full_state[name]
    identical = torch.equal(a, b)
    print(f"  {name}: shape {tuple(a.shape)}, identical={identical}, "
          f"backbone-only stats (min={a.min():.6f} max={a.max():.6f} mean={a.mean():.6f} std={a.std():.6f}), "
          f"loaded stats (min={b.min():.6f} max={b.max():.6f} mean={b.mean():.6f} std={b.std():.6f})")

# ---- Raw network output inspection for the 3 sanity-check frames ----
print(f"\n=== raw network output per frame (model_full, i.e. depth_model.pth loaded) ===")
with torch.no_grad():
    for frame in FRAMES:
        rgb_path = os.path.join(SEQ_DIR, "rgb", f"{frame}.png")
        input_image = Image.open(rgb_path).convert("RGB")
        resized = input_image.resize((feed_width, feed_height), Image.LANCZOS)
        tensor = torch.from_numpy(np.array(resized)).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)

        print(f"\n--- frame {frame} ---")
        print(f"input tensor: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
              f"min={tensor.min().item():.6f}, max={tensor.max().item():.6f}, "
              f"mean={tensor.mean().item():.6f}, std={tensor.std().item():.6f}")

        outputs = model_full(tensor)
        disp = outputs[("disp", 0)]
        disp_np = disp.detach().cpu().numpy()
        print(f"raw sigmoid disparity ('disp',0): shape={tuple(disp.shape)}, "
              f"min={disp_np.min():.8f}, max={disp_np.max():.8f}, "
              f"mean={disp_np.mean():.8f}, std={disp_np.std():.8f}")
        hist, edges = np.histogram(disp_np.flatten(), bins=10)
        print("  histogram (10 bins over observed range):")
        for h, lo, hi in zip(hist, edges[:-1], edges[1:]):
            print(f"    [{lo:.6f}, {hi:.6f}): {h}")

        _, depth = disp_to_depth(disp, 0.1, 150)
        depth_np = depth.detach().cpu().numpy()
        print(f"converted depth (disp_to_depth, min_depth=0.1 max_depth=150): "
              f"min={depth_np.min():.6f}, max={depth_np.max():.6f}, "
              f"mean={depth_np.mean():.6f}, std={depth_np.std():.6f}")
        hist_d, edges_d = np.histogram(depth_np.flatten(), bins=10)
        print("  histogram (10 bins over observed range):")
        for h, lo, hi in zip(hist_d, edges_d[:-1], edges_d[1:]):
            print(f"    [{lo:.6f}, {hi:.6f}): {h}")

        rel_std = disp_np.std() / (disp_np.mean() + 1e-12)
        print(f"disp relative std (std/mean): {rel_std:.6f} "
              f"({'NEAR-CONSTANT' if rel_std < 0.01 else 'has variation'})")
