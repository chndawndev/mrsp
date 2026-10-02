#!/usr/bin/env python
"""CUT3R Stage 2 diagnosis (docs/pipelines/cut3r.md): how much do the
image-encoder features of one frame depend on how many frames are encoded
together? The parallel path (inference) encodes all frames in one batch;
the recurrent path (forward_recurrent) encodes one frame per call. Same
weights, same inputs.

For the first K frames of a sequence: features from one batched
model._encode_image call against features from K single-frame calls, per
frame: ||batch - single|| / ||single|| and max abs difference. Also the
batched call repeated (determinism) and batches of other sizes.

No evaluation metric. Usage (cut3r env):
    CUDA_VISIBLE_DEVICES=<i> python scripts/cut3r_encoder_batch_diagnosis.py --seq-dir scratch/mast3r_slam/c1_cecum_t1_v1
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path("/data1_ycao/chua/projects/mrsp")
CUT3R_REPO = REPO / "scratch/pipelines/CUT3R"
CHECKPOINT = CUT3R_REPO / "src/cut3r_512_dpt_4_64.pth"
OUT = REPO / "results/pipelines/cut3r_stage1/encoder_batch_diagnosis.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-dir", required=True)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--no-tf32", action="store_true", help="disable TF32 (the vendored code enables it on import)")
    args = ap.parse_args()
    if "CUDA_VISIBLE_DEVICES" not in os.environ or "," in os.environ["CUDA_VISIBLE_DEVICES"]:
        raise SystemExit("set CUDA_VISIBLE_DEVICES to exactly one index")
    paths = sorted(glob.glob(str(Path(args.seq_dir).resolve() / "*.png")))[: args.k]
    os.chdir(CUT3R_REPO)
    sys.path.insert(0, str(CUT3R_REPO))
    import torch
    import demo
    from add_ckpt_path import add_path_to_dust3r

    add_path_to_dust3r(str(CHECKPOINT))
    from src.dust3r.model import ARCroco3DStereo

    if args.no_tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    views = demo.prepare_input(img_paths=paths, img_mask=[True] * len(paths), size=512, revisit=1, update=True)
    model = ARCroco3DStereo.from_pretrained(str(CHECKPOINT)).to("cuda").eval()
    imgs = torch.cat([v["img"] for v in views], 0).cuda()
    shapes = torch.cat([v["true_shape"] for v in views], 0).cuda()

    def enc(sl):
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
            out, pos, _ = model._encode_image(imgs[sl], shapes[sl])
        return out[-1].double()

    single = torch.cat([enc(slice(i, i + 1)) for i in range(len(paths))], 0)
    single2 = torch.cat([enc(slice(i, i + 1)) for i in range(len(paths))], 0)
    res = {"sequence_dir": args.seq_dir, "k": len(paths), "feature_shape": list(single.shape),
           "feature_dtype": str(model._encode_image(imgs[:1], shapes[:1])[0][-1].dtype),
           "single_frame_calls_repeated_max_abs_diff": float((single - single2).abs().max()), "batches": {}}
    for b in sorted({2, 4, 16, len(paths)}):
        full = torch.cat([enc(slice(i, min(i + b, len(paths)))) for i in range(0, len(paths), b)], 0)
        rel = ((full - single).flatten(1).norm(dim=1) / single.flatten(1).norm(dim=1)).cpu().numpy()
        res["batches"][str(b)] = {
            "relative_l2_diff_per_frame": {"min": float(rel.min()), "median": float(np.median(rel)), "max": float(rel.max())},
            "max_abs_diff": float((full - single).abs().max()),
            "feature_abs_median": float(single.abs().median()),
        }
    # patch embedding alone (one convolution): is the difference already there?
    with torch.no_grad():
        pe_b, _ = model.patch_embed(imgs, true_shape=shapes)
        pe_s = torch.cat([model.patch_embed(imgs[i:i + 1], true_shape=shapes[i:i + 1])[0] for i in range(len(paths))], 0)
    res["patch_embed_batch_vs_single_relative_l2_max"] = float(
        ((pe_b - pe_s).double().flatten(1).norm(dim=1) / pe_s.double().flatten(1).norm(dim=1)).max())
    res["torch_backends"] = {"cudnn_benchmark": torch.backends.cudnn.benchmark,
                             "cudnn_deterministic": torch.backends.cudnn.deterministic,
                             "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                             "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32}
    out = OUT.with_name(OUT.stem + ("_notf32" if args.no_tf32 else "") + ".json")
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
