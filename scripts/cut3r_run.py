#!/usr/bin/env python
"""CUT3R online inference on one frame folder, headless
(docs/pipelines/cut3r.md, Stage 1).

Copied entry script: the inference path of the vendored
scratch/pipelines/CUT3R/demo.py (run_inference -> prepare_input ->
ARCroco3DStereo.from_pretrained -> src.dust3r.inference.inference) with
the viser viewer removed and the saving rewritten. The vendored code is
imported, never edited. demo.py's own prepare_input is called as is, so
frames go through CUT3R's own loader (src/dust3r/utils/image.py
::load_images). Online inference only; no global alignment (demo_ga.py).

Fixed choices (docs/pipelines/cut3r.md): checkpoint
cut3r_512_dpt_4_64.pth, --size 512, raw fisheye frames, no undistortion.
Confidence maps are saved and never used (docs/eval_protocol.md,
2026-09-29).

Saved per frame i (frame index = position in the sorted *.png list):
  depth/<i:04d>.npz   z     (H, W) float32, camera-frame Z of
                            pts3d_in_self_view (model units)
                      conf  (H, W) float32, conf_self (recorded, unused)
  poses_c2w.npy       (N, 4, 4) float64, pose_encoding_to_camera output
  pointmap_self/<i:04d>.npy  (H, W, 3) float32, only --save-pointmap-frames
  pointmap_other/<i:04d>.npy pts3d_in_other_view (first-frame coordinates),
                             same frames; used only for the pose-convention
                             consistency check
  focal_weiszfeld.npy (N,) demo.py's own focal estimate (descriptive)
  MANIFEST.json       timing, peak GPU memory, shapes, non-finite counts

Computes no evaluation metric.

Usage (cut3r conda env; pick the GPU with scripts/gpu_status.py first):
    CUDA_VISIBLE_DEVICES=<i> python scripts/cut3r_run.py \
        --seq-dir scratch/mast3r_slam/c1_cecum_t1_v1 --sequence c1_cecum_t1_v1
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

REPO = Path("/data1_ycao/chua/projects/mrsp")
CUT3R_REPO = REPO / "scratch/pipelines/CUT3R"
CHECKPOINT = CUT3R_REPO / "src/cut3r_512_dpt_4_64.pth"
OUT_ROOT = REPO / "results/pipelines/cut3r_stage1"
SIZE = 512


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}", flush=True)


def nvidia_smi() -> str:
    return subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout


def git_head(path: Path) -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True).stdout.strip()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-dir", required=True, help="folder of zero-padded *.png frames")
    ap.add_argument("--sequence", required=True)
    ap.add_argument("--max-frames", type=int, default=None,
                    help="use only the first N frames (memory-vs-length probe); output goes to <sequence>_first<N>")
    ap.add_argument("--save-pointmap-frames", type=int, nargs="*", default=[0])
    ap.add_argument("--no-save", action="store_true", help="timing / memory probe only")
    ap.add_argument("--path", choices=["parallel", "recurrent", "recurrent_plain"], default="parallel",
                    help="parallel: demo.py's inference() (Stage 1). recurrent: vendored forward_recurrent fed one "
                         "frame at a time (scripts/cut3r_recurrent.py). recurrent_plain: vendored "
                         "inference_recurrent on the whole list. Non-default paths write to <tag>_<path>")
    ap.add_argument("--skip-checkpoint-hash", action="store_true")
    args = ap.parse_args()

    if "CUDA_VISIBLE_DEVICES" not in os.environ or "," in os.environ["CUDA_VISIBLE_DEVICES"]:
        raise SystemExit("set CUDA_VISIBLE_DEVICES to exactly one index (scripts/gpu_status.py)")
    gpu_index = int(os.environ["CUDA_VISIBLE_DEVICES"])
    log(f"CUDA_VISIBLE_DEVICES={gpu_index}; nvidia-smi at start:\n{nvidia_smi()}")

    img_paths = sorted(glob.glob(str(Path(args.seq_dir).resolve() / "*.png")))
    if not img_paths:
        raise SystemExit(f"no *.png in {args.seq_dir}")
    n_total = len(img_paths)
    if args.max_frames is not None:
        img_paths = img_paths[: args.max_frames]
    tag = args.sequence if args.max_frames is None else f"{args.sequence}_first{args.max_frames}"
    if args.path != "parallel":
        tag = f"{tag}_{args.path}"
    out_dir = OUT_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # vendored code: imported from its own directory, as demo.py expects
    os.chdir(CUT3R_REPO)
    sys.path.insert(0, str(CUT3R_REPO))
    import torch
    import demo  # vendored scratch/pipelines/CUT3R/demo.py (prepare_input)
    from add_ckpt_path import add_path_to_dust3r

    torch.manual_seed(0)
    np.random.seed(0)
    add_path_to_dust3r(str(CHECKPOINT))
    from src.dust3r.inference import inference, inference_recurrent
    sys.path.insert(0, str(REPO / "scripts"))
    from cut3r_recurrent import run_recurrent_streaming
    from src.dust3r.model import ARCroco3DStereo
    from src.dust3r.post_process import estimate_focal_knowing_depth
    from src.dust3r.utils.camera import pose_encoding_to_camera

    device = "cuda"
    manifest = {
        "sequence": args.sequence, "tag": tag, "n_frames_in_folder": n_total, "n_frames_input": len(img_paths),
        "gpu_index": gpu_index, "size": SIZE, "checkpoint": str(CHECKPOINT.relative_to(REPO)),
        "checkpoint_sha256": None if args.skip_checkpoint_hash else sha256(CHECKPOINT),
        "cut3r_commit": git_head(CUT3R_REPO), "repo_commit": git_head(REPO),
        "torch": torch.__version__, "path": args.path,
        "entry": "online inference, no global alignment",
        "status": "failed", "error": None,
    }
    try:
        t0 = time.time()
        views = demo.prepare_input(img_paths=img_paths, img_mask=[True] * len(img_paths), size=SIZE,
                                   revisit=1, update=True)
        manifest["load_seconds"] = time.time() - t0
        h, w = views[0]["img"].shape[-2:]
        log(f"{len(views)} views, internal grid {w}x{h}")

        t0 = time.time()
        model = ARCroco3DStereo.from_pretrained(str(CHECKPOINT)).to(device)
        model.eval()
        manifest["model_load_seconds"] = time.time() - t0
        manifest["gpu_mem_after_model_load_mib"] = torch.cuda.memory_allocated() / 2**20

        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        if args.path == "parallel":
            outputs, _ = inference(views, model, device)
            preds = outputs["pred"]
        elif args.path == "recurrent_plain":
            outputs, _ = inference_recurrent(views, model, device)
            preds = outputs["pred"]
        else:
            preds = run_recurrent_streaming(views, model, device, keep=lambda r: r)
        torch.cuda.synchronize()
        manifest["inference_seconds"] = time.time() - t0
        manifest["peak_gpu_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
        manifest["peak_gpu_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20
        log(f"inference {manifest['inference_seconds']:.1f}s, peak allocated {manifest['peak_gpu_allocated_mib']:.0f} MiB, "
            f"peak reserved {manifest['peak_gpu_reserved_mib']:.0f} MiB")

        manifest["n_predictions"] = len(preds)
        manifest["pred_keys"] = sorted(preds[0].keys())
        manifest["pred_shapes"] = {k: list(v.shape) for k, v in preds[0].items() if hasattr(v, "shape")}

        pts_self = torch.cat([p["pts3d_in_self_view"].cpu() for p in preds], 0)  # (N, H, W, 3)
        pts_other = torch.cat([p["pts3d_in_other_view"].cpu() for p in preds], 0)  # (N, H, W, 3), first-frame coordinates
        conf_self = torch.cat([p["conf_self"].cpu() for p in preds], 0)
        poses = torch.cat([pose_encoding_to_camera(p["camera_pose"].clone()).cpu() for p in preds], 0)
        B, H, W, _ = pts_self.shape
        pp = torch.tensor([W // 2, H // 2]).float().repeat(B, 1)
        focal = estimate_focal_knowing_depth(pts_self, pp, focal_mode="weiszfeld")  # as demo.py prepare_output

        z = pts_self[..., 2].numpy()
        poses_np = poses.numpy().astype(np.float64)
        finite_depth = np.isfinite(pts_self.numpy()).all(axis=(1, 2, 3))
        finite_pose = np.isfinite(poses_np).all(axis=(1, 2))
        manifest.update({
            "grid_h": int(H), "grid_w": int(W),
            "n_frames_with_depth": int(B), "n_frames_with_pose": int(len(poses_np)),
            "frames_nonfinite_depth": np.flatnonzero(~finite_depth).tolist(),
            "frames_nonfinite_pose": np.flatnonzero(~finite_pose).tolist(),
            "n_nonpositive_z_pixels": int((z <= 0).sum()),
            "frames_with_any_nonpositive_z": np.flatnonzero((z <= 0).any(axis=(1, 2))).tolist(),
            "focal_weiszfeld_median": float(np.median(focal.numpy())),
            "focal_weiszfeld_min_max": [float(focal.min()), float(focal.max())],
        })

        if not args.no_save:
            t0 = time.time()
            (out_dir / "depth").mkdir(exist_ok=True)
            (out_dir / "pointmap_self").mkdir(exist_ok=True)
            (out_dir / "pointmap_other").mkdir(exist_ok=True)
            conf_np = conf_self.numpy()
            for i in range(B):
                np.savez_compressed(out_dir / "depth" / f"{i:04d}.npz", z=z[i].astype(np.float32),
                                    conf=conf_np[i].astype(np.float32))
            for i in args.save_pointmap_frames:
                np.save(out_dir / "pointmap_self" / f"{i:04d}.npy", pts_self[i].numpy().astype(np.float32))
                np.save(out_dir / "pointmap_other" / f"{i:04d}.npy", pts_other[i].numpy().astype(np.float32))
            np.save(out_dir / "poses_c2w.npy", poses_np)
            np.save(out_dir / "focal_weiszfeld.npy", focal.numpy())
            manifest["save_seconds"] = time.time() - t0
            manifest["depth_dir_bytes"] = sum(p.stat().st_size for p in (out_dir / "depth").glob("*.npz"))
        manifest["status"] = "ok"
    except Exception as exc:  # CLAUDE.md: log nvidia-smi, peak memory, exception; exit non-zero; no retry
        manifest["error"] = traceback.format_exc()
        log(f"FAILED: {exc}\n{manifest['error']}")
        log(f"nvidia-smi at failure time:\n{nvidia_smi()}")
        try:
            manifest["peak_gpu_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
            log(f"peak allocated by this process: {manifest['peak_gpu_allocated_mib']:.0f} MiB")
        except Exception:
            pass
        (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
        sys.exit(1)
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    log(json.dumps({k: v for k, v in manifest.items() if k not in ("pred_shapes",)}, indent=1))


if __name__ == "__main__":
    main()
