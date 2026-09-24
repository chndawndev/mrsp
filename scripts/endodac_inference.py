#!/usr/bin/env python
"""EndoDAC depth + pose + intrinsics inference, generalized to any
registered_videos sequence. D1 Stage A (docs/d1_stage_a.md).

Same exact inference settings as the validated single-sequence run
(scratch/pipelines/endodac_full_sequence_inference.py, docs/pipelines/
endodac.md section 8): Path A baseline input (full fisheye frame, resize
only, no crop, no inpaint), same weights, same 320x256 feed resolution,
relative poses inverted before chaining (section 7's empirical finding).
No tuning, no per-sequence changes -- run_sequence_inference() is the
SAME procedure for every sequence.

Two differences from the original scratch script, both infrastructure,
not inference settings: (1) frames are read from a per-sequence temp
extraction of ONLY the rgb/ zip members, done and torn down inside
run_sequence_inference(), instead of a pre-existing full scratch
extraction; (2) a fixed seed and disabled cudnn benchmarking are set
before each sequence's forward passes, to help (not guarantee) the
reproduction gate's bit-identical requirement -- these are determinism
knobs, not a change to what the model computes.

CLI:
    endodac-env python scripts/endodac_inference.py --sequence NAME
    endodac-env python scripts/endodac_inference.py --shard-index I --shard-total K
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path

import numpy as np

REPO = Path("/data1_ycao/chua/projects/mrsp")
ENDODAC_REPO = REPO / "scratch/pipelines/EndoDAC"
DATASET_ROOT = Path("/data1_ycao/chua/datasets/C3VDv2")
REGISTERED_DIR = DATASET_ROOT / "registered_videos"
OUT_ROOT = REPO / "results/pipelines/endodac_full_run"
SCRATCH_ROOT = REPO / "scratch/endodac_full_run"
LOG_DIR = REPO / "logs"

FEED_W, FEED_H = 320, 256
MIN_DEPTH, MAX_DEPTH = 0.1, 150.0
EXPECTED_N_SEQUENCES = 169
MIN_FREE_DISK_GB = 50

sys.path.insert(0, str(ENDODAC_REPO))


def log(msg: str, tag: str = "") -> None:
    prefix = f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}]"
    if tag:
        prefix += f"[{tag}]"
    print(f"{prefix} {msg}", flush=True)


# ---------------------------------------------------------------------
# sequence discovery -- duplicated from scripts/coverage_stats.py:60-82
# (same precedent visibility_full.py follows: a small entry-point-local
# copy rather than importing across unrelated scripts).
# ---------------------------------------------------------------------

def discover_sequences() -> list[tuple[str, Path]]:
    seen: dict[str, Path] = {}
    for zpath in sorted(REGISTERED_DIR.glob("*.zip")):
        name = zpath.stem
        if name.startswith("c0_"):
            continue
        seen[name] = zpath
    seqs = sorted(seen.items())
    assert len(seqs) == EXPECTED_N_SEQUENCES, (
        f"expected {EXPECTED_N_SEQUENCES} c1/c2 registered_videos sequences, found {len(seqs)}"
    )
    return seqs


def resolve_rgb_members(zpath: Path) -> list[str]:
    """Sorted list of this archive's rgb/*.png member paths -- handles the
    c1_cecum_t1_v3.zip-style exception where everything is wrapped in a
    <seq>/ subfolder (same class of issue geometry/coverage_mesh.py's
    _resolve_zip_member already handles for single files)."""
    with zipfile.ZipFile(zpath) as zf:
        members = [n for n in zf.namelist() if n.endswith(".png") and "/rgb/" in ("/" + n)]
    if not members:
        raise RuntimeError(f"no rgb/*.png members found in {zpath}")
    return sorted(members)


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()


# ---------------------------------------------------------------------
# models
# ---------------------------------------------------------------------

def load_models(device):
    import torch
    import models.endodac as endodac
    import models.encoders as encoders
    import models.decoders as decoders

    ckpt_dir = ENDODAC_REPO / "checkpoints" / "endodac"
    pretrained_dir = ENDODAC_REPO / "pretrained_model"

    depther_dict = torch.load(ckpt_dir / "depth_model.pth", map_location="cpu")
    depther = endodac.endodac(
        backbone_size="base", r=4, lora_type="dvlora", image_shape=(224, 280),
        pretrained_path=str(pretrained_dir), residual_block_indexes=[2, 5, 8, 11],
        include_cls_token=True,
    )
    model_dict = depther.state_dict()
    depther.load_state_dict({k: v for k, v in depther_dict.items() if k in model_dict})
    depther.to(device).eval()

    pose_encoder = encoders.ResnetEncoder(18, False, 2)
    pose_encoder.load_state_dict(torch.load(ckpt_dir / "pose_encoder.pth", map_location="cpu"))
    pose_decoder = decoders.PoseDecoder(pose_encoder.num_ch_enc, 1, 2)
    pose_decoder.load_state_dict(torch.load(ckpt_dir / "pose.pth", map_location="cpu"))
    intrinsics_decoder = decoders.IntrinsicsHead(pose_encoder.num_ch_enc)
    intrinsics_decoder.load_state_dict(torch.load(ckpt_dir / "intrinsics_head.pth", map_location="cpu"))
    pose_encoder.to(device).eval()
    pose_decoder.to(device).eval()
    intrinsics_decoder.to(device).eval()

    return {
        "depther": depther, "pose_encoder": pose_encoder,
        "pose_decoder": pose_decoder, "intrinsics_decoder": intrinsics_decoder,
    }


def is_complete(name: str, expected_n_frames: int) -> bool:
    out_dir = OUT_ROOT / name
    manifest_path = out_dir / "MANIFEST.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    if manifest.get("status") != "ok":
        return False
    if manifest.get("n_frames") != expected_n_frames:
        return False
    depth_dir = out_dir / "depth"
    if not depth_dir.exists() or len(list(depth_dir.glob("*_pred_depth.npy"))) != expected_n_frames:
        return False
    poses_path = out_dir / "poses_pred.npy"
    if not poses_path.exists():
        return False
    try:
        poses = np.load(poses_path)
    except (OSError, ValueError):
        return False
    return poses.shape == (expected_n_frames, 4, 4)


# ---------------------------------------------------------------------
# per-sequence inference
# ---------------------------------------------------------------------

def run_sequence_inference(name: str, zpath: Path, models: dict, device, gpu_index: int, seed: int = 0) -> dict:
    import torch
    from PIL import Image
    from utils.layers import disp_to_depth, transformation_from_parameters

    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    t_start = time.time()
    out_dir = OUT_ROOT / name
    (out_dir / "depth").mkdir(parents=True, exist_ok=True)
    seq_scratch = SCRATCH_ROOT / name
    rgb_dir = seq_scratch / "rgb"
    rgb_dir.mkdir(parents=True, exist_ok=True)

    try:
        members = resolve_rgb_members(zpath)
        n_frames = len(members)
        with zipfile.ZipFile(zpath) as zf:
            for i, member in enumerate(members):
                data = zf.read(member)
                (rgb_dir / f"{i:04d}.png").write_bytes(data)

        def load_tensor(frame_idx):
            path = rgb_dir / f"{frame_idx:04d}.png"
            im = Image.open(path).convert("RGB")
            ow, oh = im.size
            resized = im.resize((FEED_W, FEED_H), Image.LANCZOS)
            t = torch.from_numpy(np.array(resized)).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
            return t, ow, oh

        depther = models["depther"]
        pose_encoder = models["pose_encoder"]
        pose_decoder = models["pose_decoder"]
        intrinsics_decoder = models["intrinsics_decoder"]

        n_nonfinite_depth = 0
        t0 = time.time()
        with torch.no_grad():
            for i in range(n_frames):
                tensor, ow, oh = load_tensor(i)
                tensor = tensor.to(device)
                outputs = depther(tensor)
                disp = outputs[("disp", 0)]
                _, depth = disp_to_depth(disp, MIN_DEPTH, MAX_DEPTH)
                depth_up = torch.nn.functional.interpolate(depth, (oh, ow), mode="bilinear", align_corners=False)
                depth_np = depth_up.squeeze().cpu().numpy().astype(np.float32)
                n_nonfinite_depth += int((~np.isfinite(depth_np)).sum())
                np.save(out_dir / "depth" / f"{i:04d}_pred_depth.npy", depth_np)
                if i % 100 == 0 or i == n_frames - 1:
                    log(f"{name}: depth frame {i}/{n_frames-1}, elapsed {time.time()-t0:.1f}s", tag=f"gpu{gpu_index}")
        depth_elapsed = time.time() - t0

        t0 = time.time()
        relative_T, intrinsics_list = [], []
        with torch.no_grad():
            for i in range(n_frames - 1):
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
                if i % 100 == 0 or i == n_frames - 2:
                    log(f"{name}: pose pair ({i},{i+1}), elapsed {time.time()-t0:.1f}s", tag=f"gpu{gpu_index}")
        pose_elapsed = time.time() - t0

        poses = [np.eye(4, dtype=np.float32)]
        for T in relative_T:
            T_inv = np.linalg.inv(T)
            poses.append((poses[-1] @ T_inv).astype(np.float32))
        poses = np.stack(poses)
        n_nonfinite_pose = int((~np.isfinite(poses)).sum())

        np.save(out_dir / "poses_pred.npy", poses)
        with open(out_dir / "poses_pred.txt", "w") as f:
            for M in poses:
                vals = M.T.flatten()
                f.write(",".join(f"{v:.6f}" for v in vals) + "\n")

        intrinsics_arr = np.stack(intrinsics_list).astype(np.float32)
        np.save(out_dir / "intrinsics_pred_per_pair.npy", intrinsics_arr)
        fx, fy = intrinsics_arr[:, 0, 0], intrinsics_arr[:, 1, 1]
        cx, cy = intrinsics_arr[:, 0, 2], intrinsics_arr[:, 1, 2]
        intrinsics_summary = {
            "feed_resolution_wh": [FEED_W, FEED_H],
            "fx_mean": float(fx.mean()), "fx_std": float(fx.std()),
            "fy_mean": float(fy.mean()), "fy_std": float(fy.std()),
            "cx_mean": float(cx.mean()), "cx_std": float(cx.std()),
            "cy_mean": float(cy.mean()), "cy_std": float(cy.std()),
            "n_pairs": len(intrinsics_list),
        }
        with open(out_dir / "intrinsics_pred_summary.json", "w") as f:
            json.dump(intrinsics_summary, f, indent=2)

        manifest = {
            "sequence": name,
            "n_frames": n_frames,
            "input_variant": "baseline (Path A: full fisheye frame, resize only, no crop, no inpaint)",
            "feed_resolution_wh": [FEED_W, FEED_H],
            "depth": {
                "path": "depth/{frame:04d}_pred_depth.npy", "dtype": "float32",
                "shape": "[H, W] at native frame resolution",
                "units": "network-native, scale-ambiguous (monodepth2 disp_to_depth convention, "
                         "min_depth=0.1 max_depth=150 -- NOT millimeters)",
                "recover_raw_disparity": "disp = (1/depth - 1/MAX_DEPTH) / (1/MIN_DEPTH - 1/MAX_DEPTH), "
                                          "MIN_DEPTH=0.1, MAX_DEPTH=150",
            },
            "pose": {
                "path_npy": "poses_pred.npy", "path_txt": "poses_pred.txt",
                "dtype": "float32", "shape": f"[{n_frames}, 4, 4]",
                "convention": "camera-to-world, frame 0 = identity (arbitrary origin, no absolute GT anchor).",
                "scale": "network-native, scale-ambiguous; needs a scale-recovery step before metric comparison.",
                "known_issue": "raw per-pair transformation_from_parameters output INVERTED before "
                                "chaining, per docs/pipelines/endodac.md section 7.",
            },
            "intrinsics": {
                "path_per_pair": "intrinsics_pred_per_pair.npy", "path_summary": "intrinsics_pred_summary.json",
                "dtype": "float32", "shape": f"[{n_frames - 1}, 3, 3]",
                "units": "pixel units at FEED resolution (320x256), not native resolution.",
            },
            "timing": {"depth_seconds_total": depth_elapsed, "pose_intrinsics_seconds_total": pose_elapsed},
            "git_commit": git_head(),
            "gpu_index": gpu_index,
            "runtime_seconds_total": time.time() - t_start,
            "n_nonfinite_depth": n_nonfinite_depth,
            "n_nonfinite_pose": n_nonfinite_pose,
            "status": "ok",
            "error": None,
        }
        with open(out_dir / "MANIFEST.json", "w") as f:
            json.dump(manifest, f, indent=2)
        log(f"{name}: DONE in {manifest['runtime_seconds_total']:.1f}s, "
            f"n_nonfinite_depth={n_nonfinite_depth} n_nonfinite_pose={n_nonfinite_pose}", tag=f"gpu{gpu_index}")
        return manifest

    except Exception:
        err = traceback.format_exc()
        log(f"{name}: FAILED\n{err}", tag=f"gpu{gpu_index}")
        manifest = {
            "sequence": name, "status": "failed", "error": err,
            "git_commit": git_head(), "gpu_index": gpu_index,
            "runtime_seconds_total": time.time() - t_start,
        }
        with open(out_dir / "MANIFEST.json", "w") as f:
            json.dump(manifest, f, indent=2)
        return manifest
    finally:
        shutil.rmtree(seq_scratch, ignore_errors=True)


def free_disk_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / 1e9


def run_shard(shard_index: int, shard_total: int):
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_index = int(os.environ.get("CUDA_VISIBLE_DEVICES", "-1"))
    log(f"shard {shard_index}/{shard_total} starting, device={device}, CUDA_VISIBLE_DEVICES={gpu_index}", tag=f"gpu{gpu_index}")

    sequences = discover_sequences()
    my_sequences = [(name, path) for i, (name, path) in enumerate(sequences) if i % shard_total == shard_index]
    log(f"shard {shard_index}/{shard_total}: {len(my_sequences)} of {len(sequences)} sequences assigned", tag=f"gpu{gpu_index}")

    models = load_models(device)

    for name, zpath in my_sequences:
        n_expected = len(resolve_rgb_members(zpath))
        if is_complete(name, n_expected):
            log(f"{name}: already complete, skipping", tag=f"gpu{gpu_index}")
            continue
        free_gb = free_disk_gb(REPO)
        if free_gb < MIN_FREE_DISK_GB:
            log(f"free disk {free_gb:.1f}GB < {MIN_FREE_DISK_GB}GB -- stopping this shard, "
                f"not starting {name}. Rerun later to resume.", tag=f"gpu{gpu_index}")
            return
        run_sequence_inference(name, zpath, models, device, gpu_index)

    log(f"shard {shard_index}/{shard_total}: all assigned sequences done", tag=f"gpu{gpu_index}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", help="run a single named sequence (e.g. for the reproduction gate)")
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-total", type=int)
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if args.sequence:
        import torch

        sequences = dict(discover_sequences())
        if args.sequence not in sequences:
            raise SystemExit(f"{args.sequence!r} not among the 169 registered sequences")
        zpath = sequences[args.sequence]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gpu_index = int(os.environ.get("CUDA_VISIBLE_DEVICES", "-1"))
        models = load_models(device)
        manifest = run_sequence_inference(args.sequence, zpath, models, device, gpu_index)
        if manifest["status"] != "ok":
            sys.exit(1)
        return

    if args.shard_index is None or args.shard_total is None:
        raise SystemExit("provide --sequence NAME, or both --shard-index and --shard-total")
    run_shard(args.shard_index, args.shard_total)


if __name__ == "__main__":
    main()
