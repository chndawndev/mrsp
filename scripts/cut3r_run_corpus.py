#!/usr/bin/env python
"""CUT3R Stage 3 (docs/pipelines/cut3r.md): inference over registered
sequences with the pinned configuration, and the variability runs of
docs/eval_protocol.md "2026-10-02: Numerical run-to-run variability".
No evaluation metric.

Pinned configuration (docs/pipelines/cut3r.md, Stage 3):
  - entry point: scripts/cut3r_recurrent.py::run_recurrent_streaming, the
    vendored ARCroco3DStereo.forward_recurrent fed one frame at a time
    (encoder batch size 1);
  - precision: vendor defaults unchanged (the vendored code enables
    torch.backends.cuda.matmul.allow_tf32 at import; cuDNN TF32 is the
    PyTorch default); recorded per sequence in the manifest;
  - checkpoint cut3r_512_dpt_4_64.pth, size 512, raw fisheye frames,
    CUT3R's own loader (vendored demo.py::prepare_input);
  - one GPU for the whole run; its model name is recorded per sequence.

Per sequence: frames are extracted from the archive into
scratch/cut3r/<run>/<sequence>/ (rgb members only), run, and the
extraction is deleted. Saved: depth/<frame:04d>.npz key "z" (camera-frame
Z, float32, 512x400 grid), poses_c2w.npy (N, 4, 4 camera-to-world,
float64), MANIFEST.json. Confidence is not saved (not used by the
protocol).

Variability runs: --noise-sigma S --noise-seed K adds i.i.d. Gaussian
noise of std S on the 0-1 image scale to the tensor the network receives.
That tensor is normalized to [-1, 1], so the std applied to it is 2 * S.
Frame i's noise is drawn from default_rng([K, i]). --out-root is required
with noise: variability runs never write into the primary run.

Resumable (skips sequences whose MANIFEST status is ok). No sequence is
excluded; a failing sequence gets status "failed" with the traceback and
the loop continues, except on a CUDA error (logged with nvidia-smi and
peak memory, exit non-zero, no retry). Pauses while free disk < 50 GB.

Usage (cut3r conda env, detached):
    CUDA_VISIBLE_DEVICES=<i> python scripts/cut3r_run_corpus.py 2>&1 | tee logs/cut3r_run_corpus.log
"""
from __future__ import annotations

import argparse
import glob
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
CUT3R_REPO = REPO / "scratch/pipelines/CUT3R"
CHECKPOINT = CUT3R_REPO / "src/cut3r_512_dpt_4_64.pth"
CHECKPOINT_SHA256 = "45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103"
REGISTERED_DIR = Path("/data1_ycao/chua/datasets/C3VDv2/registered_videos")
FULL_RUN_ROOT = REPO / "results/pipelines/cut3r_full_run"
SCRATCH_ROOT = REPO / "scratch/cut3r"
SIZE = 512
EXPECTED_N_SEQUENCES = 169
MIN_FREE_DISK_GB = 50.0
DISK_PAUSE_SECONDS = 600


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}", flush=True)


def nvidia_smi() -> str:
    return subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout


def git_head(path: Path) -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True).stdout.strip()


def discover_sequences() -> list[str]:
    """Registered sequences from *.zip archives only (CLAUDE.md)."""
    names = sorted(z.stem for z in REGISTERED_DIR.glob("*.zip") if not z.stem.startswith("c0_"))
    assert len(names) == EXPECTED_N_SEQUENCES, f"expected {EXPECTED_N_SEQUENCES} registered sequences, found {len(names)}"
    return names


def variability_subset() -> list[str]:
    """docs/eval_protocol.md, 2026-10-02, item 2: for each (Colon, Segment)
    mold, the first registered sequence in alphabetical order."""
    first: dict[str, str] = {}
    for name in discover_sequences():
        colon, segment = name.split("_")[:2]
        first.setdefault(f"{colon}_{segment}", name)
    subset = sorted(first.values())
    assert len(subset) == 15, f"expected 15 molds, found {len(subset)}"
    return subset


def rgb_members(zf: zipfile.ZipFile) -> list[str]:
    members = sorted(n for n in zf.namelist() if n.endswith(".png") and "/rgb/" in ("/" + n))
    if not members:
        raise RuntimeError("no rgb/*.png members in archive")
    return members


def n_gt_poses(zf: zipfile.ZipFile) -> int:
    cands = [n for n in zf.namelist() if n == "pose.txt" or n.endswith("/pose.txt")]
    if len(cands) != 1:
        raise RuntimeError(f"pose.txt members: {cands}")
    return len([ln for ln in zf.read(cands[0]).decode().splitlines() if ln.strip()])


def is_ok(out_dir: Path) -> bool:
    p = out_dir / "MANIFEST.json"
    try:
        return p.exists() and json.loads(p.read_text()).get("status") == "ok"
    except (json.JSONDecodeError, OSError):
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=str(FULL_RUN_ROOT))
    ap.add_argument("--sequence", action="append", help="repeatable; default: all 169")
    ap.add_argument("--variability-subset", action="store_true", help="the protocol's 15-sequence subset")
    ap.add_argument("--noise-sigma", type=float, default=0.0, help="std on the 0-1 image scale")
    ap.add_argument("--noise-seed", type=int, default=None)
    args = ap.parse_args()
    out_root = Path(args.out_root).resolve()
    if args.noise_sigma and (args.noise_seed is None or out_root == FULL_RUN_ROOT):
        raise SystemExit("noise runs need --noise-seed and an --out-root other than the primary run's")
    if "CUDA_VISIBLE_DEVICES" not in os.environ or "," in os.environ["CUDA_VISIBLE_DEVICES"]:
        raise SystemExit("set CUDA_VISIBLE_DEVICES to exactly one index (scripts/gpu_status.py)")
    gpu_index = int(os.environ["CUDA_VISIBLE_DEVICES"])

    all_sequences = discover_sequences()
    if args.variability_subset:
        sequences = variability_subset()
    elif args.sequence:
        unknown = [q for q in args.sequence if q not in all_sequences]
        if unknown:
            raise SystemExit(f"not among the 169 registered sequences: {unknown}")
        sequences = args.sequence
    else:
        sequences = all_sequences
    out_root.mkdir(parents=True, exist_ok=True)
    scratch_run = SCRATCH_ROOT / out_root.name / (f"seed{args.noise_seed}" if args.noise_sigma else "primary")
    log(f"{len(sequences)} sequences -> {out_root}; noise_sigma={args.noise_sigma} seed={args.noise_seed}; "
        f"CUDA_VISIBLE_DEVICES={gpu_index}; nvidia-smi at start:\n{nvidia_smi()}")

    os.chdir(CUT3R_REPO)
    sys.path.insert(0, str(CUT3R_REPO))
    sys.path.insert(0, str(REPO / "scripts"))
    import torch
    import demo  # vendored demo.py (prepare_input)
    from add_ckpt_path import add_path_to_dust3r

    add_path_to_dust3r(str(CHECKPOINT))
    from src.dust3r.model import ARCroco3DStereo
    from src.dust3r.utils.camera import pose_encoding_to_camera
    from cut3r_recurrent import run_recurrent_streaming
    from cut3r_run import sha256

    digest = sha256(CHECKPOINT)
    if digest != CHECKPOINT_SHA256:
        raise SystemExit(f"checkpoint SHA-256 {digest} != recorded {CHECKPOINT_SHA256}")
    device = "cuda"
    gpu_name = torch.cuda.get_device_name(0)
    model = ARCroco3DStereo.from_pretrained(str(CHECKPOINT)).to(device)
    model.eval()
    pinned = {
        "entry": "scripts/cut3r_recurrent.py::run_recurrent_streaming (vendored forward_recurrent, one frame per step)",
        "encoder_batch_size": 1, "size": SIZE, "checkpoint_sha256": digest,
        "tf32": {"matmul": torch.backends.cuda.matmul.allow_tf32, "cudnn": torch.backends.cudnn.allow_tf32},
        "cudnn": {"benchmark": torch.backends.cudnn.benchmark, "deterministic": torch.backends.cudnn.deterministic},
        "torch": torch.__version__, "cut3r_commit": git_head(CUT3R_REPO),
    }
    log(f"GPU: {gpu_name}; pinned configuration: {json.dumps(pinned)}")

    def keep(res):
        return (res["pts3d_in_self_view"][0, ..., 2].numpy().astype(np.float32),
                pose_encoding_to_camera(res["camera_pose"].clone())[0].numpy().astype(np.float64))

    t_run, n_done, n_failed = time.time(), 0, 0
    for k, seq in enumerate(sequences):
        out_dir = out_root / seq
        if is_ok(out_dir):
            log(f"[{k + 1}/{len(sequences)}] {seq}: already complete, skipping")
            continue
        while shutil.disk_usage(REPO).free / 1e9 < MIN_FREE_DISK_GB:
            log(f"free disk {shutil.disk_usage(REPO).free / 1e9:.1f} GB < {MIN_FREE_DISK_GB} GB -- pausing "
                f"{DISK_PAUSE_SECONDS}s, nothing deleted")
            time.sleep(DISK_PAUSE_SECONDS)
        log(f"[{k + 1}/{len(sequences)}] {seq}: start")
        t0 = time.time()
        frames_dir = scratch_run / seq
        manifest = {"sequence": seq, "status": "failed", "error": None, "gpu_index": gpu_index, "gpu_name": gpu_name,
                    "repo_commit": git_head(REPO), "noise_sigma_0_1_scale": args.noise_sigma,
                    "noise_seed": args.noise_seed, "pinned_configuration": pinned}
        cuda_error = False
        try:
            if out_dir.exists():
                shutil.rmtree(out_dir)  # a previous failed / partial attempt of this same sequence
            (out_dir / "depth").mkdir(parents=True)
            frames_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(REGISTERED_DIR / f"{seq}.zip") as zf:
                members = rgb_members(zf)
                n_gt = n_gt_poses(zf)
                for i, m in enumerate(members):
                    (frames_dir / f"{i:04d}.png").write_bytes(zf.read(m))
            manifest.update({"n_frames_rgb": len(members), "n_gt_frames": n_gt, "extract_seconds": time.time() - t0})
            if len(members) != n_gt:
                raise RuntimeError(f"{len(members)} rgb frames but {n_gt} GT poses")

            t1 = time.time()
            img_paths = sorted(glob.glob(str(frames_dir / "*.png")))
            views = demo.prepare_input(img_paths=img_paths, img_mask=[True] * len(img_paths), size=SIZE,
                                       revisit=1, update=True)
            if args.noise_sigma:
                for i, v in enumerate(views):
                    rng = np.random.default_rng([args.noise_seed, i])
                    noise = (2.0 * args.noise_sigma * rng.standard_normal(tuple(v["img"].shape))).astype(np.float32)
                    v["img"] = v["img"] + torch.from_numpy(noise)
            manifest["load_seconds"] = time.time() - t1

            torch.manual_seed(0)
            np.random.seed(0)
            torch.cuda.reset_peak_memory_stats()
            t1 = time.time()
            kept = run_recurrent_streaming(views, model, device, keep=keep)
            torch.cuda.synchronize()
            manifest["inference_seconds"] = time.time() - t1
            manifest["peak_gpu_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
            manifest["peak_gpu_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20
            del views

            t1 = time.time()
            poses = np.stack([p for _, p in kept])
            n_nonfinite_depth_frames, n_nonpositive = 0, 0
            for i, (z, _) in enumerate(kept):
                n_nonfinite_depth_frames += int(not np.isfinite(z).all())
                n_nonpositive += int((z <= 0).sum())
                np.savez_compressed(out_dir / "depth" / f"{i:04d}.npz", z=z)
            np.save(out_dir / "poses_c2w.npy", poses)
            manifest.update({
                "n_frames": len(kept), "grid_h": int(kept[0][0].shape[0]), "grid_w": int(kept[0][0].shape[1]),
                "n_frames_nonfinite_depth": n_nonfinite_depth_frames,
                "n_frames_nonfinite_pose": int((~np.isfinite(poses).all(axis=(1, 2))).sum()),
                "n_nonpositive_z_pixels": n_nonpositive,
                "save_seconds": time.time() - t1,
                "depth_dir_bytes": sum(p.stat().st_size for p in (out_dir / "depth").glob("*.npz")),
                "status": "ok",
            })
            del kept
        except Exception as exc:
            manifest["error"] = traceback.format_exc()
            cuda_error = "cuda" in manifest["error"].lower() or "out of memory" in manifest["error"].lower()
            log(f"{seq}: FAILED: {exc}\n{manifest['error']}")
            if cuda_error:  # CLAUDE.md: timestamp, nvidia-smi, peak memory, exception; exit non-zero; no retry
                manifest["peak_gpu_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
                log(f"CUDA error; peak allocated by this process {manifest['peak_gpu_allocated_mib']:.0f} MiB; "
                    f"nvidia-smi at failure time:\n{nvidia_smi()}")
        finally:
            shutil.rmtree(frames_dir, ignore_errors=True)
        manifest["runtime_seconds_total"] = time.time() - t0
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
        if cuda_error:
            sys.exit(1)
        n_done += 1
        n_failed += manifest["status"] != "ok"
        elapsed = time.time() - t_run
        remaining = sum(1 for q in sequences[k + 1:] if not is_ok(out_root / q))
        log(f"[{k + 1}/{len(sequences)}] {seq}: status={manifest['status']} frames={manifest.get('n_frames')} "
            f"{manifest['runtime_seconds_total']:.0f}s peak={manifest.get('peak_gpu_allocated_mib', float('nan')):.0f}MiB; "
            f"elapsed {elapsed / 3600:.2f}h ETA {elapsed / max(n_done, 1) * remaining / 3600:.2f}h "
            f"({n_failed} failed, free disk {shutil.disk_usage(REPO).free / 1e9:.0f} GB)")
    log(f"DONE: {n_done} run here, {n_failed} failed, wall {(time.time() - t_run) / 3600:.2f}h")


if __name__ == "__main__":
    main()
