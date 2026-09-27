#!/usr/bin/env python
"""D1.3 diagnosis, step 0: recompute per-face ignore_set for every
sequence. Stage B (scripts/eval_d1_sequence.py) never saved this array --
it only depends on ever_evaluable_hit, a GPU ray-cast quantity computed
internally by src/eval/oracle.py's run_oracle_sequence and discarded
after producing the ignore set fraction. This script recomputes just
ever_evaluable_hit (one GT-pose-only ray-cast pass per sequence, skipping
oracle.py's tau-gated depth-comparison work, which isn't needed here),
via the SAME one-line accumulation oracle.py itself uses
(src/eval/oracle.py:118), then calls the locked compute_ignore_set
unchanged.

No src/eval/ or src/gt/ edits -- reuses gt.rasterizer.build_mesh/
cast_frame and eval.evaluable.evaluable_pixel_mask directly (both already
read-only-reusable), and eval.ignore_set.compute_ignore_set as-is.

CLI:
    scratch/.venv/bin/python scripts/diagnose_d1_3_ignore_set.py --sequence NAME
    scratch/.venv/bin/python scripts/diagnose_d1_3_ignore_set.py --shard-index I --shard-total K
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
from pathlib import Path

import numpy as np

REPO = Path("/data1_ycao/chua/projects/mrsp")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from geometry.camera import CameraIntrinsics, unproject  # noqa: E402
from geometry.coverage_mesh import stream_coverage_mesh_from_zip  # noqa: E402
from geometry.pose import stream_poses_from_zip  # noqa: E402
from eval.evaluable import load_vignette_mask, evaluable_pixel_mask  # noqa: E402
from eval.ignore_set import compute_ignore_set  # noqa: E402
from gt.rasterizer import build_mesh, cast_frame, make_raycast_frame_kernel  # noqa: E402
from gpu_status import get_gpu_stats  # noqa: E402

DATASET_ROOT = Path("/data1_ycao/chua/datasets/C3VDv2/registered_videos")
INTRINSICS_PATH = "/data1_ycao/chua/datasets/C3VDv2/camera_intrinsics.txt"
OUT_ROOT = REPO / "results/d1/diagnosis/ignore_set"
EXPECTED_N_SEQUENCES = 169


def log(msg: str, tag: str = "") -> None:
    prefix = f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}]"
    if tag:
        prefix += f"[{tag}]"
    print(f"{prefix} {msg}", flush=True)


def discover_sequences() -> list[tuple[str, Path]]:
    seen: dict[str, Path] = {}
    for zpath in sorted(DATASET_ROOT.glob("*.zip")):
        name = zpath.stem
        if name.startswith("c0_"):
            continue
        seen[name] = zpath
    seqs = sorted(seen.items())
    assert len(seqs) == EXPECTED_N_SEQUENCES, (
        f"expected {EXPECTED_N_SEQUENCES} c1/c2 registered_videos sequences, found {len(seqs)}"
    )
    return seqs


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()


def is_complete(name: str) -> bool:
    manifest_path = OUT_ROOT / f"{name}.manifest.json"
    bin_path = OUT_ROOT / f"{name}.bin"
    if not manifest_path.exists() or not bin_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return manifest.get("status") == "ok"


def free_disk_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


def compute_ever_evaluable_hit(name: str, zpath: Path, wp, device: str, cam_rays, vignette_mask, kernel) -> tuple[np.ndarray, np.ndarray, int]:
    """Returns (ever_evaluable_hit, gt_observed, n_faces). One ray-cast
    pass under GT pose, no tau gating, no predicted-depth comparison."""
    mesh_data = stream_coverage_mesh_from_zip(zpath)
    gt_poses = stream_poses_from_zip(zpath)
    n_frames = len(gt_poses)
    n_faces = len(mesh_data.faces)

    wp_mesh = build_mesh(wp, mesh_data.vertices, mesh_data.faces, device)
    unused_accumulate_gpu = wp.zeros(n_faces, dtype=wp.int32, device=device)
    ever_evaluable_hit = np.zeros(n_faces, dtype=bool)

    for i in range(n_frames):
        M = gt_poses[i]
        R_c2w = M[:3, :3].T
        T_c2w = M[3, :3]
        hit_gpu, face_gpu, dist_gpu = cast_frame(wp, device, kernel, wp_mesh, cam_rays, R_c2w, T_c2w, unused_accumulate_gpu)
        wp.synchronize()
        hit = hit_gpu.numpy().astype(bool)
        face = face_gpu.numpy()
        d_hit = dist_gpu.numpy().astype(np.float64)
        evaluable = evaluable_pixel_mask(hit, d_hit, vignette_mask)
        np.logical_or.at(ever_evaluable_hit, face[evaluable], True)

    return ever_evaluable_hit, mesh_data.face_observed, n_faces


def run_one(name: str, zpath: Path, wp, device: str, cam_rays, vignette_mask, kernel, gpu_index: int) -> dict:
    t0 = time.time()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        ever_evaluable_hit, gt_observed, n_faces = compute_ever_evaluable_hit(name, zpath, wp, device, cam_rays, vignette_mask, kernel)
        ignore_set = compute_ignore_set(gt_observed, ever_evaluable_hit)
        packed = np.packbits(ignore_set, bitorder="little")
        packed.tofile(OUT_ROOT / f"{name}.bin")
        manifest = {
            "sequence": name, "n_faces": n_faces, "n_ignore_set": int(ignore_set.sum()),
            "git_commit": git_head(), "gpu_index": gpu_index,
            "runtime_seconds_total": time.time() - t0, "status": "ok", "error": None,
        }
        with open(OUT_ROOT / f"{name}.manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        log(f"{name}: DONE in {manifest['runtime_seconds_total']:.1f}s, "
            f"ignore_set={manifest['n_ignore_set']}/{n_faces}", tag=f"gpu{gpu_index}")
        return manifest
    except Exception:
        err = traceback.format_exc()
        log(f"{name}: FAILED\n{err}", tag=f"gpu{gpu_index}")
        manifest = {
            "sequence": name, "status": "failed", "error": err,
            "git_commit": git_head(), "gpu_index": gpu_index,
            "runtime_seconds_total": time.time() - t0,
        }
        try:
            OUT_ROOT.mkdir(parents=True, exist_ok=True)
            with open(OUT_ROOT / f"{name}.manifest.json", "w") as f:
                json.dump(manifest, f, indent=2)
        except OSError:
            log(f"{name}: could not write failure manifest either -- logging only", tag=f"gpu{gpu_index}")
        return manifest


def select_gpu() -> int:
    log("=== GPU availability check (scripts/gpu_status.py) ===")
    table = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
    log(table)
    gpus = get_gpu_stats()
    freest = max(gpus, key=lambda g: g["memory_free_mib"])
    log(f"Selected GPU: index {freest['index']} ({freest['name']}), "
        f"{freest['memory_free_mib']/1000:.1f} GB free, {freest['utilization_pct']}% utilized")
    if freest["memory_free_mib"] < 8000:
        raise RuntimeError(f"no GPU with >= 8GB free (best: {freest['memory_free_mib']}MiB)")
    return freest["index"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence")
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-total", type=int)
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    gpu_index = int(os.environ.get("CUDA_VISIBLE_DEVICES", "-1"))
    if gpu_index == -1:
        gpu_index = select_gpu()
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)

    import warp as wp
    wp.init()
    device = "cuda:0"
    log(f"warp version {wp.config.version}, device {device}, CUDA_VISIBLE_DEVICES={gpu_index}", tag=f"gpu{gpu_index}")

    intr = CameraIntrinsics.from_file(INTRINSICS_PATH)
    cols, rows = np.meshgrid(np.arange(intr.width), np.arange(intr.height))
    px_grid = np.stack([cols.ravel(), rows.ravel()], axis=-1).astype(np.float64)
    cam_rays = unproject(px_grid, intr).astype(np.float32)
    vignette_mask = load_vignette_mask(intr.width, intr.height)
    kernel = make_raycast_frame_kernel(wp)

    sequences = discover_sequences()

    if args.sequence:
        seq_dict = dict(sequences)
        if args.sequence not in seq_dict:
            raise SystemExit(f"{args.sequence!r} not among the 169 registered sequences")
        manifest = run_one(args.sequence, seq_dict[args.sequence], wp, device, cam_rays, vignette_mask, kernel, gpu_index)
        if manifest["status"] != "ok":
            sys.exit(1)
        return

    if args.shard_index is None or args.shard_total is None:
        raise SystemExit("provide --sequence NAME, or both --shard-index and --shard-total")

    my_sequences = [(name, zpath) for i, (name, zpath) in enumerate(sequences) if i % args.shard_total == args.shard_index]
    log(f"shard {args.shard_index}/{args.shard_total}: {len(my_sequences)} sequences assigned", tag=f"gpu{gpu_index}")
    for name, zpath in my_sequences:
        if is_complete(name):
            log(f"{name}: already complete, skipping", tag=f"gpu{gpu_index}")
            continue
        free_gb = free_disk_gb(REPO)
        if free_gb < 50:
            log(f"free disk {free_gb:.1f}GB < 50GB -- stopping shard", tag=f"gpu{gpu_index}")
            return
        try:
            run_one(name, zpath, wp, device, cam_rays, vignette_mask, kernel, gpu_index)
        except Exception:
            log(f"{name}: run_one itself raised -- logging and continuing\n{traceback.format_exc()}", tag=f"gpu{gpu_index}")
    log(f"shard {args.shard_index}/{args.shard_total}: all assigned sequences done", tag=f"gpu{gpu_index}")


if __name__ == "__main__":
    main()
