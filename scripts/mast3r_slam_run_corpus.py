#!/usr/bin/env python
"""MASt3R-SLAM Stage 3 (docs/pipelines/mast3r_slam.md): full-corpus
orchestrator, all 169 registered sequences, no evaluation metric.

Per-sequence work runs as fresh subprocesses of the already-proven
Stage 1/2 scripts (not a shared-process refactor) -- matching this
project's own established lesson from D1 Stage B (EXPERIMENTS.md: a
shared-process bug "crashed the whole shard"); per-sequence process
isolation is deliberate here, not a missed optimization. No sequence is
excluded for any reason; a sequence that errors at any step gets
`"status": "error"` with the exact exception text and the loop moves on
-- never retried with different settings.

Resumable: skips any sequence whose MANIFEST.json already has
`"status": "ok"`. Pauses (stops launching new sequences, exits non-zero)
if free disk drops below 50GB -- never deletes anything.

GPU selected ONCE at start via scripts/gpu_status.py's own freest-GPU
logic (imported, not reimplemented), single index, matching Stage 1/2's
precedent (this box's other GPUs have consistently been busy with other
users' jobs every time this session checked).

Usage (detached, e.g. tmux):
    scratch/.venv/bin/python scripts/mast3r_slam_run_corpus.py 2>&1 | tee logs/mast3r_slam_run_corpus.log
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/data1_ycao/chua/projects/mrsp")
MAST3R_REPO = REPO / "scratch/pipelines/MASt3R-SLAM"
DATASET_ROOT = Path("/data1_ycao/chua/datasets/C3VDv2")
REGISTERED_DIR = DATASET_ROOT / "registered_videos"
SCRATCH_ROOT = REPO / "scratch/mast3r_slam"
FULL_RUN_ROOT = REPO / "results/pipelines/mast3r_slam_full_run"
SCRIPTS_DIR = REPO / "scripts"

MAST3R_ENV_PYTHON = "/data1_ycao/chua/miniforge3/envs/mast3r-slam/bin/python"
CPU_ENV_PYTHON = str(REPO / "scratch/.venv/bin/python3")

EXPECTED_N_SEQUENCES = 169
MIN_FREE_DISK_GB = 50.0

sys.path.insert(0, str(SCRIPTS_DIR))


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}", flush=True)


def discover_sequences() -> list[str]:
    """Duplicated from scripts/endodac_inference.py's discover_sequences
    -- same entry-point-local-copy precedent this project already uses."""
    seen: dict[str, Path] = {}
    for zpath in sorted(REGISTERED_DIR.glob("*.zip")):
        name = zpath.stem
        if name.startswith("c0_"):
            continue
        seen[name] = zpath
    names = sorted(seen)
    assert len(names) == EXPECTED_N_SEQUENCES, (
        f"expected {EXPECTED_N_SEQUENCES} c1/c2 registered_videos sequences, found {len(names)}"
    )
    return names


def select_gpu() -> int:
    from gpu_status import get_gpu_stats
    gpus = get_gpu_stats()
    if not gpus:
        raise RuntimeError("nvidia-smi reported zero GPUs -- cannot start")
    freest = max(gpus, key=lambda g: g["memory_free_mib"])
    log(f"selected GPU {freest['index']} ({freest['name']}): "
        f"{freest['memory_free_mib']/1000:.1f}GB free, {freest['utilization_pct']}% util")
    return int(freest["index"])


def free_disk_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


def is_already_ok(seq: str) -> bool:
    manifest_path = FULL_RUN_ROOT / seq / "MANIFEST.json"
    if not manifest_path.exists():
        return False
    try:
        m = json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return m.get("status") == "ok"


def run_subprocess(cmd: list[str], cwd: Path, env: dict, log_path: Path) -> tuple[int, str]:
    proc = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    output = proc.stdout + "\n" + proc.stderr
    log_path.write_text(output)
    return proc.returncode, output


def run_one_sequence(seq: str, gpu: int) -> dict:
    t0 = time.time()
    out_dir = FULL_RUN_ROOT / seq
    out_dir.mkdir(parents=True, exist_ok=True)
    save_as = f"{seq}_full"
    env = dict(__import__("os").environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    try:
        from mast3r_slam_prepare_input import prepare
        prepare(seq)

        rc, _ = run_subprocess(
            [MAST3R_ENV_PYTHON, str(SCRIPTS_DIR / "mast3r_slam_run_perframe.py"),
             "--dataset", str(SCRATCH_ROOT / seq), "--config", "config/base.yaml",
             "--save-as", save_as, "--out-dir", str(out_dir)],
            cwd=MAST3R_REPO, env=env, log_path=out_dir / "step1_run_perframe.log",
        )
        if rc != 0:
            raise RuntimeError(f"mast3r_slam_run_perframe.py failed, rc={rc} -- see step1_run_perframe.log")

        rc, _ = run_subprocess(
            [MAST3R_ENV_PYTHON, str(SCRIPTS_DIR / "mast3r_slam_reconstruct_poses.py"),
             "--sequence", seq, "--out-root", str(FULL_RUN_ROOT), "--save-as", save_as],
            cwd=REPO, env=env, log_path=out_dir / "step2_reconstruct_poses.log",
        )
        if rc != 0:
            raise RuntimeError(f"mast3r_slam_reconstruct_poses.py failed, rc={rc} -- see step2_reconstruct_poses.log")

        rc, _ = run_subprocess(
            [CPU_ENV_PYTHON, str(SCRIPTS_DIR / "mast3r_slam_trajectory_quality_perframe.py"),
             "--sequence", seq, "--out-root", str(FULL_RUN_ROOT)],
            cwd=REPO, env=env, log_path=out_dir / "step3_trajectory_quality.log",
        )
        if rc != 0:
            raise RuntimeError(f"mast3r_slam_trajectory_quality_perframe.py failed, rc={rc} -- see step3_trajectory_quality.log")

        rc, _ = run_subprocess(
            [CPU_ENV_PYTHON, str(SCRIPTS_DIR / "mast3r_slam_sequence_descriptives.py"), "--sequence", seq],
            cwd=REPO, env=env, log_path=out_dir / "step4_descriptives.log",
        )
        if rc != 0:
            raise RuntimeError(f"mast3r_slam_sequence_descriptives.py failed, rc={rc} -- see step4_descriptives.log")

        status = "ok"
        error = None
    except Exception as e:
        status = "error"
        error = str(e)
    finally:
        rgb_dir = SCRATCH_ROOT / seq
        if rgb_dir.exists():
            shutil.rmtree(rgb_dir)

    manifest = {
        "sequence": seq, "status": status, "error": error,
        "gpu_index": gpu, "elapsed_seconds": time.time() - t0,
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip(),
    }
    with open(out_dir / "MANIFEST.json", "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def main():
    FULL_RUN_ROOT.mkdir(parents=True, exist_ok=True)
    sequences = discover_sequences()
    log(f"{len(sequences)} sequences")

    gpu = select_gpu()

    t_start = time.time()
    n_ok = n_error = n_skipped = 0
    for i, seq in enumerate(sequences):
        if is_already_ok(seq):
            n_skipped += 1
            continue

        free_gb = free_disk_gb(REPO)
        if free_gb < MIN_FREE_DISK_GB:
            log(f"PAUSING: free disk {free_gb:.1f}GB < {MIN_FREE_DISK_GB}GB floor. "
                f"{i}/{len(sequences)} sequences attempted so far ({n_ok} ok, {n_error} error, {n_skipped} skipped). "
                "Nothing deleted. Resume by rerunning this script once space is freed.")
            sys.exit(1)

        log(f"[{i+1}/{len(sequences)}] {seq}: starting")
        manifest = run_one_sequence(seq, gpu)
        if manifest["status"] == "ok":
            n_ok += 1
        else:
            n_error += 1
            log(f"[{i+1}/{len(sequences)}] {seq}: ERROR: {manifest['error']}")

        elapsed = time.time() - t_start
        done = i + 1 - n_skipped
        eta = (elapsed / done * (len(sequences) - n_skipped - done)) if done > 0 else float("nan")
        log(f"[{i+1}/{len(sequences)}] {seq}: status={manifest['status']} "
            f"elapsed_this_seq={manifest['elapsed_seconds']:.1f}s "
            f"total_elapsed={elapsed:.1f}s ETA={eta:.1f}s "
            f"(ok={n_ok} error={n_error} skipped={n_skipped})")

    log(f"DONE. {n_ok} ok, {n_error} error, {n_skipped} skipped (already complete), "
        f"out of {len(sequences)} total. Wall time this run: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
