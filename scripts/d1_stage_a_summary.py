#!/usr/bin/env python
"""D1 Stage A summary: per-sequence trajectory-quality checks and
covariates for all 169 registered sequences, over the EndoDAC predictions
in results/pipelines/endodac_full_run/. No evaluation, no D1 pass/fail
call (that's Stage B) -- this is descriptive only.

D1.1 completion is operationalized per docs/success_criteria.md section 6
"2026-09-23: Clarification: operational definition of D1.1 completion":
finite predicted depth and pose for every frame, and the Sim(3) alignment
succeeds. For a frame-to-frame pipeline like EndoDAC this measures
crash-free completion only (it cannot lose track by construction) -- the
same doc entry's own caveat, repeated here and in docs/d1_stage_a.md
wherever this count is reported.

Reuses src/eval/scale_recovery.py's compute_depth_scale /
compute_pose_alignment / aligned_predicted_pose (Umeyama) read-only --
does not modify src/eval/. Trajectory-drift definitions (endpoint error,
endpoint error as a fraction of GT path length) match
scratch/pipelines/endodac_scale_analysis.py's already-published
c1_cecum_t1_v1 numbers exactly, generalized to all 169 sequences.

Writes results/d1/stage_a_summary.csv, one row per sequence.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/data1_ycao/chua/projects/mrsp")
sys.path.insert(0, str(REPO / "src"))

from geometry.pose import stream_poses_from_zip  # noqa: E402
from gt.depth import depth_to_mm, stream_raw_depth_frames_from_zip  # noqa: E402
from eval.scale_recovery import (  # noqa: E402
    aligned_predicted_pose,
    compute_depth_scale,
    compute_pose_alignment,
)

DATASET_ROOT = Path("/data1_ycao/chua/datasets/C3VDv2")
REGISTERED_DIR = DATASET_ROOT / "registered_videos"
PRED_ROOT = REPO / "results/pipelines/endodac_full_run"
MESH_IDENTITY_CSV = REPO / "results/mesh_identity.csv"
RELEASE_CSV = REPO / "docs/release_v1.csv"
OUT_DIR = REPO / "results/d1"
LOG_PATH = REPO / "logs/d1_stage_a_summary.log"
EXPECTED_N_SEQUENCES = 169


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}"
    print(line, flush=True)


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


def process_sequence(name: str, zpath: Path) -> dict:
    out_dir = PRED_ROOT / name
    manifest_path = out_dir / "MANIFEST.json"
    row: dict = {"sequence": name}

    if not manifest_path.exists():
        row.update({"status": "missing_manifest", "d1_1_complete": False})
        return row

    manifest = json.loads(manifest_path.read_text())
    row["status"] = manifest.get("status")
    row["manifest_n_frames"] = manifest.get("n_frames")
    row["n_nonfinite_depth"] = manifest.get("n_nonfinite_depth")
    row["n_nonfinite_pose"] = manifest.get("n_nonfinite_pose")
    row["runtime_seconds_total"] = manifest.get("runtime_seconds_total")
    row["gpu_index"] = manifest.get("gpu_index")
    row["git_commit"] = manifest.get("git_commit")
    row["error"] = manifest.get("error")

    if manifest.get("status") != "ok":
        row["d1_1_complete"] = False
        return row

    n_frames = manifest["n_frames"]
    finite_depth = manifest.get("n_nonfinite_depth", 1) == 0
    finite_pose = manifest.get("n_nonfinite_pose", 1) == 0

    # ---------------- pose alignment / trajectory quality ----------------
    gt_poses = stream_poses_from_zip(zpath)
    gt_positions = gt_poses[:, 3, :3]
    pred_poses = np.load(out_dir / "poses_pred.npy")
    pred_positions = pred_poses[:, :3, 3]

    sim3_ok = True
    try:
        alignment = compute_pose_alignment(pred_positions, gt_positions)
        aligned_positions = np.stack([
            aligned_predicted_pose(alignment, pred_poses[i, :3, :3], pred_poses[i, :3, 3])[1]
            for i in range(n_frames)
        ])
        ate_mm = float(np.sqrt(np.mean(np.sum((aligned_positions - gt_positions) ** 2, axis=1))))
        gt_step_lens = np.linalg.norm(np.diff(gt_positions, axis=0), axis=1)
        gt_path_length_mm = float(gt_step_lens.sum())
        endpoint_error_mm = float(np.linalg.norm(aligned_positions[-1] - gt_positions[-1]))
        endpoint_error_frac_of_gt_path = (
            endpoint_error_mm / gt_path_length_mm if gt_path_length_mm > 0 else float("nan")
        )
        s_pose = alignment.s_pose
    except np.linalg.LinAlgError:
        sim3_ok = False
        ate_mm = gt_path_length_mm = endpoint_error_mm = endpoint_error_frac_of_gt_path = s_pose = float("nan")

    row.update({
        "s_pose": s_pose,
        "ate_mm": ate_mm,
        "gt_path_length_mm": gt_path_length_mm,
        "endpoint_error_mm": endpoint_error_mm,
        "endpoint_error_frac_of_gt_path": endpoint_error_frac_of_gt_path,
        "sim3_alignment_succeeded": sim3_ok,
    })

    # ---------------- depth scale ----------------
    gt_depth_mm_frames, gt_depth_valid_frames = [], []
    for raw in stream_raw_depth_frames_from_zip(zpath):
        d_mm, valid = depth_to_mm(raw.ravel())
        gt_depth_mm_frames.append(d_mm)
        gt_depth_valid_frames.append(valid)
    pred_depth_native_frames = [
        np.load(out_dir / "depth" / f"{i:04d}_pred_depth.npy").ravel().astype(np.float64)
        for i in range(n_frames)
    ]
    depth_scale = compute_depth_scale(gt_depth_mm_frames, gt_depth_valid_frames, pred_depth_native_frames)
    ratios = np.array(depth_scale.per_frame_ratios)
    q1, q3 = np.percentile(ratios, [25, 75])
    row.update({
        "depth_scale_median": depth_scale.median,
        "depth_scale_relative_iqr": float((q3 - q1) / depth_scale.median) if depth_scale.median else float("nan"),
    })

    row["d1_1_complete"] = bool(finite_depth and finite_pose and sim3_ok)
    return row


def load_covariates() -> pd.DataFrame:
    mesh = pd.read_csv(MESH_IDENTITY_CSV)
    release = pd.read_csv(RELEASE_CSV)
    covariates = mesh.merge(
        release[["Video Name", "Camera Speed", "Edge Enhancement", "Brightness", "Deformation",
                 "Open End Visible", "Tags", "Comments", "Qualitative Score", "Quantitative Score",
                 "Total Frames"]],
        on="Video Name", how="left", validate="one_to_one",
    )
    covariates["physical_segment_id"] = covariates["Colon"] + "_" + covariates["Segment"]
    unmatched = covariates[covariates["Camera Speed"].isna()]
    if len(unmatched):
        log(f"WARNING: {len(unmatched)} sequences matched mesh_identity.csv but not docs/release_v1.csv: "
            f"{unmatched['Video Name'].tolist()}")
    return covariates.rename(columns={"Video Name": "sequence"})


def _process_one(args: tuple[str, Path]) -> dict:
    name, zpath = args
    return process_sequence(name, zpath)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=16,
                         help="CPU-only, pure TIFF-decode + numpy work -- parallel across "
                              "sequences (each sequence is independent, read-only). "
                              "This is a shared 48-core box; 16 is a moderate default, "
                              "not the full core count.")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    sequences = discover_sequences()
    log(f"processing {len(sequences)} sequences with {args.workers} worker processes")

    rows = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_process_one, seq): seq[0] for seq in sequences}
        n_done = 0
        for future in as_completed(futures):
            name = futures[future]
            row = future.result()
            rows.append(row)
            n_done += 1
            if n_done % 10 == 0 or n_done == len(sequences):
                elapsed = time.time() - t0
                eta = elapsed / n_done * (len(sequences) - n_done)
                log(f"{n_done}/{len(sequences)}: {name} status={row.get('status')} "
                    f"d1_1_complete={row.get('d1_1_complete')} elapsed={elapsed:.1f}s ETA={eta:.1f}s")

    df = pd.DataFrame(rows).sort_values("sequence").reset_index(drop=True)
    covariates = load_covariates()
    merged = df.merge(covariates, on="sequence", how="left", validate="one_to_one")

    unmatched_pred = merged[merged["mesh_hash"].isna()]
    if len(unmatched_pred):
        log(f"WARNING: {len(unmatched_pred)} predicted sequences have no covariate match: "
            f"{unmatched_pred['sequence'].tolist()}")

    out_path = OUT_DIR / "stage_a_summary.csv"
    merged.to_csv(out_path, index=False)
    log(f"wrote {out_path} ({len(merged)} rows, {len(merged.columns)} columns)")
    log(f"D1.1 complete: {int(merged['d1_1_complete'].sum())}/{len(merged)}")
    log(f"distinct mesh_hash: {merged['mesh_hash'].nunique()}")
    log(f"distinct physical_segment_id: {merged['physical_segment_id'].nunique()}")
    log("ALL DONE")


if __name__ == "__main__":
    main()
