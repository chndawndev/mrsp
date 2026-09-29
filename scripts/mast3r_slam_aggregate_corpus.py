#!/usr/bin/env python
"""MASt3R-SLAM Stage 3 step 4 (docs/pipelines/mast3r_slam.md): corpus
aggregation. Reads every sequence's MANIFEST.json + descriptives.json
(whether or not that sequence completed -- failures are rows too, per
instructions), joins in covariate columns from the already-published
results/d1/stage_a_summary.csv (plain CSV read + merge on `sequence`,
not re-deriving them), writes results/mast3r/stage3_summary.csv.

Does not compute any evaluation metric, does not import from src/eval/
or src/eval_ext/.

Usage:
    python scripts/mast3r_slam_aggregate_corpus.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO = Path("/data1_ycao/chua/projects/mrsp")
FULL_RUN_ROOT = REPO / "results/pipelines/mast3r_slam_full_run"
STAGE_A_CSV = REPO / "results/d1/stage_a_summary.csv"
OUT_DIR = REPO / "results/mast3r"

COVARIATE_COLS = [
    "sequence", "mesh_hash", "n_vertices", "n_faces", "n_unobserved_components",
    "unobserved_area_total_mm2", "Colon", "Segment", "Phantom Number", "Video Number",
    "Debris", "Camera Speed", "Edge Enhancement", "Brightness", "Deformation",
    "Open End Visible", "Tags", "Comments", "Qualitative Score", "Quantitative Score",
    "Total Frames", "physical_segment_id",
]


def load_sequence_row(seq_dir: Path) -> dict:
    manifest_path = seq_dir / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    row = {
        "sequence": manifest["sequence"], "status": manifest["status"], "error": manifest.get("error"),
        "gpu_index": manifest.get("gpu_index"), "elapsed_seconds": manifest.get("elapsed_seconds"),
        "git_commit": manifest.get("git_commit"),
    }
    desc_path = seq_dir / "descriptives.json"
    if manifest["status"] == "ok" and desc_path.exists():
        d = json.loads(desc_path.read_text())
        row.update({
            "n_total_frames": d["completion"]["n_total_frames"],
            "n_reconstructed": d["completion"]["n_reconstructed"],
            "sim3_alignment_succeeded": d["completion"]["sim3_alignment_succeeded"],
            "d1_1_complete": d["completion"]["d1_1_complete"],
            "ate_rmse_mm": d["ate_rmse_mm"],
            "s_pose": d["s_pose"],
            "endpoint_drift_frac_of_gt_path": d["endpoint_drift_frac_of_gt_path"],
            "gt_path_length_mm": d["gt_path_length_mm"],
            "depth_scale_median": d["depth_scale"]["depth_scale_median"],
            "depth_scale_relative_iqr": d["depth_scale"]["depth_scale_relative_iqr"],
            "n_keyframes": d["keyframe_gap"]["n_keyframes"],
            "longest_gap_frames": d["keyframe_gap"]["longest_gap_frames"],
            "longest_gap_range_start": d["keyframe_gap"]["longest_gap_range"][0] if d["keyframe_gap"]["longest_gap_range"] else None,
            "longest_gap_range_end": d["keyframe_gap"]["longest_gap_range"][1] if d["keyframe_gap"]["longest_gap_range"] else None,
            "longest_gap_gt_path_length_mm": d["keyframe_gap"]["longest_gap_gt_motion"]["total_path_length_mm"] if d["keyframe_gap"]["longest_gap_gt_motion"] else None,
            "longest_gap_gt_rotation_deg": d["keyframe_gap"]["longest_gap_gt_motion"]["total_rotation_deg"] if d["keyframe_gap"]["longest_gap_gt_motion"] else None,
        })
    else:
        for col in ["n_total_frames", "n_reconstructed", "sim3_alignment_succeeded", "d1_1_complete",
                    "ate_rmse_mm", "s_pose", "endpoint_drift_frac_of_gt_path", "gt_path_length_mm",
                    "depth_scale_median", "depth_scale_relative_iqr", "n_keyframes", "longest_gap_frames",
                    "longest_gap_range_start", "longest_gap_range_end",
                    "longest_gap_gt_path_length_mm", "longest_gap_gt_rotation_deg"]:
            row[col] = None
    return row


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seq_dirs = sorted(d for d in FULL_RUN_ROOT.iterdir() if d.is_dir())
    rows = [load_sequence_row(d) for d in seq_dirs]
    df = pd.DataFrame(rows)

    stage_a = pd.read_csv(STAGE_A_CSV)
    covariates = stage_a[[c for c in COVARIATE_COLS if c in stage_a.columns]]
    df = df.merge(covariates, on="sequence", how="left")

    out_path = OUT_DIR / "stage3_summary.csv"
    df.to_csv(out_path, index=False)

    n_total = len(df)
    n_ok = int((df["status"] == "ok").sum())
    n_error = int((df["status"] == "error").sum())
    n_d1_1 = int((df["d1_1_complete"] == True).sum())  # noqa: E712
    print(f"wrote {out_path} ({n_total} rows)")
    print(f"status: ok={n_ok} error={n_error}")
    print(f"D1.1 complete: {n_d1_1}/{n_total}")
    if n_error:
        print("errored sequences:")
        for _, row in df[df["status"] == "error"].iterrows():
            print(f"  {row['sequence']}: {row['error']}")


if __name__ == "__main__":
    main()
