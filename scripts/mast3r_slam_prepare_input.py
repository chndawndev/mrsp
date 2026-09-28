#!/usr/bin/env python
"""Extract one sequence's raw fisheye rgb/*.png frames into scratch/, for
MASt3R-SLAM's uncalibrated RGBFiles dataset loader (docs/pipelines/mast3r_slam.md).

Same selective-extraction pattern as scripts/endodac_inference.py
(resolve_rgb_members / per-frame zf.read): extracts only the rgb/ zip
members, never the whole archive, and never touches the read-only dataset
root -- output goes to scratch/mast3r_slam/<sequence>/rgb/, reproducible
from the archive at any time.

No undistortion, no resizing: CLAUDE.md's frozen input-format rule
(original fisheye frames) and MASt3R-SLAM's uncalibrated mode both want
the raw frame as-is; RGBFiles just globs *.png and natsorts them.

Usage:
    mast3r-slam-env python scripts/mast3r_slam_prepare_input.py --sequence c1_cecum_t1_v1
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

REPO = Path("/data1_ycao/chua/projects/mrsp")
DATASET_ROOT = Path("/data1_ycao/chua/datasets/C3VDv2")
REGISTERED_DIR = DATASET_ROOT / "registered_videos"
SCRATCH_ROOT = REPO / "scratch/mast3r_slam"


def resolve_rgb_members(zpath: Path) -> list[str]:
    with zipfile.ZipFile(zpath) as zf:
        members = [n for n in zf.namelist() if n.endswith(".png") and "/rgb/" in ("/" + n)]
    if not members:
        raise RuntimeError(f"no rgb/*.png members found in {zpath}")
    return sorted(members)


def prepare(sequence: str) -> Path:
    """Frames land directly in scratch/mast3r_slam/<sequence>/*.png (not a
    nested rgb/ subfolder): MASt3R-SLAM's RGBFiles dataset loader globs
    *.png directly under --dataset and names outputs after
    dataset_path.stem, so the frame folder itself must be named after the
    sequence for output files to come out as c1_cecum_t1_v1.txt etc.
    instead of rgb.txt."""
    zpath = REGISTERED_DIR / f"{sequence}.zip"
    if not zpath.exists():
        raise FileNotFoundError(zpath)
    members = resolve_rgb_members(zpath)
    out_dir = SCRATCH_ROOT / sequence
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath) as zf:
        for i, member in enumerate(members):
            data = zf.read(member)
            (out_dir / f"{i:04d}.png").write_bytes(data)
    print(f"{sequence}: extracted {len(members)} frames -> {out_dir}")
    return out_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence", required=True)
    args = ap.parse_args()
    prepare(args.sequence)
