"""Part 1 of the oracle-ceiling measurement requested for docs/eval_protocol.md.
Step 1: build the fixed camera valid-pixel mask (GT depth raw==0 region)
and verify it is identical across every frame of all 3 sequences. If not
identical, report where/by how much and STOP -- do not proceed to step 2.
See docs/eval_protocol_oracle_gap.md.
"""
import sys
from pathlib import Path

import numpy as np
import tifffile

SEQUENCES = ["c1_cecum_t1_v1", "c1_ascending_t3_v1", "c2_rectum_t1_v1"]
REPO = Path("/data1_ycao/chua/projects/mrsp")


def log(msg): print(msg, flush=True)


reference_mask = None
reference_source = None
all_identical = True

for seq in SEQUENCES:
    depth_dir = REPO / "scratch" / seq / "depth"
    frame_files = sorted(depth_dir.glob("*_depth.tiff"))
    log(f"{seq}: {len(frame_files)} depth frames")
    for fpath in frame_files:
        raw = tifffile.imread(fpath)
        mask = (raw == 0)  # True where invalid (vignette / no-hit)
        if reference_mask is None:
            reference_mask = mask
            reference_source = f"{seq}/{fpath.name}"
            log(f"  reference mask set from {reference_source}: "
                f"{mask.sum()} invalid pixels ({mask.mean()*100:.4f}%), shape {mask.shape}")
            continue
        if mask.shape != reference_mask.shape:
            log(f"  MISMATCH (shape): {seq}/{fpath.name} has shape {mask.shape}, "
                f"reference {reference_mask.shape}")
            all_identical = False
            continue
        if not np.array_equal(mask, reference_mask):
            diff = mask != reference_mask
            n_diff = int(diff.sum())
            log(f"  MISMATCH: {seq}/{fpath.name} differs from reference at {n_diff} pixels "
                f"({n_diff/mask.size*100:.6f}%); this-frame invalid count={mask.sum()}, "
                f"reference invalid count={reference_mask.sum()}")
            all_identical = False

log("")
if all_identical:
    log(f"RESULT: mask IDENTICAL across all frames of all {len(SEQUENCES)} sequences.")
    log(f"Reference: {reference_source}, {reference_mask.sum()} invalid pixels "
        f"({reference_mask.mean()*100:.4f}% of {reference_mask.size} total).")
    np.save(REPO / "results/pipelines/oracle_gap_valid_mask.npy", ~reference_mask)  # save VALID mask (inverted)
    log("Saved valid-pixel mask (True=valid) to results/pipelines/oracle_gap_valid_mask.npy")
    sys.exit(0)
else:
    log("RESULT: mask NOT identical across all frames/sequences. STOPPING per instructions -- "
        "do not proceed to step 2 without resolving this.")
    sys.exit(1)
