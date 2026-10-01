#!/usr/bin/env python
"""Pipeline 2 evaluation pre-flight checks (no metric computed by the
adapter check).

  gate-diff     Pre-flight 1: exact, NaN-aware diff of every leaf of
                results/eval_stage4/summary.json against the metrics.json
                produced by scripts/eval_pipeline_sequence.py --pipeline
                endodac on c1_cecum_t1_v1.
  adapter-check Pre-flight 2: MASt3R-SLAM adapter on c1_cecum_t1_v1 --
                frames with pose/depth, missing frames, fraction of valid
                (non-vignette) pixels with no depth, and a round-trip check
                of the bilinear mapping at 10 random valid pixels against an
                independent scalar implementation fed by Stage 2's own
                documented mapping function. Also a corpus-wide count of
                frames with pose/depth vs the GT frame count (file listing
                only), and the rotation residual after Umeyama alignment as
                a pose-convention check.

CLI:
    scratch/.venv/bin/python scripts/pipeline2_preflight.py gate-diff --candidate PATH
    scratch/.venv/bin/python scripts/pipeline2_preflight.py adapter-check
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import zipfile
from pathlib import Path

import numpy as np

REPO = Path("/data1_ycao/chua/projects/mrsp")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

REFERENCE = REPO / "results/eval_stage4/summary.json"
OUT_DIR = REPO / "results/pipeline2_eval/preflight"
DATASET_ROOT = Path("/data1_ycao/chua/datasets/C3VDv2/registered_videos")
SEED = 20260930


def leaves(x, path=()):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from leaves(v, path + (k,))
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from leaves(v, path + (i,))
    else:
        yield path, x


def same(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return type(a) is type(b) and a == b


def gate_diff(candidate: Path) -> dict:
    ref = dict(leaves(json.loads(REFERENCE.read_text())))
    cand = dict(leaves(json.loads(candidate.read_text())))
    missing = [p for p in ref if p not in cand]
    mismatches = [(p, ref[p], cand[p]) for p in ref if p in cand and not same(ref[p], cand[p])]
    out = {
        "reference": str(REFERENCE.relative_to(REPO)), "candidate": str(candidate),
        "n_reference_leaves": len(ref), "n_missing": len(missing), "n_mismatches": len(mismatches),
        "missing": [list(map(str, p)) for p in missing[:50]],
        "mismatches": [[list(map(str, p)), repr(a), repr(b)] for p, a, b in mismatches[:50]],
        "bit_identical": not missing and not mismatches,
    }
    return out


def gt_frame_count(name: str) -> int:
    with zipfile.ZipFile(DATASET_ROOT / f"{name}.zip") as zf:
        members = [n for n in zf.namelist() if n == "pose.txt" or n.endswith("/pose.txt")]
        if len(members) != 1:
            raise RuntimeError(f"{name}: pose.txt members {members}")
        return len([ln for ln in zf.read(members[0]).decode().splitlines() if ln.strip()])


def adapter_check() -> dict:
    from geometry.pose import stream_poses_from_zip
    from eval.evaluable import load_vignette_mask
    from eval.scale_recovery import aligned_predicted_pose, compute_pose_alignment
    from mast3r_slam_resolution_mapping import original_to_model as stage2_original_to_model
    from pipeline_adapters import (
        BilinearGridMap, Mast3rSlamAdapter, bilinear_reference, load_mast3r_mapping, original_to_model,
        ORIGINAL_H, ORIGINAL_W,
    )

    name = "c1_cecum_t1_v1"
    m = load_mast3r_mapping()
    grid_map = BilinearGridMap(m)
    n_gt = gt_frame_count(name)
    ad = Mast3rSlamAdapter(name, n_gt, grid_map)

    vignette = load_vignette_mask(ORIGINAL_W, ORIGINAL_H)
    valid = ~vignette
    no_depth_valid = valid & ~grid_map.available
    out = {"sequence": name, "mapping": m}

    # same formula as Stage 2's documented function
    xs = np.random.default_rng(SEED).uniform(0, ORIGINAL_W - 1, 1000)
    ys = np.random.default_rng(SEED + 1).uniform(0, ORIGINAL_H - 1, 1000)
    a = np.stack(original_to_model(xs, ys, m))
    b = np.stack(stage2_original_to_model(xs, ys, m))
    out["mapping_formula_max_abs_diff_vs_stage2_function"] = float(np.abs(a - b).max())

    both = sorted(set(ad.frames_with_depth) & set(ad.frames_with_pose))
    out["counts"] = {
        "n_gt_frames": n_gt,
        "n_frames_with_depth": len(ad.frames_with_depth),
        "n_frames_with_pose": len(ad.frames_with_pose),
        "n_frames_with_pose_and_depth": len(both),
        "n_frames_missing_depth": n_gt - len(ad.frames_with_depth),
        "n_frames_missing_pose": n_gt - len(ad.frames_with_pose),
        "n_frames_missing_either": n_gt - len(both),
    }
    out["pixels"] = {
        "n_valid_non_vignette": int(valid.sum()),
        "n_valid_with_no_depth": int(no_depth_valid.sum()),
        "frac_valid_with_no_depth": float(no_depth_valid.sum() / valid.sum()),
        "n_available_total": int(grid_map.available.sum()),
        "n_available_inside_vignette": int((grid_map.available & vignette).sum()),
    }
    # breakdown of the no-depth valid pixels by cause
    X, Y = np.meshgrid(np.arange(ORIGINAL_W, dtype=np.float64), np.arange(ORIGINAL_H, dtype=np.float64))
    px, py = original_to_model(X.ravel(), Y.ravel(), m)
    gw, gh = int(m["model_grid_w"]), int(m["model_grid_h"])
    out["pixels"]["no_depth_valid_by_cause"] = {
        "py_lt_0_top_crop": int((no_depth_valid & (py < 0)).sum()),
        "py_gt_h_minus_1": int((no_depth_valid & (py > gh - 1)).sum()),
        "px_gt_w_minus_1": int((no_depth_valid & (px > gw - 1)).sum()),
        "px_lt_0": int((no_depth_valid & (px < 0)).sum()),
    }

    # round trip at 10 random valid, available pixels, two frames
    rng = np.random.default_rng(SEED)
    cand = np.flatnonzero(valid & grid_map.available)
    picks = rng.choice(cand, size=10, replace=False)
    frames = [0, int(rng.choice(ad.frames_with_depth))]
    rt = []
    max_abs = 0.0
    for f in frames:
        z = ad.depth_grid(f)
        full, _ = ad.depth_native(f)
        for p in picks:
            y, x = divmod(int(p), ORIGINAL_W)
            mpx, mpy = stage2_original_to_model(np.array([float(x)]), np.array([float(y)]), m)
            ref = bilinear_reference(z, float(mpx[0]), float(mpy[0]))
            got = float(full[p])
            d = abs(got - ref)
            max_abs = max(max_abs, d)
            rt.append({"frame": f, "x": x, "y": y, "model_px": float(mpx[0]), "model_py": float(mpy[0]),
                       "adapter_full_res": got, "independent_bilinear": ref, "abs_diff": d,
                       "rel_diff": d / abs(ref) if ref else None})
    out["round_trip"] = {"seed": SEED, "frames": frames, "n_pixels": 10, "max_abs_diff": max_abs, "rows": rt}
    # a pixel with no depth really is NaN in the full-res map
    full0, avail0 = ad.depth_native(0)
    out["round_trip"]["unavailable_pixels_all_nan"] = bool(np.isnan(full0[~avail0]).all())
    out["round_trip"]["available_pixels_all_finite"] = bool(np.isfinite(full0[avail0]).all())

    # pose convention check: rotation residual after position-only Umeyama
    gt_poses = stream_poses_from_zip(DATASET_ROOT / f"{name}.zip")
    gt_pos = np.array([M[3, :3] for M in gt_poses])
    fp = ad.frames_with_pose
    al = compute_pose_alignment(np.array([ad.pose(i)[1] for i in fp]), gt_pos[fp])
    resid = []
    for i in fp:
        Rw, _ = aligned_predicted_pose(al, *ad.pose(i))
        R_gt = gt_poses[i][:3, :3].T
        c = np.clip((np.trace(Rw.T @ R_gt) - 1) / 2, -1, 1)
        resid.append(float(np.degrees(np.arccos(c))))
    out["pose_convention_check"] = {
        "s_pose": al.s_pose, "rotation_residual_deg_median": float(np.median(resid)),
        "rotation_residual_deg_p95": float(np.percentile(resid, 95)),
        "rotation_residual_deg_max": float(np.max(resid)),
    }

    # corpus-wide: frames with pose/depth vs GT frame count (file listing + CSV only)
    from eval_pipeline_sequence import discover_sequences
    corpus = []
    for seq, _ in discover_sequences():
        n = gt_frame_count(seq)
        a2 = Mast3rSlamAdapter(seq, n, grid_map)
        corpus.append({
            "sequence": seq, "n_gt_frames": n, "n_frames_with_depth": len(a2.frames_with_depth),
            "n_frames_with_pose": len(a2.frames_with_pose),
            "n_missing_pose": n - len(a2.frames_with_pose), "n_missing_depth": n - len(a2.frames_with_depth),
            "pose_rows_not_reconstructed": a2.pose_rows_not_reconstructed,
            "first_missing_pose_frame": next((i for i in range(n) if a2.pose(i) is None), None),
        })
    out["corpus_frame_counts"] = corpus
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gate-diff")
    g.add_argument("--candidate", required=True)
    sub.add_parser("adapter-check")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.cmd == "gate-diff":
        res = gate_diff(Path(args.candidate))
        (OUT_DIR / "gate1_reproduction_diff.json").write_text(json.dumps(res, indent=2))
        print(json.dumps({k: v for k, v in res.items() if k not in ("missing", "mismatches")}, indent=2))
        if res["mismatches"] or res["missing"]:
            print(json.dumps(res["mismatches"][:20], indent=2))
        sys.exit(0 if res["bit_identical"] else 1)
    res = adapter_check()
    (OUT_DIR / "gate2_adapter_check.json").write_text(json.dumps(res, indent=2))
    short = {k: v for k, v in res.items() if k != "corpus_frame_counts"}
    short["round_trip"] = {k: v for k, v in res["round_trip"].items() if k != "rows"}
    print(json.dumps(short, indent=2))
    cc = res["corpus_frame_counts"]
    print("corpus: sequences with any missing pose:", sum(r["n_missing_pose"] > 0 for r in cc),
          "total missing-pose frames:", sum(r["n_missing_pose"] for r in cc),
          "sequences with any missing depth:", sum(r["n_missing_depth"] > 0 for r in cc),
          "total GT frames:", sum(r["n_gt_frames"] for r in cc))


if __name__ == "__main__":
    main()
