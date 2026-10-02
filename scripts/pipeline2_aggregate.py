#!/usr/bin/env python
"""Pipeline 2 evaluation: corpus aggregation (CPU only).

  A. MASt3R-SLAM corpus tables, every config x tau x metric, built with
     scripts/eval_d1_aggregate.py::build_corpus_tables (imported, same
     seed, same bootstrap) -- D1 Stage B's exact aggregation. The same
     call on EndoDAC's Stage B outputs is checked equal to
     results/d1/corpus_tables.json first (aggregation reproduction check).
     Diagnostics a-c, missing frames and the no-depth pixel fraction.
  B. Region IoU for both pipelines next to the area-matched random
     baseline at that pipeline's own predicted-unobserved area
     (rows from scripts/pipeline2_region_iou.py).
  C. H5 and H6 on MASt3R-SLAM: paired (same sequences) differences of
     pooled ratios, cluster bootstrap by mesh.
  D. Descriptive pairwise MASt3R-SLAM minus EndoDAC differences, paired by
     sequence, cluster bootstrap by mesh.

Aggregation rule (docs/eval_protocol.md, 2026-09-23): pooled point
estimates; mesh-level cluster bootstrap, 10,000 replicates, 95% percentile
interval; (Colon, Segment) and (Colon, Segment, Phantom Number) intervals
alongside every CI. No p-values. Bootstrap seed 20260925 (logged).

CLI: scratch/.venv/bin/python scripts/pipeline2_aggregate.py
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/data1_ycao/chua/projects/mrsp")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import eval_d1_aggregate as agg  # noqa: E402
import region_iou_gate as rig  # noqa: E402

SEED = agg.SEED
N_BOOT = agg.N_BOOT
TAUS = agg.TAUS
CONFIG_NAMES = agg.CONFIG_NAMES
ROOTS = {
    "endodac": REPO / "results/d1/per_sequence",
    "mast3r_slam": REPO / "results/pipeline2_eval/per_sequence/mast3r_slam",
}
OUT_DIR = REPO / "results/pipeline2_eval"
IOU_DIR = OUT_DIR / "region_iou"
CLUSTERS = ["mesh_hash", "colon_segment", "colon_segment_phantom"]
SIZE_CLASSES = ["small", "medium", "large", "medium_plus_large"]
IOU_CELLS = [(c, 0.25) for c in CONFIG_NAMES] + [("fully_predicted", t) for t in (0.15, 0.35, 0.50)]
AREA_METRICS = {
    "false_reassurance_rate": ("false_reassurance_numerator_area_mm2", "gt_unobserved_area_mm2"),
    "false_alarm_rate": ("false_alarm_numerator_area_mm2", "false_alarm_denominator_area_mm2"),
    "area_fraction": ("pred_unobserved_area_mm2", "total_mesh_area_mm2"),
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}", flush=True)


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def load_pipeline(pipeline: str, mesh_df: pd.DataFrame):
    agg.PER_SEQ_ROOT = ROOTS[pipeline]  # load_all/discover read this module global at call time
    ok, failures = agg.discover_completed_sequences()
    if failures or len(ok) != 169:
        raise SystemExit(f"{pipeline}: {len(ok)} ok sequences, failures {failures} -- not aggregating a partial corpus")
    regions_df, area_df = agg.load_all(ok)
    return ok, agg.add_clusters(regions_df, mesh_df), agg.add_clusters(area_df, mesh_df)


def leaves(x, p=()):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from leaves(v, p + (str(k),))
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            yield from leaves(v, p + (i,))
    else:
        yield p, x


def same_leaf(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return float(a) == float(b)
    return a == b


# --------------------------------------------------------------------------
# paired bootstraps
# --------------------------------------------------------------------------

def paired_ratio_diff(dfA: pd.DataFrame, dfB: pd.DataFrame, num: str, den: str, cluster: str,
                      rng: np.random.Generator) -> dict:
    """ratio(A) - ratio(B), each a pooled sum(num)/sum(den); A and B cover the
    same sequences. One mesh-cluster resample drives both ratios."""
    if sorted(dfA["sequence"]) != sorted(dfB["sequence"]):
        raise RuntimeError("paired difference on different sequence sets")
    gA = dfA.groupby(cluster)[[num, den]].sum()
    gB = dfB.groupby(cluster)[[num, den]].sum().reindex(gA.index)
    nA, dA, nB, dB = (gA[num].to_numpy(), gA[den].to_numpy(), gB[num].to_numpy(), gB[den].to_numpy())
    point_A, point_B = nA.sum() / dA.sum(), nB.sum() / dB.sum()
    n = len(gA)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    with np.errstate(invalid="ignore", divide="ignore"):
        diffs = nA[idx].sum(1) / dA[idx].sum(1) - nB[idx].sum(1) / dB[idx].sum(1)
    diffs = diffs[np.isfinite(diffs)]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"A": float(point_A), "B": float(point_B), "diff": float(point_A - point_B),
            "ci": [float(lo), float(hi)], "n_clusters": int(n), "n_sequences": int(dfA["sequence"].nunique())}


def paired_mean_diff(values: pd.DataFrame, col: str, cluster: str, rng: np.random.Generator) -> dict:
    """Pooled mean over rows of `col` (a per-region paired difference), mesh-cluster bootstrap."""
    g = values.groupby(cluster)[col].agg(["sum", "count"])
    s, c = g["sum"].to_numpy(), g["count"].to_numpy(dtype=float)
    idx = rng.integers(0, len(g), size=(N_BOOT, len(g)))
    boot = s[idx].sum(1) / c[idx].sum(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"mean": float(s.sum() / c.sum()), "ci": [float(lo), float(hi)], "n_regions": int(c.sum()),
            "n_clusters": int(len(g))}


def with_sensitivity(fn, *args, rng) -> dict:
    out = fn(*args, "mesh_hash", rng)
    for cl in ("colon_segment", "colon_segment_phantom"):
        out[f"sensitivity_{cl}_ci"] = fn(*args, cl, rng)["ci"]
    return out


def size_filter(df: pd.DataFrame, size_class: str) -> pd.DataFrame:
    if size_class == "medium_plus_large":
        return df[df["size_class"].isin(["medium", "large"])]
    return df[df["size_class"] == size_class]


# --------------------------------------------------------------------------

def section_a(mesh_df) -> tuple[dict, dict]:
    log("=== A0: aggregation reproduction on EndoDAC vs results/d1/corpus_tables.json ===")
    _, reg_e, area_e = load_pipeline("endodac", mesh_df)
    tables_e = agg.build_corpus_tables(reg_e, area_e, np.random.default_rng(SEED))
    ref = json.loads((REPO / "results/d1/corpus_tables.json").read_text())["tables"]
    ref_leaves = dict(leaves(ref))
    new_leaves = dict(leaves(json.loads(json.dumps(jsonable(tables_e)))))
    mism = [p for p in ref_leaves if p not in new_leaves or not same_leaf(ref_leaves[p], new_leaves[p])]
    repro = {"n_reference_leaves": len(ref_leaves), "n_mismatch_or_missing": len(mism), "examples": [list(map(str, p)) for p in mism[:10]]}
    log(f"aggregation reproduction: {repro}")
    if mism:
        raise SystemExit("EndoDAC aggregation does not reproduce results/d1/corpus_tables.json -- stopping")

    tables_out, reg_m, area_m = corpus_section("mast3r_slam", mesh_df, repro)
    return {"endodac": (reg_e, area_e), "mast3r_slam": (reg_m, area_m)}, tables_out


def corpus_section(pipeline: str, mesh_df, repro: dict | None = None, out_dir: Path | None = None,
                   prefix: str = "mast3r") -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Corpus tables, diagnostics b and c, missing frames, no-depth pixel
    fraction and invalid-depth counts of one adapter-evaluated pipeline
    (ROOTS[pipeline]). Defaults write the pipeline 2 evaluation's files."""
    out_dir = out_dir or OUT_DIR
    log(f"=== A: {pipeline} corpus tables ===")
    seqs, reg_m, area_m = load_pipeline(pipeline, mesh_df)
    tables_m = agg.build_corpus_tables(reg_m, area_m, np.random.default_rng(SEED))

    # diagnostics b (per config, over sequences) and c (pooled face counts by tau); missing frames
    diag_b, diag_c, missing_rows = {}, {}, []
    per_seq = []
    for s in seqs:
        m = json.loads((ROOTS[pipeline] / s / "metrics.json").read_text())
        mp = m["missing_predictions"]
        missing_rows.append({
            "sequence": s, **{k: v for k, v in mp.items() if not isinstance(v, (list, dict))},
            "first_missing_pose_frame": mp["frames_missing_pose"][0] if mp["frames_missing_pose"] else None,
            "depth_scale_median": m["depth_scale_median"], "pose_alignment_s_pose": m["pose_alignment_s_pose"],
            "ate_after_alignment_mm": m["ate_after_alignment_mm"],
            "rotation_residual_median_deg": m["rotation_residual_after_alignment_deg"]["median"],
            "cross_check_ignore_set_diff": m["cross_checks"].get("ignore_set_n_faces_differing_from_d1_diagnosis"),
            "cross_check_oracle_bits_diff_max": max(
                m["cross_checks"]["oracle_bits_n_faces_differing_from_d1_stage_b_by_tau"].values()),
            "n_invalid_depth_pixels": mp["adapter"].get("n_invalid_depth_pixels"),
            "n_frames_with_invalid_depth": mp["adapter"].get("n_frames_with_invalid_depth"),
        })
        for c, cd in m["configurations"].items():
            per_seq.append({"sequence": s, "config": c, "ray_miss_frac": cd["ray_miss_frac"],
                            "d_pred_unavailable_frac_of_evaluable": cd["d_pred_unavailable_frac_of_evaluable"],
                            "n_frames_raycast": cd["n_frames_raycast"]})
        for r in m["diagnostic_c_face_diff"]:
            t = str(r["tau"])
            d = diag_c.setdefault(t, {"n_only_fully_predicted_unobserved": 0, "n_also_pred_depth_only_unobserved": 0})
            d["n_only_fully_predicted_unobserved"] += r["n_only_fully_predicted_unobserved"]
            d["n_also_pred_depth_only_unobserved"] += r["n_also_pred_depth_only_unobserved"]
    for d in diag_c.values():
        d["fraction"] = d["n_also_pred_depth_only_unobserved"] / d["n_only_fully_predicted_unobserved"] \
            if d["n_only_fully_predicted_unobserved"] else float("nan")
    ps = pd.DataFrame(per_seq)
    ps.to_csv(out_dir / f"{prefix}_diagnostic_b.csv", index=False)
    for c in CONFIG_NAMES:
        sub = ps[ps["config"] == c]
        diag_b[c] = {col: {"mean": float(sub[col].mean()), "median": float(sub[col].median()), "max": float(sub[col].max()),
                           "argmax_sequence": sub.loc[sub[col].idxmax(), "sequence"]}
                     for col in ["ray_miss_frac", "d_pred_unavailable_frac_of_evaluable"]}
    mdf = pd.DataFrame(missing_rows).merge(
        mesh_df[["Video Name", "mesh_hash"]].rename(columns={"Video Name": "sequence"}), on="sequence")
    mdf.to_csv(out_dir / f"{prefix}_missing_frames.csv", index=False)
    aff = mdf[mdf["n_frames_missing_either"] > 0]
    missing_summary = {
        "n_sequences": len(mdf), "n_gt_frames_total": int(mdf["n_gt_frames"].sum()),
        "n_frames_missing_pose": int(mdf["n_frames_missing_pose"].sum()),
        "n_frames_missing_depth": int(mdf["n_frames_missing_depth"].sum()),
        "n_frames_missing_either": int(mdf["n_frames_missing_either"].sum()),
        "n_sequences_affected": int(len(aff)), "n_meshes_affected": int(aff["mesh_hash"].nunique()),
        "affected_sequences": aff[["sequence", "n_gt_frames", "n_frames_with_pose", "n_frames_missing_pose",
                                   "first_missing_pose_frame"]].to_dict("records"),
        "no_depth_frac_of_valid_pixels_frames_with_depth": {
            "min": float(mdf["no_depth_frac_of_valid_pixels_frames_with_depth"].min()),
            "max": float(mdf["no_depth_frac_of_valid_pixels_frames_with_depth"].max())},
        "no_depth_frac_of_valid_pixels_all_gt_frames_pooled": float(
            (mdf["no_depth_frac_of_valid_pixels_all_gt_frames"] * mdf["n_gt_frames"]).sum() / mdf["n_gt_frames"].sum()),
        "n_sequences_every_frame_pose_and_depth": int((mdf["n_frames_missing_either"] == 0).sum()),
        "cross_check_ignore_set_max_diff": int(mdf["cross_check_ignore_set_diff"].max()),
        "cross_check_oracle_bits_max_diff": int(mdf["cross_check_oracle_bits_diff_max"].max()),
        "n_invalid_depth_pixels_total": (None if mdf["n_invalid_depth_pixels"].isna().all()
                                         else int(mdf["n_invalid_depth_pixels"].sum())),
        "sequences_with_invalid_depth": mdf.loc[mdf["n_invalid_depth_pixels"].fillna(0) > 0,
                                                ["sequence", "n_invalid_depth_pixels", "n_frames_with_invalid_depth"]].to_dict("records"),
    }
    log(f"missing frames: { {k: v for k, v in missing_summary.items() if k != 'affected_sequences'} }")

    seg_all = reg_m[reg_m["headline"] & reg_m["segment_intersects_mesh"].notna()]
    out = {
        "seed": SEED, "n_boot": N_BOOT, "aggregation_reproduction_endodac": repro,
        "tables": tables_m, "diagnostic_b": diag_b, "diagnostic_c": diag_c, "missing_frames": missing_summary,
        "segment_intersect_fraction_overall": float(seg_all["segment_intersects_mesh"].astype(bool).mean()),
        "n_meshes": int(reg_m["mesh_hash"].nunique()),
        "n_colon_segment": int(reg_m["colon_segment"].nunique()),
        "n_colon_segment_phantom": int(reg_m["colon_segment_phantom"].nunique()),
    }
    (out_dir / f"{prefix}_corpus_tables.json").write_text(json.dumps(jsonable(out), indent=2))
    return out, reg_m, area_m


def section_b(mesh_df, pipelines=("endodac", "mast3r_slam"), iou_dir: Path | None = None,
              out_path: Path | None = None) -> dict:
    """pipelines / iou_dir / out_path: defaults are the pipeline 2 evaluation;
    scripts/d2b_aggregate.py passes the three-pipeline values."""
    log("=== B: region IoU with matched random baselines ===")
    iou_dir = iou_dir or IOU_DIR
    rows = agg.add_clusters(pd.read_csv(iou_dir / "region_iou_rows.csv.gz"), mesh_df)
    areas = agg.add_clusters(pd.read_csv(iou_dir / "area_rows.csv"), mesh_df)
    rng = np.random.default_rng(SEED)
    out = {"seed": SEED, "n_boot": N_BOOT, "n_random_seeds": 20, "cells": {}}

    # check: EndoDAC fully_predicted@0.25 random rows reproduce the gate's area-matched random rows
    gate = pd.read_csv(REPO / "results/region_iou_gate/region_iou_raw.csv")
    g = gate[gate["baseline"] == "area_matched_random"][["sequence", "region_id", "seed", "iou"]]
    mine = rows[(rows.pipeline == "endodac") & (rows.config == "fully_predicted") & (rows.tau == 0.25)
                & (rows.source == "random")][["sequence", "region_id", "seed", "iou"]]
    j = g.merge(mine, on=["sequence", "region_id", "seed"], suffixes=("_gate", "_new"))
    out["check_reproduces_gate_random_rows"] = {
        "n_gate_rows": len(g), "n_matched": len(j), "max_abs_diff": float((j.iou_gate - j.iou_new).abs().max())}
    log(f"gate random-row reproduction: {out['check_reproduces_gate_random_rows']}")

    for pipeline in pipelines:
        for config, tau in IOU_CELLS:
            key = f"{pipeline}|{config}|{tau}"
            cell_rows = rows[(rows.pipeline == pipeline) & (rows.config == config) & (rows.tau == tau)]
            pipe = cell_rows[cell_rows.source == "pipeline"]
            rand = cell_rows[cell_rows.source == "random"]
            rand_avg = rand.groupby(["sequence", "region_id"], as_index=False).agg(
                size_class=("size_class", "first"), iou=("iou", "mean"), mesh_hash=("mesh_hash", "first"),
                colon_segment=("colon_segment", "first"), colon_segment_phantom=("colon_segment_phantom", "first"))
            paired = pipe.merge(rand_avg[["sequence", "region_id", "iou"]], on=["sequence", "region_id"],
                                suffixes=("", "_rand"))
            if len(paired) != len(pipe):
                raise RuntimeError(f"{key}: pairing lost rows")
            paired["diff"] = paired["iou"] - paired["iou_rand"]
            cell = {"n_sequences": int(pipe.sequence.nunique()), "n_meshes": int(pipe.mesh_hash.nunique()),
                    "by_size_class": {}}
            for sc in SIZE_CLASSES:
                p_cell = rig.pooled_cell(pipe, sc, "mesh_hash", rng)
                for cl in ("colon_segment", "colon_segment_phantom"):
                    s = rig.pooled_cell(pipe, sc, cl, rng)
                    p_cell[f"sensitivity_{cl}_mean_ci"] = s["mean_ci"]
                    p_cell[f"sensitivity_{cl}_median_ci"] = s["median_ci"]
                r_cell = rig.pooled_cell(rand_avg, sc, "mesh_hash", rng)
                per_seed = []
                for seed, sg in rand.groupby("seed"):
                    v = size_filter(sg, sc)["iou"].to_numpy()
                    per_seed.append((float(v.mean()), float(np.median(v))) if len(v) else (np.nan, np.nan))
                per_seed = np.array(per_seed)
                d_cell = with_sensitivity(paired_mean_diff, size_filter(paired, sc), "diff", rng=rng) \
                    if len(size_filter(paired, sc)) else None
                cell["by_size_class"][sc] = {
                    "pipeline": p_cell,
                    "random_seed_averaged": {"mean": r_cell["mean"], "median": r_cell["median"],
                                             "mean_ci": r_cell["mean_ci"], "median_ci": r_cell["median_ci"],
                                             "per_seed_mean_range": [float(np.nanmin(per_seed[:, 0])), float(np.nanmax(per_seed[:, 0]))],
                                             "per_seed_median_range": [float(np.nanmin(per_seed[:, 1])), float(np.nanmax(per_seed[:, 1]))]},
                    "paired_diff_pipeline_minus_random_mean": d_cell,
                }
            # false alarm and area fraction next to IoU (pipeline: pooled + CI; random: seed-averaged pooled + range)
            a_cell = areas[(areas.pipeline == pipeline) & (areas.config == config) & (areas.tau == tau)]
            a_pipe = a_cell[a_cell.source == "pipeline"]
            a_rand = a_cell[a_cell.source == "random"]
            ctx = {}
            for metric, (num, den) in [("false_alarm_rate", AREA_METRICS["false_alarm_rate"]),
                                       ("area_fraction", AREA_METRICS["area_fraction"])]:
                c = agg.area_metric_cell(a_pipe.assign(config=config, tau=tau), config, tau, num, den, "mesh_hash", rng)
                per_seed = a_rand.groupby("seed").apply(lambda x: x[num].sum() / x[den].sum(), include_groups=False)
                ctx[metric] = {"pipeline": c["point"], "pipeline_ci": [c["ci_lo"], c["ci_hi"]],
                               "random_seed_mean": float(per_seed.mean()),
                               "random_seed_range": [float(per_seed.min()), float(per_seed.max())]}
            cell["context"] = ctx
            out["cells"][key] = cell
            ml = cell["by_size_class"]["medium_plus_large"]
            log(f"{key}: IoU mean {ml['pipeline']['mean']:.4f} vs random {ml['random_seed_averaged']['mean']:.4f}, "
                f"diff {ml['paired_diff_pipeline_minus_random_mean']['mean']:+.4f} "
                f"{ml['paired_diff_pipeline_minus_random_mean']['ci']}, FA {ctx['false_alarm_rate']['pipeline']:.4f}, "
                f"area {ctx['area_fraction']['pipeline']:.4f}")
    (out_path or OUT_DIR / "region_iou_summary.json").write_text(json.dumps(jsonable(out), indent=2))
    return out


def section_c(dfs, pipeline: str = "mast3r_slam", out_path: Path | None = None) -> dict:
    log(f"=== C: H5 and H6 on {pipeline} ===")
    _, area_m = dfs[pipeline]
    rng = np.random.default_rng(SEED)
    out = {"seed": SEED, "n_boot": N_BOOT, "H5": {}, "H6": {}}
    for tau in TAUS:
        sub = area_m[area_m.tau == tau]
        num, den = AREA_METRICS["false_reassurance_rate"]
        h5 = with_sensitivity(paired_ratio_diff, sub[sub.config == "fully_predicted"], sub[sub.config == "pred_pose_only"],
                              num, den, rng=rng)
        h5["criterion_met_mesh_level"] = bool(h5["diff"] < 0 and h5["ci"][1] < 0)
        out["H5"][str(tau)] = h5
        num, den = AREA_METRICS["false_alarm_rate"]
        h6 = with_sensitivity(paired_ratio_diff, sub[sub.config == "pred_depth_only"], sub[sub.config == "pred_pose_only"],
                              num, den, rng=rng)
        h6["criterion_met_mesh_level"] = bool(h6["diff"] > 0 and h6["ci"][0] > 0)
        out["H6"][str(tau)] = h6
        log(f"tau {tau}: H5 diff {h5['diff']:+.5f} {h5['ci']} met={h5['criterion_met_mesh_level']}; "
            f"H6 diff {h6['diff']:+.5f} {h6['ci']} met={h6['criterion_met_mesh_level']}")
    (out_path or OUT_DIR / "h5_h6.json").write_text(json.dumps(jsonable(out), indent=2))
    return out


def section_d(dfs, iou_rows_path: Path, mesh_df, a: str = "mast3r_slam", b: str = "endodac",
              out_path: Path | None = None) -> dict:
    """Paired differences a minus b (variables below keep the names of the
    original MASt3R-SLAM minus EndoDAC case: *_m is pipeline a, *_e is b)."""
    log(f"=== D: pairwise {a} minus {b} (descriptive) ===")
    rng = np.random.default_rng(SEED)
    reg_e, area_e = dfs[b]
    reg_m, area_m = dfs[a]
    iou = agg.add_clusters(pd.read_csv(iou_rows_path), mesh_df)
    iou = iou[iou.source == "pipeline"]
    out = {"seed": SEED, "n_boot": N_BOOT, "direction": f"{a} minus {b}", "cells": {}}
    for config, tau in IOU_CELLS:
        key = f"{config}|{tau}"
        cell = {}
        for metric, (num, den) in AREA_METRICS.items():
            A = area_m[(area_m.config == config) & (area_m.tau == tau)]
            B = area_e[(area_e.config == config) & (area_e.tau == tau)]
            cell[metric] = with_sensitivity(paired_ratio_diff, A, B, num, den, rng=rng)
        # region recall, medium + large, 50% threshold: same regions in both pipelines
        rA = size_filter(reg_m[(reg_m.config == config) & (reg_m.tau == tau)], "medium_plus_large").assign(one=1.0)
        rB = size_filter(reg_e[(reg_e.config == config) & (reg_e.tau == tau)], "medium_plus_large").assign(one=1.0)
        rA = rA.assign(det=rA["detected_at_0.5"].astype(float))
        rB = rB.assign(det=rB["detected_at_0.5"].astype(float))
        cell["recall_medium_plus_large_at_50pct"] = with_sensitivity(paired_ratio_diff, rA, rB, "det", "one", rng=rng)
        # region IoU, medium + large, per-region paired difference
        iA = size_filter(iou[(iou.pipeline == a) & (iou.config == config) & (iou.tau == tau)], "medium_plus_large")
        iB = size_filter(iou[(iou.pipeline == b) & (iou.config == config) & (iou.tau == tau)], "medium_plus_large")
        pr = iA.merge(iB[["sequence", "region_id", "iou"]], on=["sequence", "region_id"], suffixes=("", "_endodac"))
        if len(pr) != len(iA) or len(pr) != len(iB):
            raise RuntimeError(f"{key}: IoU pairing mismatch")
        pr["diff"] = pr["iou"] - pr["iou_endodac"]
        d = with_sensitivity(paired_mean_diff, pr, "diff", rng=rng)
        d.update({"A": float(pr["iou"].mean()), "B": float(pr["iou_endodac"].mean())})
        cell["region_iou_mean_medium_plus_large"] = d
        out["cells"][key] = cell
        log(f"{key}: FR {cell['false_reassurance_rate']['diff']:+.4f} {cell['false_reassurance_rate']['ci']}, "
            f"IoU {d['mean']:+.4f} {d['ci']}")
    (out_path or OUT_DIR / "pairwise_mast3r_minus_endodac.json").write_text(json.dumps(jsonable(out), indent=2))
    return out


def main():
    log(f"bootstrap seed {SEED}, replicates {N_BOOT}")
    mesh_df = pd.read_csv(REPO / "results/mesh_identity.csv")
    dfs, _ = section_a(mesh_df)
    section_b(mesh_df)
    section_c(dfs)
    section_d(dfs, IOU_DIR / "region_iou_rows.csv.gz", mesh_df)
    log("ALL DONE")


if __name__ == "__main__":
    main()
