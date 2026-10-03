#!/usr/bin/env python
"""D2b evaluation: aggregation over three pipelines (EndoDAC,
MASt3R-SLAM, CUT3R). CPU only; reads saved per-sequence evaluation
outputs. docs/d2b_eval.md.

  A. CUT3R corpus tables (scripts/eval_d1_aggregate.py::build_corpus_tables
     via scripts/pipeline2_aggregate.py::corpus_section), diagnostics,
     missing frames, invalid-depth counts.
  B. Region IoU with area-matched random baselines, three pipelines.
  C. H5 and H6 on CUT3R.
  D. Pairwise differences for the three pairs.
  E. D2b.1, D2b.2 (mechanical, mesh-level CI as the verdict basis) and
     D2b.3 (record only).
  F. Variability runs (docs/eval_protocol.md, 2026-10-02, items 3 and 4).

Aggregation rule (docs/eval_protocol.md, 2026-09-23): pooled point
estimates; mesh-level cluster bootstrap, 10,000 replicates, 95% percentile
interval, seed 20260925; (Colon, Segment) and (Colon, Segment, Phantom
Number) intervals alongside. All verdicts are judged on the primary runs.

CLI: scratch/.venv/bin/python scripts/d2b_aggregate.py [--skip-variability]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/data1_ycao/chua/projects/mrsp")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import eval_d1_aggregate as agg  # noqa: E402
import pipeline2_aggregate as p2  # noqa: E402
from cut3r_run_corpus import variability_subset  # noqa: E402

OUT = REPO / "results/d2b_eval"
IOU_DIR = OUT / "region_iou"
VAR_EVAL = OUT / "variability"
VAR_IOU_DIR = OUT / "variability_region_iou"
PIPELINES = ["endodac", "mast3r_slam", "cut3r"]
p2.ROOTS["cut3r"] = OUT / "per_sequence/cut3r"
SEED, TAUS = p2.SEED, p2.TAUS
PRIMARY_CONFIG, PRIMARY_TAU = "fully_predicted", 0.25
D2B1_THRESHOLD, D2B2_THRESHOLD = 0.01, 0.10  # docs/success_criteria.md lines 412-416
RUNS = ["primary", "seed1", "seed2", "seed3", "seed4", "seed5"]
log = p2.log


def ci_excludes_zero(c) -> bool:
    return bool(c[0] > 0 or c[1] < 0)


# -------------------------------------------------------------------------- E. D2b

def d2b(dfs, iou: pd.DataFrame) -> dict:
    rng = np.random.default_rng(SEED)
    num, den = p2.AREA_METRICS["false_reassurance_rate"]
    out = {"seed": SEED, "n_boot": p2.N_BOOT, "config": PRIMARY_CONFIG, "tau": PRIMARY_TAU, "by_tau": {}}

    def fr_points(tau):
        return {p: (lambda a: float(a[num].sum() / a[den].sum()))(
            dfs[p][1][(dfs[p][1].config == PRIMARY_CONFIG) & (dfs[p][1].tau == tau)]) for p in PIPELINES}

    def iou_points(tau):
        r = p2.size_filter(iou[(iou.source == "pipeline") & (iou.config == PRIMARY_CONFIG) & (iou.tau == tau)], "medium_plus_large")
        return {p: float(r[r.pipeline == p]["iou"].mean()) for p in PIPELINES}

    for tau in TAUS:
        fr, io = fr_points(tau), iou_points(tau)
        out["by_tau"][str(tau)] = {
            "false_reassurance": fr, "false_reassurance_best": min(fr, key=fr.get), "false_reassurance_worst": max(fr, key=fr.get),
            "region_iou_mean": io, "region_iou_best": max(io, key=io.get), "region_iou_worst": min(io, key=io.get),
        }
    prim = out["by_tau"][str(PRIMARY_TAU)]

    # D2b.1: worst minus best pooled false reassurance (lower is better), paired by sequence
    best, worst = prim["false_reassurance_best"], prim["false_reassurance_worst"]
    A = dfs[worst][1][(dfs[worst][1].config == PRIMARY_CONFIG) & (dfs[worst][1].tau == PRIMARY_TAU)]
    B = dfs[best][1][(dfs[best][1].config == PRIMARY_CONFIG) & (dfs[best][1].tau == PRIMARY_TAU)]
    r = p2.with_sensitivity(p2.paired_ratio_diff, A, B, num, den, rng=rng)
    out["D2b.1"] = {
        "endpoint": "pooled false reassurance", "best": best, "worst": worst,
        "best_value": r["B"], "worst_value": r["A"], "abs_difference": abs(r["diff"]),
        "paired_difference_worst_minus_best": r["diff"], "mesh_ci": r["ci"],
        "colon_segment_ci": r["sensitivity_colon_segment_ci"],
        "colon_segment_phantom_ci": r["sensitivity_colon_segment_phantom_ci"],
        "n_sequences": r["n_sequences"], "n_meshes": r["n_clusters"],
        "threshold": D2B1_THRESHOLD, "difference_meets_threshold": bool(abs(r["diff"]) >= D2B1_THRESHOLD),
        "mesh_ci_excludes_zero": ci_excludes_zero(r["ci"]),
        "colon_segment_ci_excludes_zero": ci_excludes_zero(r["sensitivity_colon_segment_ci"]),
        "colon_segment_phantom_ci_excludes_zero": ci_excludes_zero(r["sensitivity_colon_segment_phantom_ci"]),
    }
    out["D2b.1"]["verdict"] = "PASS" if (out["D2b.1"]["difference_meets_threshold"] and out["D2b.1"]["mesh_ci_excludes_zero"]) else "FAIL"

    # D2b.2: best minus worst mean region IoU (medium + large), paired per region
    best, worst = prim["region_iou_best"], prim["region_iou_worst"]
    rows = p2.size_filter(iou[(iou.source == "pipeline") & (iou.config == PRIMARY_CONFIG) & (iou.tau == PRIMARY_TAU)], "medium_plus_large")
    iA, iB = rows[rows.pipeline == best], rows[rows.pipeline == worst]
    pr = iA.merge(iB[["sequence", "region_id", "iou"]], on=["sequence", "region_id"], suffixes=("", "_worst"))
    if len(pr) != len(iA) or len(pr) != len(iB):
        raise RuntimeError("D2b.2: region pairing mismatch")
    pr["diff"] = pr["iou"] - pr["iou_worst"]
    r = p2.with_sensitivity(p2.paired_mean_diff, pr, "diff", rng=rng)
    out["D2b.2"] = {
        "endpoint": "mean region IoU, medium + large", "best": best, "worst": worst,
        "best_value": float(pr["iou"].mean()), "worst_value": float(pr["iou_worst"].mean()),
        "abs_difference": abs(r["mean"]), "paired_difference_best_minus_worst": r["mean"], "mesh_ci": r["ci"],
        "colon_segment_ci": r["sensitivity_colon_segment_ci"],
        "colon_segment_phantom_ci": r["sensitivity_colon_segment_phantom_ci"],
        "n_regions": r["n_regions"], "n_meshes": r["n_clusters"], "n_sequences": int(pr["sequence"].nunique()),
        "threshold": D2B2_THRESHOLD, "difference_meets_threshold": bool(abs(r["mean"]) >= D2B2_THRESHOLD),
        "mesh_ci_excludes_zero": ci_excludes_zero(r["ci"]),
        "colon_segment_ci_excludes_zero": ci_excludes_zero(r["sensitivity_colon_segment_ci"]),
        "colon_segment_phantom_ci_excludes_zero": ci_excludes_zero(r["sensitivity_colon_segment_phantom_ci"]),
    }
    out["D2b.2"]["verdict"] = "PASS" if (out["D2b.2"]["difference_meets_threshold"] and out["D2b.2"]["mesh_ci_excludes_zero"]) else "FAIL"

    # D2b.3 (record only)
    rec = {}
    for ep, bk, wk in [("false_reassurance", "false_reassurance_best", "false_reassurance_worst"),
                       ("region_iou", "region_iou_best", "region_iou_worst")]:
        tops = {t: out["by_tau"][t][bk] for t in out["by_tau"]}
        bottoms = {t: out["by_tau"][t][wk] for t in out["by_tau"]}
        rec[ep] = {"top_by_tau": tops, "bottom_by_tau": bottoms,
                   "top_same_at_every_tau": len(set(tops.values())) == 1,
                   "bottom_same_at_every_tau": len(set(bottoms.values())) == 1,
                   "top_and_bottom_do_not_swap": len(set(tops.values())) == 1 and len(set(bottoms.values())) == 1}
    out["D2b.3_record_only"] = rec
    return out


# -------------------------------------------------------------------------- F. variability

def seq_cell(metrics_path: Path) -> dict:
    m = json.loads(metrics_path.read_text())
    t = m["configurations"][PRIMARY_CONFIG]["by_tau"][str(PRIMARY_TAU)]
    return {"fr_num": t["false_reassurance_numerator_area_mm2"], "fr_den": t["gt_unobserved_area_mm2"],
            "fa_num": t["false_alarm_numerator_area_mm2"], "fa_den": t["false_alarm_denominator_area_mm2"],
            "af_num": t["pred_unobserved_area_mm2"], "af_den": t["total_mesh_area_mm2"],
            # D1 Stage B's EndoDAC metrics.json (the EndoDAC primary run) predates the adapter fields
            "n_invalid_depth_pixels": m.get("missing_predictions", {}).get("adapter", {}).get("n_invalid_depth_pixels")}


def variability(primary_iou: pd.DataFrame) -> dict:
    subset = variability_subset()
    var_iou = pd.read_csv(VAR_IOU_DIR / "region_iou_rows.csv.gz")
    var_iou = var_iou[(var_iou.config == PRIMARY_CONFIG) & (var_iou.tau == PRIMARY_TAU)
                      & var_iou.size_class.isin(["medium", "large"])]
    prim_iou = primary_iou[(primary_iou.config == PRIMARY_CONFIG) & (primary_iou.tau == PRIMARY_TAU)
                           & primary_iou.size_class.isin(["medium", "large"]) & primary_iou.sequence.isin(subset)]
    primary_roots = {"endodac": REPO / "results/d1/per_sequence",
                     "mast3r_slam": REPO / "results/pipeline2_eval/per_sequence/mast3r_slam",
                     "cut3r": OUT / "per_sequence/cut3r"}
    metrics = ["false_reassurance", "false_alarm", "area_fraction", "region_iou_mean", "random_iou_mean"]
    out = {"subset": subset, "config": PRIMARY_CONFIG, "tau": PRIMARY_TAU, "runs": RUNS, "pipelines": {}}
    for p in PIPELINES:
        runs = RUNS + (["rerun_unperturbed"] if p == "mast3r_slam" else [])
        per_run_seq, pooled = {}, {}
        for run in runs:
            if run == "primary":
                root, rows = primary_roots[p], prim_iou[prim_iou.pipeline == p]
            else:
                root, rows = VAR_EVAL / p / run, var_iou[var_iou.pipeline == f"{p}__{run}"]
            cells = {s: seq_cell(root / s / "metrics.json") for s in subset}
            pipe, rand = rows[rows.source == "pipeline"], rows[rows.source == "random"]
            if sorted(pipe.sequence.unique()) != sorted(s for s in subset if s in set(prim_iou.sequence)):
                raise RuntimeError(f"{p}/{run}: region IoU rows do not cover the subset")
            seq_rows = {}
            for s in subset:
                c = cells[s]
                seq_rows[s] = {
                    "false_reassurance": c["fr_num"] / c["fr_den"], "false_alarm": c["fa_num"] / c["fa_den"],
                    "area_fraction": c["af_num"] / c["af_den"],
                    "region_iou_mean": float(pipe[pipe.sequence == s]["iou"].mean()) if (pipe.sequence == s).any() else None,
                    "random_iou_mean": float(rand[rand.sequence == s]["iou"].mean()) if (rand.sequence == s).any() else None,
                    "n_regions_medium_plus_large": int((pipe.sequence == s).sum()),
                    "n_invalid_depth_pixels": c["n_invalid_depth_pixels"],
                }
            per_run_seq[run] = seq_rows
            pooled[run] = {
                "false_reassurance": sum(c["fr_num"] for c in cells.values()) / sum(c["fr_den"] for c in cells.values()),
                "false_alarm": sum(c["fa_num"] for c in cells.values()) / sum(c["fa_den"] for c in cells.values()),
                "area_fraction": sum(c["af_num"] for c in cells.values()) / sum(c["af_den"] for c in cells.values()),
                "region_iou_mean": float(pipe["iou"].mean()), "random_iou_mean": float(rand["iou"].mean()),
                "n_regions_medium_plus_large": int(len(pipe)),
            }

        def spread(vals):
            v = np.array([x for x in vals if x is not None], dtype=float)
            if len(v) != len(RUNS):
                return None
            return {"sd": float(v.std(ddof=1)), "range": float(v.max() - v.min()), "min": float(v.min()), "max": float(v.max())}

        out["pipelines"][p] = {
            "per_run_pooled": pooled, "per_run_per_sequence": per_run_seq,
            "pooled_spread_six_runs": {m: spread([pooled[r][m] for r in RUNS]) for m in metrics},
            "per_sequence_spread_six_runs": {
                s: {m: spread([per_run_seq[r][s][m] for r in RUNS]) for m in metrics} for s in subset},
        }
        ps = out["pipelines"][p]["pooled_spread_six_runs"]
        log(f"variability {p}: pooled range FR {ps['false_reassurance']['range']:.5f} FA {ps['false_alarm']['range']:.5f} "
            f"area {ps['area_fraction']['range']:.5f} IoU {ps['region_iou_mean']['range']:.5f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-variability", action="store_true")
    args = ap.parse_args()
    log(f"bootstrap seed {SEED}, replicates {p2.N_BOOT}")
    mesh_df = pd.read_csv(REPO / "results/mesh_identity.csv")

    dfs = {}
    for p in ("endodac", "mast3r_slam"):
        _, reg, area = p2.load_pipeline(p, mesh_df)
        dfs[p] = (reg, area)
    _, reg_c, area_c = p2.corpus_section("cut3r", mesh_df, None, OUT, "cut3r")
    dfs["cut3r"] = (reg_c, area_c)

    # the EndoDAC and MASt3R-SLAM region IoU rows recomputed here must equal the pipeline 2 evaluation's
    iou_raw = pd.read_csv(IOU_DIR / "region_iou_rows.csv.gz")
    old = pd.read_csv(REPO / "results/pipeline2_eval/region_iou/region_iou_rows.csv.gz")
    key = ["pipeline", "sequence", "config", "tau", "source", "seed", "region_id"]
    j = old.merge(iou_raw, on=key, suffixes=("_old", "_new"), how="left")
    check = {"n_old_rows": int(len(old)), "n_matched": int(j["iou_new"].notna().sum()),
             "max_abs_diff": float((j["iou_old"] - j["iou_new"]).abs().max())}
    log(f"region IoU rows vs pipeline 2 evaluation (EndoDAC, MASt3R-SLAM): {check}")
    if check["n_matched"] != check["n_old_rows"] or check["max_abs_diff"] != 0.0:
        raise SystemExit("recomputed EndoDAC / MASt3R-SLAM region IoU rows differ from the pipeline 2 evaluation")

    p2.section_b(mesh_df, pipelines=PIPELINES, iou_dir=IOU_DIR, out_path=OUT / "region_iou_summary.json")
    p2.section_c(dfs, "cut3r", OUT / "h5_h6_cut3r.json")
    for a, b in [("cut3r", "endodac"), ("cut3r", "mast3r_slam"), ("mast3r_slam", "endodac")]:
        p2.section_d(dfs, IOU_DIR / "region_iou_rows.csv.gz", mesh_df, a, b, OUT / f"pairwise_{a}_minus_{b}.json")

    iou = agg.add_clusters(iou_raw, mesh_df)
    res = d2b(dfs, iou)
    res["region_iou_rows_check_vs_pipeline2_eval"] = check
    log(f"D2b.1: {json.dumps({k: v for k, v in res['D2b.1'].items()})}")
    log(f"D2b.2: {json.dumps({k: v for k, v in res['D2b.2'].items()})}")
    log(f"D2b.3: {json.dumps(res['D2b.3_record_only'])}")

    if not args.skip_variability:
        var = variability(iou_raw)
        (OUT / "variability_summary.json").write_text(json.dumps(p2.jsonable(var), indent=2))
        # item 4 of the 2026-10-02 entry
        for name, metric in [("D2b.1", "false_reassurance"), ("D2b.2", "region_iou_mean")]:
            d = res[name]
            ranges = {p: var["pipelines"][p]["pooled_spread_six_runs"][metric]["range"] for p in (d["best"], d["worst"])}
            d["pooled_run_to_run_range"] = ranges
            d["difference_smaller_than_run_to_run_range_of_either"] = bool(d["abs_difference"] < max(ranges.values()))
            log(f"{name}: |difference| {d['abs_difference']:.5f}; pooled run-to-run ranges {ranges}")
    (OUT / "d2b.json").write_text(json.dumps(p2.jsonable(res), indent=2))
    log("ALL DONE")


if __name__ == "__main__":
    main()
