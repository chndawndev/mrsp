#!/usr/bin/env python
"""POST-HOC DESCRIPTIVE SENSITIVITY ANALYSIS (not pre-registered, no
pass/fail): the pipeline 2 evaluation restricted to the sequences on
which MASt3R-SLAM predicted every frame (pose and depth), for BOTH
pipelines on exactly those sequences.

Re-aggregates saved outputs only (per-sequence metrics.json / regions.csv
of both pipelines and the region IoU rows of
scripts/pipeline2_region_iou.py). No ray casting, no metric code.

  1. Subset: sequences with n_frames_missing_either == 0 in MASt3R-SLAM's
     per-sequence metrics.json; meshes, trajectories, molds ((Colon,
     Segment) groups) and (Colon, Segment, Phantom Number) groups covered.
  2. Per pipeline, every configuration at tau = 0.25: false reassurance,
     false alarm, predicted-unobserved area fraction, region IoU (medium +
     large) next to the area-matched random baseline, on the subset.
  3. Paired MASt3R-SLAM minus EndoDAC differences on the subset.
  4. Stage A covariates (results/d1/stage_a_summary.csv) of the incomplete
     sequences next to the complete ones, descriptive only.

All-169 values are not recomputed here: they are read from the main
aggregation (scripts/pipeline2_aggregate.py outputs,
results/d1/corpus_tables.json), after checking that this script's own
all-169 point estimates equal them.

Bootstrap: scripts/pipeline2_aggregate.py's functions, mesh-level
clusters, 10,000 replicates, seed 20260925; (Colon, Segment) interval
alongside.

CLI: scratch/.venv/bin/python scripts/pipeline2_complete_sequence_sensitivity.py
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

SEED = agg.SEED
TAU = 0.25
CONFIG_NAMES = agg.CONFIG_NAMES
PIPELINES = ["endodac", "mast3r_slam"]
EXPECTED_N_COMPLETE = 143
EXPECTED_N_SEQUENCES = 169
EXPECTED_N_TRAJECTORIES = 113
STAGE_A_CSV = REPO / "results/d1/stage_a_summary.csv"
CLUSTERS = ["mesh_hash", "colon_segment"]
CATEGORICAL_COVARIATES = ["Colon", "Segment", "physical_segment_id", "Video Number", "Debris", "Edge Enhancement",
                          "Open End Visible"]
NUMERIC_COVARIATES = ["Total Frames", "Camera Speed", "Brightness", "gt_path_length_mm", "unobserved_area_total_mm2",
                      "n_unobserved_components", "Qualitative Score", "Quantitative Score",
                      # EndoDAC's own Stage A trajectory / depth-scale quantities (not MASt3R-SLAM's)
                      "ate_mm", "endpoint_error_frac_of_gt_path", "depth_scale_relative_iqr"]


def trajectory_id(df: pd.DataFrame) -> pd.Series:
    """docs/regions.md line 50: v3 shares v2's trajectory (README
    definition); v1 has its own."""
    vn = df["Video Number"].where(df["Video Number"] == "v1", "v2v3")
    return df["Colon"] + "_" + df["Segment"] + "_" + df["Phantom Number"] + "_" + vn


def coverage(mesh_df: pd.DataFrame, seqs: list[str]) -> dict:
    sub = mesh_df[mesh_df["Video Name"].isin(seqs)]
    if len(sub) != len(seqs):
        raise RuntimeError("sequence missing from mesh_identity.csv")
    return {
        "n_sequences": int(len(sub)),
        "n_meshes": int(sub["mesh_hash"].nunique()),
        "n_trajectories": int(trajectory_id(sub).nunique()),
        "n_molds_colon_segment": int((sub["Colon"] + "_" + sub["Segment"]).nunique()),
        "n_colon_segment_phantom": int((sub["Colon"] + "_" + sub["Segment"] + "_" + sub["Phantom Number"]).nunique()),
        "sequences_per_mold": (sub["Colon"] + "_" + sub["Segment"]).value_counts().sort_index().to_dict(),
    }


def ratio_cell(area_df: pd.DataFrame, config: str, num: str, den: str, rng) -> dict:
    out = {}
    for cl in CLUSTERS:
        c = agg.area_metric_cell(area_df, config, TAU, num, den, cl, rng)
        out.setdefault("point", c["point"])
        out[f"{cl}_ci"] = [c["ci_lo"], c["ci_hi"]]
        out[f"n_{cl}"] = int(area_df[(area_df.config == config) & (area_df.tau == TAU)][cl].nunique())
    return out


def iou_cell(rows: pd.DataFrame, pipeline: str, config: str, rng) -> dict:
    cell = rows[(rows.pipeline == pipeline) & (rows.config == config) & (rows.tau == TAU)]
    cell = p2.size_filter(cell, "medium_plus_large")
    pipe = cell[cell.source == "pipeline"]
    rand = cell[cell.source == "random"]
    rand_avg = rand.groupby(["sequence", "region_id"], as_index=False)["iou"].mean()
    paired = pipe.merge(rand_avg, on=["sequence", "region_id"], suffixes=("", "_rand"))
    if len(paired) != len(pipe):
        raise RuntimeError(f"{pipeline}/{config}: pairing lost rows")
    paired["diff"] = paired["iou"] - paired["iou_rand"]
    out = {"n_regions": int(len(pipe)), "n_meshes": int(pipe["mesh_hash"].nunique()),
           "n_colon_segment": int(pipe["colon_segment"].nunique()),
           "iou_mean": float(pipe["iou"].mean()), "random_mean": float(paired["iou_rand"].mean()),
           "random_per_seed_mean_range": [float(rand.groupby("seed")["iou"].mean().min()),
                                          float(rand.groupby("seed")["iou"].mean().max())]}
    for col, name in [("iou", "iou_mean"), ("diff", "paired_diff")]:
        for cl in CLUSTERS:
            r = p2.paired_mean_diff(paired, col, cl, rng)
            out.setdefault(name, r["mean"])
            out[f"{name}_{cl}_ci"] = r["ci"]
    return out


def pairwise_cell(dfs, iou: pd.DataFrame, config: str, rng) -> dict:
    (_, area_e), (_, area_m) = dfs["endodac"], dfs["mast3r_slam"]
    cell = {}
    for metric, (num, den) in p2.AREA_METRICS.items():
        A = area_m[(area_m.config == config) & (area_m.tau == TAU)]
        B = area_e[(area_e.config == config) & (area_e.tau == TAU)]
        r = {}
        for cl in CLUSTERS:
            x = p2.paired_ratio_diff(A, B, num, den, cl, rng)
            r.setdefault("A", x["A"]); r.setdefault("B", x["B"]); r.setdefault("diff", x["diff"])
            r[f"{cl}_ci"] = x["ci"]
        cell[metric] = r
    pipe = p2.size_filter(iou[(iou.source == "pipeline") & (iou.config == config) & (iou.tau == TAU)], "medium_plus_large")
    iA, iB = pipe[pipe.pipeline == "mast3r_slam"], pipe[pipe.pipeline == "endodac"]
    pr = iA.merge(iB[["sequence", "region_id", "iou"]], on=["sequence", "region_id"], suffixes=("", "_endodac"))
    if len(pr) != len(iA) or len(pr) != len(iB):
        raise RuntimeError(f"{config}: IoU pairing mismatch")
    pr["diff"] = pr["iou"] - pr["iou_endodac"]
    r = {"A": float(pr["iou"].mean()), "B": float(pr["iou_endodac"].mean()), "n_regions": int(len(pr))}
    for cl in CLUSTERS:
        x = p2.paired_mean_diff(pr, "diff", cl, rng)
        r.setdefault("diff", x["mean"])
        r[f"{cl}_ci"] = x["ci"]
    cell["region_iou_mean_medium_plus_large"] = r
    return cell


def analyse(dfs, iou: pd.DataFrame, seqs: list[str], seed_offset: int) -> dict:
    sub_dfs = {p: (reg[reg.sequence.isin(seqs)], area[area.sequence.isin(seqs)]) for p, (reg, area) in dfs.items()}
    sub_iou = iou[iou.sequence.isin(seqs)]
    rng = np.random.default_rng(SEED + seed_offset)
    out = {"per_pipeline": {}, "pairwise_mast3r_minus_endodac": {}}
    for p in PIPELINES:
        out["per_pipeline"][p] = {}
        for config in CONFIG_NAMES:
            cell = {m: ratio_cell(sub_dfs[p][1], config, num, den, rng) for m, (num, den) in p2.AREA_METRICS.items()}
            cell["region_iou"] = iou_cell(sub_iou, p, config, rng)
            out["per_pipeline"][p][config] = cell
    for config in CONFIG_NAMES:
        out["pairwise_mast3r_minus_endodac"][config] = pairwise_cell(sub_dfs, sub_iou, config, rng)
    return out


def check_against_main(all169: dict, root: Path) -> dict:
    """This script's all-169 point estimates must equal the main
    aggregation's. Returns the main aggregation's all-169 values (with its
    CIs) in this script's layout."""
    main_tables = {"mast3r_slam": json.loads((root / "mast3r_corpus_tables.json").read_text())["tables"],
                   "endodac": json.loads((REPO / "results/d1/corpus_tables.json").read_text())["tables"]}
    iou_main = json.loads((root / "region_iou_summary.json").read_text())["cells"]
    pair_main = json.loads((root / "pairwise_mast3r_minus_endodac.json").read_text())["cells"]
    worst = 0.0

    def chk(a, b):
        nonlocal worst
        worst = max(worst, abs(a - b))

    out = {"per_pipeline": {}, "pairwise_mast3r_minus_endodac": {}}
    for p in PIPELINES:
        out["per_pipeline"][p] = {}
        for config in CONFIG_NAMES:
            mine = all169["per_pipeline"][p][config]
            cell = {}
            for m in p2.AREA_METRICS:
                x = main_tables[p][config][str(TAU)][m]
                chk(mine[m]["point"], x["point"])
                cell[m] = {"point": x["point"], "mesh_hash_ci": [x["ci_lo"], x["ci_hi"]],
                           "colon_segment_ci": x["sensitivity_colon_segment_ci"]}
            x = iou_main[f"{p}|{config}|{TAU}"]["by_size_class"]["medium_plus_large"]
            d = x["paired_diff_pipeline_minus_random_mean"]
            chk(mine["region_iou"]["iou_mean"], x["pipeline"]["mean"])
            chk(mine["region_iou"]["random_mean"], x["random_seed_averaged"]["mean"])
            chk(mine["region_iou"]["paired_diff"], d["mean"])
            cell["region_iou"] = {
                "n_regions": x["pipeline"]["n_regions"], "iou_mean": x["pipeline"]["mean"],
                "iou_mean_mesh_hash_ci": x["pipeline"]["mean_ci"],
                "iou_mean_colon_segment_ci": x["pipeline"]["sensitivity_colon_segment_mean_ci"],
                "random_mean": x["random_seed_averaged"]["mean"], "paired_diff": d["mean"],
                "paired_diff_mesh_hash_ci": d["ci"], "paired_diff_colon_segment_ci": d["sensitivity_colon_segment_ci"]}
            out["per_pipeline"][p][config] = cell
    for config in CONFIG_NAMES:
        cell = {}
        for m, x in pair_main[f"{config}|{TAU}"].items():
            if m not in all169["pairwise_mast3r_minus_endodac"][config]:
                continue
            diff = x.get("diff", x.get("mean"))
            chk(all169["pairwise_mast3r_minus_endodac"][config][m]["diff"], diff)
            cell[m] = {"A": x["A"], "B": x["B"], "diff": diff, "mesh_hash_ci": x["ci"],
                       "colon_segment_ci": x["sensitivity_colon_segment_ci"]}
        out["pairwise_mast3r_minus_endodac"][config] = cell
    if worst > 1e-12:
        raise SystemExit(f"all-169 point estimates differ from the main aggregation by up to {worst}")
    out["max_abs_point_diff_vs_this_script"] = worst
    return out


def covariates(complete: list[str], incomplete: list[str], missing: pd.DataFrame) -> dict:
    sa = pd.read_csv(STAGE_A_CSV)
    if len(sa) != EXPECTED_N_SEQUENCES:
        raise RuntimeError(f"stage_a_summary.csv has {len(sa)} rows")
    sa = sa.merge(missing[["sequence", "ate_after_alignment_mm", "rotation_residual_median_deg"]].rename(
        columns={"ate_after_alignment_mm": "mast3r_ate_after_alignment_mm",
                 "rotation_residual_median_deg": "mast3r_rotation_residual_median_deg"}), on="sequence", validate="one_to_one")
    groups = {"complete": sa[sa.sequence.isin(complete)], "incomplete": sa[sa.sequence.isin(incomplete)]}
    out = {"categorical": {}, "numeric": {}}
    def norm(series: pd.Series) -> pd.Series:
        # the summary sheet has trailing blanks ("no " next to "no")
        return series.fillna("NaN").astype(str).str.strip()

    for col in CATEGORICAL_COVARIATES:
        levels = sorted(norm(sa[col]).unique())
        out["categorical"][col] = {lv: {g: int((norm(df[col]) == lv).sum()) for g, df in groups.items()} for lv in levels}

    # Tags is a ";"-separated multi-label field: count sequences carrying each tag
    def tag_set(v) -> set[str]:
        return {t.strip() for t in str(v).split(";") if t.strip()} if isinstance(v, str) else set()

    tags = sorted(set().union(*[tag_set(v) for v in sa["Tags"]]))
    out["categorical"]["Tags (sequences carrying the tag)"] = {
        t: {g: int(sum(t in tag_set(v) for v in df["Tags"])) for g, df in groups.items()} for t in tags}
    out["categorical"]["Tags (sequences carrying the tag)"]["(no tag)"] = {
        g: int(sum(not tag_set(v) for v in df["Tags"])) for g, df in groups.items()}
    for col in NUMERIC_COVARIATES + ["mast3r_ate_after_alignment_mm", "mast3r_rotation_residual_median_deg"]:
        out["numeric"][col] = {
            g: {"n": int(df[col].notna().sum()), "median": float(df[col].median()),
                "q25": float(df[col].quantile(0.25)), "q75": float(df[col].quantile(0.75)),
                "min": float(df[col].min()), "max": float(df[col].max())} for g, df in groups.items()}
    return out


# -------------------------------------------------------------------------- markdown

def f(x, d=4):
    return f"{x:.{d}f}"


def ci(c, d=4):
    return f"[{f(c[0], d)}, {f(c[1], d)}]"


def sci(c, d=4):
    return f"[{c[0]:+.{d}f}, {c[1]:+.{d}f}]"


def render(res: dict) -> str:
    n = res["subset"]["complete"]["n_sequences"]
    L = []
    for p in PIPELINES:
        L += [f"#### {p}: {n} complete sequences, tau = 0.25 (post-hoc descriptive)", "",
              "| config | false_reassurance [mesh CI] ((C,S) CI) | false_alarm [mesh CI] ((C,S) CI) | area_fraction [mesh CI] ((C,S) CI) | all-169: FR / FA / area |",
              "|---|---|---|---|---|"]
        for c in CONFIG_NAMES:
            x, a = res["complete"]["per_pipeline"][p][c], res["all_169_main_aggregation"]["per_pipeline"][p][c]
            cells = []
            for m, d in [("false_reassurance_rate", 5), ("false_alarm_rate", 4), ("area_fraction", 4)]:
                cells.append(f"{f(x[m]['point'], d)} {ci(x[m]['mesh_hash_ci'], d)} ({ci(x[m]['colon_segment_ci'], d)})")
            L.append(f"| {c} | " + " | ".join(cells) + f" | {f(a['false_reassurance_rate']['point'], 5)} / "
                     f"{f(a['false_alarm_rate']['point'])} / {f(a['area_fraction']['point'])} |")
        L += ["", f"#### {p}: region IoU (medium + large) vs area-matched random, {n} complete sequences, tau = 0.25 (post-hoc descriptive)", "",
              "| config | n regions / meshes / molds | IoU mean [mesh CI] | random mean (seed range) | paired diff [mesh CI] | (C,S) CI | all-169 paired diff [mesh CI] ((C,S) CI) |",
              "|---|---|---|---|---|---|---|"]
        for c in CONFIG_NAMES:
            x = res["complete"]["per_pipeline"][p][c]["region_iou"]
            a = res["all_169_main_aggregation"]["per_pipeline"][p][c]["region_iou"]
            L.append(f"| {c} | {x['n_regions']} / {x['n_meshes']} / {x['n_colon_segment']} | {f(x['iou_mean'])} {ci(x['iou_mean_mesh_hash_ci'])} "
                     f"| {f(x['random_mean'])} ({f(x['random_per_seed_mean_range'][0])}-{f(x['random_per_seed_mean_range'][1])}) "
                     f"| {x['paired_diff']:+.4f} {sci(x['paired_diff_mesh_hash_ci'])} | {sci(x['paired_diff_colon_segment_ci'])} "
                     f"| {a['paired_diff']:+.4f} {sci(a['paired_diff_mesh_hash_ci'])} ({sci(a['paired_diff_colon_segment_ci'])}) |")
        L.append("")
    L += [f"#### Paired MASt3R-SLAM minus EndoDAC, tau = 0.25: {n} complete sequences next to all 169 (post-hoc descriptive)", "",
          f"| config | metric | {n}: MASt3R | {n}: EndoDAC | {n}: diff [mesh CI] | {n}: (C,S) CI | 169: diff [mesh CI] | 169: (C,S) CI |",
          "|---|---|---|---|---|---|---|---|"]
    for c in CONFIG_NAMES:
        for m in ["false_reassurance_rate", "false_alarm_rate", "area_fraction", "region_iou_mean_medium_plus_large"]:
            x = res["complete"]["pairwise_mast3r_minus_endodac"][c][m]
            a = res["all_169_main_aggregation"]["pairwise_mast3r_minus_endodac"][c][m]
            L.append(f"| {c} | {m} | {f(x['A'], 5)} | {f(x['B'], 5)} | {x['diff']:+.5f} {sci(x['mesh_hash_ci'], 5)} "
                     f"| {sci(x['colon_segment_ci'], 5)} | {a['diff']:+.5f} {sci(a['mesh_hash_ci'], 5)} | {sci(a['colon_segment_ci'], 5)} |")
    cov = res["covariates"]
    L += ["", "#### Stage A covariates: incomplete vs complete sequences (descriptive only)", "",
          "| covariate | level | complete | incomplete |", "|---|---|---|---|"]
    nc, ni = res["subset"]["complete"]["n_sequences"], res["subset"]["incomplete"]["n_sequences"]
    for col, levels in cov["categorical"].items():
        for lv, g in levels.items():
            L.append(f"| {col} | {lv} | {g['complete']} ({g['complete'] / nc:.0%}) | {g['incomplete']} ({g['incomplete'] / ni:.0%}) |")
    L += ["", "| covariate | complete: median (q25-q75) [min, max] | incomplete: median (q25-q75) [min, max] |", "|---|---|---|"]
    for col, g in cov["numeric"].items():
        cells = [f"{x['median']:.4g} ({x['q25']:.4g}-{x['q75']:.4g}) [{x['min']:.4g}, {x['max']:.4g}]"
                 for x in (g["complete"], g["incomplete"])]
        L.append(f"| {col} | {cells[0]} | {cells[1]} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(REPO / "results/pipeline2_eval"),
                    help="pipeline 2 aggregation root (per_sequence/mast3r_slam, region_iou/, aggregation JSONs)")
    args = ap.parse_args()
    root = Path(args.root)
    p2.ROOTS["mast3r_slam"] = root / "per_sequence/mast3r_slam"
    out_dir = root / "complete_sequence_sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)
    p2.log(f"POST-HOC DESCRIPTIVE SENSITIVITY ANALYSIS; root {root}; bootstrap seed {SEED}, replicates {agg.N_BOOT}")

    mesh_df = pd.read_csv(REPO / "results/mesh_identity.csv")
    if trajectory_id(mesh_df).nunique() != EXPECTED_N_TRAJECTORIES:
        raise RuntimeError(f"trajectory definition gives {trajectory_id(mesh_df).nunique()}, expected {EXPECTED_N_TRAJECTORIES}")
    dfs = {}
    for p in PIPELINES:
        seqs, reg, area = p2.load_pipeline(p, mesh_df)
        dfs[p] = (reg, area)
    all_seqs = sorted(seqs)

    missing = pd.read_csv(root / "mast3r_missing_frames.csv")
    if sorted(missing["sequence"]) != all_seqs:
        raise RuntimeError("mast3r_missing_frames.csv does not list the 169 sequences")
    complete = sorted(missing.loc[missing["n_frames_missing_either"] == 0, "sequence"])
    incomplete = sorted(missing.loc[missing["n_frames_missing_either"] > 0, "sequence"])
    if len(complete) != EXPECTED_N_COMPLETE or len(complete) + len(incomplete) != EXPECTED_N_SEQUENCES:
        raise RuntimeError(f"{len(complete)} complete / {len(incomplete)} incomplete sequences, expected "
                           f"{EXPECTED_N_COMPLETE} / {EXPECTED_N_SEQUENCES - EXPECTED_N_COMPLETE}")

    iou = agg.add_clusters(pd.read_csv(root / "region_iou/region_iou_rows.csv.gz"), mesh_df)
    iou = iou[iou.tau == TAU]

    cov_c, cov_i, cov_all = coverage(mesh_df, complete), coverage(mesh_df, incomplete), coverage(mesh_df, all_seqs)
    m_c = set(mesh_df[mesh_df["Video Name"].isin(complete)]["mesh_hash"])
    m_i = set(mesh_df[mesh_df["Video Name"].isin(incomplete)]["mesh_hash"])
    res = {
        "label": "post-hoc descriptive sensitivity analysis; not pre-registered; no pass/fail",
        "seed": SEED, "n_boot": agg.N_BOOT, "tau": TAU, "root": str(root),
        "subset": {"complete": cov_c, "incomplete": cov_i, "all": cov_all,
                   "n_meshes_only_in_complete": len(m_c - m_i), "n_meshes_only_in_incomplete": len(m_i - m_c),
                   "n_meshes_in_both": len(m_c & m_i),
                   "complete_sequences": complete, "incomplete_sequences": incomplete},
    }
    p2.log(f"complete: {[(k, v) for k, v in cov_c.items() if k != 'sequences_per_mold']}")
    p2.log(f"incomplete: {[(k, v) for k, v in cov_i.items() if k != 'sequences_per_mold']}")

    res["complete"] = analyse(dfs, iou, complete, seed_offset=0)
    mine_all = analyse(dfs, iou, all_seqs, seed_offset=1)
    res["all_169_main_aggregation"] = check_against_main(mine_all, root)
    p2.log(f"all-169 point estimates vs main aggregation: max abs diff "
           f"{res['all_169_main_aggregation']['max_abs_point_diff_vs_this_script']:.2e}")
    res["covariates"] = covariates(complete, incomplete, missing)

    (out_dir / "complete_sequence_sensitivity.json").write_text(json.dumps(p2.jsonable(res), indent=2))
    md = render(res)
    (out_dir / "tables.md").write_text(md)
    print(md)
    p2.log("ALL DONE")


if __name__ == "__main__":
    main()
