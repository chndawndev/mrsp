#!/usr/bin/env python
"""Render the markdown tables of docs/d2b_eval.md from the JSON outputs of
scripts/d2b_aggregate.py. Formatting only.

CLI: scratch/.venv/bin/python scripts/d2b_report_tables.py > results/d2b_eval/report_tables.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path("/data1_ycao/chua/projects/mrsp")
sys.path.insert(0, str(REPO / "scripts"))
import pipeline2_report_tables as rt  # noqa: E402

OUT = REPO / "results/d2b_eval"
PIPELINES = ["endodac", "mast3r_slam", "cut3r"]
NAMES = {"endodac": "EndoDAC", "mast3r_slam": "MASt3R-SLAM", "cut3r": "CUT3R"}
f, ci = rt.f, rt.ci


def load(name):
    return json.loads((OUT / name).read_text())


def summary_table(iou: dict, tables: dict) -> str:
    L = ["| pipeline | config | tau | false reassurance | false alarm | area fraction | recall m+l @50% | region IoU mean | matched random IoU | IoU - random [mesh CI] | (C,S) CI |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in PIPELINES:
        for key, cell in iou["cells"].items():
            q, c, tau = key.split("|")
            if q != p:
                continue
            t = tables[p][c][tau]
            x = cell["by_size_class"]["medium_plus_large"]
            d = x["paired_diff_pipeline_minus_random_mean"]
            L.append(f"| {NAMES[p]} | {c} | {tau} | {f(t['false_reassurance_rate']['point'], 5)} | {f(t['false_alarm_rate']['point'])} "
                     f"| {f(t['area_fraction']['point'])} | {f(t['region_recall']['medium_plus_large']['0.5']['point'])} "
                     f"| {f(x['pipeline']['mean'])} | {f(x['random_seed_averaged']['mean'])} | {d['mean']:+.4f} {ci(d['ci'])} "
                     f"| {ci(d['sensitivity_colon_segment_ci'])} |")
    return "\n".join(L)


def d2b_tables(d: dict) -> str:
    L = ["| tau | FR EndoDAC | FR MASt3R-SLAM | FR CUT3R | FR best / worst | IoU EndoDAC | IoU MASt3R-SLAM | IoU CUT3R | IoU best / worst |",
         "|---|---|---|---|---|---|---|---|---|"]
    for tau, x in d["by_tau"].items():
        fr, io = x["false_reassurance"], x["region_iou_mean"]
        L.append(f"| {tau} | " + " | ".join(f(fr[p], 5) for p in PIPELINES)
                 + f" | {NAMES[x['false_reassurance_best']]} / {NAMES[x['false_reassurance_worst']]} | "
                 + " | ".join(f(io[p]) for p in PIPELINES)
                 + f" | {NAMES[x['region_iou_best']]} / {NAMES[x['region_iou_worst']]} |")
    L += ["", "| criterion | best | worst | best value | worst value | abs difference | threshold | mesh CI of the paired difference | (C,S) CI | (C,S,P) CI | difference >= threshold | mesh CI excludes zero | verdict |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k in ["D2b.1", "D2b.2"]:
        x = d[k]
        L.append(f"| {k} ({x['endpoint']}) | {NAMES[x['best']]} | {NAMES[x['worst']]} | {f(x['best_value'], 5)} | {f(x['worst_value'], 5)} "
                 f"| {f(x['abs_difference'], 5)} | {x['threshold']} | {ci(x['mesh_ci'], 5)} | {ci(x['colon_segment_ci'], 5)} "
                 f"| {ci(x['colon_segment_phantom_ci'], 5)} | {'yes' if x['difference_meets_threshold'] else 'no'} "
                 f"| {'yes' if x['mesh_ci_excludes_zero'] else 'no'} | **{x['verdict']}** |")
    L.append("")
    for k in ["D2b.1", "D2b.2"]:
        x = d[k]
        if "pooled_run_to_run_range" in x:
            rr = ", ".join(f"{NAMES[p]} {v:.5f}" for p, v in x["pooled_run_to_run_range"].items())
            L.append(f"- {k}: |difference| {x['abs_difference']:.5f}; pooled run-to-run range (six runs, 15-sequence subset): {rr}; "
                     f"difference smaller than the range of either pipeline: "
                     f"{'YES' if x['difference_smaller_than_run_to_run_range_of_either'] else 'no'}; "
                     f"(C,S) CI excludes zero: {'yes' if x['colon_segment_ci_excludes_zero'] else 'no'}; "
                     f"(C,S,P) CI excludes zero: {'yes' if x['colon_segment_phantom_ci_excludes_zero'] else 'no'}")
    return "\n".join(L)


def variability_tables(v: dict) -> str:
    mets = [("false_reassurance", 5), ("false_alarm", 5), ("area_fraction", 5), ("region_iou_mean", 5), ("random_iou_mean", 5)]
    L = ["### Pooled over the 15-sequence subset, per run", "",
         "| pipeline | run | false reassurance | false alarm | area fraction | region IoU mean (m+l) | matched random IoU | n regions |",
         "|---|---|---|---|---|---|---|---|"]
    for p in PIPELINES:
        for run, x in v["pipelines"][p]["per_run_pooled"].items():
            L.append(f"| {NAMES[p]} | {run} | " + " | ".join(f(x[m], d) for m, d in mets) + f" | {x['n_regions_medium_plus_large']} |")
    L += ["", "### Pooled: SD and range across the six runs (primary + seeds 1 to 5)", "",
          "| pipeline | metric | SD | range | min | max |", "|---|---|---|---|---|---|"]
    for p in PIPELINES:
        for m, d in mets[:4]:
            s = v["pipelines"][p]["pooled_spread_six_runs"][m]
            L.append(f"| {NAMES[p]} | {m} | {s['sd']:.2e} | {s['range']:.2e} | {f(s['min'], 5)} | {f(s['max'], 5)} |")
    L += ["", "### Per sequence: range (SD) across the six runs", "",
          "| sequence | pipeline | primary FR | FR range (SD) | primary FA | FA range (SD) | primary area | area range (SD) | primary IoU (n regions) | IoU range (SD) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for s_ in v["subset"]:
        for p in PIPELINES:
            sp = v["pipelines"][p]["per_sequence_spread_six_runs"][s_]
            pr = v["pipelines"][p]["per_run_per_sequence"]["primary"][s_]
            cells = []
            for m in ["false_reassurance", "false_alarm", "area_fraction"]:
                cells += [f(pr[m], 5), f"{sp[m]['range']:.2e} ({sp[m]['sd']:.2e})"]
            if pr["region_iou_mean"] is None or sp["region_iou_mean"] is None:
                cells += ["no m+l region", "n/a"]
            else:
                cells += [f"{f(pr['region_iou_mean'])} ({pr['n_regions_medium_plus_large']})",
                          f"{sp['region_iou_mean']['range']:.2e} ({sp['region_iou_mean']['sd']:.2e})"]
            L.append(f"| {s_} | {NAMES[p]} | " + " | ".join(cells) + " |")
    return "\n".join(L)


def main():
    cut = load("cut3r_corpus_tables.json")
    iou = load("region_iou_summary.json")
    tables = {"cut3r": cut["tables"],
              "mast3r_slam": json.loads((REPO / "results/pipeline2_eval/mast3r_corpus_tables.json").read_text())["tables"],
              "endodac": json.loads((REPO / "results/d1/corpus_tables.json").read_text())["tables"]}
    parts = ["## CORPUS (CUT3R)", rt.corpus_tables(cut), "## IOU", rt.iou_tables(iou, PIPELINES),
             "## BASELINE STATEMENTS", rt.baseline_statements(iou), "## SUMMARY", summary_table(iou, tables),
             "## H5H6 (CUT3R)", rt.h_tables(load("h5_h6_cut3r.json"))]
    for a, b in [("cut3r", "endodac"), ("cut3r", "mast3r_slam"), ("mast3r_slam", "endodac")]:
        parts += [f"## PAIRWISE {a} minus {b}", rt.pairwise_table(load(f"pairwise_{a}_minus_{b}.json"))
                  .replace("| MASt3R | EndoDAC | MASt3R - EndoDAC |", f"| {NAMES[a]} | {NAMES[b]} | {NAMES[a]} - {NAMES[b]} |")]
    parts += ["## D2B", d2b_tables(load("d2b.json"))]
    if (OUT / "variability_summary.json").exists():
        parts += ["## VARIABILITY", variability_tables(load("variability_summary.json"))]
    print("\n\n".join(parts))


if __name__ == "__main__":
    main()
