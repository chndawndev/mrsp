#!/usr/bin/env python
"""Render the markdown tables for docs/pipeline2_eval.md from the JSON
outputs of scripts/pipeline2_aggregate.py. Pure formatting, no computation
beyond reading values.

CLI: scratch/.venv/bin/python scripts/pipeline2_report_tables.py [--root results/pipeline2_eval]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path("/data1_ycao/chua/projects/mrsp")
CONFIGS = ["oracle", "pred_depth_only", "pred_pose_only", "fully_predicted"]
TAUS = ["0.15", "0.25", "0.35", "0.5"]


def f(x, d=4):
    return "n/a" if x is None else f"{x:.{d}f}"


def ci(c, d=4):
    return f"[{f(c[0], d)}, {f(c[1], d)}]"


def corpus_tables(t: dict) -> str:
    T = t["tables"]
    out = ["### Corpus table, tau = 0.25 (point estimate, mesh-level 95% CI)", "",
           "| config | recall m+l @50% | area_fraction | calibration (raw / ignore-excl.) | false_reassurance | false_alarm | loc_err median mm (n) | segment_intersect |",
           "|---|---|---|---|---|---|---|---|"]
    for c in CONFIGS:
        x = T[c]["0.25"]
        r = x["region_recall"]["medium_plus_large"]["0.5"]
        out.append(
            f"| {c} | {f(r['point'])} {ci([r['ci_lo'], r['ci_hi']])} (n={r['n_regions']}, {r['n_meshes']} meshes) "
            f"| {f(x['area_fraction']['point'])} {ci([x['area_fraction']['ci_lo'], x['area_fraction']['ci_hi']])} "
            f"| {f(x['calibration_ratio']['point'], 3)} / {f(x['calibration_ratio_ignore_excluded']['point'], 3)} "
            f"| {f(x['false_reassurance_rate']['point'], 5)} {ci([x['false_reassurance_rate']['ci_lo'], x['false_reassurance_rate']['ci_hi']], 5)} "
            f"| {f(x['false_alarm_rate']['point'])} {ci([x['false_alarm_rate']['ci_lo'], x['false_alarm_rate']['ci_hi']])} "
            f"| {f(x['localization_error']['median'], 3)} (n={x['localization_error']['n_regions']}) "
            f"| {f(x['segment_intersect_fraction']['point'])} |")
    out += ["", "### All taus: recall (m+l, 50%), area fraction, false reassurance, false alarm (point estimates)", "",
            "| config | tau | recall m+l | recall small | recall large | area_fraction | false_reassurance | false_alarm |",
            "|---|---|---|---|---|---|---|---|"]
    for c in CONFIGS:
        for tau in TAUS:
            x = T[c][tau]
            rr = x["region_recall"]
            out.append(f"| {c} | {tau} | {f(rr['medium_plus_large']['0.5']['point'])} | {f(rr['small']['0.5']['point'])} "
                       f"| {f(rr['large']['0.5']['point'])} | {f(x['area_fraction']['point'])} "
                       f"| {f(x['false_reassurance_rate']['point'], 5)} | {f(x['false_alarm_rate']['point'])} |")
    out += ["", "### Sensitivity intervals, tau = 0.25: (Colon, Segment) and (Colon, Segment, Phantom Number)", "",
            "| config | metric | mesh CI | (Colon, Segment) CI | (Colon, Segment, Phantom) CI |", "|---|---|---|---|---|"]
    for c in CONFIGS:
        x = T[c]["0.25"]
        r = x["region_recall"]["medium_plus_large"]["0.5"]
        out.append(f"| {c} | recall m+l | {ci([r['ci_lo'], r['ci_hi']])} | {ci(r['sensitivity_colon_segment_ci'])} | {ci(r['sensitivity_colon_segment_phantom_ci'])} |")
        for m in ["false_reassurance_rate", "false_alarm_rate", "area_fraction"]:
            y = x[m]
            out.append(f"| {c} | {m} | {ci([y['ci_lo'], y['ci_hi']], 5)} | {ci(y['sensitivity_colon_segment_ci'], 5)} | {ci(y['sensitivity_colon_segment_phantom_ci'], 5)} |")
    out += ["", "### Diagnostic b (over 169 sequences: mean / median / max)", "",
            "| config | ray_miss_frac | d_pred_unavailable_frac_of_evaluable |", "|---|---|---|"]
    for c in CONFIGS:
        b = t["diagnostic_b"][c]
        out.append(f"| {c} | {f(b['ray_miss_frac']['mean'])} / {f(b['ray_miss_frac']['median'])} / {f(b['ray_miss_frac']['max'])} ({b['ray_miss_frac']['argmax_sequence']}) "
                   f"| {f(b['d_pred_unavailable_frac_of_evaluable']['mean'])} / {f(b['d_pred_unavailable_frac_of_evaluable']['median'])} / {f(b['d_pred_unavailable_frac_of_evaluable']['max'])} |")
    out += ["", "### Diagnostic c (pooled faces)", "",
            "| tau | only-unobserved under fully_predicted, not pred_pose_only | also unobserved under pred_depth_only | fraction |",
            "|---|---|---|---|"]
    for tau, d in t["diagnostic_c"].items():
        out.append(f"| {tau} | {d['n_only_fully_predicted_unobserved']:,} | {d['n_also_pred_depth_only_unobserved']:,} | {d['fraction']:.3f} |")
    return "\n".join(out)


def missing_table(t: dict) -> str:
    m = t["missing_frames"]
    out = ["| sequence | GT frames | frames with pose | missing pose | first missing-pose frame |", "|---|---|---|---|---|"]
    for r in m["affected_sequences"]:
        out.append(f"| {r['sequence']} | {r['n_gt_frames']} | {r['n_frames_with_pose']} | {r['n_frames_missing_pose']} | {r['first_missing_pose_frame']} |")
    return "\n".join(out)


def iou_tables(b: dict) -> str:
    out = []
    for pipeline in ["endodac", "mast3r_slam"]:
        out += [f"### {pipeline}: region IoU (medium + large) vs area-matched random baseline", "",
                "| config | tau | IoU mean [mesh CI] | IoU median | random mean, seed-avg (range over 20 seeds) | random median | paired diff (pipeline - random) [mesh CI] | (C,S) CI | (C,S,P) CI | false_alarm (random) | area_fraction | n regions / meshes |",
                "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for key, cell in b["cells"].items():
            p, c, tau = key.split("|")
            if p != pipeline:
                continue
            x = cell["by_size_class"]["medium_plus_large"]
            pp, rr, dd, cx = x["pipeline"], x["random_seed_averaged"], x["paired_diff_pipeline_minus_random_mean"], cell["context"]
            out.append(
                f"| {c} | {tau} | {f(pp['mean'])} {ci(pp['mean_ci'])} | {f(pp['median'])} | {f(rr['mean'])} ({f(rr['per_seed_mean_range'][0])}-{f(rr['per_seed_mean_range'][1])}) "
                f"| {f(rr['median'])} | {dd['mean']:+.4f} {ci(dd['ci'])} | {ci(dd['sensitivity_colon_segment_ci'])} | {ci(dd['sensitivity_colon_segment_phantom_ci'])} "
                f"| {f(cx['false_alarm_rate']['pipeline'])} ({f(cx['false_alarm_rate']['random_seed_mean'])}) | {f(cx['area_fraction']['pipeline'])} "
                f"| {pp['n_regions']} / {cell['n_meshes']} |")
        out.append("")
        out += [f"#### {pipeline}: by size class, fully_predicted tau = 0.25", "",
                "| size class | IoU mean | random mean | paired diff [mesh CI] | n |", "|---|---|---|---|---|"]
        cell = b["cells"][f"{pipeline}|fully_predicted|0.25"]
        for sc in ["small", "medium", "large", "medium_plus_large"]:
            x = cell["by_size_class"][sc]
            dd = x["paired_diff_pipeline_minus_random_mean"]
            out.append(f"| {sc} | {f(x['pipeline']['mean'])} | {f(x['random_seed_averaged']['mean'])} | {dd['mean']:+.4f} {ci(dd['ci'])} | {x['pipeline']['n_regions']} |")
        out.append("")
    return "\n".join(out)


def h_tables(h: dict) -> str:
    out = []
    for name, a, b_, direction in [("H5", "fully_predicted", "pred_pose_only", "false reassurance, A - B < 0"),
                                   ("H6", "pred_depth_only", "pred_pose_only", "false alarm, A - B > 0")]:
        out += [f"**{name}** ({direction}; A = {a}, B = {b_})", "",
                "| tau | A | B | A - B | mesh CI | (C,S) CI | (C,S,P) CI | criterion met (mesh CI) |", "|---|---|---|---|---|---|---|---|"]
        for tau, x in h[name].items():
            out.append(f"| {tau} | {f(x['A'], 5)} | {f(x['B'], 5)} | {x['diff']:+.5f} | {ci(x['ci'], 5)} | {ci(x['sensitivity_colon_segment_ci'], 5)} "
                       f"| {ci(x['sensitivity_colon_segment_phantom_ci'], 5)} | {'yes' if x['criterion_met_mesh_level'] else 'no'} |")
        out.append("")
    return "\n".join(out)


def pairwise_table(d: dict) -> str:
    out = ["| config | tau | metric | MASt3R | EndoDAC | MASt3R - EndoDAC | mesh CI | (C,S) CI | (C,S,P) CI |",
           "|---|---|---|---|---|---|---|---|---|"]
    for key, cell in d["cells"].items():
        c, tau = key.split("|")
        for m, x in cell.items():
            diff = x.get("diff", x.get("mean"))
            out.append(f"| {c} | {tau} | {m} | {f(x['A'], 5)} | {f(x['B'], 5)} | {diff:+.5f} | {ci(x['ci'], 5)} "
                       f"| {ci(x['sensitivity_colon_segment_ci'], 5)} | {ci(x['sensitivity_colon_segment_phantom_ci'], 5)} |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(REPO / "results/pipeline2_eval"))
    args = ap.parse_args()
    root = Path(args.root)
    parts = []
    if (root / "mast3r_corpus_tables.json").exists():
        t = json.loads((root / "mast3r_corpus_tables.json").read_text())
        parts += ["## CORPUS", corpus_tables(t), "", "## MISSING", missing_table(t)]
    parts += ["## IOU", iou_tables(json.loads((root / "region_iou_summary.json").read_text())),
              "## H5H6", h_tables(json.loads((root / "h5_h6.json").read_text())),
              "## PAIRWISE", pairwise_table(json.loads((root / "pairwise_mast3r_minus_endodac.json").read_text()))]
    text = "\n\n".join(parts)
    (root / "report_tables.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
