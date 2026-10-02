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
        out.append(f"| {r['sequence']} | {r['n_gt_frames']} | {r['n_frames_with_pose']} | {r['n_frames_missing_pose']} | {int(r['first_missing_pose_frame'])} |")
    return "\n".join(out)


def iou_tables(b: dict, pipelines=("endodac", "mast3r_slam")) -> str:
    out = []
    for pipeline in pipelines:
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


def sign_of(c) -> str:
    return "above" if c[0] > 0 else "below" if c[1] < 0 else "includes zero"


def baseline_statements(b: dict) -> str:
    """One line per (pipeline, cell): position of the pipeline's region IoU
    relative to its area-matched random baseline at the mesh level AND at
    the (Colon, Segment) level. A direction is stated only when both
    intervals exclude zero on the same side; otherwise the disagreement
    (or the absence of a direction) is stated."""
    out = ["| pipeline | config | tau | paired diff | mesh CI (103 meshes) | (Colon, Segment) CI (15 molds) | statement |",
           "|---|---|---|---|---|---|---|"]
    for key, cell in b["cells"].items():
        p, c, tau = key.split("|")
        d = cell["by_size_class"]["medium_plus_large"]["paired_diff_pipeline_minus_random_mean"]
        m, cs = sign_of(d["ci"]), sign_of(d["sensitivity_colon_segment_ci"])
        if m == cs and m != "includes zero":
            stmt = f"{m} its baseline at both levels"
        elif m == cs:
            stmt = "not distinguishable from its baseline at either level"
        elif m != "includes zero" and cs == "includes zero":
            stmt = (f"levels disagree: mesh CI is {m} zero, (Colon, Segment) CI includes zero; "
                    "no direction declared")
        elif m == "includes zero":
            stmt = (f"levels disagree: mesh CI includes zero, (Colon, Segment) CI is {cs} zero; "
                    "no direction declared")
        else:
            stmt = f"levels disagree: mesh CI {m} zero, (Colon, Segment) CI {cs} zero; no direction declared"
        out.append(f"| {p} | {c} | {tau} | {d['mean']:+.4f} | {ci(d['ci'])} | {ci(d['sensitivity_colon_segment_ci'])} | {stmt} |")
    return "\n".join(out)


def h_tables(h: dict) -> str:
    """H5/H6 per docs/success_criteria.md section 6, 2026-10-01
    clarification: judged at the primary tau = 0.25 on the mesh-level CI;
    the other taus are reported alongside."""
    out = []
    for name, a, b_, direction, want in [("H5", "fully_predicted", "pred_pose_only", "false reassurance, A - B < 0", "below"),
                                         ("H6", "pred_depth_only", "pred_pose_only", "false alarm, A - B > 0", "above")]:
        out += [f"**{name}** ({direction}; A = {a}, B = {b_})", "",
                "| tau | role | A | B | A - B | mesh CI | (C,S) CI | (C,S,P) CI | mesh CI excludes zero in the stated direction | (C,S) CI does |",
                "|---|---|---|---|---|---|---|---|---|---|"]
        for tau, x in h[name].items():
            role = "**primary**" if float(tau) == 0.25 else "alongside"
            cs = "yes" if sign_of(x["sensitivity_colon_segment_ci"]) == want else "no"
            out.append(f"| {tau} | {role} | {f(x['A'], 5)} | {f(x['B'], 5)} | {x['diff']:+.5f} | {ci(x['ci'], 5)} | {ci(x['sensitivity_colon_segment_ci'], 5)} "
                       f"| {ci(x['sensitivity_colon_segment_phantom_ci'], 5)} | {'yes' if x['criterion_met_mesh_level'] else 'no'} | {cs} |")
        out.append("")
    return "\n".join(out)


def old_new_table(root: Path, old_root: Path) -> str:
    """Record of the grid-mapping correction: corner-aligned (old, archived)
    next to pixel-center aligned (new) values. Formatting only."""
    def load(r, n):
        return json.loads((r / n).read_text())
    out = ["| quantity | old (corner aligned) | new (center aligned) | new - old |", "|---|---|---|---|"]

    def row(label, o, n, d=5):
        out.append(f"| {label} | {f(o, d)} | {f(n, d)} | {n - o:+.{d}f} |")

    to, tn = load(old_root, "mast3r_corpus_tables.json"), load(root, "mast3r_corpus_tables.json")
    for c in CONFIGS:
        for tau in TAUS:
            for m in ["false_reassurance_rate", "false_alarm_rate", "area_fraction"]:
                row(f"{c} tau {tau} {m}", to["tables"][c][tau][m]["point"], tn["tables"][c][tau][m]["point"])
            row(f"{c} tau {tau} recall m+l @50%", to["tables"][c][tau]["region_recall"]["medium_plus_large"]["0.5"]["point"],
                tn["tables"][c][tau]["region_recall"]["medium_plus_large"]["0.5"]["point"])
    bo, bn = load(old_root, "region_iou_summary.json"), load(root, "region_iou_summary.json")
    for key in bn["cells"]:
        if not key.startswith("mast3r_slam"):
            continue
        xo, xn = (b["cells"][key]["by_size_class"]["medium_plus_large"] for b in (bo, bn))
        row(f"{key} IoU mean", xo["pipeline"]["mean"], xn["pipeline"]["mean"])
        row(f"{key} random mean", xo["random_seed_averaged"]["mean"], xn["random_seed_averaged"]["mean"])
        row(f"{key} paired diff", xo["paired_diff_pipeline_minus_random_mean"]["mean"], xn["paired_diff_pipeline_minus_random_mean"]["mean"])
    ho, hn = load(old_root, "h5_h6.json"), load(root, "h5_h6.json")
    for name in ["H5", "H6"]:
        for tau in TAUS:
            row(f"{name} tau {tau} A - B", ho[name][tau]["diff"], hn[name][tau]["diff"])
            out.append(f"| {name} tau {tau} mesh CI | {ci(ho[name][tau]['ci'], 5)} | {ci(hn[name][tau]['ci'], 5)} | |")
    mo, mn = to["missing_frames"], tn["missing_frames"]
    row("valid pixels with no depth (fraction)", mo["no_depth_frac_of_valid_pixels_all_gt_frames_pooled"],
        mn["no_depth_frac_of_valid_pixels_all_gt_frames_pooled"])
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
    ap.add_argument("--old-root", default=None,
                    help="archived aggregation root to tabulate against (grid-mapping correction record)")
    args = ap.parse_args()
    root = Path(args.root)
    parts = []
    if (root / "mast3r_corpus_tables.json").exists():
        t = json.loads((root / "mast3r_corpus_tables.json").read_text())
        parts += ["## CORPUS", corpus_tables(t), "", "## MISSING", missing_table(t)]
    iou_summary = json.loads((root / "region_iou_summary.json").read_text())
    parts += ["## IOU", iou_tables(iou_summary), "## BASELINE STATEMENTS", baseline_statements(iou_summary),
              "## H5H6", h_tables(json.loads((root / "h5_h6.json").read_text())),
              "## PAIRWISE", pairwise_table(json.loads((root / "pairwise_mast3r_minus_endodac.json").read_text()))]
    if args.old_root:
        parts += ["## OLD VS NEW", old_new_table(root, Path(args.old_root))]
    text = "\n\n".join(parts)
    (root / "report_tables.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
