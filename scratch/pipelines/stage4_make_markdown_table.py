"""Renders results/eval_stage4/summary.json into a readable markdown
table, docs/eval_stage4.md -- one table per configuration, rows=metric,
columns=tau, plus a tau-sensitivity note per config. Pure formatting, no
recomputation.
"""
import json
from pathlib import Path

REPO = Path("/data1_ycao/chua/projects/mrsp")
d = json.loads((REPO / "results/eval_stage4/summary.json").read_text())

TAUS = ["0.15", "0.25", "0.35", "0.5"]
CONFIG_ORDER = ["oracle", "pred_depth_only", "pred_pose_only", "fully_predicted"]
CONFIG_LABEL = {
    "oracle": "Oracle (GT depth + GT pose)",
    "pred_depth_only": "Predicted depth + GT pose",
    "pred_pose_only": "GT depth + predicted pose",
    "fully_predicted": "Fully predicted",
}


def fmt(x, pct=False, nd=4):
    if x is None:
        return "—"
    try:
        if x != x:  # nan
            return "n/a"
    except TypeError:
        pass
    if pct:
        return f"{x*100:.{nd}f}%"
    return f"{x:.{nd}f}"


def sensitivity(values, near_zero_abs=None):
    vals = [v for v in values if v is not None and v == v]
    if len(vals) < 2:
        return "n/a"
    lo, hi = min(vals), max(vals)
    if hi == 0:
        return "flat (0)"
    # Near-zero absolute range: relative change can be huge (e.g. 0.0001->0)
    # while being practically meaningless noise, not a real tau effect --
    # classify by the absolute range instead of misleadingly as "sensitive".
    if near_zero_abs is not None and hi < near_zero_abs:
        return "flat (near zero)"
    rel = (hi - lo) / max(abs(hi), abs(lo), 1e-12)
    if rel < 0.02:
        return "flat"
    if rel < 0.25:
        return "mild"
    return "sensitive"


lines = []
lines.append("# Stage 4 full per-tau table\n")
lines.append(
    f"`{d['sequence']}`, {d['n_faces']} faces, {d['n_frames']} frames. "
    f"depth_scale.median={d['depth_scale_median']:.6f}, pose s_pose={d['pose_alignment_s_pose']:.6f}. "
    f"ignore_set = {d['ignore_set_frac_faces']*100:.4f}% of faces (sequence-level constant, "
    "repeated in every row below since it is required alongside every recall number). "
    f"GT regions: {d['n_gt_regions_total']} total, {d['n_gt_regions_headline']} headline (d>=5mm; "
    "this sequence has 2 small, 0 medium, 1 large).\n"
)
lines.append(
    "Pulled directly from `results/eval_stage4/summary.json`, no recomputation. "
    "**No D1 pass/fail commentary** -- one sequence is not D1.\n"
)
lines.append("---\n")

for cfg_name in CONFIG_ORDER:
    cfg = d["configurations"][cfg_name]
    lines.append(f"## {CONFIG_LABEL[cfg_name]} (`{cfg_name}`)\n")
    lines.append(
        f"ray_miss_frac={fmt(cfg['ray_miss_frac'], pct=True, nd=5)}, "
        f"evaluable_frac={fmt(cfg['evaluable_frac'], pct=True, nd=4)}, "
        f"d_pred_unavailable_count={cfg['d_pred_unavailable_count']}\n"
    )

    rows = [
        ("ignore_set_frac_faces", "%", lambda t: t["ignore_set_frac_faces"]),
        ("area_fraction", "%", lambda t: t["area_fraction"]),
        ("calibration_ratio", "x", lambda t: t["calibration_ratio"]),
        ("recall@50% small", "%", lambda t: t["recall_at_50pct"]["small"]["recall"]),
        ("recall@50% medium", "%", lambda t: t["recall_at_50pct"]["medium"]["recall"]),
        ("recall@50% large", "%", lambda t: t["recall_at_50pct"]["large"]["recall"]),
        ("recall@50% below_headline (appendix)", "%", lambda t: t["recall_at_50pct"]["below_headline"]["recall"]),
        ("false_reassurance_rate", "%_nz", lambda t: t["false_reassurance_rate"]),
        ("false_alarm_rate", "%_nz", lambda t: t["false_alarm_rate"]),
        ("localization_error median (mm)", "raw", lambda t: t["localization_error_median_mm"]),
        ("localization_error mean (mm)", "raw", lambda t: t["localization_error_mean_mm"]),
        ("n_headline_regions_undetected", "raw", lambda t: t["n_headline_regions_undetected"]),
        ("segment_intersect_fraction", "%", lambda t: t["segment_intersect_fraction"]),
        ("n_predicted_unobserved_faces", "raw", lambda t: t["n_predicted_unobserved_faces"]),
        ("tau_reject_count", "raw", lambda t: t["tau_reject_count"]),
    ]

    header = "| metric | " + " | ".join(f"tau={t}" for t in TAUS) + " | tau-sensitivity |"
    sep = "|---" * (len(TAUS) + 2) + "|"
    lines.append(header)
    lines.append(sep)
    for name, kind, getter in rows:
        values = [getter(cfg["by_tau"][t]) for t in TAUS]
        if kind in ("%", "%_nz"):
            cells = [fmt(v, pct=True, nd=4) for v in values]
        else:
            cells = [fmt(v) if isinstance(v, float) else str(v) for v in values]
        near_zero_abs = 0.001 if kind == "%_nz" else None  # 0.1 percentage points
        sens = sensitivity(values, near_zero_abs=near_zero_abs)
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {sens} |")

    # detection sweep 25/50/75 for headline classes, one small sub-table
    lines.append("\n**25/50/75% detection threshold sweep (small/medium/large), at each tau:**\n")
    lines.append("| tau | thresh | small | medium | large |")
    lines.append("|---|---|---|---|---|")
    for t in TAUS:
        sweep = cfg["by_tau"][t]["detection_sweep"]
        for thresh in ["0.25", "0.5", "0.75"]:
            s = sweep[thresh]
            lines.append(
                f"| {t} | {thresh} | {fmt(s['small']['recall'], pct=True, nd=2)} "
                f"({s['small']['n_detected']}/{s['small']['n_regions']}) | "
                f"{fmt(s['medium']['recall'], pct=True, nd=2)} ({s['medium']['n_detected']}/{s['medium']['n_regions']}) | "
                f"{fmt(s['large']['recall'], pct=True, nd=2)} ({s['large']['n_detected']}/{s['large']['n_regions']}) |"
            )
    lines.append("")

lines.append("---\n")
lines.append("## Tau-sensitivity summary, MEASURED\n")
lines.append(
    "- **Flat across the sweep, every configuration**: `area_fraction`/`calibration_ratio` for the oracle "
    "config only (predicted-depth configs move it substantially); `recall@50% large`; "
    "`segment_intersect_fraction` (0.00% everywhere, all 4 configs, all 4 tau -- see the separate "
    "geodesic-vs-Euclidean report); `n_headline_regions_undetected` (0 everywhere -- every headline "
    "region has *some* overlapping predicted component at every tau, in every configuration).\n"
)
lines.append(
    "- **Sensitive to tau**: `tau_reject_count` in every configuration (by construction -- a looser tau "
    "admits more pixels); `area_fraction`/`calibration_ratio`/`false_alarm_rate` for every configuration "
    "that includes predicted depth (`pred_depth_only`, `fully_predicted`) -- looser tau lets more predicted "
    "faces pass the depth-agreement test, shrinking the predicted-unobserved set and area fraction, and "
    "with it false alarm rate; `recall@50% small` for `pred_pose_only` (drops exactly between tau=0.15 and "
    "0.25) and `fully_predicted` (drops exactly between tau=0.25 and 0.35) -- see the separate small-region "
    "diagnosis for the face-level mechanism.\n"
)
lines.append(
    "- **INTERPRETATION, not confirmed further here**: `localization_error` mean is more tau-sensitive "
    "than its median in the predicted-depth configurations (e.g. `pred_depth_only`'s mean moves "
    "3.60mm->2.34mm across the sweep while its median barely moves, 1.14mm->1.10mm) -- consistent with a "
    "small number of large-error outlier region-matches whose *identity* (which predicted component gets "
    "matched to which GT region) shifts with tau, dragging the mean while the bulk of matches stay stable. "
    "Not traced to specific regions here; would need the same per-region breakdown as the small-region "
    "diagnosis to confirm.\n"
)

out_path = REPO / "docs/eval_stage4.md"
out_path.write_text("\n".join(lines) + "\n")
print(f"wrote {out_path}")
