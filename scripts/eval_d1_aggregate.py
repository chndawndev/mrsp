#!/usr/bin/env python
"""D1 Stage B: corpus aggregation and D1.1-D1.5, per docs/eval_protocol.md's
"2026-09-23: Aggregation rule for corpus-level metrics" (line 338) and
docs/success_criteria.md sections 2 + 6. Reads every completed sequence's
results/d1/per_sequence/<seq>/{metrics.json,regions.csv}.

Point estimate: pooled over regions/sequences (region recall: detected/
total per class, equal weight per region; area-based metrics: summed
numerator area / summed denominator area across sequences -- NOT a mean
of per-sequence ratios, which the aggregation rule explicitly rules out).

Uncertainty: mesh-level cluster bootstrap (resample mesh_hash with
replacement, 10,000 replicates, 95% percentile interval), plus two
sensitivity clusterings -- (Colon, Segment) and (Colon, Segment, Phantom
Number) -- reported, not decision-driving. Seed fixed and logged.

Two explicit stops, per instructions (both checked after computing the
tables, since both need the full corpus):
  - segment_intersects_mesh fraction > 20% of matched pairs -> stop,
    report the fraction, do not implement geodesic distance this session.
  - D1.3 fails -> report it and stop, no further analysis this session.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/data1_ycao/chua/projects/mrsp")
PER_SEQ_ROOT = REPO / "results/d1/per_sequence"
MESH_IDENTITY_CSV = REPO / "results/mesh_identity.csv"
STAGE_A_CSV = REPO / "results/d1/stage_a_summary.csv"
OUT_DIR = REPO / "results/d1"
SEED = 20260925
N_BOOT = 10000
TAUS = [0.15, 0.25, 0.35, 0.50]
CONFIG_NAMES = ["oracle", "pred_depth_only", "pred_pose_only", "fully_predicted"]
SIZE_CLASSES = ["small", "medium", "large", "medium_plus_large", "below_headline"]
DETECTION_THRESHOLDS = [0.25, 0.50, 0.75]
D13_TRIGGER_PP = 10.0
SEGMENT_INTERSECT_TRIGGER = 0.20


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}", flush=True)


def discover_completed_sequences() -> tuple[list[str], list[dict]]:
    """Returns (ok_sequence_names, failure_records). Never silently drops
    a non-ok sequence -- callers report failure_records explicitly."""
    ok, failures = [], []
    for d in sorted(PER_SEQ_ROOT.iterdir()):
        if not d.is_dir():
            continue
        manifest_path = d / "MANIFEST.json"
        if not manifest_path.exists():
            failures.append({"sequence": d.name, "status": "no_manifest"})
            continue
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("status") == "ok":
            ok.append(d.name)
        else:
            failures.append({"sequence": d.name, "status": manifest.get("status"), "error": manifest.get("error")})
    return ok, failures


def load_all(sequences: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (regions_df, area_df). regions_df: one row per
    (sequence, config, tau, region). area_df: one row per
    (sequence, config, tau), the raw area components."""
    region_frames, area_rows = [], []
    for seq in sequences:
        d = PER_SEQ_ROOT / seq
        region_frames.append(pd.read_csv(d / "regions.csv"))
        metrics = json.loads((d / "metrics.json").read_text())
        for config_name, cdata in metrics["configurations"].items():
            for tau_str, t in cdata["by_tau"].items():
                area_rows.append({
                    "sequence": seq, "config": config_name, "tau": float(tau_str),
                    "pred_unobserved_area_mm2": t["pred_unobserved_area_mm2"],
                    "total_mesh_area_mm2": t["total_mesh_area_mm2"],
                    "gt_unobserved_area_mm2": t["gt_unobserved_area_mm2"],
                    "false_reassurance_numerator_area_mm2": t["false_reassurance_numerator_area_mm2"],
                    "false_alarm_numerator_area_mm2": t["false_alarm_numerator_area_mm2"],
                    "false_alarm_denominator_area_mm2": t["false_alarm_denominator_area_mm2"],
                    "pred_unobserved_area_ignore_excluded_mm2": t["pred_unobserved_area_ignore_excluded_mm2"],
                    "gt_unobserved_area_ignore_excluded_mm2": t["gt_unobserved_area_ignore_excluded_mm2"],
                    "d_pred_unavailable_frac_of_evaluable": cdata["d_pred_unavailable_frac_of_evaluable"],
                    "ray_miss_frac": cdata["ray_miss_frac"],
                })
    regions_df = pd.concat(region_frames, ignore_index=True)
    regions_df["size_class_effective"] = regions_df["size_class"]
    area_df = pd.DataFrame(area_rows)
    return regions_df, area_df


def add_clusters(df: pd.DataFrame, mesh_df: pd.DataFrame) -> pd.DataFrame:
    cov = mesh_df[["Video Name", "mesh_hash", "Colon", "Segment", "Phantom Number"]].rename(
        columns={"Video Name": "sequence"}
    )
    cov["colon_segment"] = cov["Colon"] + "_" + cov["Segment"]
    cov["colon_segment_phantom"] = cov["Colon"] + "_" + cov["Segment"] + "_" + cov["Phantom Number"]
    out = df.merge(cov[["sequence", "mesh_hash", "colon_segment", "colon_segment_phantom"]], on="sequence", how="left")
    if out["mesh_hash"].isna().any():
        missing = out.loc[out["mesh_hash"].isna(), "sequence"].unique().tolist()
        raise RuntimeError(f"sequences with no mesh_identity.csv match: {missing}")
    return out


# --------------------------------------------------------------------------
# vectorized cluster bootstrap for ratio-of-sums metrics
# --------------------------------------------------------------------------

def bootstrap_ratio(numer_by_cluster: np.ndarray, denom_by_cluster: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    n = len(numer_by_cluster)
    if n == 0 or denom_by_cluster.sum() == 0:
        return float("nan"), float("nan")
    idx = rng.integers(0, n, size=(N_BOOT, n))
    num_boot = numer_by_cluster[idx].sum(axis=1)
    den_boot = denom_by_cluster[idx].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio_boot = num_boot / den_boot
    ratio_boot = ratio_boot[np.isfinite(ratio_boot)]
    if len(ratio_boot) == 0:
        return float("nan"), float("nan")
    lo, hi = np.percentile(ratio_boot, [2.5, 97.5])
    return float(lo), float(hi)


def bootstrap_median(values_by_cluster: list[np.ndarray], rng: np.random.Generator) -> tuple[float, float]:
    """values_by_cluster: list (one per cluster) of 1-D arrays of individual
    values (e.g. per-region localization errors). Cluster bootstrap of a
    pooled median can't be a simple sum-of-numerator ratio, so this loops
    over replicates (still fast: N_BOOT iterations over small per-cluster
    arrays, not over individual pixels/faces)."""
    n = len(values_by_cluster)
    non_empty = [v for v in values_by_cluster if len(v) > 0]
    if n == 0 or not non_empty:
        return float("nan"), float("nan")
    medians = np.empty(N_BOOT)
    idx_matrix = rng.integers(0, n, size=(N_BOOT, n))
    for b in range(N_BOOT):
        pooled = np.concatenate([values_by_cluster[i] for i in idx_matrix[b] if len(values_by_cluster[i]) > 0])
        medians[b] = np.median(pooled) if len(pooled) else np.nan
    medians = medians[np.isfinite(medians)]
    if len(medians) == 0:
        return float("nan"), float("nan")
    lo, hi = np.percentile(medians, [2.5, 97.5])
    return float(lo), float(hi)


# --------------------------------------------------------------------------

def region_recall_cell(regions_df: pd.DataFrame, config: str, tau: float, size_class: str, threshold: float,
                        cluster_col: str, rng: np.random.Generator) -> dict:
    sub = regions_df[(regions_df["config"] == config) & (regions_df["tau"] == tau)]
    if size_class == "medium_plus_large":
        sub = sub[sub["size_class"].isin(["medium", "large"])]
    else:
        sub = sub[sub["size_class"] == size_class]
    det_col = f"detected_at_{threshold}"
    n_total = len(sub)
    n_detected = int(sub[det_col].sum())
    point = n_detected / n_total if n_total else float("nan")

    by_cluster = sub.groupby(cluster_col)[det_col].agg(["sum", "count"])
    lo, hi = bootstrap_ratio(by_cluster["sum"].to_numpy(dtype=float), by_cluster["count"].to_numpy(dtype=float), rng)
    return {
        "point": point, "ci_lo": lo, "ci_hi": hi,
        "n_regions": n_total, "n_detected": n_detected,
        "n_meshes": sub["mesh_hash"].nunique(),
    }


def area_metric_cell(area_df: pd.DataFrame, config: str, tau: float, numer_col: str, denom_col: str,
                      cluster_col: str, rng: np.random.Generator) -> dict:
    sub = area_df[(area_df["config"] == config) & (area_df["tau"] == tau)]
    numer_sum = float(sub[numer_col].sum())
    denom_sum = float(sub[denom_col].sum())
    point = numer_sum / denom_sum if denom_sum else float("nan")
    by_cluster = sub.groupby(cluster_col)[[numer_col, denom_col]].sum()
    lo, hi = bootstrap_ratio(by_cluster[numer_col].to_numpy(), by_cluster[denom_col].to_numpy(), rng)
    return {"point": point, "ci_lo": lo, "ci_hi": hi, "n_sequences": len(sub), "n_meshes": sub["mesh_hash"].nunique()}


def loc_error_cell(regions_df: pd.DataFrame, config: str, tau: float, cluster_col: str, rng: np.random.Generator) -> dict:
    sub = regions_df[(regions_df["config"] == config) & (regions_df["tau"] == tau) & regions_df["headline"]
                      & regions_df["localization_error_mm"].notna()]
    vals = sub["localization_error_mm"].to_numpy()
    if len(vals) == 0:
        return {"median": None, "iqr": None, "ci_lo": None, "ci_hi": None, "n_regions": 0, "n_meshes": 0}
    median = float(np.median(vals))
    q1, q3 = np.percentile(vals, [25, 75])
    by_cluster = [g["localization_error_mm"].to_numpy() for _, g in sub.groupby(cluster_col)]
    lo, hi = bootstrap_median(by_cluster, rng)
    return {
        "median": median, "iqr": float(q3 - q1), "ci_lo": lo, "ci_hi": hi,
        "n_regions": len(vals), "n_meshes": sub["mesh_hash"].nunique(),
    }


def segment_intersect_cell(regions_df: pd.DataFrame, config: str, tau: float, cluster_col: str, rng: np.random.Generator) -> dict:
    sub = regions_df[(regions_df["config"] == config) & (regions_df["tau"] == tau) & regions_df["headline"]
                      & regions_df["segment_intersects_mesh"].notna()]
    if len(sub) == 0:
        return {"point": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "n_pairs": 0}
    flag = sub["segment_intersects_mesh"].astype(bool)
    point = float(flag.mean())
    by_cluster = sub.assign(flag=flag.astype(int)).groupby(cluster_col)["flag"].agg(["sum", "count"])
    lo, hi = bootstrap_ratio(by_cluster["sum"].to_numpy(dtype=float), by_cluster["count"].to_numpy(dtype=float), rng)
    return {"point": point, "ci_lo": lo, "ci_hi": hi, "n_pairs": len(sub)}


def build_corpus_tables(regions_df: pd.DataFrame, area_df: pd.DataFrame, rng: np.random.Generator) -> dict:
    tables = {}
    for config in CONFIG_NAMES:
        tables[config] = {}
        for tau in TAUS:
            cell = {}
            cell["region_recall"] = {}
            for size_class in SIZE_CLASSES:
                cell["region_recall"][size_class] = {}
                for thresh in DETECTION_THRESHOLDS:
                    primary = region_recall_cell(regions_df, config, tau, size_class, thresh, "mesh_hash", rng)
                    sens1 = region_recall_cell(regions_df, config, tau, size_class, thresh, "colon_segment", rng)
                    sens2 = region_recall_cell(regions_df, config, tau, size_class, thresh, "colon_segment_phantom", rng)
                    primary["sensitivity_colon_segment_ci"] = [sens1["ci_lo"], sens1["ci_hi"]]
                    primary["sensitivity_colon_segment_phantom_ci"] = [sens2["ci_lo"], sens2["ci_hi"]]
                    cell["region_recall"][size_class][str(thresh)] = primary

            for metric, (numer, denom) in {
                "false_reassurance_rate": ("false_reassurance_numerator_area_mm2", "gt_unobserved_area_mm2"),
                "false_alarm_rate": ("false_alarm_numerator_area_mm2", "false_alarm_denominator_area_mm2"),
                "area_fraction": ("pred_unobserved_area_mm2", "total_mesh_area_mm2"),
                "calibration_ratio": ("pred_unobserved_area_mm2", "gt_unobserved_area_mm2"),
                "calibration_ratio_ignore_excluded": (
                    "pred_unobserved_area_ignore_excluded_mm2", "gt_unobserved_area_ignore_excluded_mm2"),
            }.items():
                primary = area_metric_cell(area_df, config, tau, numer, denom, "mesh_hash", rng)
                sens1 = area_metric_cell(area_df, config, tau, numer, denom, "colon_segment", rng)
                sens2 = area_metric_cell(area_df, config, tau, numer, denom, "colon_segment_phantom", rng)
                primary["sensitivity_colon_segment_ci"] = [sens1["ci_lo"], sens1["ci_hi"]]
                primary["sensitivity_colon_segment_phantom_ci"] = [sens2["ci_lo"], sens2["ci_hi"]]
                cell[metric] = primary

            loc_primary = loc_error_cell(regions_df, config, tau, "mesh_hash", rng)
            loc_sens1 = loc_error_cell(regions_df, config, tau, "colon_segment", rng)
            loc_sens2 = loc_error_cell(regions_df, config, tau, "colon_segment_phantom", rng)
            loc_primary["sensitivity_colon_segment_ci"] = [loc_sens1["ci_lo"], loc_sens1["ci_hi"]]
            loc_primary["sensitivity_colon_segment_phantom_ci"] = [loc_sens2["ci_lo"], loc_sens2["ci_hi"]]
            cell["localization_error"] = loc_primary

            seg_primary = segment_intersect_cell(regions_df, config, tau, "mesh_hash", rng)
            seg_sens1 = segment_intersect_cell(regions_df, config, tau, "colon_segment", rng)
            seg_sens2 = segment_intersect_cell(regions_df, config, tau, "colon_segment_phantom", rng)
            seg_primary["sensitivity_colon_segment_ci"] = [seg_sens1["ci_lo"], seg_sens1["ci_hi"]]
            seg_primary["sensitivity_colon_segment_phantom_ci"] = [seg_sens2["ci_lo"], seg_sens2["ci_hi"]]
            cell["segment_intersect_fraction"] = seg_primary

            tables[config][str(tau)] = cell
    return tables


def compute_d1(tables: dict, stage_a_d11: int, n_sequences_total: int) -> dict:
    d1 = {}
    # D1.1: from Stage A, reused here (see docs/success_criteria.md section 6, 2026-09-23 entry)
    d1["D1.1"] = {
        "value": f"{stage_a_d11}/{n_sequences_total}",
        "threshold": ">= 70% of 169 sequences",
        "result": "pass" if stage_a_d11 / n_sequences_total >= 0.70 else "fail",
        "note": "computed in Stage A per the docs/success_criteria.md section 6 2026-09-23 operational "
                "definition; for a frame-to-frame pipeline like EndoDAC this measures crash-free "
                "completion only, not track-loss detection (same entry's own caveat).",
    }
    # D1.2: oracle recall (medium+large) at primary tau=0.25, 50% threshold
    oracle_recall = tables["oracle"]["0.25"]["region_recall"]["medium_plus_large"]["0.5"]["point"]
    d1["D1.2"] = {
        "value": oracle_recall, "threshold": ">= 0.80",
        "result": "pass" if (oracle_recall == oracle_recall and oracle_recall >= 0.80) else "fail",
        "tau": 0.25,
    }
    # D1.3: gap oracle - fully_predicted (medium+large recall), same tau/threshold
    fp_recall = tables["fully_predicted"]["0.25"]["region_recall"]["medium_plus_large"]["0.5"]["point"]
    gap_pp = (oracle_recall - fp_recall) * 100
    d1["D1.3"] = {
        "value_pp": gap_pp, "threshold": f">= {D13_TRIGGER_PP} percentage points",
        "result": "pass" if (gap_pp == gap_pp and gap_pp >= D13_TRIGGER_PP) else "fail",
        "oracle_recall": oracle_recall, "fully_predicted_recall": fp_recall, "tau": 0.25,
    }
    # D1.4: small recall < large recall, gap >= 15pp (record-only per success_criteria.md)
    small_recall = tables["fully_predicted"]["0.25"]["region_recall"]["small"]["0.5"]["point"]
    large_recall = tables["fully_predicted"]["0.25"]["region_recall"]["large"]["0.5"]["point"]
    gap14 = (large_recall - small_recall) * 100
    d1["D1.4"] = {
        "small_recall": small_recall, "large_recall": large_recall, "gap_pp": gap14,
        "threshold": ">= 15 percentage points (record-only, does not block)",
        "result": "record_only_meets_gap" if (gap14 == gap14 and gap14 >= 15) else "record_only_does_not_meet_gap",
        "tau": 0.25,
    }
    # D1.5: false reassurance under fully_predicted, strictly between 0.02 and 0.98
    fr = tables["fully_predicted"]["0.25"]["false_reassurance_rate"]["point"]
    d1["D1.5"] = {
        "value": fr, "threshold": "strictly between 0.02 and 0.98",
        "result": "pass" if (fr == fr and 0.02 < fr < 0.98) else "fail",
        "tau": 0.25,
    }
    return d1


def main():
    log("=== D1 Stage B aggregation ===")
    rng = np.random.default_rng(SEED)
    log(f"bootstrap seed: {SEED}, replicates: {N_BOOT}")

    sequences, failures = discover_completed_sequences()
    log(f"{len(sequences)} sequences with status=ok, {len(failures)} not ok")
    if failures:
        log(f"NOT-OK sequences (reported, not dropped): {failures}")

    mesh_df = pd.read_csv(MESH_IDENTITY_CSV)
    regions_df, area_df = load_all(sequences)
    regions_df = add_clusters(regions_df, mesh_df)
    area_df = add_clusters(area_df, mesh_df)
    log(f"loaded {len(regions_df)} region rows, {len(area_df)} area rows")

    tables = build_corpus_tables(regions_df, area_df, rng)

    # ---------------- stop condition: segment-intersect trigger ----------------
    all_headline = regions_df[regions_df["headline"] & regions_df["segment_intersects_mesh"].notna()]
    seg_frac_overall = float(all_headline["segment_intersects_mesh"].astype(bool).mean()) if len(all_headline) else float("nan")
    log(f"corpus-wide segment_intersects_mesh fraction (all configs/taus pooled): {seg_frac_overall:.4f}")
    trigger_hit = seg_frac_overall == seg_frac_overall and seg_frac_overall > SEGMENT_INTERSECT_TRIGGER

    stage_a = pd.read_csv(STAGE_A_CSV)
    d11 = int(stage_a["d1_1_complete"].sum())
    d1_results = compute_d1(tables, d11, len(stage_a))
    log(f"D1.1: {d1_results['D1.1']}")
    log(f"D1.2: {d1_results['D1.2']}")
    log(f"D1.3: {d1_results['D1.3']}")
    log(f"D1.4: {d1_results['D1.4']}")
    log(f"D1.5: {d1_results['D1.5']}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    corpus_out = {
        "seed": SEED, "n_boot": N_BOOT, "n_sequences_included": len(sequences),
        "failures": failures, "tables": tables, "D1": d1_results,
        "segment_intersect_fraction_overall": seg_frac_overall,
        "segment_intersect_trigger_hit": bool(trigger_hit),
    }
    with open(OUT_DIR / "corpus_tables.json", "w") as f:
        json.dump(corpus_out, f, indent=2)
    log(f"wrote {OUT_DIR / 'corpus_tables.json'}")

    # ---------------- per-sequence metrics CSV, joined with Stage A covariates ----------------
    per_seq_rows = []
    for seq in sequences:
        metrics = json.loads((PER_SEQ_ROOT / seq / "metrics.json").read_text())
        for config_name, cdata in metrics["configurations"].items():
            for tau_str, t in cdata["by_tau"].items():
                per_seq_rows.append({
                    "sequence": seq, "config": config_name, "tau": float(tau_str),
                    "area_fraction": t["area_fraction"], "calibration_ratio": t["calibration_ratio"],
                    "false_reassurance_rate": t["false_reassurance_rate"], "false_alarm_rate": t["false_alarm_rate"],
                    "localization_error_median_mm": t["localization_error_median_mm"],
                    "segment_intersect_fraction": t["segment_intersect_fraction"],
                    "n_headline_regions_undetected": t["n_headline_regions_undetected"],
                })
    per_seq_df = pd.DataFrame(per_seq_rows).merge(stage_a, on="sequence", how="left")
    per_seq_df.to_csv(OUT_DIR / "per_sequence_metrics.csv", index=False)
    log(f"wrote {OUT_DIR / 'per_sequence_metrics.csv'} ({len(per_seq_df)} rows)")

    # ---------------- explicit stops ----------------
    if trigger_hit:
        log(f"*** STOP: segment_intersects_mesh fraction {seg_frac_overall:.4f} > {SEGMENT_INTERSECT_TRIGGER} "
            f"-- reporting only, NOT implementing geodesic distance this session ***")
        sys.exit(2)
    if d1_results["D1.3"]["result"] == "fail":
        log(f"*** STOP: D1.3 FAILED ({d1_results['D1.3']}) -- reporting only, no further analysis this session ***")
        sys.exit(3)

    log("ALL DONE")


if __name__ == "__main__":
    main()
