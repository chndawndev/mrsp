#!/usr/bin/env python
"""D1.3 diagnosis, step 1 (post-hoc, descriptive): CPU-only diagnostics
separating D1.3's two pre-registered candidate causes (docs/success_
criteria.md section 2's fail branch) -- the detection definition is
insensitive, or the regions are too easy. NOT a new pass/fail verdict,
NOT a replacement metric: D1's verdict stays FAIL regardless of anything
computed here.

Reuses results/d1/per_sequence/ (Stage B, scripts/eval_d1_sequence.py),
results/d1/diagnosis/ignore_set/ (step 0, scripts/diagnose_d1_3_ignore_set.py),
and results/mesh_identity.csv. Recomputes only what those outputs don't
contain (mesh geometry, adjacency, GT regions, predicted components) via
the exact locked src/eval functions Stage B already used -- no new metric
logic, no edits to src/eval/, src/gt/, or the frozen docs.

Imports scripts/eval_d1_aggregate.py as a module to reuse its already-
validated bootstrap and aggregation helpers (bootstrap_ratio,
bootstrap_median, region_recall_cell, add_clusters, discover_completed_
sequences, load_all) rather than re-deriving that math.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path("/data1_ycao/chua/projects/mrsp")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from geometry.coverage_mesh import stream_coverage_mesh_from_zip  # noqa: E402
from geometry.mesh_stats import face_adjacency, face_areas  # noqa: E402
from eval.regions import compute_regions  # noqa: E402
from eval.region_metrics import (  # noqa: E402
    build_face_to_component_id,
    detection_at_threshold,
    false_alarm_rate,
    false_reassurance_rate,
    localization_error,
    matching_predicted_component,
)

import eval_d1_aggregate as agg  # noqa: E402

DATASET_ROOT = Path("/data1_ycao/chua/datasets/C3VDv2/registered_videos")
PER_SEQ_ROOT = REPO / "results/d1/per_sequence"
IGNORE_SET_DIR = REPO / "results/d1/diagnosis/ignore_set"
MESH_IDENTITY_CSV = REPO / "results/mesh_identity.csv"
OUT_DIR = REPO / "results/d1/diagnosis"
SEED = 20260925
N_BOOT = 10000
N_RANDOM_SEEDS = 20
CONFIG_NAMES = ["oracle", "pred_depth_only", "pred_pose_only", "fully_predicted"]
TAUS = [0.15, 0.25, 0.35, 0.50]
DETECTION_THRESHOLDS = [0.25, 0.50, 0.75]
PRIMARY_TAU = 0.25


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}", flush=True)


def discover_sequences_with_ignore_set() -> list[str]:
    """Sequences with both a complete Stage B manifest AND a complete
    ignore_set manifest -- never silently proceeds on a partial pair."""
    ok_stage_b, failures_stage_b = agg.discover_completed_sequences()
    ok = []
    missing_ignore_set = []
    for name in ok_stage_b:
        manifest_path = IGNORE_SET_DIR / f"{name}.manifest.json"
        if not manifest_path.exists():
            missing_ignore_set.append(name)
            continue
        m = json.loads(manifest_path.read_text())
        if m.get("status") == "ok":
            ok.append(name)
        else:
            missing_ignore_set.append(name)
    if failures_stage_b:
        log(f"Stage B non-ok sequences excluded: {failures_stage_b}")
    if missing_ignore_set:
        log(f"missing/failed ignore_set sequences excluded: {missing_ignore_set}")
    return ok


def load_sequence_geometry(name: str):
    zpath = DATASET_ROOT / f"{name}.zip"
    mesh_data = stream_coverage_mesh_from_zip(zpath)
    areas = face_areas(mesh_data.vertices, mesh_data.faces)
    adjacency = face_adjacency(mesh_data.vertices, mesh_data.faces)
    gt_observed = mesh_data.face_observed
    gt_unobserved = ~gt_observed
    n_faces = len(mesh_data.faces)
    gt_regions = compute_regions(n_faces, adjacency, gt_unobserved, areas)
    return mesh_data, areas, adjacency, gt_observed, gt_unobserved, gt_regions, n_faces


def load_ignore_set(name: str, n_faces: int) -> np.ndarray:
    packed = np.fromfile(IGNORE_SET_DIR / f"{name}.bin", dtype=np.uint8)
    return np.unpackbits(packed, bitorder="little")[:n_faces].astype(bool)


def load_predicted_observed(name: str, n_faces: int) -> np.ndarray:
    n_bytes = (n_faces + 7) // 8
    packed = np.fromfile(PER_SEQ_ROOT / name / "predicted_observed_packed.bin", dtype=np.uint8)
    return packed.reshape(len(CONFIG_NAMES), len(TAUS), n_bytes)


def region_rows_for_mask(predicted_unobserved, gt_regions, gt_observed, gt_unobserved, ignore_set, areas,
                          adjacency, mesh_data, n_faces) -> tuple[list[dict], float, float]:
    """One evaluation of the locked region_metrics functions against a
    given predicted_unobserved mask. Returns (per-region rows,
    false_reassurance_rate, false_alarm_rate)."""
    predicted_observed = ~predicted_unobserved
    predicted_components = compute_regions(n_faces, adjacency, predicted_unobserved & ~ignore_set, areas)
    face_to_component_id = build_face_to_component_id(n_faces, predicted_components)
    f_reassur = false_reassurance_rate(gt_unobserved, predicted_observed, areas)
    f_alarm = false_alarm_rate(predicted_unobserved, gt_observed, ignore_set, areas)

    det = {t: detection_at_threshold(gt_regions, predicted_unobserved, areas, t) for t in DETECTION_THRESHOLDS}
    rows = []
    for region_id, r in enumerate(gt_regions):
        is_headline = r.size_class != "below_headline"
        row = {
            "region_id": region_id, "size_class": r.size_class, "headline": is_headline,
            "region_area_mm2": r.area,
            "coverage_fraction": det[0.25].coverage[region_id],
            "detected_at_0.25": det[0.25].detected[region_id],
            "detected_at_0.5": det[0.50].detected[region_id],
            "detected_at_0.75": det[0.75].detected[region_id],
            "localization_error_mm": None,
            "matched_component_area_mm2": None,
        }
        if is_headline:
            err = localization_error(r, predicted_components, face_to_component_id, mesh_data.vertices, mesh_data.faces, areas)
            if err is not None:
                row["localization_error_mm"] = err
                comp = matching_predicted_component(r, predicted_components, face_to_component_id)
                row["matched_component_area_mm2"] = float(areas[comp.faces].sum())
        rows.append(row)
    return rows, f_reassur, f_alarm


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng_root = np.random.default_rng(SEED)

    sequences = discover_sequences_with_ignore_set()
    log(f"{len(sequences)} sequences with both Stage B and ignore_set complete")

    mesh_df = pd.read_csv(MESH_IDENTITY_CSV)

    # ---------------- Stage B outputs, reused as-is (diagnostics 2 & 3) ----------------
    regions_df, area_df = agg.load_all(sequences)
    regions_df = agg.add_clusters(regions_df, mesh_df)
    area_df = agg.add_clusters(area_df, mesh_df)
    log(f"loaded {len(regions_df)} Stage B region rows, {len(area_df)} area rows")

    # ==================== Diagnostic 2: recall sensitivity table ====================
    log("=== Diagnostic 2: recall sensitivity table ===")
    rng = np.random.default_rng(SEED)
    d2_rows = []
    size_classes = ["small", "medium", "large", "medium_plus_large"]
    for config in CONFIG_NAMES:
        for tau in TAUS:
            for size_class in size_classes:
                for thresh in DETECTION_THRESHOLDS:
                    cell = agg.region_recall_cell(regions_df, config, tau, size_class, thresh, "mesh_hash", rng)
                    d2_rows.append({
                        "config": config, "tau": tau, "size_class": size_class, "threshold": thresh,
                        "recall": cell["point"], "ci_lo": cell["ci_lo"], "ci_hi": cell["ci_hi"],
                        "n_regions": cell["n_regions"], "n_detected": cell["n_detected"], "n_meshes": cell["n_meshes"],
                    })
    d2 = pd.DataFrame(d2_rows)
    oracle_recall = d2[d2["config"] == "oracle"].set_index(["tau", "size_class", "threshold"])["recall"]
    d2["oracle_minus_this_pp"] = d2.apply(
        lambda row: (oracle_recall.get((row["tau"], row["size_class"], row["threshold"]), np.nan) - row["recall"]) * 100,
        axis=1,
    )
    d2.to_csv(OUT_DIR / "recall_sensitivity.csv", index=False)
    log(f"wrote {OUT_DIR / 'recall_sensitivity.csv'} ({len(d2)} rows)")

    # ==================== Diagnostic 3: region difficulty ====================
    log("=== Diagnostic 3: region difficulty ===")
    d3_rows = []
    sub3 = regions_df[(regions_df["tau"] == PRIMARY_TAU) & (regions_df["size_class"] != "below_headline")]
    for config in CONFIG_NAMES:
        for size_class in ["small", "medium", "large"]:
            vals = sub3[(sub3["config"] == config) & (sub3["size_class"] == size_class)]["coverage_fraction"].to_numpy()
            if len(vals) == 0:
                continue
            q = np.percentile(vals, [0, 25, 50, 75, 100])
            near_threshold = float(np.mean((vals >= 0.40) & (vals <= 0.60)))
            d3_rows.append({
                "config": config, "size_class": size_class, "n_regions": len(vals),
                "min": q[0], "q25": q[1], "median": q[2], "q75": q[3], "max": q[4],
                "frac_near_50pct_threshold": near_threshold,
            })
    d3 = pd.DataFrame(d3_rows)
    d3.to_csv(OUT_DIR / "region_difficulty.csv", index=False)
    with open(OUT_DIR / "region_difficulty_summary.json", "w") as f:
        json.dump(d3_rows, f, indent=2)
    log(f"wrote {OUT_DIR / 'region_difficulty.csv'} ({len(d3)} rows)")

    # ==================== Diagnostics 1 & 4: per-sequence face-level work ====================
    log("=== Diagnostics 1 & 4: per-sequence geometry pass ===")
    baseline_rows = []  # diagnostic 1
    merging_rows = []  # diagnostic 4
    t0 = time.time()
    for i, name in enumerate(sequences):
        mesh_data, areas, adjacency, gt_observed, gt_unobserved, gt_regions, n_faces = load_sequence_geometry(name)
        ignore_set = load_ignore_set(name, n_faces)
        packed = load_predicted_observed(name, n_faces)

        # ---------------- diagnostic 1: location-blind baselines, tau=0.25 ----------------
        seq_metrics = json.loads((PER_SEQ_ROOT / name / "metrics.json").read_text())
        target_area = seq_metrics["configurations"]["fully_predicted"]["by_tau"][str(PRIMARY_TAU)]["pred_unobserved_area_mm2"]

        # (a) all-unobserved
        pu_all = np.ones(n_faces, dtype=bool)
        rows, f_reassur, f_alarm = region_rows_for_mask(pu_all, gt_regions, gt_observed, gt_unobserved, ignore_set,
                                                           areas, adjacency, mesh_data, n_faces)
        for r in rows:
            r.update({"sequence": name, "baseline": "all_unobserved", "seed": None,
                       "false_reassurance_rate": f_reassur, "false_alarm_rate": f_alarm})
            baseline_rows.append(r)

        # (b) area-matched random, 20 seeds
        for seed in range(N_RANDOM_SEEDS):
            seed_rng = np.random.default_rng(1000 * SEED + seed)
            perm = seed_rng.permutation(n_faces)
            cum = np.cumsum(areas[perm])
            k = int(np.searchsorted(cum, target_area) + 1)
            k = min(k, n_faces)
            pu_rand = np.zeros(n_faces, dtype=bool)
            pu_rand[perm[:k]] = True
            rows, f_reassur, f_alarm = region_rows_for_mask(pu_rand, gt_regions, gt_observed, gt_unobserved, ignore_set,
                                                               areas, adjacency, mesh_data, n_faces)
            for r in rows:
                r.update({"sequence": name, "baseline": "area_matched_random", "seed": seed,
                           "false_reassurance_rate": f_reassur, "false_alarm_rate": f_alarm})
                baseline_rows.append(r)

        # ---------------- diagnostic 4: region-merging check, tau=0.25, every config ----------------
        ti = TAUS.index(PRIMARY_TAU)
        for ci, config in enumerate(CONFIG_NAMES):
            bits = np.unpackbits(packed[ci, ti], bitorder="little")[:n_faces].astype(bool)
            predicted_unobserved = ~bits
            predicted_components = compute_regions(n_faces, adjacency, predicted_unobserved & ~ignore_set, areas)
            face_to_component_id = build_face_to_component_id(n_faces, predicted_components)
            for region_id, r in enumerate(gt_regions):
                if r.size_class == "below_headline":
                    continue
                err = localization_error(r, predicted_components, face_to_component_id, mesh_data.vertices, mesh_data.faces, areas)
                if err is None:
                    continue
                comp = matching_predicted_component(r, predicted_components, face_to_component_id)
                comp_area = float(areas[comp.faces].sum())
                merging_rows.append({
                    "sequence": name, "config": config, "region_id": region_id, "size_class": r.size_class,
                    "region_area_mm2": r.area, "matched_component_area_mm2": comp_area,
                    "area_ratio": comp_area / r.area if r.area > 0 else float("nan"),
                    "localization_error_mm": err,
                })

        if (i + 1) % 10 == 0 or i == len(sequences) - 1:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(sequences) - i - 1)
            log(f"{i+1}/{len(sequences)}: {name} elapsed={elapsed:.1f}s ETA={eta:.1f}s")

    baseline_df = pd.DataFrame(baseline_rows)
    baseline_df.to_csv(OUT_DIR / "baselines.csv", index=False)
    log(f"wrote {OUT_DIR / 'baselines.csv'} ({len(baseline_df)} rows)")

    merging_df = pd.DataFrame(merging_rows)
    merging_df.to_csv(OUT_DIR / "region_merging.csv", index=False)
    log(f"wrote {OUT_DIR / 'region_merging.csv'} ({len(merging_df)} rows)")

    # ==================== Diagnostic 1 aggregation (bootstrap) ====================
    log("=== Diagnostic 1: aggregation ===")
    baseline_df = baseline_df.merge(
        mesh_df[["Video Name", "mesh_hash", "Colon", "Segment", "Phantom Number"]].rename(columns={"Video Name": "sequence"}),
        on="sequence", how="left",
    )
    baseline_df["colon_segment"] = baseline_df["Colon"] + "_" + baseline_df["Segment"]
    baseline_df["colon_segment_phantom"] = baseline_df["Colon"] + "_" + baseline_df["Segment"] + "_" + baseline_df["Phantom Number"]

    def baseline_cell(sub: pd.DataFrame, size_class: str, threshold: float, cluster_col: str, rng_local) -> dict:
        if size_class == "medium_plus_large":
            sc = sub[sub["size_class"].isin(["medium", "large"])]
        else:
            sc = sub[sub["size_class"] == size_class]
        det_col = f"detected_at_{threshold}"
        by_cluster = sc.groupby(cluster_col)[det_col].agg(["sum", "count"])
        point = by_cluster["sum"].sum() / by_cluster["count"].sum() if by_cluster["count"].sum() else float("nan")
        lo, hi = agg.bootstrap_ratio(by_cluster["sum"].to_numpy(dtype=float), by_cluster["count"].to_numpy(dtype=float), rng_local)
        return {"point": float(point), "ci_lo": lo, "ci_hi": hi, "n_regions": int(by_cluster["count"].sum())}

    bootstrap_out = {"seed": SEED, "n_boot": N_BOOT, "n_random_seeds": N_RANDOM_SEEDS, "baselines": {}}
    rng = np.random.default_rng(SEED)
    for baseline_name in ["all_unobserved", "area_matched_random"]:
        sub = baseline_df[baseline_df["baseline"] == baseline_name]

        per_seed_stats = None
        if baseline_name == "area_matched_random":
            # per-seed corpus-level point estimates first (mean/range over the
            # 20 seeds is a DIFFERENT quantity than the bootstrap CI of the
            # seed-averaged value computed below -- both are reported).
            per_seed_stats = {"recall": {sc: {str(t): [] for t in DETECTION_THRESHOLDS} for sc in size_classes},
                               "false_reassurance_rate": [], "false_alarm_rate": [], "localization_error_median_mm": []}
            for seed in range(N_RANDOM_SEEDS):
                seed_sub = sub[sub["seed"] == seed]
                for size_class in size_classes:
                    scsub = seed_sub[seed_sub["size_class"].isin(["medium", "large"])] if size_class == "medium_plus_large" \
                        else seed_sub[seed_sub["size_class"] == size_class]
                    for thresh in DETECTION_THRESHOLDS:
                        col = f"detected_at_{thresh}"
                        per_seed_stats["recall"][size_class][str(thresh)].append(
                            float(scsub[col].sum()) / len(scsub) if len(scsub) else float("nan")
                        )
                per_seq_seed = seed_sub.drop_duplicates("sequence")
                per_seed_stats["false_reassurance_rate"].append(float(per_seq_seed["false_reassurance_rate"].mean()))
                per_seed_stats["false_alarm_rate"].append(float(per_seq_seed["false_alarm_rate"].mean()))
                loc_seed = seed_sub[seed_sub["localization_error_mm"].notna()]["localization_error_mm"]
                per_seed_stats["localization_error_median_mm"].append(float(loc_seed.median()) if len(loc_seed) else float("nan"))

            # seed-average second: one row per (sequence, region, seed) -> average across seeds per (sequence, region)
            sub = sub.groupby(["sequence", "region_id"], as_index=False).agg({
                "size_class": "first", "detected_at_0.25": "mean", "detected_at_0.5": "mean", "detected_at_0.75": "mean",
                "false_reassurance_rate": "mean", "false_alarm_rate": "mean", "localization_error_mm": "mean",
                "mesh_hash": "first", "colon_segment": "first", "colon_segment_phantom": "first",
            })
        bootstrap_out["baselines"][baseline_name] = {"recall": {}, "false_reassurance_rate": {}, "false_alarm_rate": {}, "localization_error": {}}
        if per_seed_stats is not None:
            bootstrap_out["baselines"][baseline_name]["per_seed_mean_range"] = {
                "recall": {sc: {t: {"mean": float(np.nanmean(v)), "min": float(np.nanmin(v)), "max": float(np.nanmax(v))}
                                for t, v in thr.items()} for sc, thr in per_seed_stats["recall"].items()},
                "false_reassurance_rate": {"mean": float(np.mean(per_seed_stats["false_reassurance_rate"])),
                                            "min": float(np.min(per_seed_stats["false_reassurance_rate"])),
                                            "max": float(np.max(per_seed_stats["false_reassurance_rate"]))},
                "false_alarm_rate": {"mean": float(np.mean(per_seed_stats["false_alarm_rate"])),
                                      "min": float(np.min(per_seed_stats["false_alarm_rate"])),
                                      "max": float(np.max(per_seed_stats["false_alarm_rate"]))},
                "localization_error_median_mm": {"mean": float(np.nanmean(per_seed_stats["localization_error_median_mm"])),
                                                  "min": float(np.nanmin(per_seed_stats["localization_error_median_mm"])),
                                                  "max": float(np.nanmax(per_seed_stats["localization_error_median_mm"]))},
            }
        for size_class in size_classes:
            bootstrap_out["baselines"][baseline_name]["recall"][size_class] = {}
            for thresh in DETECTION_THRESHOLDS:
                primary = baseline_cell(sub, size_class, thresh, "mesh_hash", rng)
                sens1 = baseline_cell(sub, size_class, thresh, "colon_segment", rng)
                sens2 = baseline_cell(sub, size_class, thresh, "colon_segment_phantom", rng)
                primary["sensitivity_colon_segment_ci"] = [sens1["ci_lo"], sens1["ci_hi"]]
                primary["sensitivity_colon_segment_phantom_ci"] = [sens2["ci_lo"], sens2["ci_hi"]]
                bootstrap_out["baselines"][baseline_name]["recall"][size_class][str(thresh)] = primary

        # false_reassurance/false_alarm are constant per (sequence,region) group by construction; take per-sequence mean
        per_seq = baseline_df[baseline_df["baseline"] == baseline_name].drop_duplicates(["sequence"] + (["seed"] if baseline_name == "area_matched_random" else []))
        if baseline_name == "area_matched_random":
            per_seq = per_seq.groupby("sequence", as_index=False).agg({
                "false_reassurance_rate": "mean", "false_alarm_rate": "mean",
                "mesh_hash": "first", "colon_segment": "first", "colon_segment_phantom": "first",
            })
        for metric in ["false_reassurance_rate", "false_alarm_rate"]:
            by_cluster = per_seq.groupby("mesh_hash")[metric].mean().to_numpy()
            n = len(by_cluster)
            idx = rng.integers(0, n, size=(N_BOOT, n))
            means = by_cluster[idx].mean(axis=1)
            lo, hi = np.percentile(means, [2.5, 97.5])
            bootstrap_out["baselines"][baseline_name][metric] = {
                "point": float(per_seq[metric].mean()), "ci_lo": float(lo), "ci_hi": float(hi), "n_sequences": len(per_seq),
            }

        loc_sub = sub[sub["localization_error_mm"].notna()]
        by_cluster_loc = [g["localization_error_mm"].to_numpy() for _, g in loc_sub.groupby("mesh_hash")]
        lo, hi = agg.bootstrap_median(by_cluster_loc, rng)
        bootstrap_out["baselines"][baseline_name]["localization_error"] = {
            "median": float(loc_sub["localization_error_mm"].median()) if len(loc_sub) else None,
            "ci_lo": lo, "ci_hi": hi, "n_regions": len(loc_sub),
        }

    with open(OUT_DIR / "baselines_bootstrap.json", "w") as f:
        json.dump(bootstrap_out, f, indent=2)
    log(f"wrote {OUT_DIR / 'baselines_bootstrap.json'}")

    # ==================== Diagnostic 4 aggregation (Spearman + quartiles) ====================
    log("=== Diagnostic 4: aggregation ===")
    merging_df = merging_df.merge(
        mesh_df[["Video Name", "mesh_hash"]].rename(columns={"Video Name": "sequence"}), on="sequence", how="left"
    )
    merging_summary = {"seed": SEED, "n_boot": N_BOOT, "by_config": {}}
    for config in CONFIG_NAMES:
        sub = merging_df[merging_df["config"] == config]
        if len(sub) < 3:
            merging_summary["by_config"][config] = {"n": len(sub)}
            continue
        rho, pval = spearmanr(sub["area_ratio"], sub["localization_error_mm"])

        mesh_ids = sub["mesh_hash"].unique()
        by_mesh = {mh: g[["area_ratio", "localization_error_mm"]].to_numpy() for mh, g in sub.groupby("mesh_hash")}
        n = len(mesh_ids)
        rhos = np.empty(N_BOOT)
        idx_matrix = rng.integers(0, n, size=(N_BOOT, n))
        for b in range(N_BOOT):
            pooled = np.concatenate([by_mesh[mesh_ids[i]] for i in idx_matrix[b]], axis=0)
            if len(pooled) < 3 or np.std(pooled[:, 0]) == 0 or np.std(pooled[:, 1]) == 0:
                rhos[b] = np.nan
                continue
            rhos[b], _ = spearmanr(pooled[:, 0], pooled[:, 1])
        rhos = rhos[np.isfinite(rhos)]
        ci_lo, ci_hi = (float(np.percentile(rhos, 2.5)), float(np.percentile(rhos, 97.5))) if len(rhos) else (None, None)

        quartile_edges = np.percentile(sub["area_ratio"], [0, 25, 50, 75, 100])
        quartile_labels = pd.cut(sub["area_ratio"], bins=np.unique(quartile_edges), include_lowest=True, labels=False)
        quartile_stats = []
        for q in sorted(quartile_labels.dropna().unique()):
            vals = sub.loc[quartile_labels == q, "localization_error_mm"]
            ratio_vals = sub.loc[quartile_labels == q, "area_ratio"]
            quartile_stats.append({
                "quartile": int(q), "area_ratio_range": [float(ratio_vals.min()), float(ratio_vals.max())],
                "n": len(vals), "loc_err_median_mm": float(vals.median()), "loc_err_iqr_mm": float(vals.quantile(0.75) - vals.quantile(0.25)),
            })

        merging_summary["by_config"][config] = {
            "n": len(sub), "n_meshes": n, "spearman_rho": float(rho), "spearman_pvalue": float(pval),
            "bootstrap_ci": [ci_lo, ci_hi], "quartile_stats": quartile_stats,
        }
    with open(OUT_DIR / "region_merging_summary.json", "w") as f:
        json.dump(merging_summary, f, indent=2)
    log(f"wrote {OUT_DIR / 'region_merging_summary.json'}")

    log("ALL DONE")


if __name__ == "__main__":
    main()
