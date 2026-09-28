#!/usr/bin/env python
"""Region IoU corpus computation + validity gate, `docs/success_criteria.md`
section 6 (2026-09-27 deviation, items 1-2). All 169 registered sequences,
tau=0.25, three configurations only: oracle, all-unobserved baseline, and
the area-matched random baseline (same 20 seeds as scripts/
diagnose_d1_3.py). Does NOT score EndoDAC, MASt3R-SLAM, or any other real
pipeline -- the gate is defined on oracle + the two location-blind
baselines only.

Does not edit and does not need to edit src/eval/, src/gt/,
docs/success_criteria.md, or docs/eval_protocol.md. Reuses:
  - eval.regions.compute_regions (the locked component-construction
    function, imported not reimplemented -- builds both GT regions and
    predicted components).
  - eval.region_metrics.build_face_to_component_id.
  - eval_ext.region_iou.region_iou (this task's new metric).
  - scripts/diagnose_d1_3.py's geometry/mask-construction helpers
    (load_sequence_geometry, load_ignore_set, load_predicted_observed,
    discover_sequences_with_ignore_set) and its exact area-matched-random
    baseline construction (same target-area source, same seed formula),
    imported as a module -- not re-derived.
  - scripts/eval_d1_aggregate.py's mesh-cluster helpers (add_clusters,
    bootstrap_ratio) and SEED/N_BOOT constants, imported as a module.

No GPU needed (pure CPU geometry/component pass over already-computed
ray-cast outputs). No src/eval/src/gt ray casting is re-run.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/data1_ycao/chua/projects/mrsp")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from eval.region_metrics import build_face_to_component_id  # noqa: E402
from eval.regions import compute_regions  # noqa: E402
from eval_ext.region_iou import region_iou  # noqa: E402

import diagnose_d1_3 as d13  # noqa: E402
import eval_d1_aggregate as agg  # noqa: E402

MESH_IDENTITY_CSV = REPO / "results/mesh_identity.csv"
OUT_DIR = REPO / "results/region_iou_gate"
SEED = agg.SEED  # 20260925, same as diagnose_d1_3.py -- gate reuses that seed, not a new one
N_BOOT = agg.N_BOOT  # 10000
N_RANDOM_SEEDS = d13.N_RANDOM_SEEDS  # 20
TAUS = d13.TAUS
CONFIG_NAMES = d13.CONFIG_NAMES
PRIMARY_TAU = 0.25
SIZE_CLASSES = ["small", "medium", "large", "medium_plus_large"]
EXPECTED_N_SEQUENCES = 169

# Gate thresholds, docs/success_criteria.md section 6, item 2 -- quoted:
# "Pass: oracle median IoU >= 0.80, and oracle median IoU exceeds each
# baseline's median by >= 0.30."
GATE_ORACLE_MIN_MEDIAN = 0.80
GATE_ORACLE_MARGIN_OVER_BASELINE = 0.30


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}", flush=True)


def rows_for_mask(name, predicted_unobserved, gt_regions, ignore_set, areas, adjacency, n_faces):
    predicted_components = compute_regions(n_faces, adjacency, predicted_unobserved & ~ignore_set, areas)
    face_to_component_id = build_face_to_component_id(n_faces, predicted_components)
    rows = []
    for region_id, r in enumerate(gt_regions):
        if r.size_class == "below_headline":
            continue
        iou = region_iou(r, predicted_components, face_to_component_id, areas)
        rows.append({
            "sequence": name, "region_id": region_id, "size_class": r.size_class,
            "region_area_mm2": r.area, "iou": iou,
        })
    return rows


def bootstrap_mean_and_median(values_by_cluster: list[np.ndarray], rng: np.random.Generator, n_boot: int = N_BOOT) -> dict:
    """Cluster bootstrap of the POOLED mean and POOLED median together (one
    resampling loop, both statistics per replicate) -- same structure as
    scripts/eval_d1_aggregate.py's bootstrap_median, extended to also
    track the mean, since region IoU's corpus value (section 6, item 1)
    is "mean over regions", not a ratio of summed areas, so
    bootstrap_ratio does not apply."""
    n = len(values_by_cluster)
    non_empty = [v for v in values_by_cluster if len(v) > 0]
    if n == 0 or not non_empty:
        return {"mean_ci": (float("nan"), float("nan")), "median_ci": (float("nan"), float("nan"))}
    means = np.empty(n_boot)
    medians = np.empty(n_boot)
    idx_matrix = rng.integers(0, n, size=(n_boot, n))
    for b in range(n_boot):
        pooled = np.concatenate([values_by_cluster[i] for i in idx_matrix[b] if len(values_by_cluster[i]) > 0])
        if len(pooled):
            means[b] = pooled.mean()
            medians[b] = np.median(pooled)
        else:
            means[b] = np.nan
            medians[b] = np.nan
    means_f = means[np.isfinite(means)]
    medians_f = medians[np.isfinite(medians)]
    mean_ci = tuple(np.percentile(means_f, [2.5, 97.5])) if len(means_f) else (float("nan"), float("nan"))
    median_ci = tuple(np.percentile(medians_f, [2.5, 97.5])) if len(medians_f) else (float("nan"), float("nan"))
    return {"mean_ci": (float(mean_ci[0]), float(mean_ci[1])), "median_ci": (float(median_ci[0]), float(median_ci[1]))}


def pooled_cell(df: pd.DataFrame, size_class: str, cluster_col: str, rng: np.random.Generator) -> dict:
    sub = df[df["size_class"].isin(["medium", "large"])] if size_class == "medium_plus_large" \
        else df[df["size_class"] == size_class]
    vals = sub["iou"].to_numpy()
    if len(vals) == 0:
        return {"n_regions": 0, "mean": float("nan"), "median": float("nan"),
                "mean_ci": [float("nan"), float("nan")], "median_ci": [float("nan"), float("nan")]}
    by_cluster = [g["iou"].to_numpy() for _, g in sub.groupby(cluster_col)]
    boot = bootstrap_mean_and_median(by_cluster, rng)
    return {
        "n_regions": int(len(vals)), "n_clusters": int(sub[cluster_col].nunique()),
        "mean": float(vals.mean()), "median": float(np.median(vals)),
        "mean_ci": list(boot["mean_ci"]), "median_ci": list(boot["median_ci"]),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sequences = d13.discover_sequences_with_ignore_set()
    assert len(sequences) == EXPECTED_N_SEQUENCES, (
        f"expected {EXPECTED_N_SEQUENCES} sequences with complete Stage B + ignore_set, got {len(sequences)}"
    )
    log(f"{len(sequences)} sequences")

    mesh_df = pd.read_csv(MESH_IDENTITY_CSV)
    tau_idx = TAUS.index(PRIMARY_TAU)
    oracle_idx = CONFIG_NAMES.index("oracle")

    all_rows = []
    t0 = time.time()
    for i, name in enumerate(sequences):
        mesh_data, areas, adjacency, gt_observed, gt_unobserved, gt_regions, n_faces = d13.load_sequence_geometry(name)
        ignore_set = d13.load_ignore_set(name, n_faces)
        packed = d13.load_predicted_observed(name, n_faces)
        seq_metrics = json.loads((d13.PER_SEQ_ROOT / name / "metrics.json").read_text())
        target_area = seq_metrics["configurations"]["fully_predicted"]["by_tau"][str(PRIMARY_TAU)]["pred_unobserved_area_mm2"]

        # oracle: already-computed ray-cast output, unpack only, no new ray cast
        bits = np.unpackbits(packed[oracle_idx, tau_idx], bitorder="little")[:n_faces].astype(bool)
        pu_oracle = ~bits
        for r in rows_for_mask(name, pu_oracle, gt_regions, ignore_set, areas, adjacency, n_faces):
            r.update({"baseline": "oracle", "seed": None})
            all_rows.append(r)

        # all-unobserved
        pu_all = np.ones(n_faces, dtype=bool)
        for r in rows_for_mask(name, pu_all, gt_regions, ignore_set, areas, adjacency, n_faces):
            r.update({"baseline": "all_unobserved", "seed": None})
            all_rows.append(r)

        # area-matched random, 20 seeds -- identical construction to scripts/diagnose_d1_3.py
        for seed in range(N_RANDOM_SEEDS):
            seed_rng = np.random.default_rng(1000 * SEED + seed)
            perm = seed_rng.permutation(n_faces)
            cum = np.cumsum(areas[perm])
            k = int(np.searchsorted(cum, target_area) + 1)
            k = min(k, n_faces)
            pu_rand = np.zeros(n_faces, dtype=bool)
            pu_rand[perm[:k]] = True
            for r in rows_for_mask(name, pu_rand, gt_regions, ignore_set, areas, adjacency, n_faces):
                r.update({"baseline": "area_matched_random", "seed": seed})
                all_rows.append(r)

        if (i + 1) % 10 == 0 or i == len(sequences) - 1:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(sequences) - i - 1)
            log(f"{i+1}/{len(sequences)}: {name} elapsed={elapsed:.1f}s ETA={eta:.1f}s")

    df = pd.DataFrame(all_rows)
    df = agg.add_clusters(df, mesh_df)
    df.to_csv(OUT_DIR / "region_iou_raw.csv", index=False)
    log(f"wrote {OUT_DIR / 'region_iou_raw.csv'} ({len(df)} rows)")

    # ==================== corpus tables: oracle + all_unobserved (single pass) ====================
    rng = np.random.default_rng(SEED)
    summary = {"seed": SEED, "n_boot": N_BOOT, "n_random_seeds": N_RANDOM_SEEDS, "tau": PRIMARY_TAU, "by_baseline": {}}

    for baseline in ["oracle", "all_unobserved"]:
        sub = df[df["baseline"] == baseline]
        summary["by_baseline"][baseline] = {}
        for size_class in SIZE_CLASSES:
            primary = pooled_cell(sub, size_class, "mesh_hash", rng)
            sens_cs = pooled_cell(sub, size_class, "colon_segment", rng)
            sens_csp = pooled_cell(sub, size_class, "colon_segment_phantom", rng)
            primary["sensitivity_colon_segment_median_ci"] = sens_cs["median_ci"]
            primary["sensitivity_colon_segment_mean_ci"] = sens_cs["mean_ci"]
            primary["sensitivity_colon_segment_phantom_median_ci"] = sens_csp["median_ci"]
            primary["sensitivity_colon_segment_phantom_mean_ci"] = sens_csp["mean_ci"]
            summary["by_baseline"][baseline][size_class] = primary
        log(f"{baseline}: {summary['by_baseline'][baseline]['medium_plus_large']}")

    # ==================== area_matched_random: per-seed mean/range, plus seed-averaged pooled CI ====================
    rand = df[df["baseline"] == "area_matched_random"]
    per_seed = {sc: {"mean": [], "median": []} for sc in SIZE_CLASSES}
    for seed in range(N_RANDOM_SEEDS):
        seed_sub = rand[rand["seed"] == seed]
        for size_class in SIZE_CLASSES:
            sc_sub = seed_sub[seed_sub["size_class"].isin(["medium", "large"])] if size_class == "medium_plus_large" \
                else seed_sub[seed_sub["size_class"] == size_class]
            vals = sc_sub["iou"].to_numpy()
            per_seed[size_class]["mean"].append(float(vals.mean()) if len(vals) else float("nan"))
            per_seed[size_class]["median"].append(float(np.median(vals)) if len(vals) else float("nan"))

    # seed-averaged per (sequence, region): average the 20 seeds' iou for the same region, then pool+bootstrap
    rand_seed_avg = rand.groupby(["sequence", "region_id"], as_index=False).agg({
        "size_class": "first", "iou": "mean", "mesh_hash": "first",
        "colon_segment": "first", "colon_segment_phantom": "first",
    })
    summary["by_baseline"]["area_matched_random"] = {}
    for size_class in SIZE_CLASSES:
        seed_averaged_cell = pooled_cell(rand_seed_avg, size_class, "mesh_hash", rng)
        summary["by_baseline"]["area_matched_random"][size_class] = {
            "per_seed_mean_of_means": float(np.nanmean(per_seed[size_class]["mean"])),
            "per_seed_range_of_means": [float(np.nanmin(per_seed[size_class]["mean"])), float(np.nanmax(per_seed[size_class]["mean"]))],
            "per_seed_mean_of_medians": float(np.nanmean(per_seed[size_class]["median"])),
            "per_seed_range_of_medians": [float(np.nanmin(per_seed[size_class]["median"])), float(np.nanmax(per_seed[size_class]["median"]))],
            "seed_averaged_pooled_mean": seed_averaged_cell["mean"],
            "seed_averaged_pooled_median": seed_averaged_cell["median"],
            "seed_averaged_pooled_mean_ci": seed_averaged_cell["mean_ci"],
            "seed_averaged_pooled_median_ci": seed_averaged_cell["median_ci"],
            "n_regions": seed_averaged_cell["n_regions"],
        }
    log(f"area_matched_random: {summary['by_baseline']['area_matched_random']['medium_plus_large']}")

    with open(OUT_DIR / "region_iou_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(f"wrote {OUT_DIR / 'region_iou_summary.json'}")

    # ==================== gate, docs/success_criteria.md section 6 item 2 (mechanical) ====================
    oracle_median = summary["by_baseline"]["oracle"]["medium_plus_large"]["median"]
    all_unobs_median = summary["by_baseline"]["all_unobserved"]["medium_plus_large"]["median"]
    random_median = summary["by_baseline"]["area_matched_random"]["medium_plus_large"]["per_seed_mean_of_medians"]

    margin_all_unobs = oracle_median - all_unobs_median
    margin_random = oracle_median - random_median

    pass_min = oracle_median >= GATE_ORACLE_MIN_MEDIAN
    pass_margin_all_unobs = margin_all_unobs >= GATE_ORACLE_MARGIN_OVER_BASELINE
    pass_margin_random = margin_random >= GATE_ORACLE_MARGIN_OVER_BASELINE
    gate_pass = pass_min and pass_margin_all_unobs and pass_margin_random

    gate = {
        "tau": PRIMARY_TAU, "size_class": "medium_plus_large",
        "oracle_median_iou": oracle_median,
        "all_unobserved_median_iou": all_unobs_median,
        "area_matched_random_median_iou_seed_averaged": random_median,
        "margin_oracle_minus_all_unobserved": margin_all_unobs,
        "margin_oracle_minus_area_matched_random": margin_random,
        "threshold_oracle_min_median": GATE_ORACLE_MIN_MEDIAN,
        "threshold_margin": GATE_ORACLE_MARGIN_OVER_BASELINE,
        "pass_oracle_min_median": bool(pass_min),
        "pass_margin_over_all_unobserved": bool(pass_margin_all_unobs),
        "pass_margin_over_area_matched_random": bool(pass_margin_random),
        "GATE_RESULT": "PASS" if gate_pass else "FAIL",
    }
    with open(OUT_DIR / "gate_result.json", "w") as f:
        json.dump(gate, f, indent=2)
    log(f"GATE RESULT: {gate['GATE_RESULT']}")
    log(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
