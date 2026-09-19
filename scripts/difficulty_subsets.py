#!/usr/bin/env python
"""Classify the 169 registered sequences into coverage-difficulty subsets
from already-computed results/coverage_stats.csv + results/coverage_components.json
(no mesh re-analysis).

  - "monolithic": largest unobserved component holds >70% of total unobserved area
  - "fragmented": >100 unobserved components AND largest component <40% of area
  - "mixed": everything else

Adds a "difficulty_subset" column to results/coverage_stats.csv in place.
Writes docs/difficulty_subsets.md (after inspecting this script's output).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

CSV_PATH = "results/coverage_stats.csv"
COMPONENTS_PATH = "results/coverage_components.json"

MONOLITHIC_FRAC_THRESHOLD = 0.70
FRAGMENTED_COUNT_THRESHOLD = 100
FRAGMENTED_FRAC_THRESHOLD = 0.40


def classify(row: pd.Series) -> str:
    if row["largest_component_frac"] > MONOLITHIC_FRAC_THRESHOLD:
        return "monolithic"
    if (
        row["n_unobserved_components"] > FRAGMENTED_COUNT_THRESHOLD
        and row["largest_component_frac"] < FRAGMENTED_FRAC_THRESHOLD
    ):
        return "fragmented"
    return "mixed"


def main():
    df = pd.read_csv(CSV_PATH)
    components = json.load(open(COMPONENTS_PATH))

    # Cross-check comp_area_top1_mm2/unobserved_area_mm2 (CSV) against a
    # direct recomputation from the raw per-sequence component area lists.
    recomputed = df["Video Name"].map(lambda n: max(components[n]) / sum(components[n]))
    csv_frac = df["comp_area_top1_mm2"] / df["unobserved_area_mm2"]
    max_diff = float((recomputed - csv_frac).abs().max())
    print(f"cross-check: max |recomputed - CSV| largest-component-fraction diff = {max_diff:.2e}")
    assert max_diff < 1e-6, "CSV comp_area_top1_mm2 disagrees with coverage_components.json"

    df["largest_component_frac"] = csv_frac
    df["difficulty_subset"] = df.apply(classify, axis=1)

    df.to_csv(CSV_PATH, index=False)
    print(f"wrote {CSV_PATH} with difficulty_subset column")

    print("\n=== counts ===")
    print(df["difficulty_subset"].value_counts())

    print("\n=== per-subset medians ===")
    metrics = [
        "unobserved_frac_by_area",
        "n_components_gt5mm_diam",
        "n_components_gt10mm_diam",
        "n_fully_interior_components",
    ]
    print(df.groupby("difficulty_subset")[metrics].median())

    print("\n=== per-subset Segment distribution ===")
    print(pd.crosstab(df["Segment"], df["difficulty_subset"]))

    print("\n=== 5 closest to monolithic boundary (largest_component_frac vs 0.70) ===")
    d = (df["largest_component_frac"] - MONOLITHIC_FRAC_THRESHOLD).abs().sort_values()
    print(df.loc[d.index[:5], ["Video Name", "largest_component_frac", "n_unobserved_components", "difficulty_subset"]])

    print("\n=== 5 closest to fragmented count boundary (n_unobserved_components vs 100) ===")
    d = (df["n_unobserved_components"] - FRAGMENTED_COUNT_THRESHOLD).abs().sort_values()
    print(df.loc[d.index[:5], ["Video Name", "n_unobserved_components", "largest_component_frac", "difficulty_subset"]])

    print("\n=== 5 closest to fragmented frac boundary (largest_component_frac vs 0.40) ===")
    d = (df["largest_component_frac"] - FRAGMENTED_FRAC_THRESHOLD).abs().sort_values()
    print(df.loc[d.index[:5], ["Video Name", "largest_component_frac", "n_unobserved_components", "difficulty_subset"]])


if __name__ == "__main__":
    main()
