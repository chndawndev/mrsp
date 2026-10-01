#!/usr/bin/env python
"""Region IoU for EndoDAC and MASt3R-SLAM, each next to its own
area-matched random baseline (docs/eval_protocol.md, "2026-09-28:
Reporting rule for region IoU").

Cells per pipeline: all four configurations at tau = 0.25, and
fully_predicted at tau in {0.15, 0.35, 0.50} (7 cells).

For every (pipeline, sequence, cell): region IoU of the pipeline's own
predicted-unobserved set, and of 20 area-matched random sets whose target
area is THAT cell's predicted-unobserved area on THAT sequence. The random
construction is scripts/diagnose_d1_3.py's / scripts/region_iou_gate.py's,
unchanged: seed_rng = default_rng(1000 * 20260925 + seed), a permutation of
all faces, faces taken in that order until the cumulative area reaches the
target (k = searchsorted(cumsum, target) + 1). Only the target area differs
per cell.

Metric code: src/eval_ext/region_iou.py::region_iou (unchanged), with the
locked component construction eval.regions.compute_regions on
predicted_unobserved & ~ignore_set, exactly as scripts/region_iou_gate.py.
The ignore set is results/d1/diagnosis/ignore_set/ (method-independent,
GT pose only); the MASt3R evaluation run recomputes it per sequence and
records that it is identical (metrics.json cross_checks).

Also records, per (pipeline, sequence, cell, source, seed), the raw areas
needed to pool false alarm and area fraction for the random baselines.

CPU only. CLI:
    scratch/.venv/bin/python scripts/pipeline2_region_iou.py --workers 12
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
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

SEED = d13.SEED  # 20260925
N_RANDOM_SEEDS = d13.N_RANDOM_SEEDS  # 20
TAUS = d13.TAUS
CONFIG_NAMES = d13.CONFIG_NAMES
CELLS = [(c, 0.25) for c in CONFIG_NAMES] + [("fully_predicted", t) for t in (0.15, 0.35, 0.50)]
PER_SEQ_ROOTS = {
    "endodac": REPO / "results/d1/per_sequence",
    "mast3r_slam": REPO / "results/pipeline2_eval/per_sequence/mast3r_slam",
}
OUT_DIR = REPO / "results/pipeline2_eval/region_iou"
EXPECTED_N_SEQUENCES = 169


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}", flush=True)


def load_packed(root: Path, name: str, n_faces: int) -> np.ndarray:
    n_bytes = (n_faces + 7) // 8
    packed = np.fromfile(root / name / "predicted_observed_packed.bin", dtype=np.uint8)
    return packed.reshape(len(CONFIG_NAMES), len(TAUS), n_bytes)


def iou_rows(predicted_unobserved, gt_regions, ignore_set, areas, adjacency, n_faces):
    comps = compute_regions(n_faces, adjacency, predicted_unobserved & ~ignore_set, areas)
    f2c = build_face_to_component_id(n_faces, comps)
    return [
        (region_id, r.size_class, r.area, region_iou(r, comps, f2c, areas))
        for region_id, r in enumerate(gt_regions) if r.size_class != "below_headline"
    ]


def area_row(predicted_unobserved, gt_observed, ignore_set, areas) -> dict:
    scored = predicted_unobserved & ~ignore_set
    return {
        "pred_unobserved_area_mm2": float(areas[predicted_unobserved].sum()),
        "false_alarm_numerator_area_mm2": float(areas[scored & gt_observed].sum()),
        "false_alarm_denominator_area_mm2": float(areas[scored].sum()),
    }


def process_sequence(name: str) -> tuple[list[dict], list[dict]]:
    t0 = time.time()
    mesh_data, areas, adjacency, gt_observed, gt_unobserved, gt_regions, n_faces = d13.load_sequence_geometry(name)
    ignore_set = d13.load_ignore_set(name, n_faces)
    total_area = float(areas.sum())

    # one permutation per seed (depends only on seed and n_faces), reused for every cell
    perms = []
    for seed in range(N_RANDOM_SEEDS):
        seed_rng = np.random.default_rng(1000 * SEED + seed)
        perm = seed_rng.permutation(n_faces)
        perms.append((perm, np.cumsum(areas[perm])))

    region_out, area_out = [], []
    for pipeline, root in PER_SEQ_ROOTS.items():
        packed = load_packed(root, name, n_faces)
        metrics = json.loads((root / name / "metrics.json").read_text())
        for config, tau in CELLS:
            ci, ti = CONFIG_NAMES.index(config), TAUS.index(tau)
            observed = np.unpackbits(packed[ci, ti], bitorder="little")[:n_faces].astype(bool)
            pu = ~observed
            target_area = metrics["configurations"][config]["by_tau"][str(tau)]["pred_unobserved_area_mm2"]
            if abs(float(areas[pu].sum()) - target_area) > 1e-6 * max(target_area, 1.0):
                raise RuntimeError(f"{pipeline}/{name}/{config}/{tau}: packed bits disagree with metrics.json area")
            base = {"pipeline": pipeline, "sequence": name, "config": config, "tau": tau}
            for rid, sc, ra, iou in iou_rows(pu, gt_regions, ignore_set, areas, adjacency, n_faces):
                region_out.append({**base, "source": "pipeline", "seed": -1, "region_id": rid,
                                   "size_class": sc, "region_area_mm2": ra, "iou": iou})
            area_out.append({**base, "source": "pipeline", "seed": -1, "total_mesh_area_mm2": total_area,
                             **area_row(pu, gt_observed, ignore_set, areas)})
            for seed, (perm, cum) in enumerate(perms):
                k = min(int(np.searchsorted(cum, target_area) + 1), n_faces)
                pr = np.zeros(n_faces, dtype=bool)
                pr[perm[:k]] = True
                for rid, sc, ra, iou in iou_rows(pr, gt_regions, ignore_set, areas, adjacency, n_faces):
                    region_out.append({**base, "source": "random", "seed": seed, "region_id": rid,
                                       "size_class": sc, "region_area_mm2": ra, "iou": iou})
                area_out.append({**base, "source": "random", "seed": seed, "total_mesh_area_mm2": total_area,
                                 **area_row(pr, gt_observed, ignore_set, areas)})
    log(f"{name}: {time.time() - t0:.1f}s")
    return region_out, area_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--sequence", action="append")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    from eval_pipeline_sequence import discover_sequences
    sequences = args.sequence if args.sequence else [n for n, _ in discover_sequences()]
    for pipeline, root in PER_SEQ_ROOTS.items():
        not_ok = [n for n in sequences if not (root / n / "MANIFEST.json").exists()
                  or json.loads((root / n / "MANIFEST.json").read_text()).get("status") != "ok"]
        if not_ok:
            raise SystemExit(f"{pipeline}: per-sequence evaluation incomplete or failed: {not_ok}")
    log(f"{len(sequences)} sequences, cells={CELLS}, seeds={N_RANDOM_SEEDS}, workers={args.workers}")

    t0 = time.time()
    region_rows, area_rows = [], []
    with Pool(args.workers) as pool:
        for k, (r, a) in enumerate(pool.imap_unordered(process_sequence, sequences)):
            region_rows.extend(r)
            area_rows.extend(a)
            el = time.time() - t0
            log(f"progress {k+1}/{len(sequences)} elapsed {el/60:.1f}min ETA {el/(k+1)*(len(sequences)-k-1)/60:.1f}min")
    suffix = "" if not args.sequence else "_subset"
    pd.DataFrame(region_rows).to_csv(OUT_DIR / f"region_iou_rows{suffix}.csv.gz", index=False)
    pd.DataFrame(area_rows).to_csv(OUT_DIR / f"area_rows{suffix}.csv", index=False)
    log(f"wrote {len(region_rows)} region rows, {len(area_rows)} area rows in {(time.time()-t0)/60:.1f}min")


if __name__ == "__main__":
    main()
