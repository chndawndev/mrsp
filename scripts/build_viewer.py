#!/usr/bin/env python
"""Build the self-contained results viewer HTML for c1_cecum_t1_v1 from
the Stage 1 export (results/viewer/c1_cecum_t1_v1/) and the viewer source
in tools/viewer/.

Reads the Stage 1 export ONLY (never writes to it). Embeds float32/int
arrays plus the three JSON files, gzip+base64, verified against the
export's own manifest.json sha256 list before embedding. Does NOT embed
the float64 verification copies (those exist for Stage 1's own check A,
not for the viewer).

Also writes results/viewer/category_counts_python.json: per-face category
counts computed directly in numpy from the exported arrays, for every
config x tau plus one diff pair -- the independent reference
scripts/check_viewer.py compares the browser's JS counts against. This
file is NOT embedded in the HTML.
"""
from __future__ import annotations

import base64
import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path("/data1_ycao/chua/projects/mrsp")
EXPORT_DIR = REPO / "results/viewer/c1_cecum_t1_v1"
VIEWER_SRC = REPO / "tools/viewer"
OUT_HTML = REPO / "results/viewer/c1_cecum_t1_v1_viewer.html"
COUNTS_OUT = REPO / "results/viewer/category_counts_python.json"

MAX_SIZE_MB = 60

BIN_KEYS = [
    "vertices_f32",
    "faces_i32",
    "gt_observed_u8",
    "ignore_set_u8",
    "gt_region_id_i32",
    "predicted_observed_packed",
]
JSON_FILES = {
    "metrics_json": "metrics.json",
    "regions_json": "regions.json",
    "trajectory_json": "trajectory.json",
}


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gzip_b64(data: bytes) -> str:
    return base64.b64encode(gzip.compress(data, 9)).decode("ascii")


def load_and_verify_manifest() -> dict:
    manifest = json.loads((EXPORT_DIR / "manifest.json").read_text())
    for key in BIN_KEYS:
        info = manifest["files"][key]
        path = EXPORT_DIR / info["path"]
        actual = sha256_of(path)
        if actual != info["sha256"]:
            raise RuntimeError(
                f"sha256 mismatch for {key} ({path}): manifest says {info['sha256']}, "
                f"file is {actual} -- refusing to embed a changed export"
            )
    return manifest


def build_data_blocks(manifest: dict) -> tuple[str, dict]:
    blocks = []
    payload_hashes = {}
    for key in BIN_KEYS:
        info = manifest["files"][key]
        raw = (EXPORT_DIR / info["path"]).read_bytes()
        b64 = gzip_b64(raw)
        payload_hashes[key] = hashlib.sha256(raw).hexdigest()
        blocks.append(f'<script type="application/octet-stream" id="data-{key}">{b64}</script>')
    for key, filename in JSON_FILES.items():
        raw = (EXPORT_DIR / filename).read_bytes()
        b64 = gzip_b64(raw)
        payload_hashes[key] = hashlib.sha256(raw).hexdigest()
        blocks.append(f'<script type="application/octet-stream" id="data-{key}">{b64}</script>')
    return "\n".join(blocks), payload_hashes


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()


def compute_python_category_counts(manifest: dict) -> dict:
    n_faces = manifest["n_faces"]
    gt_observed = np.fromfile(EXPORT_DIR / "gt_observed_u8.bin", dtype=np.uint8).astype(bool)
    ignore_set = np.fromfile(EXPORT_DIR / "ignore_set_u8.bin", dtype=np.uint8).astype(bool)
    info = manifest["files"]["predicted_observed_packed"]
    packed = np.fromfile(EXPORT_DIR / info["path"], dtype=info["dtype"]).reshape(info["shape"])
    config_names = manifest["configurations"]
    taus = manifest["taus"]

    def predicted_observed(config_name: str, tau: float) -> np.ndarray:
        ci = config_names.index(config_name)
        ti = taus.index(tau)
        return np.unpackbits(packed[ci, ti], bitorder="little")[:n_faces].astype(bool)

    def categorize(po: np.ndarray) -> np.ndarray:
        pu = ~po
        cat = np.full(n_faces, 3, dtype=np.uint8)  # default: correctly observed
        cat[~gt_observed & pu] = 0
        cat[~gt_observed & ~pu] = 1
        cat[gt_observed & pu & ~ignore_set] = 2
        cat[ignore_set] = 4
        return cat

    out = {"single_config": {}, "diff_example": {}}
    for config_name in config_names:
        out["single_config"][config_name] = {}
        for tau in taus:
            cat = categorize(predicted_observed(config_name, tau))
            counts = np.bincount(cat, minlength=5).tolist()
            out["single_config"][config_name][str(tau)] = counts

    pu_a = ~predicted_observed("fully_predicted", 0.25)
    pu_b = ~predicted_observed("pred_pose_only", 0.25)
    diff_cat = np.full(n_faces, 3, dtype=np.uint8)
    diff_cat[pu_a & ~pu_b] = 0
    diff_cat[pu_b & ~pu_a] = 1
    diff_cat[pu_a & pu_b] = 2
    diff_cat[~pu_a & ~pu_b] = 3
    out["diff_example"] = {
        "a": "fully_predicted", "b": "pred_pose_only", "tau": 0.25,
        "counts": np.bincount(diff_cat, minlength=4).tolist(),
    }
    return out


def main():
    manifest = load_and_verify_manifest()
    print(f"verified sha256 for {len(BIN_KEYS)} binary files against manifest")

    data_blocks, payload_hashes = build_data_blocks(manifest)
    css = (VIEWER_SRC / "viewer.css").read_text()
    core_js = (VIEWER_SRC / "core.js").read_text()
    viewer_js = (VIEWER_SRC / "viewer.js").read_text()
    template = (VIEWER_SRC / "template.html").read_text()

    build_info = (
        f"<!-- built from commit {git_head()}, source commit for the export in manifest below; "
        f"payload sha256s: {json.dumps(payload_hashes)} -->"
    )

    html = template
    html = html.replace("<!-- BUILD_INFO -->", build_info)
    html = html.replace("/* CSS_INLINE */", css)
    html = html.replace("MANIFEST_JSON", json.dumps(manifest))
    html = html.replace("<!-- DATA_BLOCKS -->", data_blocks)
    html = html.replace(
        "/* APP_JS_INLINE (core.js followed by viewer.js, concatenated by build_viewer.py --\n"
        "   one module scope so viewer.js can call core.js's top-level functions directly) */",
        core_js + "\n" + viewer_js,
    )

    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(html)

    size_mb = OUT_HTML.stat().st_size / 1e6
    print(f"wrote {OUT_HTML} ({size_mb:.2f} MB)")
    if size_mb > MAX_SIZE_MB:
        print(f"*** size {size_mb:.2f} MB exceeds {MAX_SIZE_MB} MB -- stopping before optimizing, per instructions ***")
        sys.exit(1)

    counts = compute_python_category_counts(manifest)
    COUNTS_OUT.write_text(json.dumps(counts, indent=2))
    print(f"wrote {COUNTS_OUT}")
    print("oracle @ tau=0.25 counts:", counts["single_config"]["oracle"]["0.25"])


if __name__ == "__main__":
    main()
