"""Part 1 (corrected) step 1: per-sequence vignette mask = complement of the
INTERSECTION of GT depth raw==0 across every frame of that sequence -- not
a single frame's zero set. Computed for all 169 registered sequences,
streaming depth TIFFs directly from the archives (no bulk extraction).
Then compares all 169 per-sequence masks for identity, with attention to
v1/v2 and c1/c2 groupings. See docs/eval_protocol_oracle_gap.md.
"""
import io
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import tifffile

DATASET_ROOT = Path("/data1_ycao/chua/datasets/C3VDv2/registered_videos")
OUT_DIR = Path("/data1_ycao/chua/projects/mrsp/results/pipelines/oracle_gap_vignette")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sequence_names():
    zips = sorted(DATASET_ROOT.glob("*.zip"))
    names = [z.stem for z in zips if not z.stem.startswith("c0_")]
    assert len(names) == 169, f"expected 169 registered sequences, got {len(names)}"
    return names


def depth_members(zf: zipfile.ZipFile) -> list[str]:
    return sorted(
        n for n in zf.namelist()
        if n.endswith("_depth.tiff") and ("/depth/" in n or n.startswith("depth/"))
    )


def sequence_vignette_mask(seq_name: str) -> tuple[np.ndarray, int]:
    """Returns (intersection_mask boolean [True=vignette], n_frames_used)."""
    zpath = DATASET_ROOT / f"{seq_name}.zip"
    with zipfile.ZipFile(zpath) as zf:
        members = depth_members(zf)
        if not members:
            raise RuntimeError(f"{seq_name}: no depth/*_depth.tiff members found")
        intersection = None
        for m in members:
            data = zf.read(m)
            raw = tifffile.imread(io.BytesIO(data))
            zero = (raw == 0)
            intersection = zero if intersection is None else (intersection & zero)
        return intersection, len(members)


def main():
    names = sequence_names()
    log(f"{len(names)} registered sequences (c1_*/c2_*, c0_* excluded)")

    t0 = time.time()
    summary = []
    for i, name in enumerate(names):
        try:
            mask, n_frames = sequence_vignette_mask(name)
        except Exception as e:
            log(f"  ERROR on {name}: {e}")
            summary.append({"name": name, "error": str(e)})
            continue
        n_vignette = int(mask.sum())
        packed = np.packbits(mask)
        np.save(OUT_DIR / f"{name}_vignette_mask.npy", packed)
        summary.append({
            "name": name, "n_frames": n_frames, "n_vignette_px": n_vignette,
            "frac": n_vignette / mask.size, "shape": mask.shape,
        })
        if i % 10 == 0 or i == len(names) - 1:
            elapsed = time.time() - t0
            log(f"  [{i+1}/{len(names)}] {name}: {n_frames} frames, "
                f"{n_vignette} vignette px ({n_vignette/mask.size*100:.4f}%), "
                f"elapsed {elapsed:.0f}s")

    log(f"\nall sequences processed in {time.time()-t0:.0f}s")

    # ---- compare all masks for identity ----
    log("\n=== comparing all per-sequence vignette masks ===")
    ok = [s for s in summary if "error" not in s]
    if len(ok) < len(summary):
        log(f"WARNING: {len(summary)-len(ok)} sequences errored, see above")

    ref_name = ok[0]["name"]
    ref_packed = np.load(OUT_DIR / f"{ref_name}_vignette_mask.npy")
    log(f"reference: {ref_name}, {ok[0]['n_vignette_px']} vignette px")

    groups: dict[bytes, list[str]] = {}
    for s in ok:
        packed = np.load(OUT_DIR / f"{s['name']}_vignette_mask.npy")
        key = packed.tobytes()
        groups.setdefault(key, []).append(s["name"])

    log(f"number of DISTINCT masks across {len(ok)} sequences: {len(groups)}")
    if len(groups) == 1:
        log("RESULT: all 169 per-sequence vignette masks are IDENTICAL. This is the global mask.")
    else:
        log("RESULT: masks DIFFER. Reporting groups:")
        for key, members in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            packed = np.frombuffer(key, dtype=np.uint8)
            mask = np.unpackbits(packed)[: ok[0]["shape"][0] * ok[0]["shape"][1]].reshape(ok[0]["shape"])
            n = int(mask.sum())
            c1_count = sum(1 for m in members if m.startswith("c1_"))
            c2_count = sum(1 for m in members if m.startswith("c2_"))
            v1_count = sum(1 for m in members if m.endswith("_v1"))
            v2_count = sum(1 for m in members if m.endswith("_v2"))
            v3_count = sum(1 for m in members if m.endswith("_v3"))
            v4_count = sum(1 for m in members if m.endswith("_v4"))
            log(f"  group with {n} vignette px, {len(members)} sequences "
                f"(c1={c1_count} c2={c2_count}, v1={v1_count} v2={v2_count} v3={v3_count} v4={v4_count}): "
                f"{members[:5]}{'...' if len(members) > 5 else ''}")
        # pixel-level diff vs reference for each other group
        ref_mask = np.unpackbits(ref_packed)[: ok[0]["shape"][0] * ok[0]["shape"][1]].reshape(ok[0]["shape"])
        for key, members in groups.items():
            if members[0] == ref_name:
                continue
            packed = np.frombuffer(key, dtype=np.uint8)
            mask = np.unpackbits(packed)[: ok[0]["shape"][0] * ok[0]["shape"][1]].reshape(ok[0]["shape"])
            diff = mask != ref_mask
            log(f"  group starting with {members[0]}: differs from reference ({ref_name}) "
                f"at {int(diff.sum())} pixels")

    import json
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log(f"\nwrote {OUT_DIR / 'summary.json'}")
    log("ALL DONE")


if __name__ == "__main__":
    main()
