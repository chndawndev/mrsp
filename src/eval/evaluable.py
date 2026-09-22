"""Evaluable pixel gate, `docs/eval_protocol.md` section 2a.

A pixel is evaluable iff (1) it is outside the fixed 135-sequence-majority
vignette mask (`docs/eval_protocol.md` section 8) AND (2) the ray's hit
distance -- camera-frame Z-depth, `src/gt/rasterizer.py` -- is <=100mm. A
pixel whose ray misses the mesh entirely is never evaluable either
(section 3 handles misses separately -- discarded and counted, not folded
into "evaluable" one way or the other).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

MAX_HIT_DISTANCE_MM = 100.0

# The fixed camera valid-pixel mask: 135/169 registered sequences share this
# exact vignette mask (intersection of raw==0 across every frame of that
# sequence), 102,049px (6.9992%) -- docs/eval_protocol.md section 8,
# established in scratch/pipelines/oracle_gap_vignette_169.py (169-sequence
# comparison, ~28 min run, logs/oracle_gap_vignette_169.log). Any of the 135
# matching sequences' saved masks is bit-identical; c1_cecum_t1_v1's is used
# as the canonical copy here, the same file already referenced this way by
# scratch/pipelines/oracle_gap_part1_steps234.py and
# scratch/pipelines/oracle_gap_item3_vignette_only_169.py -- not
# recomputed, since results/ already holds this validated artifact and
# regenerating it costs ~28 minutes.
VIGNETTE_MASK_PATH = Path(
    "/data1_ycao/chua/projects/mrsp/results/pipelines/oracle_gap_vignette/"
    "c1_cecum_t1_v1_vignette_mask.npy"
)
EXPECTED_VIGNETTE_PIXEL_COUNT = 102049


def load_vignette_mask(width: int, height: int) -> np.ndarray:
    """Returns a flat `(width*height,)` bool array, True = vignette
    (non-evaluable).

    Fails loudly (no silent fallback, `CLAUDE.md` code style) if the loaded
    mask's pixel count doesn't match the frozen 102,049px convention -- this
    asset must never silently drift out from under locked eval code.
    """
    packed = np.load(VIGNETTE_MASK_PATH)
    mask = np.unpackbits(packed)[: width * height].astype(bool)
    if mask.size != width * height:
        raise RuntimeError(
            f"vignette mask at {VIGNETTE_MASK_PATH} unpacks to {mask.size} px, "
            f"expected {width * height} ({width}x{height})"
        )
    n = int(mask.sum())
    if n != EXPECTED_VIGNETTE_PIXEL_COUNT:
        raise RuntimeError(
            f"vignette mask at {VIGNETTE_MASK_PATH} has {n} px set, expected exactly "
            f"{EXPECTED_VIGNETTE_PIXEL_COUNT} (docs/eval_protocol.md section 8) -- refusing "
            "to proceed with a mask that doesn't match the frozen convention"
        )
    return mask


def evaluable_pixel_mask(hit: np.ndarray, d_hit: np.ndarray, vignette_mask: np.ndarray) -> np.ndarray:
    """hit, d_hit: this frame's per-pixel ray-cast output (`src/gt/rasterizer.py`,
    `cast_frame`'s `hit`/`dist` arrays). vignette_mask: fixed, frame-independent,
    True = non-evaluable (`load_vignette_mask`).

    Returns a bool array, True = evaluable (section 2a): a real mesh hit,
    outside the vignette, at <=100mm camera-frame Z-depth.
    """
    return hit & ~vignette_mask & (d_hit <= MAX_HIT_DISTANCE_MM)
