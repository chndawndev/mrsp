"""The ignore set, `docs/eval_protocol.md` section 6 ("The ignore set --
ADOPTED (2026-09-21, approved by Chen)"). Faces never reachable through an
evaluable pixel (section 2a) under GT pose -- computed once per sequence,
from GT pose alone, method-independent, the same set for every
configuration and every pipeline evaluated on that sequence.

Excluded from false alarm rate and predicted-unobserved connected-component
construction (Stage 3's job); GT regions (`coverage_mesh.obj`'s own `vt`
flags) are never touched by it.
"""
from __future__ import annotations

import numpy as np


def compute_ignore_set(ever_evaluable_hit: np.ndarray) -> np.ndarray:
    """`ever_evaluable_hit`: `(n_faces,)` bool, True if the face was ever
    hit by an evaluable ray under GT pose, any frame -- this is also,
    definitionally, the evaluable-pixel-restricted GT rasterization used as
    the oracle configuration's own section 6 validation target
    (`src/eval/oracle.py`'s `OracleSequenceResult.ever_evaluable_hit`).

    Returns the ignore set: its complement. A one-line function so the
    definition -- "the ignore set is the complement of the
    evaluable-pixel-restricted GT rasterization" -- is stated once, by
    name, rather than inlined at each call site.
    """
    return ~ever_evaluable_hit
