"""Face-set comparison metrics used to validate `src/eval/`'s configurations
against their `docs/eval_protocol.md` section 6 targets. Region-level
metrics (`docs/success_criteria.md` section 1) are Stage 3's job; this
module is only the face-set-level code-correctness check.
"""
from __future__ import annotations

import numpy as np


def face_set_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Two `(n_faces,)` bool arrays -> IoU. NaN if the union is empty (no
    faces in either set -- an undefined ratio, not zero)."""
    inter = np.sum(a & b)
    union = np.sum(a | b)
    return float(inter / union) if union else float("nan")


def face_set_disagreement(a: np.ndarray, b: np.ndarray) -> tuple[int, int]:
    """`a`=predicted/candidate, `b`=target/reference. Returns
    `(a_not_b, b_not_a)` raw counts -- mirrors `scripts/visibility_full.py`'s
    `false_observed`/`false_unobserved` reporting style."""
    return int(np.sum(a & ~b)), int(np.sum(~a & b))
