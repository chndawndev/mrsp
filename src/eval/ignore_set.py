"""The ignore set, `docs/eval_protocol.md` section 6 ("The ignore set --
ADOPTED (2026-09-21, approved by Chen)"). Computed once per sequence, from
GT pose alone, method-independent, the same set for every configuration
and every pipeline evaluated on that sequence.

**Scope, resolved 2026-09-22 (approved by Chen) after Stage 3 uncovered a
real inconsistency between two sentences of section 6**: the definition
sentence reads "the set of faces never reachable through an evaluable
pixel under GT pose" (unrestricted -- would include GT-unobserved faces
too, since a genuinely-missed face is also never reachable), but the same
paragraph calls this "exactly the 'gap' set already measured in
`docs/eval_protocol_oracle_gap.md` Part 1" -- and Part 1's gap was
explicitly `gt_observed & ~ever_evaluable` (restricted to GT-OBSERVED
faces only). These are not the same set: on `c1_cecum_t1_v1`, 99.5% of
GT-unobserved faces (63,814 of 64,155) are themselves unreachable, so the
unrestricted reading swallows almost all of GT-unobserved into the
ignore set, leaving nothing to validate against (measured: false alarm
rate degenerates to 1.000000, predicted components collapse to 0-1,
instead of matching GT regions). The restricted reading (implemented
below) reproduces the already-validated section 6 evidence table almost
exactly (false alarm rate ~0.00001, not 1.0; 3/3 matching headline
components on `c1_cecum_t1_v1`) -- confirmed by direct measurement before
this fix was applied, not assumed.

Excluded from false alarm rate and predicted-unobserved connected-component
construction (Stage 3's job); GT regions (`coverage_mesh.obj`'s own `vt`
flags) are never touched by it.
"""
from __future__ import annotations

import numpy as np


def compute_ignore_set(gt_observed: np.ndarray, ever_evaluable_hit: np.ndarray) -> np.ndarray:
    """`gt_observed`: `(n_faces,)` bool, the released mesh's own `vt`-flag
    observed set. `ever_evaluable_hit`: `(n_faces,)` bool, True if the face
    was ever hit by an evaluable ray under GT pose, any frame
    (`src/eval/oracle.py`'s `OracleSequenceResult.ever_evaluable_hit`) --
    also, definitionally, the evaluable-pixel-restricted GT rasterization
    used as the oracle configuration's own section 6 validation target.

    Returns the ignore set: `gt_observed & ~ever_evaluable_hit` -- the
    literal Part 1 "gap" (GT-observed faces the released mesh credits as
    observed but that no evaluable ray, under GT pose, can ever reach).
    Faces that are GT-unobserved and also unreachable (the common,
    expected case for a genuinely missed region) are NOT in the ignore
    set -- see module docstring for why the unrestricted reading breaks
    the section 6 validation this set exists to support.
    """
    return gt_observed & ~ever_evaluable_hit
