"""Oracle configuration (GT depth + GT pose) end-to-end, `docs/eval_protocol.md`
sections 1 (N/A -- the oracle needs no scale recovery, since it isn't a
prediction), 2 (ray-cast + tau-gated depth agreement), 2a (evaluable
pixel), and 3 (ray misses). Predicted configurations (scale recovery,
post-hoc substitution) are Stage 4's job; this module only ever consumes
GT depth and GT pose.

Does NOT compute the ignore set itself (`src/eval/ignore_set.py` does,
from this module's `ever_evaluable_hit` plus `gt_observed` -- the latter
isn't otherwise needed by this module, and keeping the ignore-set
definition in one place avoided a real bug: an earlier version computed
`ignore_set = ~ever_evaluable_hit` here directly, which turned out to be
the wrong, unrestricted reading of section 6 -- see
`src/eval/ignore_set.py`'s docstring for the full diagnosis).

No file I/O, no GPU/device selection here -- callers (`scripts/`) load the
mesh/poses/depth and pick a GPU; this module is the locked procedure, not
the runnable entry point (`CLAUDE.md` layout: `src/eval/` holds metric
definitions, `scripts/` holds entry points).

Design point from `docs/eval_protocol.md` section 7: ray-casting happens
once per frame; all tau values are evaluated post-hoc from the same cached
`(hit, face, d_hit, d_pred)` arrays, at zero extra ray-casting cost -- this
is why `run_oracle_sequence` takes a list of taus and returns one
predicted-observed set per tau from a single frame loop, rather than
looping over the sequence once per tau.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gt.depth import depth_to_mm
from gt.rasterizer import build_mesh, cast_frame, make_raycast_frame_kernel

from .evaluable import evaluable_pixel_mask


@dataclass
class OracleSequenceResult:
    n_faces: int
    n_frames: int
    n_rays_per_frame: int
    taus: list[float]
    predicted_observed: dict[float, np.ndarray]  # tau -> (n_faces,) bool, OR-accumulated
    ever_evaluable_hit: np.ndarray  # (n_faces,) bool -- evaluable-pixel-restricted GT rasterization
    ray_miss_count: int  # section 3: rays that never hit the mesh at all, summed over all frames
    evaluable_count: int  # evaluable pixels (section 2a), summed over all frames
    d_pred_unavailable_count: int  # evaluable pixels where GT depth itself is raw==0/65535
    tau_reject_count: dict[float, int]  # per tau: usable pixels that failed the tau test
    n_rays_total: int  # n_rays_per_frame * n_frames


def run_oracle_sequence(
    wp,
    device: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    poses: np.ndarray,  # (n_frames, 4, 4) raw pose.txt reshape (CLAUDE.md "raw" convention)
    depth_frames,  # iterable of raw uint16 (H, W) depth arrays, pose.txt frame order
    cam_rays: np.ndarray,  # (n_rays, 3) unit camera rays, src/geometry/camera.py unproject()
    vignette_mask: np.ndarray,  # (n_rays,) bool, True = non-evaluable (src/eval/evaluable.py)
    taus: list[float],
) -> OracleSequenceResult:
    n_faces = len(faces)
    n_rays = len(cam_rays)

    kernel = make_raycast_frame_kernel(wp)
    wp_mesh = build_mesh(wp, vertices, faces, device)
    # cast_frame's accumulate slot is unused here (per-frame face-level
    # accumulation happens below, gated by evaluability + tau, not by
    # cast_frame's own unconditional hit/miss accumulation) -- allocated
    # once and passed through so cast_frame's signature stays a single
    # shared contract with src/gt/rasterizer.py's other callers.
    unused_accumulate_gpu = wp.zeros(n_faces, dtype=wp.int32, device=device)

    ever_evaluable_hit = np.zeros(n_faces, dtype=bool)
    predicted_observed = {tau: np.zeros(n_faces, dtype=bool) for tau in taus}

    ray_miss_count = 0
    evaluable_count = 0
    d_pred_unavailable_count = 0
    tau_reject_count = {tau: 0 for tau in taus}
    n_frames = 0

    for M, raw_depth in zip(poses, depth_frames):
        n_frames += 1
        R_c2w = M[:3, :3].T
        T_c2w = M[3, :3]

        hit_gpu, face_gpu, dist_gpu = cast_frame(
            wp, device, kernel, wp_mesh, cam_rays, R_c2w, T_c2w, unused_accumulate_gpu
        )
        wp.synchronize()
        hit = hit_gpu.numpy().astype(bool)
        face = face_gpu.numpy()
        d_hit = dist_gpu.numpy().astype(np.float64)

        ray_miss_count += int((~hit).sum())  # section 3: discard, count -- not folded into evaluable

        evaluable = evaluable_pixel_mask(hit, d_hit, vignette_mask)
        evaluable_count += int(evaluable.sum())

        d_pred, d_pred_valid = depth_to_mm(raw_depth.ravel())

        # section 2a: "predicted depth is masked with the same valid-pixel
        # mask before use" -- an evaluable pixel whose GT depth reading
        # itself is raw==0/65535 (rare: expected only right at the 100mm
        # boundary, where our own mesh ray-cast and the released depth
        # image's encoding can disagree by a hair -- see Stage 1's
        # silhouette-aliasing finding) has no usable d_pred, so it cannot
        # enter the tau test either way. Counted, not silently dropped,
        # mirroring section 3's ray-miss bookkeeping.
        usable = evaluable & d_pred_valid
        d_pred_unavailable_count += int((evaluable & ~d_pred_valid).sum())

        np.logical_or.at(ever_evaluable_hit, face[evaluable], True)

        rel_err = np.full(n_rays, np.inf)
        rel_err[usable] = np.abs(d_pred[usable] - d_hit[usable]) / d_hit[usable]

        for tau in taus:
            tau_pass = usable & (rel_err < tau)  # strict '<', docs/eval_protocol.md section 2
            tau_reject_count[tau] += int((usable & ~tau_pass).sum())
            np.logical_or.at(predicted_observed[tau], face[tau_pass], True)

    return OracleSequenceResult(
        n_faces=n_faces,
        n_frames=n_frames,
        n_rays_per_frame=n_rays,
        taus=list(taus),
        predicted_observed=predicted_observed,
        ever_evaluable_hit=ever_evaluable_hit,
        ray_miss_count=ray_miss_count,
        evaluable_count=evaluable_count,
        d_pred_unavailable_count=d_pred_unavailable_count,
        tau_reject_count=tau_reject_count,
        n_rays_total=n_rays * n_frames,
    )
