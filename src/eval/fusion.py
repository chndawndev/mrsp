"""General ray-cast + tau-gated depth-agreement fusion, `docs/eval_protocol.md`
sections 2 (procedure), 2a (evaluable pixel), and 3 (ray misses) --
generalized beyond `src/eval/oracle.py`'s GT-only version to accept an
arbitrary camera-to-world pose and an arbitrary (already-scaled) depth
source per frame, per section 4's four configurations via post-hoc
substitution.

Deliberately NOT a refactor of `src/eval/oracle.py`: that module is
Stage 2's locked, already-validated artifact (`CLAUDE.md`'s hard rule --
do not modify locked eval code without explicit approval, even to
consolidate). This module duplicates a small amount of the same
ray-cast/evaluable/tau-gate logic rather than risk Stage 2/3's validated
numbers; `tests/eval/test_fusion.py` cross-checks that running the oracle
configuration (GT depth + GT pose) through THIS module reproduces
`run_oracle_sequence`'s numbers exactly on an identical synthetic
scenario, as the correctness check for that duplication.

Design point from `docs/eval_protocol.md` section 7: ray-casting depends
only on `(frame, pose-variant)`, not on `(frame, configuration)` -- two
pose variants (GT pose, aligned predicted pose), not four, "tau is
applied post-hoc to cached hit distances, exactly as instructed, at zero
extra ray-casting cost." `run_pose_variant_sequence` takes ONE pose
provider but a whole dict of depth providers, casting rays once per frame
and evaluating every depth source (and every tau) against that same
cached ray-cast -- so the Stage 4 driver calls it exactly twice per
sequence (once per pose variant), covering all four configurations with
two ray-casting passes, not four.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gt.rasterizer import build_mesh, cast_frame, make_raycast_frame_kernel

from .evaluable import evaluable_pixel_mask


@dataclass
class ConfigurationResult:
    n_faces: int
    n_frames: int
    n_rays_per_frame: int
    taus: list[float]
    predicted_observed: dict[float, np.ndarray]  # tau -> (n_faces,) bool, OR-accumulated
    ray_miss_count: int  # section 3 -- shared across every depth source at this pose variant
    evaluable_count: int  # section 2a -- shared across every depth source at this pose variant
    d_pred_unavailable_count: int  # depends on this specific depth source
    tau_reject_count: dict[float, int]  # depends on this specific depth source
    n_rays_total: int


def run_pose_variant_sequence(
    wp,
    device: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    n_frames: int,
    pose_provider,  # callable(frame_idx) -> (R_c2w (3,3), T_c2w (3,))
    depth_providers: dict[str, object],  # name -> callable(frame_idx) -> (d_pred_mm (n_rays,) float64, d_pred_valid (n_rays,) bool)
    cam_rays: np.ndarray,
    vignette_mask: np.ndarray,
    taus: list[float],
) -> dict[str, ConfigurationResult]:
    """One pose variant, ray-cast once per frame, evaluated against every
    entry of `depth_providers` (and every tau) from that same cached
    ray-cast. Returns one `ConfigurationResult` per `depth_providers` key.
    """
    n_faces = len(faces)
    n_rays = len(cam_rays)
    names = list(depth_providers.keys())

    kernel = make_raycast_frame_kernel(wp)
    wp_mesh = build_mesh(wp, vertices, faces, device)
    unused_accumulate_gpu = wp.zeros(n_faces, dtype=wp.int32, device=device)

    predicted_observed = {name: {tau: np.zeros(n_faces, dtype=bool) for tau in taus} for name in names}
    d_pred_unavailable_count = {name: 0 for name in names}
    tau_reject_count = {name: {tau: 0 for tau in taus} for name in names}
    ray_miss_count = 0
    evaluable_count = 0

    for frame_idx in range(n_frames):
        R_c2w, T_c2w = pose_provider(frame_idx)

        hit_gpu, face_gpu, dist_gpu = cast_frame(
            wp, device, kernel, wp_mesh, cam_rays, R_c2w, T_c2w, unused_accumulate_gpu
        )
        wp.synchronize()
        hit = hit_gpu.numpy().astype(bool)
        face = face_gpu.numpy()
        d_hit = dist_gpu.numpy().astype(np.float64)

        ray_miss_count += int((~hit).sum())

        evaluable = evaluable_pixel_mask(hit, d_hit, vignette_mask)
        evaluable_count += int(evaluable.sum())

        for name in names:
            d_pred, d_pred_valid = depth_providers[name](frame_idx)

            usable = evaluable & d_pred_valid
            d_pred_unavailable_count[name] += int((evaluable & ~d_pred_valid).sum())

            rel_err = np.full(n_rays, np.inf)
            rel_err[usable] = np.abs(d_pred[usable] - d_hit[usable]) / d_hit[usable]

            for tau in taus:
                tau_pass = usable & (rel_err < tau)
                tau_reject_count[name][tau] += int((usable & ~tau_pass).sum())
                np.logical_or.at(predicted_observed[name][tau], face[tau_pass], True)

    return {
        name: ConfigurationResult(
            n_faces=n_faces,
            n_frames=n_frames,
            n_rays_per_frame=n_rays,
            taus=list(taus),
            predicted_observed=predicted_observed[name],
            ray_miss_count=ray_miss_count,
            evaluable_count=evaluable_count,
            d_pred_unavailable_count=d_pred_unavailable_count[name],
            tau_reject_count=tau_reject_count[name],
            n_rays_total=n_rays * n_frames,
        )
        for name in names
    }
