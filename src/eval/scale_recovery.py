"""Scale recovery, `docs/eval_protocol.md` section 1. Both steps use GT,
for every predicted configuration -- every number this protocol produces
is therefore an upper bound on deployed performance, not a deployability
estimate (section 0, restated at the point of use here as instructed).

**Depth**: a single global per-sequence factor, median-of-per-frame-medians
(NOT a single ratio pooled across all pixels of all frames -- a different,
not-comparable number). For each frame with GT depth available, compute
`median(GT depth) / median(pred depth)` over GT-valid pixels, then take
the median of those per-frame ratios.

**Pose**: Umeyama (rotation + translation + scale) alignment of the
predicted camera trajectory's positions to GT's, fit over the *entire*
sequence (not a window). The fitted `(R_u, t_u, s_pose)` is then applied
to every predicted per-frame `(R_i, t_i)` via the world-frame anchoring
formula (`aligned_predicted_pose` below) -- required, not optional
bookkeeping, since a bare scalar multiply by `s_pose` only fixes the
*scale* of the predicted trajectory's arbitrary reference frame, not its
position/orientation relative to the mesh's real-world coordinates.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DepthScale:
    median: float  # the sequence's single global depth scale factor
    per_frame_ratios: list[float]


def compute_depth_scale(gt_depth_mm_frames, gt_valid_frames, pred_depth_native_frames) -> DepthScale:
    """`gt_depth_mm_frames`, `gt_valid_frames`, `pred_depth_native_frames`:
    equal-length iterables of `(H, W)` or flat arrays, one per frame --
    GT depth in mm, GT-valid mask (`raw != 0` and `raw != 65535`), and the
    method's own native-scale predicted depth, all for the SAME frame and
    pixel grid.

    Returns the per-sequence global scale (median of per-frame
    `median(GT)/median(pred)` ratios, both restricted to GT-valid pixels).
    A pred_depth of exactly 0 at every GT-valid pixel in a frame (never
    observed in practice, but not excluded here) would make that frame's
    ratio undefined -- fails loudly (raises) rather than silently skipping
    the frame, per `CLAUDE.md`'s "no silent fallbacks" rule.
    """
    ratios = []
    for gt_mm, valid, pred_native in zip(gt_depth_mm_frames, gt_valid_frames, pred_depth_native_frames):
        gt_mm = np.asarray(gt_mm).ravel()
        valid = np.asarray(valid).ravel()
        pred_native = np.asarray(pred_native).ravel()
        if not valid.any():
            raise ValueError("a frame has zero GT-valid pixels -- cannot compute its depth ratio")
        pred_median = float(np.median(pred_native[valid]))
        if pred_median == 0.0:
            raise ValueError("predicted depth median is exactly 0 for a GT-valid frame -- undefined ratio")
        ratios.append(float(np.median(gt_mm[valid])) / pred_median)
    return DepthScale(median=float(np.median(ratios)), per_frame_ratios=ratios)


@dataclass
class PoseAlignment:
    R_u: np.ndarray  # (3, 3)
    t_u: np.ndarray  # (3,)
    s_pose: float


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Umeyama (1991) similarity transform: finds `(R, t, c)` minimizing
    `sum(||dst_i - (c*R@src_i + t)||^2)`. `src`, `dst`: `(N, 3)` matched
    point sets. Same algorithm already validated in this project
    (`scratch/pipelines/endodac_fusion_check.py`, cross-checked there
    against the independently-saved `s_pose=557.87`)."""
    N = src.shape[0]
    mu_src, mu_dst = src.mean(axis=0), dst.mean(axis=0)
    src_c, dst_c = src - mu_src, dst - mu_dst
    var_src = (src_c**2).sum() / N
    sigma = (dst_c.T @ src_c) / N
    U, D, Vt = np.linalg.svd(sigma)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    c = np.trace(np.diag(D) @ S) / var_src
    t = mu_dst - c * R @ mu_src
    return R, t, c


def compute_pose_alignment(pred_positions: np.ndarray, gt_positions: np.ndarray) -> PoseAlignment:
    """`pred_positions`, `gt_positions`: `(n_frames, 3)` camera-center
    positions, same frame order, full trajectory (not a window)."""
    R_u, t_u, s_pose = umeyama(pred_positions, gt_positions)
    return PoseAlignment(R_u=R_u, t_u=t_u, s_pose=float(s_pose))


def aligned_predicted_pose(alignment: PoseAlignment, R_i: np.ndarray, t_i: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """World-frame anchoring formula, `docs/eval_protocol.md` section 1:
    `R_world_i = R_u @ R_i`, `T_world_i = s_pose * (R_u @ t_i) + t_u`.
    `R_i`, `t_i`: the predicted camera's own per-frame rotation/translation
    (camera-to-world, in the pipeline's own arbitrary reference frame)."""
    R_world = alignment.R_u @ R_i
    T_world = alignment.s_pose * (alignment.R_u @ t_i) + alignment.t_u
    return R_world, T_world
