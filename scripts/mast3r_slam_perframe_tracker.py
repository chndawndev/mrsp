#!/usr/bin/env python
"""InstrumentedFrameTracker: a wrapper subclass around MASt3R-SLAM's own
`mast3r_slam.tracker.FrameTracker`, for Stage 2 (docs/pipelines/
mast3r_slam.md) per-frame output. Does NOT edit the vendored
FrameTracker in place -- this is a new file that subclasses it and copies
`track()`'s body verbatim (read directly from scratch/pipelines/
MASt3R-SLAM/mast3r_slam/tracker.py), adding only a `self.records.append`
at the two points the method already returns: the "Skipped frame" early
return, and the normal end-of-track return.

Every quantity recorded already exists as a local variable inside the
original `track()`:
  - `keyframe.frame_id` (the reference keyframe's original input-frame
    index -- read at the very top, before any of this frame's own
    tracking/keyframe-promotion logic runs).
  - `T_CkCf = T_WCk.inv() * T_WCf` -- exactly the task's
    `T_ref_to_frame = inv(T_WC_ref) @ T_WC_frame` (docs/pipelines/
    mast3r_slam.md Stage 2 step 2). Recorded as the raw 8-float Sim3
    embedded vector (tx,ty,tz,qx,qy,qz,qw,s) -- scale is NOT dropped
    here, only at the final SE3 conversion step
    (scripts/mast3r_slam_reconstruct_poses.py), matching
    `mast3r_slam/lietorch_utils.py::as_SE3`'s own convention.
  - `match_frac`, `match_frac_k` -- MASt3R-SLAM's own tracking-quality
    signals, used internally for the skip-frame and new-keyframe
    decisions respectively.
"""
from __future__ import annotations

import torch
from mast3r_slam.frame import Frame
from mast3r_slam.geometry import (
    act_Sim3,
    point_to_ray_dist,
    get_pixel_coords,
    constrain_points_to_ray,
    project_calib,
)
from mast3r_slam.nonlinear_optimizer import check_convergence, huber
from mast3r_slam.config import config
from mast3r_slam.mast3r_utils import mast3r_match_asymmetric
from mast3r_slam.tracker import FrameTracker


class InstrumentedFrameTracker(FrameTracker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records: list[dict] = []

    def track(self, frame: Frame):
        keyframe = self.keyframes.last_keyframe()
        ref_keyframe_frame_id = keyframe.frame_id

        idx_f2k, valid_match_k, Xff, Cff, Qff, Xkf, Ckf, Qkf = mast3r_match_asymmetric(
            self.model, frame, keyframe, idx_i2j_init=self.idx_f2k
        )
        self.idx_f2k = idx_f2k.clone()

        idx_f2k = idx_f2k[0]
        valid_match_k = valid_match_k[0]

        Qk = torch.sqrt(Qff[idx_f2k] * Qkf)

        frame.update_pointmap(Xff, Cff)

        use_calib = config["use_calib"]
        img_size = frame.img.shape[-2:]
        if use_calib:
            K = keyframe.K
        else:
            K = None

        Xf, Xk, T_WCf, T_WCk, Cf, Ck, meas_k, valid_meas_k = self.get_points_poses(
            frame, keyframe, idx_f2k, img_size, use_calib, K
        )

        valid_Cf = Cf > self.cfg["C_conf"]
        valid_Ck = Ck > self.cfg["C_conf"]
        valid_Q = Qk > self.cfg["Q_conf"]

        valid_opt = valid_match_k & valid_Cf & valid_Ck & valid_Q
        valid_kf = valid_match_k & valid_Q

        match_frac = float(valid_opt.sum() / valid_opt.numel())
        if match_frac < self.cfg["min_match_frac"]:
            print(f"Skipped frame {frame.frame_id}")
            self.records.append({
                "frame_id": frame.frame_id,
                "ref_keyframe_frame_id": ref_keyframe_frame_id,
                "T_ref_to_frame": None,
                "match_frac": match_frac,
                "match_frac_k": None,
                "skipped": True,
            })
            return False, [], True

        try:
            if not use_calib:
                T_WCf, T_CkCf = self.opt_pose_ray_dist_sim3(
                    Xf, Xk, T_WCf, T_WCk, Qk, valid_opt
                )
            else:
                T_WCf, T_CkCf = self.opt_pose_calib_sim3(
                    Xf, Xk, T_WCf, T_WCk, Qk, valid_opt, meas_k, valid_meas_k, K, img_size,
                )
        except Exception:
            print(f"Cholesky failed {frame.frame_id}")
            self.records.append({
                "frame_id": frame.frame_id,
                "ref_keyframe_frame_id": ref_keyframe_frame_id,
                "T_ref_to_frame": None,
                "match_frac": match_frac,
                "match_frac_k": None,
                "skipped": True,
            })
            return False, [], True

        frame.T_WC = T_WCf

        Xkk = T_CkCf.act(Xkf)
        keyframe.update_pointmap(Xkk, Ckf)
        self.keyframes[len(self.keyframes) - 1] = keyframe

        n_valid = valid_kf.sum()
        match_frac_k = float(n_valid / valid_kf.numel())
        unique_frac_f = (
            torch.unique(idx_f2k[valid_match_k[:, 0]]).shape[0] / valid_kf.numel()
        )

        new_kf = min(match_frac_k, unique_frac_f) < self.cfg["match_frac_thresh"]

        if new_kf:
            self.reset_idx_f2k()

        self.records.append({
            "frame_id": frame.frame_id,
            "ref_keyframe_frame_id": ref_keyframe_frame_id,
            "T_ref_to_frame": T_CkCf.data.detach().cpu().numpy().reshape(-1).tolist(),
            "match_frac": match_frac,
            "match_frac_k": match_frac_k,
            "skipped": False,
        })

        return (
            new_kf,
            [
                keyframe.X_canon,
                keyframe.get_average_conf(),
                frame.X_canon,
                frame.get_average_conf(),
                Qkf,
                Qff,
            ],
            False,
        )
