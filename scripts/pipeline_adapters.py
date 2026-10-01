"""Pipeline adapters for the generalized evaluation driver
(scripts/eval_pipeline_sequence.py). An adapter exposes a pipeline's own
saved per-frame outputs in one shape, so the evaluation itself (scale
recovery, fusion, every metric) is the same code for every pipeline.

Interface, per frame index i of the GT sequence (0 .. n_gt_frames-1):
  - depth_native(i) -> (d_native (n_rays,) float64, available (n_rays,) bool)
    on the 1350x1080 input grid, flattened row-major; or None when the
    pipeline produced no depth for frame i.
  - pose(i) -> (R_i (3,3), t_i (3,)) camera-to-world in the pipeline's own
    arbitrary frame; or None when the pipeline produced no pose for frame i.
  - frames_with_depth / frames_with_pose: sorted lists of frame indices.

docs/eval_protocol.md, "2026-09-29: Missing predictions, internal crops,
confidence maps" is applied here and in the driver, not in locked code:
  - a missing frame is reported as None and never filled or interpolated;
  - a pixel with no prediction is `available=False` (treated exactly like
    d_pred unavailable by src/eval/fusion.py);
  - depth on a pipeline's internal grid is mapped to 1350x1080 by that
    pipeline's documented mapping with bilinear interpolation that never
    reads a grid cell outside the prediction;
  - confidence maps are never read.

EndoDACAdapter reproduces scripts/eval_d1_sequence.py's loading exactly
(same files, same dtype conversions), so the EndoDAC path through the new
driver is bit-identical to D1 Stage B (pre-flight 1 checks this).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

REPO = Path("/data1_ycao/chua/projects/mrsp")
ENDODAC_ROOT = REPO / "results/pipelines/endodac_full_run"
MAST3R_ROOT = REPO / "results/pipelines/mast3r_slam_full_run"
# Stage 2's documented, round-trip-tested resize/crop mapping
# (docs/pipelines/mast3r_slam.md, Stage 2 step 5). Identical for every
# sequence: fixed 1350x1080 input and MASt3R's fixed 512 px target.
MAST3R_MAPPING_JSON = REPO / "results/pipelines/mast3r_slam_perframe/c1_cecum_t1_v1/resolution_mapping.json"
ORIGINAL_W, ORIGINAL_H = 1350, 1080


class EndoDACAdapter:
    name = "endodac"

    def __init__(self, sequence: str, n_gt_frames: int):
        self.sequence = sequence
        self.dir = ENDODAC_ROOT / sequence
        self.n_gt_frames = n_gt_frames
        # Exactly scripts/eval_d1_sequence.py's loading: every frame
        # preloaded, .ravel().astype(np.float64).
        self._depth = [
            np.load(self.dir / "depth" / f"{i:04d}_pred_depth.npy").ravel().astype(np.float64)
            for i in range(n_gt_frames)
        ]
        self._poses = np.load(self.dir / "poses_pred.npy")
        if len(self._poses) != n_gt_frames:
            raise RuntimeError(f"{sequence}: EndoDAC poses {len(self._poses)} != GT frames {n_gt_frames}")
        self._available = np.ones(len(self._depth[0]), dtype=bool)
        self.frames_with_depth = list(range(n_gt_frames))
        self.frames_with_pose = list(range(n_gt_frames))

    def depth_native(self, i: int):
        return self._depth[i], self._available

    def pose(self, i: int):
        return self._poses[i, :3, :3], self._poses[i, :3, 3]

    def descriptives(self) -> dict:
        return {"internal_grid": None}


# --------------------------------------------------------------------------
# MASt3R-SLAM
# --------------------------------------------------------------------------

def load_mast3r_mapping() -> dict:
    return json.loads(MAST3R_MAPPING_JSON.read_text())["resolution_mapping"]


def original_to_model(ox: np.ndarray, oy: np.ndarray, m: dict) -> tuple[np.ndarray, np.ndarray]:
    """Stage 2's documented forward mapping, original 1350x1080 pixel ->
    MASt3R grid pixel (scripts/mast3r_slam_resolution_mapping.py
    ::original_to_model, same formula; that script imports MASt3R's
    torch-side code lazily, so the formula is restated here and checked
    equal to the imported one in pre-flight 2)."""
    px = ox / m["scale_w"] - m["half_crop_w"]
    py = oy / m["scale_h"] - m["half_crop_h"]
    return px, py


class BilinearGridMap:
    """Precomputed bilinear sampling of a (grid_h, grid_w) array at every
    1350x1080 pixel's mapped coordinate. A pixel is available iff its mapped
    coordinate lies in [0, grid_w-1] x [0, grid_h-1]: then all four bilinear
    neighbours are grid cells that hold a prediction. Outside that box
    (the vertical crop band, and the sub-pixel strip past the last grid
    column/row) no neighbour set exists without extrapolating, so the pixel
    is unavailable."""

    def __init__(self, m: dict):
        self.m = m
        gw, gh = int(m["model_grid_w"]), int(m["model_grid_h"])
        self.grid_w, self.grid_h = gw, gh
        xs, ys = np.meshgrid(np.arange(ORIGINAL_W, dtype=np.float64), np.arange(ORIGINAL_H, dtype=np.float64))
        px, py = original_to_model(xs.ravel(), ys.ravel(), m)
        self.available = (px >= 0) & (px <= gw - 1) & (py >= 0) & (py <= gh - 1)
        pxa, pya = px[self.available], py[self.available]
        x0 = np.clip(np.floor(pxa).astype(np.int64), 0, gw - 2)
        y0 = np.clip(np.floor(pya).astype(np.int64), 0, gh - 2)
        self.wx = pxa - x0
        self.wy = pya - y0
        self.x0, self.y0 = x0, y0
        if not ((self.wx >= 0) & (self.wx <= 1) & (self.wy >= 0) & (self.wy <= 1)).all():
            raise RuntimeError("bilinear weights outside [0, 1] -- mapping or clipping is wrong")

    def sample(self, grid: np.ndarray) -> np.ndarray:
        """grid: (grid_h, grid_w). Returns (1350*1080,) float64, NaN where
        unavailable."""
        if grid.shape != (self.grid_h, self.grid_w):
            raise RuntimeError(f"grid shape {grid.shape} != ({self.grid_h}, {self.grid_w})")
        g = grid.astype(np.float64)
        x0, y0, wx, wy = self.x0, self.y0, self.wx, self.wy
        v = (
            g[y0, x0] * (1 - wx) * (1 - wy)
            + g[y0, x0 + 1] * wx * (1 - wy)
            + g[y0 + 1, x0] * (1 - wx) * wy
            + g[y0 + 1, x0 + 1] * wx * wy
        )
        out = np.full(ORIGINAL_W * ORIGINAL_H, np.nan)
        out[self.available] = v
        return out


def bilinear_reference(grid: np.ndarray, px: float, py: float) -> float:
    """Independent scalar bilinear lookup, for the pre-flight round-trip
    check only. Written separately from BilinearGridMap on purpose."""
    gh, gw = grid.shape
    if not (0 <= px <= gw - 1 and 0 <= py <= gh - 1):
        return float("nan")
    xl = int(np.floor(px)); yl = int(np.floor(py))
    xh = min(xl + 1, gw - 1); yh = min(yl + 1, gh - 1)
    fx = px - xl; fy = py - yl
    top = (1 - fx) * float(grid[yl, xl]) + fx * float(grid[yl, xh])
    bot = (1 - fx) * float(grid[yh, xl]) + fx * float(grid[yh, xh])
    return (1 - fy) * top + fy * bot


class Mast3rSlamAdapter:
    """Reads results/pipelines/mast3r_slam_full_run/<seq>/ (Stage 3).

    Depth: depth/<frame:04d>.npz key "z" (camera-frame Z at MASt3R's
    512x400 grid). Key "conf" is never read. A frame with no .npz has no
    depth. Non-finite z inside a saved grid fails loudly (malformed input,
    not a missing prediction).

    Pose: poses_per_frame.csv rows with reconstructed == "True"; (tx,ty,tz,
    qx,qy,qz,qw) is lietorch SE3 data order, camera-to-world (Stage 1 Q3).
    A frame with no row, or a row with reconstructed == "False", has no
    pose. The GT frame count, not the CSV row count, defines the frame
    range: frames the tracker never reported (e.g. relocalization
    failures) are missing poses."""

    name = "mast3r_slam"

    def __init__(self, sequence: str, n_gt_frames: int, grid_map: BilinearGridMap | None = None):
        self.sequence = sequence
        self.dir = MAST3R_ROOT / sequence
        self.n_gt_frames = n_gt_frames
        self.grid_map = grid_map if grid_map is not None else BilinearGridMap(load_mast3r_mapping())

        depth_files = {int(p.stem): p for p in (self.dir / "depth").glob("*.npz")}
        extra = sorted(k for k in depth_files if not (0 <= k < n_gt_frames))
        if extra:
            raise RuntimeError(f"{sequence}: depth files outside GT frame range: {extra[:10]}")
        self._depth_files = depth_files
        self.frames_with_depth = sorted(depth_files)

        self._poses: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.pose_rows_not_reconstructed: list[int] = []
        with open(self.dir / "poses_per_frame.csv") as f:
            for row in csv.DictReader(f):
                fid = int(row["frame_id"])
                if not (0 <= fid < n_gt_frames):
                    raise RuntimeError(f"{sequence}: pose row frame_id {fid} outside GT range")
                if fid in self._poses:
                    raise RuntimeError(f"{sequence}: duplicate pose row for frame {fid}")
                if row["reconstructed"] != "True":
                    self.pose_rows_not_reconstructed.append(fid)
                    continue
                vals = np.array([float(row[k]) for k in ["tx", "ty", "tz", "qx", "qy", "qz", "qw"]])
                if not np.isfinite(vals).all():
                    raise RuntimeError(f"{sequence}: non-finite reconstructed pose at frame {fid}")
                R = Rotation.from_quat(vals[3:7]).as_matrix()  # scipy: scalar-last (x, y, z, w)
                self._poses[fid] = (R, vals[:3].copy())
        self.frames_with_pose = sorted(self._poses)
        self._cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def depth_grid(self, i: int) -> np.ndarray | None:
        p = self._depth_files.get(i)
        if p is None:
            return None
        with np.load(p) as npz:
            z = npz["z"]
        if not np.isfinite(z).all():
            raise RuntimeError(f"{self.sequence}: non-finite MASt3R z in frame {i}")
        return z

    def depth_native(self, i: int):
        # Cached: the driver reads each frame up to four times (scale recovery,
        # two pose variants, no-depth count). Same array every time; values are
        # unchanged by caching (checked against an uncached run).
        if i in self._cache:
            return self._cache[i]
        z = self.depth_grid(i)
        if z is None:
            return None
        out = (self.grid_map.sample(z), self.grid_map.available)
        self._cache[i] = out
        return out

    def pose(self, i: int):
        return self._poses.get(i)

    def descriptives(self) -> dict:
        return {
            "internal_grid": [self.grid_map.grid_h, self.grid_map.grid_w],
            "pose_rows_not_reconstructed": self.pose_rows_not_reconstructed,
        }


ADAPTERS = {"endodac": EndoDACAdapter, "mast3r_slam": Mast3rSlamAdapter}
