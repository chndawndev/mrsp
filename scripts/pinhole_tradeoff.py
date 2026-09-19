#!/usr/bin/env python
"""Quantify the cost of undistorting the omnidirectional camera to a pinhole
model, before choosing a pipeline input format. Diagnostic only -- does NOT
touch the frozen visibility pipeline (scripts/visibility_full.py) or any
frozen geometry code; the pinhole model here is local to this script.

For 3 sequences spanning segments (cecum, descending, sigmoid2), using GT
depth^Wpose (poses only -- coverage is a pure visibility/geometry question,
depth isn't needed) and the released coverage_mesh.obj:
  1. Derive a pinhole camera preserving as much FOV as practical -- found by
     an empirical perimeter-validity search (see below), not a naive
     corner-to-vignette-angle formula, which turned out to overshoot the
     actual captured frame (see chat log).
  2. Cast GT-criterion visibility twice per sequence: full omnidirectional
     model (reuses src/geometry/camera.py, already-validated), and the
     derived pinhole model restricted to real captured data.
  3. Report observed face fraction under each; connected components of
     faces observed under omni but NOT under pinhole ("newly lost"),
     filtered to >5mm equivalent diameter; and GT-unobserved >5mm regions'
     status under each (sanity check -- expected unchanged).
  4. Corner pixel resampling stretch factor (source-omni-pixel to
     destination-pinhole-pixel local magnification).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from geometry.camera import CameraIntrinsics, project, unproject  # noqa: E402
from geometry.coverage_mesh import stream_coverage_mesh_from_zip  # noqa: E402
from geometry.mesh_stats import area_to_diameter, connected_components_of_subset, face_adjacency, face_areas  # noqa: E402
from geometry.pose import stream_poses_from_zip  # noqa: E402

REGISTERED_DIR = Path("/data1_ycao/chua/datasets/C3VDv2/registered_videos")
INTRINSICS_PATH = "/data1_ycao/chua/datasets/C3VDv2/camera_intrinsics.txt"
OUT_DIR = Path("results/pinhole_tradeoff")

SEQUENCES = ["c2_cecum_t4_v1", "c1_descending_t1_v1", "c1_sigmoid2_t1_v1"]

VIGNETTE_RADIUS_PX = 719.81  # measured from depth TIFFs, see chat log (radius from true optical center cx,cy)
PINHOLE_F = 541.29  # empirically found: smallest f (widest FOV) whose full image perimeter stays
                     # within the captured frame AND the vignette circle -- see find_max_fov_f() below


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def find_max_fov_f(intr: CameraIntrinsics, vignette_r: float, width: int, height: int, n_perimeter: int = 400) -> float:
    """Empirical search (not used at runtime here, f is hardcoded above from
    this search -- kept for reproducibility/inspection): binary-searches the
    smallest pinhole focal length f such that a `width` x `height` pinhole
    image, principal point at image center, has its ENTIRE perimeter mapping
    (via the omni model's project()) to pixel positions that are both inside
    the actual captured frame [0,width)x[0,height) and inside the vignette
    circle of radius `vignette_r` around (cx,cy). A naive corner-only match
    to the vignette angle is insufficient: the omni model's stretch matrix
    makes it mildly anisotropic, and edge midpoints (not just corners) can
    require source pixels outside the captured frame even when the corner
    constraint alone is satisfied -- confirmed empirically before adopting
    this search.
    """
    cx_pin, cy_pin = width / 2, height / 2
    n = n_perimeter
    top = np.stack([np.linspace(0, width - 1, n), np.zeros(n)], axis=1)
    bottom = np.stack([np.linspace(0, width - 1, n), np.full(n, height - 1)], axis=1)
    left = np.stack([np.zeros(n), np.linspace(0, height - 1, n)], axis=1)
    right = np.stack([np.full(n, width - 1), np.linspace(0, height - 1, n)], axis=1)
    perimeter = np.concatenate([top, bottom, left, right], axis=0)

    def fully_valid(f: float) -> bool:
        X, Y = perimeter[:, 0] - cx_pin, perimeter[:, 1] - cy_pin
        cam_pts = np.stack([X, Y, np.full(len(X), f)], axis=1)
        omni_px = project(cam_pts, intr)
        r = np.hypot(omni_px[:, 0] - intr.cx, omni_px[:, 1] - intr.cy)
        in_frame = (omni_px[:, 0] >= 0) & (omni_px[:, 0] <= width - 1) & (omni_px[:, 1] >= 0) & (omni_px[:, 1] <= height - 1)
        return bool((in_frame & (r <= vignette_r)).all())

    lo, hi = 200.0, 800.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if fully_valid(mid):
            hi = mid
        else:
            lo = mid
    return hi


def pinhole_rays(width: int, height: int, f: float) -> np.ndarray:
    """(width*height, 3) unit camera-space ray directions for a pinhole
    model, principal point at image center, in the SAME (col, row)
    pixel-major order as geometry.camera.unproject's grid convention."""
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    X = (cols.ravel() - width / 2).astype(np.float64)
    Y = (rows.ravel() - height / 2).astype(np.float64)
    Z = np.full(X.shape, f)
    dirs = np.stack([X, Y, Z], axis=1)
    return dirs / np.linalg.norm(dirs, axis=1, keepdims=True)


def corner_stretch_factor(intr: CameraIntrinsics, f: float, width: int, height: int) -> float:
    """Local resampling magnification at the image corner: for a 1-pixel
    step in the OMNI source image (near the omni pixel corresponding to the
    pinhole's own corner), how many PINHOLE destination pixels does that
    step cover? >1 means the corner is stretched (magnified) when
    undistorting omni -> pinhole -- the classic wide-FOV correction
    artifact.
    """
    cx_pin, cy_pin = width / 2, height / 2
    X, Y = (width - 1) - cx_pin, (height - 1) - cy_pin
    omni_corner_px = project(np.array([[X, Y, f]]), intr)[0]

    eps = 1.0
    p0 = omni_corner_px
    p1 = omni_corner_px + np.array([eps, 0.0])

    ray0 = unproject(p0[None, :], intr)[0]
    ray1 = unproject(p1[None, :], intr)[0]

    def to_pinhole_px(ray):
        x, y, z = ray
        return np.array([x / z * f + cx_pin, y / z * f + cy_pin])

    pin0 = to_pinhole_px(ray0)
    pin1 = to_pinhole_px(ray1)
    pinhole_dist = np.linalg.norm(pin1 - pin0)
    return pinhole_dist / eps


def analyze_sequence(name: str, intr: CameraIntrinsics, omni_rays: np.ndarray, pin_rays: np.ndarray) -> dict:
    log(f"=== {name} ===")
    archive = REGISTERED_DIR / f"{name}.zip"
    mesh_data = stream_coverage_mesh_from_zip(archive)
    poses = stream_poses_from_zip(archive)
    n_faces = len(mesh_data.faces)
    gt_observed = mesh_data.face_observed
    areas = face_areas(mesh_data.vertices, mesh_data.faces)

    mesh = trimesh.Trimesh(vertices=mesh_data.vertices, faces=mesh_data.faces.astype(np.int32), process=False)
    intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)

    omni_observed = np.zeros(n_faces, dtype=bool)
    pin_observed = np.zeros(n_faces, dtype=bool)

    t0 = time.time()
    for M in poses:
        R, origin = M[:3, :3], M[3, :3]

        world_dirs_omni = omni_rays @ R
        origins = np.tile(origin, (len(omni_rays), 1))
        hit = intersector.intersects_first(origins, world_dirs_omni)
        omni_observed[hit[hit >= 0]] = True

        world_dirs_pin = pin_rays @ R
        hit = intersector.intersects_first(origins, world_dirs_pin)
        pin_observed[hit[hit >= 0]] = True
    log(f"  rasterized {len(poses)} frames x 2 models in {time.time()-t0:.1f}s")

    omni_frac = float(omni_observed.mean())
    pin_frac = float(pin_observed.mean())
    log(f"  observed face fraction: omni={omni_frac:.4f} pinhole={pin_frac:.4f}")

    adjacency = face_adjacency(mesh_data.vertices, mesh_data.faces)

    # Newly-lost: seen under full omni model, not seen under pinhole-restricted.
    newly_lost_mask = omni_observed & ~pin_observed
    newly_lost_components = connected_components_of_subset(n_faces, adjacency, newly_lost_mask)
    nl_areas = sorted((float(areas[c].sum()) for c in newly_lost_components), reverse=True)
    nl_over5mm = [a for a in nl_areas if area_to_diameter(a) > 5.0]
    log(f"  newly-lost (omni-observed, pinhole-unobserved): {int(newly_lost_mask.sum())} faces, "
        f"{len(newly_lost_components)} components, {len(nl_over5mm)} with equiv diam >5mm "
        f"(total area {sum(nl_over5mm):.1f}mm2)" if nl_over5mm else
        f"  newly-lost: {int(newly_lost_mask.sum())} faces, {len(newly_lost_components)} components, none >5mm")

    # GT-unobserved >5mm regions: sanity check status under each model.
    gt_unobserved_mask = ~gt_observed
    gt_components = connected_components_of_subset(n_faces, adjacency, gt_unobserved_mask)
    gt_over5mm = [c for c in gt_components if area_to_diameter(float(areas[c].sum())) > 5.0]
    n_flip_omni = sum(1 for c in gt_over5mm if omni_observed[c].any())
    n_flip_pin = sum(1 for c in gt_over5mm if pin_observed[c].any())
    log(f"  GT-unobserved >5mm regions: {len(gt_over5mm)} total; "
        f"{n_flip_omni} touched by >=1 omni-observed face; {n_flip_pin} touched by >=1 pinhole-observed face")

    return {
        "name": name,
        "n_faces": n_faces,
        "n_frames": len(poses),
        "omni_observed_frac": omni_frac,
        "pinhole_observed_frac": pin_frac,
        "n_newly_lost_faces": int(newly_lost_mask.sum()),
        "n_newly_lost_components": len(newly_lost_components),
        "n_newly_lost_components_gt5mm": len(nl_over5mm),
        "newly_lost_components_gt5mm_areas": nl_over5mm,
        "n_gt_unobserved_components_gt5mm": len(gt_over5mm),
        "n_gt_unobserved_gt5mm_touched_by_omni": n_flip_omni,
        "n_gt_unobserved_gt5mm_touched_by_pinhole": n_flip_pin,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    intr = CameraIntrinsics.from_file(INTRINSICS_PATH)
    width, height = intr.width, intr.height

    log(f"pinhole model: f={PINHOLE_F:.2f}px, image {width}x{height}")
    hfov = 2 * np.degrees(np.arctan((width / 2) / PINHOLE_F))
    vfov = 2 * np.degrees(np.arctan((height / 2) / PINHOLE_F))
    log(f"pinhole FOV: H={hfov:.2f}deg V={vfov:.2f}deg")

    stretch = corner_stretch_factor(intr, PINHOLE_F, width, height)
    log(f"corner resampling stretch factor: {stretch:.2f}x (pinhole px per 1 omni source px)")

    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    px = np.stack([cols.ravel(), rows.ravel()], axis=-1).astype(np.float64)
    omni_rays = unproject(px, intr)
    pin_rays = pinhole_rays(width, height, PINHOLE_F)

    results = []
    for name in SEQUENCES:
        results.append(analyze_sequence(name, intr, omni_rays, pin_rays))

    out = {
        "pinhole_f": PINHOLE_F,
        "pinhole_width": width,
        "pinhole_height": height,
        "pinhole_hfov_deg": hfov,
        "pinhole_vfov_deg": vfov,
        "corner_stretch_factor": stretch,
        "sequences": results,
    }
    out_path = OUT_DIR / "pinhole_tradeoff.json"
    out_path.write_text(json.dumps(out, indent=2))
    log(f"wrote {out_path}")
    log("ALL DONE")


if __name__ == "__main__":
    main()
