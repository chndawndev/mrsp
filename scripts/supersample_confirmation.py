#!/usr/bin/env python
"""Single confirmation experiment (see chat log): does 2x2 sub-pixel
supersampling recover the previously-false_unobserved faces at the specific
frames identified in docs/visibility_outliers.md as each sequence's dominant
"best" frame?

  - c1_sigmoid1_t2_v2 frame 122, c1_sigmoid1_t1_v1 frame 103   (group B)
  - c1_ascending_t4_v2 frames 58 and 681                        (group A1)

Diagnostic only. Does NOT change the production visibility pipeline (that
remains scripts/visibility_full.py, single ray per pixel). CPU/embree is
plenty fast for 4 single-frame casts; no GPU needed for this tiny scope.

Reuses results/visibility_full/observed_masks.npz for the "previously
false_unobserved" ground truth (no rerun of the full pipeline), and the same
camera-model / pose-transform shortcut already validated in
scripts/visibility_full.py (world_dir = cam_ray @ M[:3,:3], origin = M[3,:3]).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from geometry.camera import CameraIntrinsics, unproject  # noqa: E402
from geometry.coverage_mesh import stream_coverage_mesh_from_zip  # noqa: E402
from geometry.pose import stream_poses_from_zip  # noqa: E402

REGISTERED_DIR = Path("/data1_ycao/chua/datasets/C3VDv2/registered_videos")
INTRINSICS_PATH = "/data1_ycao/chua/datasets/C3VDv2/camera_intrinsics.txt"
OUT_DIR = Path("results/visibility_outliers")

TEST_POINTS = [
    ("c1_sigmoid1_t2_v2", 122, "B"),
    ("c1_sigmoid1_t1_v1", 103, "B"),
    ("c1_ascending_t4_v2", 58, "A1"),
    ("c1_ascending_t4_v2", 681, "A1"),
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cast_frame(intersector, intr: CameraIntrinsics, M: np.ndarray, offsets: list[tuple[float, float]]) -> set[int]:
    """Casts one ray per pixel per (dx, dy) sub-pixel offset at this frame's
    pose, returns the set of unique face indices hit across all offsets."""
    hit_faces: set[int] = set()
    cols, rows = np.meshgrid(np.arange(intr.width), np.arange(intr.height))
    R = M[:3, :3].astype(np.float64)
    origin = M[3, :3].astype(np.float64)

    for dx, dy in offsets:
        px = np.stack([cols.ravel() + dx, rows.ravel() + dy], axis=-1).astype(np.float64)
        cam_rays = unproject(px, intr)
        world_dirs = cam_rays @ R
        origins = np.tile(origin, (len(px), 1))
        face_id = intersector.intersects_first(origins, world_dirs)
        hit_faces.update(face_id[face_id >= 0].tolist())

    return hit_faces


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    masks = dict(np.load("results/visibility_full/observed_masks.npz"))
    intr = CameraIntrinsics.from_file(INTRINSICS_PATH)

    # 1x (single ray through pixel center) and 2x2 supersample (4 sub-pixel offsets).
    offsets_1x = [(0.5, 0.5)]
    offsets_4x = [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)]

    mesh_cache: dict[str, tuple] = {}
    results = []

    for name, frame_idx, group in TEST_POINTS:
        log(f"=== {name} frame {frame_idx} (group {group}) ===")
        if name not in mesh_cache:
            archive = REGISTERED_DIR / f"{name}.zip"
            mesh_data = stream_coverage_mesh_from_zip(archive)
            poses = stream_poses_from_zip(archive)
            mesh = trimesh.Trimesh(vertices=mesh_data.vertices, faces=mesh_data.faces.astype(np.int32), process=False)
            intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)
            n_faces = len(mesh_data.faces)
            pred_observed = np.unpackbits(masks[name])[:n_faces].astype(bool)
            gt_observed = mesh_data.face_observed
            false_unobserved_mask = gt_observed & ~pred_observed
            false_unobserved_idx = set(np.where(false_unobserved_mask)[0].tolist())
            mesh_cache[name] = (intersector, poses, false_unobserved_idx, n_faces)
        intersector, poses, false_unobserved_idx, n_faces = mesh_cache[name]
        n_false_unobserved = len(false_unobserved_idx)

        M = poses[frame_idx]

        t0 = time.time()
        hit_1x = cast_frame(intersector, intr, M, offsets_1x)
        recovered_1x = false_unobserved_idx & hit_1x
        log(f"  1x sanity check: {len(recovered_1x)}/{n_false_unobserved} false_unobserved faces hit "
            f"(expected 0, by definition of false_unobserved across the whole sequence) [{time.time()-t0:.1f}s]")

        t0 = time.time()
        hit_4x = cast_frame(intersector, intr, M, offsets_4x)
        recovered_4x = false_unobserved_idx & hit_4x
        frac = len(recovered_4x) / n_false_unobserved if n_false_unobserved else float("nan")
        log(f"  4x supersample: {len(recovered_4x)}/{n_false_unobserved} recovered ({frac:.1%}) [{time.time()-t0:.1f}s]")

        results.append(
            {
                "name": name,
                "frame": frame_idx,
                "group": group,
                "n_false_unobserved_total": n_false_unobserved,
                "n_recovered_1x_sanity": len(recovered_1x),
                "n_recovered_4x": len(recovered_4x),
                "frac_recovered_4x": frac,
                "recovered_4x_face_ids": sorted(recovered_4x),
            }
        )

    # Combined (union) recovery for c1_ascending_t4_v2 across frames 58 and 681.
    asc_results = [r for r in results if r["name"] == "c1_ascending_t4_v2"]
    union_recovered = set(asc_results[0]["recovered_4x_face_ids"]) | set(asc_results[1]["recovered_4x_face_ids"])
    n_total = asc_results[0]["n_false_unobserved_total"]
    log(f"c1_ascending_t4_v2 combined (frames 58+681): {len(union_recovered)}/{n_total} "
        f"({len(union_recovered)/n_total:.1%}) recovered")

    out_path = OUT_DIR / "supersample_confirmation.json"
    out_path.write_text(json.dumps({
        "per_frame_results": results,
        "c1_ascending_t4_v2_combined_58_681": {
            "n_recovered": len(union_recovered),
            "n_total": n_total,
            "frac_recovered": len(union_recovered) / n_total,
        },
    }, indent=2))
    log(f"wrote {out_path}")
    log("ALL DONE")


if __name__ == "__main__":
    main()
