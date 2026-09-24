"""Diagnosis step 3: for c1_ascending_t3_v1 frame 152, back-project the
5,436 no-hit-but-valid-GT-depth pixels to world coordinates and check
where they actually lie: distance to coverage_mesh.obj, inside/outside
its bbox, and distance to the registered mold meshes (registration
transform from raw lumen.obj -> coverage_mesh.obj, RMS 0.094mm, computed
separately). See docs/eval_protocol_oracle_gap.md.
"""
import sys
from pathlib import Path

import numpy as np
import tifffile
import trimesh

REPO = Path("/data1_ycao/chua/projects/mrsp")
sys.path.insert(0, str(REPO / "src"))
from geometry.camera import CameraIntrinsics, unproject, backproject_depth  # noqa: E402
from geometry.coverage_mesh import load_coverage_mesh  # noqa: E402
from geometry.pose import load_poses, transform_points  # noqa: E402

SEQ_DIR = REPO / "scratch" / "c1_ascending_t3_v1"
FRAME = 152
INTRINSICS_PATH = "/data1_ycao/chua/datasets/C3VDv2/camera_intrinsics.txt"

intr = CameraIntrinsics.from_file(INTRINSICS_PATH)
W, H = intr.width, intr.height
n_pixels = W * H

vignette_packed = np.load(REPO / "results/pipelines/oracle_gap_vignette/c1_cecum_t1_v1_vignette_mask.npy")
vignette_mask = np.unpackbits(vignette_packed)[:n_pixels].reshape(H, W).astype(bool)

mesh_data = load_coverage_mesh(SEQ_DIR / "coverage_mesh.obj")
mesh = trimesh.Trimesh(vertices=mesh_data.vertices, faces=mesh_data.faces, process=False)
poses = load_poses(SEQ_DIR / "pose.txt")
M = poses[FRAME]

cols, rows = np.meshgrid(np.arange(W), np.arange(H))
px_grid = np.stack([cols.ravel(), rows.ravel()], axis=-1).astype(np.float64)
cam_rays = unproject(px_grid, intr)

origin = transform_points(np.zeros((1, 3)), M, "transposed")[0]
endpoints = transform_points(cam_rays, M, "transposed")
directions = endpoints - origin
directions /= np.linalg.norm(directions, axis=1, keepdims=True)
origins = np.broadcast_to(origin, directions.shape)

_, index_ray, _ = mesh.ray.intersects_location(origins, directions, multiple_hits=False)
hit_flat = np.zeros(n_pixels, dtype=bool)
hit_flat[index_ray] = True
nohit_flat = (~hit_flat) & ~vignette_mask.ravel()

raw = tifffile.imread(SEQ_DIR / "depth" / f"{FRAME:04d}_depth.tiff")
raw_flat = raw.ravel()
valid_depth = (raw_flat != 0) & (raw_flat != 65535)
target = nohit_flat & valid_depth  # the 5,436-pixel set
print(f"target pixel count: {target.sum()} (expected 5436)")

depth_mm = raw_flat[target].astype(np.float64) / 65535.0 * 100.0
px_target = px_grid[target]
cam_pts = backproject_depth(px_target, depth_mm, intr)
world_pts = transform_points(cam_pts, M, "transposed")

# ---- (a) distance to coverage_mesh.obj, and bbox check ----
closest, dist_to_coverage, face_idx = mesh.nearest.on_surface(world_pts)
bbmin, bbmax = mesh.bounds
inside_bbox = np.all((world_pts >= bbmin) & (world_pts <= bbmax), axis=1)

print(f"\ndistance to coverage_mesh.obj: median={np.median(dist_to_coverage):.3f}mm, "
      f"mean={np.mean(dist_to_coverage):.3f}mm, max={np.max(dist_to_coverage):.3f}mm, "
      f"min={np.min(dist_to_coverage):.3f}mm")
print(f"fraction inside coverage_mesh.obj's bbox: {inside_bbox.mean():.4f} "
      f"({inside_bbox.sum()}/{len(inside_bbox)})")

# ---- (b) distance to registered mold ----
transform = np.load(REPO / "results/pipelines/oracle_gap_lumen_to_coverage_transform.npy")
mold_dir = REPO / "scratch/3D_models/c1_ascending_mold/c1_ascending"
mold_meshes = []
for name in ["c1_ascending_bottom.stl", "c1_ascending_core.stl", "c1_ascending_top.stl"]:
    m = trimesh.load(mold_dir / name)
    m.apply_transform(transform)
    mold_meshes.append((name, m))

combined_mold = trimesh.util.concatenate([m for _, m in mold_meshes])
print(f"\ncombined registered mold: {len(combined_mold.vertices)} verts, "
      f"bbox {combined_mold.bounds.tolist()}")

closest_mold, dist_to_mold, _ = combined_mold.nearest.on_surface(world_pts)
print(f"distance to registered mold (all 3 pieces combined): median={np.median(dist_to_mold):.3f}mm, "
      f"mean={np.mean(dist_to_mold):.3f}mm, max={np.max(dist_to_mold):.3f}mm, "
      f"min={np.min(dist_to_mold):.3f}mm")

for name, m in mold_meshes:
    _, d, _ = m.nearest.on_surface(world_pts)
    print(f"  vs {name}: median={np.median(d):.3f}mm, min={np.min(d):.3f}mm, "
          f"frac<1mm={np.mean(d < 1.0):.4f}, frac<5mm={np.mean(d < 5.0):.4f}")

print(f"\nfraction closer to mold than to coverage_mesh.obj: "
      f"{(dist_to_mold < dist_to_coverage).mean():.4f}")

np.savez(REPO / "results/pipelines/oracle_gap_diagnosis_step3.npz",
         world_pts=world_pts, dist_to_coverage=dist_to_coverage,
         dist_to_mold=dist_to_mold, inside_bbox=inside_bbox, depth_mm=depth_mm)
print("\nsaved results/pipelines/oracle_gap_diagnosis_step3.npz")
