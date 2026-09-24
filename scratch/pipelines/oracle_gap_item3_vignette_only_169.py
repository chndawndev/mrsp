"""Item 3: quantify vignette-only-observed faces across all 169 registered
sequences -- faces the released GT marks observed but that our own
ray-cast (GT pose) only ever reaches through vignette-region pixels
(never through a non-vignette pixel). Candidate first level for a
criteria-sensitivity analysis ("released GT minus faces never actually
imaged"). Streams coverage_mesh.obj + pose.txt directly from the
archives, no bulk extraction, no depth/rgb needed.

Reduced pixel density (stride 4) used for full-corpus tractability --
docs/oracle_check.md already characterized this specific stride's gap
vs. full density (IoU 0.944 vs 0.9988) -- flagged explicitly as an
approximation, not a full-density result. See
docs/eval_protocol_oracle_gap.md.
"""
import io
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import trimesh

REPO = Path("/data1_ycao/chua/projects/mrsp")
sys.path.insert(0, str(REPO / "src"))
from geometry.camera import CameraIntrinsics, unproject  # noqa: E402
from geometry.coverage_mesh import parse_coverage_mesh_lines  # noqa: E402
from geometry.pose import _load_poses_from_lines, transform_points  # noqa: E402

DATASET_ROOT = Path("/data1_ycao/chua/datasets/C3VDv2/registered_videos")
INTRINSICS_PATH = "/data1_ycao/chua/datasets/C3VDv2/camera_intrinsics.txt"
OUT_DIR = REPO / "results/pipelines/oracle_gap_item3"
OUT_DIR.mkdir(parents=True, exist_ok=True)
STRIDE = 4


def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


intr = CameraIntrinsics.from_file(INTRINSICS_PATH)
W, H = intr.width, intr.height

vignette_packed = np.load(REPO / "results/pipelines/oracle_gap_vignette/c1_cecum_t1_v1_vignette_mask.npy")
vignette_full = np.unpackbits(vignette_packed)[: W * H].reshape(H, W).astype(bool)

rows_s = np.arange(0, H, STRIDE)
cols_s = np.arange(0, W, STRIDE)
grid_cols, grid_rows = np.meshgrid(cols_s, rows_s)
px_grid = np.stack([grid_cols.ravel(), grid_rows.ravel()], axis=-1).astype(np.float64)
cam_rays = unproject(px_grid, intr)
vignette_sub = vignette_full[np.ix_(rows_s, cols_s)].ravel()
n_rays = len(px_grid)
log(f"stride {STRIDE}: {n_rays} rays/frame ({vignette_sub.sum()} vignette, "
    f"{(~vignette_sub).sum()} non-vignette)")


def sequence_names():
    zips = sorted(DATASET_ROOT.glob("*.zip"))
    return [z.stem for z in zips if not z.stem.startswith("c0_")]


def resolve_member(zf, basename):
    candidates = [n for n in zf.namelist() if n.endswith(basename)]
    if len(candidates) != 1:
        raise RuntimeError(f"expected exactly 1 match for {basename}, got {candidates}")
    return candidates[0]


results = {}
names = sequence_names()
assert len(names) == 169
log(f"{len(names)} registered sequences")

t_start = time.time()
for i, name in enumerate(names):
    zpath = DATASET_ROOT / f"{name}.zip"
    try:
        with zipfile.ZipFile(zpath) as zf:
            mesh_member = resolve_member(zf, "coverage_mesh.obj")
            mesh_bytes = zf.read(mesh_member).decode("utf-8", errors="ignore").splitlines()
            mesh_data = parse_coverage_mesh_lines(mesh_bytes)
            pose_member = resolve_member(zf, "pose.txt")
            pose_lines = zf.read(pose_member).decode("utf-8", errors="ignore").splitlines()
            poses = _load_poses_from_lines(pose_lines)
    except Exception as e:
        log(f"  [{i+1}/{len(names)}] {name}: ERROR loading: {e}")
        results[name] = {"error": str(e)}
        continue

    mesh = trimesh.Trimesh(vertices=mesh_data.vertices, faces=mesh_data.faces, process=False)
    n_faces = len(mesh_data.faces)
    gt_observed = mesh_data.face_observed
    n_frames = len(poses)

    ever_nonvignette = np.zeros(n_faces, dtype=bool)
    ever_vignette = np.zeros(n_faces, dtype=bool)

    t0 = time.time()
    for frame_idx in range(n_frames):
        M = poses[frame_idx]
        origin = transform_points(np.zeros((1, 3)), M, "transposed")[0]
        endpoints = transform_points(cam_rays, M, "transposed")
        directions = endpoints - origin
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        origins = np.broadcast_to(origin, directions.shape)

        _, index_ray, index_tri = mesh.ray.intersects_location(
            origins, directions, multiple_hits=False
        )
        if len(index_ray) == 0:
            continue
        is_vig = vignette_sub[index_ray]
        np.logical_or.at(ever_nonvignette, index_tri[~is_vig], True)
        np.logical_or.at(ever_vignette, index_tri[is_vig], True)

    elapsed = time.time() - t0
    vignette_only = gt_observed & ~ever_nonvignette & ever_vignette
    n_gt_obs = int(gt_observed.sum())
    n_vo = int(vignette_only.sum())

    results[name] = {
        "n_faces": n_faces, "n_frames": n_frames, "n_gt_observed": n_gt_obs,
        "n_vignette_only": n_vo, "frac_of_gt_observed": n_vo / n_gt_obs if n_gt_obs else float("nan"),
        "elapsed_s": elapsed,
    }
    if i % 5 == 0 or i == len(names) - 1:
        total_elapsed = time.time() - t_start
        log(f"  [{i+1}/{len(names)}] {name}: {n_frames}f, GT_obs={n_gt_obs}, "
            f"vignette_only={n_vo} ({n_vo/n_gt_obs*100 if n_gt_obs else 0:.3f}%), "
            f"{elapsed:.1f}s, total {total_elapsed:.0f}s")

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(results, f, indent=2)

log(f"\nall done in {time.time()-t_start:.0f}s")

ok = [v for v in results.values() if "error" not in v]
fracs = np.array([v["frac_of_gt_observed"] for v in ok if not np.isnan(v["frac_of_gt_observed"])])
log(f"\n=== distribution of vignette-only fraction of GT-observed faces, {len(fracs)} sequences ===")
log(f"median={np.median(fracs)*100:.3f}%, mean={np.mean(fracs)*100:.3f}%, "
    f"min={fracs.min()*100:.3f}%, max={fracs.max()*100:.3f}%, "
    f"p25={np.percentile(fracs,25)*100:.3f}%, p75={np.percentile(fracs,75)*100:.3f}%")
log("ALL DONE")
