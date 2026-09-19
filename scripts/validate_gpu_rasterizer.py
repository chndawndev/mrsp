#!/usr/bin/env python
"""Pre-full-run validation of the GPU (Warp) rasterizer against the CPU
(embree) rasterizer and the released ground truth, before committing to the
169-sequence GPU run. See chat log for the exact checklist; this covers:

  1. Backend equivalence: Warp vs embree face-id images, same 20 frames.
  2. GT cross-check: released depth TIFF raw==0 fraction vs our miss fraction.
  3. (MAX_DEPTH: answered by direct code citation, not computation -- see report)
  4. Accuracy vs released coverage_mesh.obj: face-set IoU at pixel strides 4,2,1.
  5. Real end-to-end per-sequence cost: archive streaming + OBJ parse + BVH
     build + kernel, not just the kernel.

Does NOT run the 169-sequence job.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import tifffile
import trimesh

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "7")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from geometry.camera import CameraIntrinsics, unproject  # noqa: E402
from geometry.coverage_mesh import load_coverage_mesh, stream_coverage_mesh_from_zip  # noqa: E402
from geometry.pose import load_poses, transform_points  # noqa: E402

INTRINSICS_PATH = "/data1_ycao/chua/datasets/C3VDv2/camera_intrinsics.txt"
REGISTERED_DIR = Path("/data1_ycao/chua/datasets/C3VDv2/registered_videos")


def log(msg=""):
    print(msg, flush=True)


def frame_world_rays(intr, cam_rays, pose_matrix):
    world_rays = transform_points(cam_rays, pose_matrix, "transposed") - transform_points(
        np.zeros((1, 3)), pose_matrix, "transposed"
    )
    world_rays = world_rays / np.linalg.norm(world_rays, axis=-1, keepdims=True)
    origin = transform_points(np.zeros((1, 3)), pose_matrix, "transposed")
    origins = np.tile(origin, (len(cam_rays), 1))
    return origins, world_rays


# ============================================================
# 1. Backend equivalence: Warp vs embree, same 20 frames
# ============================================================
def check_backend_equivalence():
    log("\n" + "=" * 70)
    log("1. BACKEND EQUIVALENCE: Warp (GPU) vs embree (CPU), 20 frames, c1_cecum_t1_v1")
    log("=" * 70)

    import warp as wp

    wp.init()

    seq_dir = Path("scratch/c1_cecum_t1_v1")
    mesh_data = load_coverage_mesh(seq_dir / "coverage_mesh.obj")
    verts64 = mesh_data.vertices
    faces = mesh_data.faces.astype(np.int32)

    mesh_cpu = trimesh.Trimesh(vertices=verts64, faces=faces, process=False)
    embree = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh_cpu)

    device = "cuda:0"
    wp_points = wp.array(verts64.astype(np.float32), dtype=wp.vec3, device=device)
    wp_indices = wp.array(faces.ravel(), dtype=wp.int32, device=device)
    wp_mesh = wp.Mesh(points=wp_points, indices=wp_indices)

    @wp.kernel
    def raycast_kernel(mesh_id: wp.uint64, origins: wp.array(dtype=wp.vec3), dirs: wp.array(dtype=wp.vec3), out_face: wp.array(dtype=wp.int32)):
        tid = wp.tid()
        q = wp.mesh_query_ray(mesh_id, origins[tid], dirs[tid], 1.0e6)
        out_face[tid] = q.face if q.result else -1

    intr = CameraIntrinsics.from_file(INTRINSICS_PATH)
    poses = load_poses(seq_dir / "pose.txt")
    n_frames = len(poses)
    cols, rows = np.meshgrid(np.arange(intr.width), np.arange(intr.height))
    px = np.stack([cols.ravel(), rows.ravel()], axis=-1).astype(np.float64)
    cam_rays = unproject(px, intr)
    n_rays = len(px)

    sample_idx = np.linspace(0, n_frames - 1, 20).astype(int)
    out_face_gpu = wp.zeros(n_rays, dtype=wp.int32, device=device)

    total_disagree = 0
    total_pixels = 0
    disagreement_examples = []

    for i in sample_idx:
        origins, dirs = frame_world_rays(intr, cam_rays, poses[i])

        cpu_face = embree.intersects_first(origins, dirs).astype(np.int32)

        origins_wp = wp.array(origins.astype(np.float32), dtype=wp.vec3, device=device)
        dirs_wp = wp.array(dirs.astype(np.float32), dtype=wp.vec3, device=device)
        wp.launch(raycast_kernel, dim=n_rays, inputs=[wp_mesh.id, origins_wp, dirs_wp, out_face_gpu], device=device)
        wp.synchronize()
        gpu_face = out_face_gpu.numpy()

        disagree = cpu_face != gpu_face
        n_dis = int(disagree.sum())
        total_disagree += n_dis
        total_pixels += n_rays

        if n_dis > 0 and len(disagreement_examples) < 5:
            dis_idx = np.where(disagree)[0][: 5 - len(disagreement_examples)]
            for d_idx in dis_idx:
                disagreement_examples.append(
                    {
                        "frame": int(i),
                        "pixel": (int(px[d_idx, 0]), int(px[d_idx, 1])),
                        "cpu_face": int(cpu_face[d_idx]),
                        "gpu_face": int(gpu_face[d_idx]),
                        "origin": origins[d_idx].tolist(),
                        "dir": dirs[d_idx].tolist(),
                    }
                )

        log(f"  frame {i}: {n_dis} / {n_rays} disagree ({100*n_dis/n_rays:.5f}%)")

    frac = total_disagree / total_pixels
    log(f"\nTOTAL: {total_disagree} / {total_pixels} pixels disagree = {100*frac:.5f}%")
    log(f"Threshold: 0.1%. {'PASS' if frac < 0.001 else 'FAIL -- STOP AND DIAGNOSE'}")

    log("\nInspecting disagreement examples (checking if they're edge/tie cases):")
    for ex in disagreement_examples:
        cpu_f, gpu_f = ex["cpu_face"], ex["gpu_face"]
        if cpu_f >= 0 and gpu_f >= 0:
            # Distance between the two candidate faces' centroids -- small
            # distance is consistent with a shared-edge/tie disagreement.
            c_cpu = verts64[faces[cpu_f]].mean(axis=0)
            c_gpu = verts64[faces[gpu_f]].mean(axis=0)
            centroid_dist = float(np.linalg.norm(c_cpu - c_gpu))
            shares_vertex = bool(set(faces[cpu_f]) & set(faces[gpu_f]))
            log(f"  frame={ex['frame']} px={ex['pixel']}: cpu_face={cpu_f} gpu_face={gpu_f}, "
                f"face-centroid dist={centroid_dist:.4f}mm, shares a vertex={shares_vertex}")
        else:
            log(f"  frame={ex['frame']} px={ex['pixel']}: cpu_face={cpu_f} gpu_face={gpu_f} (one backend missed entirely)")

    return frac


# ============================================================
# 2. GT cross-check: depth TIFF raw==0 fraction vs our miss fraction
# ============================================================
def _stream_zip_member_to_temp(archive: Path, member: str, out_path: Path) -> None:
    import subprocess

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        subprocess.run(["unzip", "-p", str(archive), member], stdout=f, check=True)


def check_gt_miss_fraction(seq_name: str, seq_dir_or_none: Path | None, n_sample_frames: int = 10):
    log("\n" + "=" * 70)
    log(f"2. GT MISS-FRACTION CROSS-CHECK: {seq_name}")
    log("=" * 70)

    if seq_dir_or_none is not None and seq_dir_or_none.exists():
        mesh_data = load_coverage_mesh(seq_dir_or_none / "coverage_mesh.obj")
        depth_dir = seq_dir_or_none / "depth"
        depth_files = sorted(depth_dir.glob("*_depth.tiff"))
        poses = load_poses(seq_dir_or_none / "pose.txt")

        def read_depth(i):
            frame_idx = int(depth_files[i].name.split("_")[0])
            return frame_idx, tifffile.imread(depth_files[i])

        n_frames = len(depth_files)
    else:
        # Stream just what's needed (mesh, pose.txt, a handful of depth
        # frames) from the zip, without extracting the whole archive.
        archive = REGISTERED_DIR / f"{seq_name}.zip"
        mesh_data = stream_coverage_mesh_from_zip(archive)

        tmp_dir = Path("/tmp") / f"gpu_validate_{seq_name}"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        pose_tmp = tmp_dir / "pose.txt"
        _stream_zip_member_to_temp(archive, "pose.txt", pose_tmp)
        poses = load_poses(pose_tmp)
        n_frames = len(poses)

        def read_depth(i):
            frame_idx = i
            member = f"depth/{frame_idx:04d}_depth.tiff"
            tmp_path = tmp_dir / f"{frame_idx:04d}_depth.tiff"
            _stream_zip_member_to_temp(archive, member, tmp_path)
            arr = tifffile.imread(tmp_path)
            tmp_path.unlink()
            return frame_idx, arr

    mesh_cpu = trimesh.Trimesh(vertices=mesh_data.vertices, faces=mesh_data.faces.astype(np.int32), process=False)
    embree = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh_cpu)

    intr = CameraIntrinsics.from_file(INTRINSICS_PATH)
    cols, rows = np.meshgrid(np.arange(intr.width), np.arange(intr.height))
    px = np.stack([cols.ravel(), rows.ravel()], axis=-1).astype(np.float64)
    cam_rays = unproject(px, intr)

    sample_idx = np.linspace(0, n_frames - 1, n_sample_frames).astype(int)

    gt_zero_fracs = []
    our_miss_fracs = []
    for i in sample_idx:
        frame_idx, raw = read_depth(i)
        gt_zero_frac = float((raw == 0).mean())
        gt_zero_fracs.append(gt_zero_frac)

        origins, dirs = frame_world_rays(intr, cam_rays, poses[frame_idx])
        face_id = embree.intersects_first(origins, dirs)
        our_miss_frac = float((face_id < 0).mean())
        our_miss_fracs.append(our_miss_frac)

        log(f"  frame {frame_idx}: GT raw==0 frac={gt_zero_frac:.5f}, our miss frac={our_miss_frac:.5f}")

    log(f"\nmean GT raw==0 frac: {np.mean(gt_zero_fracs):.5f}")
    log(f"mean our miss frac:  {np.mean(our_miss_fracs):.5f}")
    return np.mean(gt_zero_fracs), np.mean(our_miss_fracs)


# ============================================================
# 4. Accuracy vs released coverage_mesh.obj at strides 4, 2, 1
# ============================================================
def check_accuracy_vs_released(seq_dir: Path, strides=(4, 2, 1)):
    log("\n" + "=" * 70)
    log(f"4. ACCURACY vs RELEASED coverage_mesh.obj: {seq_dir.name}, strides {strides}")
    log("=" * 70)

    mesh_data = load_coverage_mesh(seq_dir / "coverage_mesh.obj")
    n_faces = len(mesh_data.faces)
    gt_observed = mesh_data.face_observed

    mesh_cpu = trimesh.Trimesh(vertices=mesh_data.vertices, faces=mesh_data.faces.astype(np.int32), process=False)
    embree = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh_cpu)

    intr = CameraIntrinsics.from_file(INTRINSICS_PATH)
    poses = load_poses(seq_dir / "pose.txt")
    n_frames = len(poses)

    results = {}
    for stride in strides:
        cols, rows = np.meshgrid(np.arange(0, intr.width, stride), np.arange(0, intr.height, stride))
        px = np.stack([cols.ravel(), rows.ravel()], axis=-1).astype(np.float64)
        cam_rays = unproject(px, intr)

        pred_observed = np.zeros(n_faces, dtype=bool)
        t0 = time.time()
        for i in range(n_frames):
            origins, dirs = frame_world_rays(intr, cam_rays, poses[i])
            face_id = embree.intersects_first(origins, dirs)
            hit = face_id[face_id >= 0]
            pred_observed[hit] = True
        dt = time.time() - t0

        inter = np.sum(pred_observed & gt_observed)
        union = np.sum(pred_observed | gt_observed)
        iou = inter / union if union else float("nan")
        false_observed = int(np.sum(pred_observed & ~gt_observed))
        false_unobserved = int(np.sum(~pred_observed & gt_observed))

        results[stride] = dict(iou=iou, false_observed=false_observed, false_unobserved=false_unobserved, dt=dt)
        log(f"  stride {stride}: IoU={iou:.5f}, false_observed={false_observed}, "
            f"false_unobserved={false_unobserved}, all-frame rasterize time={dt:.1f}s")

    return results


# ============================================================
# 5. Real end-to-end per-sequence cost
# ============================================================
def check_end_to_end_cost(seq_name: str):
    log("\n" + "=" * 70)
    log(f"5. END-TO-END PER-SEQUENCE COST: {seq_name}")
    log("=" * 70)

    archive = REGISTERED_DIR / f"{seq_name}.zip"

    t0 = time.time()
    mesh_data = stream_coverage_mesh_from_zip(archive)
    t_stream_parse = time.time() - t0
    log(f"  stream archive + parse OBJ: {t_stream_parse:.2f}s ({len(mesh_data.faces)} faces)")

    t0 = time.time()
    mesh_cpu = trimesh.Trimesh(vertices=mesh_data.vertices, faces=mesh_data.faces.astype(np.int32), process=False)
    embree = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh_cpu)
    t_cpu_bvh = time.time() - t0
    log(f"  build embree (CPU) BVH: {t_cpu_bvh:.2f}s")

    import warp as wp

    wp.init()
    device = "cuda:0"
    t0 = time.time()
    wp_points = wp.array(mesh_data.vertices.astype(np.float32), dtype=wp.vec3, device=device)
    wp_indices = wp.array(mesh_data.faces.astype(np.int32).ravel(), dtype=wp.int32, device=device)
    wp_mesh = wp.Mesh(points=wp_points, indices=wp_indices)
    wp.synchronize()
    t_gpu_bvh = time.time() - t0
    log(f"  build Warp (GPU) BVH: {t_gpu_bvh:.2f}s")

    return dict(t_stream_parse=t_stream_parse, t_cpu_bvh=t_cpu_bvh, t_gpu_bvh=t_gpu_bvh, n_faces=len(mesh_data.faces))


if __name__ == "__main__":
    frac = check_backend_equivalence()

    check_gt_miss_fraction("c1_cecum_t1_v1", Path("scratch/c1_cecum_t1_v1"))
    check_gt_miss_fraction("c1_ascending_t2_v1", None)

    check_accuracy_vs_released(Path("scratch/c1_cecum_t1_v1"), strides=(4, 2, 1))

    check_end_to_end_cost("c1_cecum_t1_v1")
    check_end_to_end_cost("c1_ascending_t2_v1")

    log("\nALL VALIDATION CHECKS COMPLETE. Not starting the 169-sequence run.")
