#!/usr/bin/env python
"""Reusable off-screen mesh renderer: highlight a set of face groups on a
mesh and render 2 exterior views + 1 "cut-open" (lengthwise-halved) view.

Uses pyrender with the EGL backend (headless, GPU-accelerated -- set
PYOPENGL_PLATFORM=egl before importing pyrender, done at import time below).
Not a full slicing/retriangulation "cut": the cut-open view keeps only
faces on one side of a plane through the mesh centroid (face-level clip,
not geometric re-triangulation), which is enough for visual inspection and
avoids losing the face-index<->color correspondence used to highlight
specific components.

Reused by scripts/openend_artifact_check.py; can also be run standalone:
    python scripts/render_coverage_views.py <coverage_mesh.obj> <out_prefix>
"""
from __future__ import annotations

import os

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import sys
from pathlib import Path

import numpy as np
import PIL.Image
import pyrender
import trimesh

DEFAULT_BASE_COLOR = np.array([190, 190, 190, 255], dtype=np.uint8)


def principal_axes(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (centroid, axes) where axes is a (3,3) array of unit PCA
    eigenvectors sorted by decreasing eigenvalue (axes[0] = longest axis,
    the tube's long axis for a colon-segment-shaped mesh)."""
    centroid = vertices.mean(axis=0)
    centered = vertices - centroid
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    axes = eigvecs[:, order].T  # rows are axes, descending eigenvalue
    return centroid, axes


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Camera-to-world 4x4 pose, OpenGL convention (camera looks down -Z, +Y up)."""
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    true_up = np.cross(right, forward)

    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = true_up
    pose[:3, 2] = -forward
    pose[:3, 3] = eye
    return pose


def _make_scene(vertices: np.ndarray, faces: np.ndarray, face_colors: np.ndarray) -> pyrender.Scene:
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.visual.face_colors = face_colors
    pr_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    # coverage_mesh.obj face-normal winding may not always face the camera
    # (e.g. the interior lumen surface seen from outside after a lengthwise
    # cut), so single-sided culling renders blank; force double-sided on the
    # auto-generated (color-preserving) material rather than replacing it.
    for primitive in pr_mesh.primitives:
        primitive.material.doubleSided = True
    scene = pyrender.Scene(bg_color=[255, 255, 255, 255], ambient_light=[0.4, 0.4, 0.4])
    scene.add(pr_mesh)
    return scene


def _render(scene: pyrender.Scene, cam_pose: np.ndarray, width: int, height: int) -> np.ndarray:
    scene_cam_light = scene
    cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.5)
    cam_node = scene_cam_light.add(cam, pose=cam_pose)
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=4.0)
    light_node = scene_cam_light.add(light, pose=cam_pose)
    r = pyrender.OffscreenRenderer(width, height)
    try:
        color, _ = r.render(scene_cam_light)
    finally:
        r.delete()
        scene_cam_light.remove_node(cam_node)
        scene_cam_light.remove_node(light_node)
    return color


def render_three_views(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_colors: np.ndarray,
    out_prefix: str | Path,
    width: int = 900,
    height: int = 700,
) -> dict[str, Path]:
    """Renders view1 (exterior), view2 (exterior, rotated ~100 deg around the
    long axis), and cutopen (one lengthwise half, camera looking into the
    cut face) and saves 3 PNGs at <out_prefix>_{view1,view2,cutopen}.png.
    Returns {name: path}.
    """
    centroid, axes = principal_axes(vertices)
    long_axis, mid_axis, short_axis = axes[0], axes[1], axes[2]

    bbox_diag = np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))
    cam_dist = bbox_diag * 1.1

    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs = {}

    # View 1: from along the short axis.
    eye1 = centroid + short_axis * cam_dist
    pose1 = look_at(eye1, centroid, long_axis)
    scene1 = _make_scene(vertices, faces, face_colors)
    img1 = _render(scene1, pose1, width, height)
    p1 = Path(f"{out_prefix}_view1.png")
    PIL.Image.fromarray(img1).save(p1)
    outputs["view1"] = p1

    # View 2: rotated ~100deg around the long axis from view 1.
    theta = np.deg2rad(100)
    rot_dir = mid_axis * np.cos(theta) + short_axis * np.sin(theta)
    eye2 = centroid + rot_dir * cam_dist
    pose2 = look_at(eye2, centroid, long_axis)
    scene2 = _make_scene(vertices, faces, face_colors)
    img2 = _render(scene2, pose2, width, height)
    p2 = Path(f"{out_prefix}_view2.png")
    PIL.Image.fromarray(img2).save(p2)
    outputs["view2"] = p2

    # Cut-open view: keep faces on one lengthwise half. The dataset's colon
    # segments are often curved/bent, not straight tubes, so a single global
    # plane through the overall centroid can slice off nearly the whole tube
    # away from its middle. Instead, bin vertices along the long axis and use
    # each face's LOCAL bin centroid as the cut reference, approximating a
    # cutting surface that follows the tube's bend.
    face_centroids = vertices[faces].mean(axis=1)
    n_bins = 24
    long_coord = (vertices - centroid) @ long_axis
    bin_edges = np.linspace(long_coord.min(), long_coord.max(), n_bins + 1)
    bin_idx = np.clip(np.digitize(long_coord, bin_edges) - 1, 0, n_bins - 1)
    local_centroid = np.zeros((n_bins, 3))
    for b in range(n_bins):
        m = bin_idx == b
        local_centroid[b] = vertices[m].mean(axis=0) if m.any() else centroid

    face_long_coord = (face_centroids - centroid) @ long_axis
    face_bin = np.clip(np.digitize(face_long_coord, bin_edges) - 1, 0, n_bins - 1)
    side = ((face_centroids - local_centroid[face_bin]) * mid_axis).sum(axis=1)
    keep = side >= 0
    cut_faces = faces[keep]
    cut_colors = face_colors[keep]

    eye3 = centroid + mid_axis * cam_dist * 0.9
    pose3 = look_at(eye3, centroid, long_axis)
    scene3 = _make_scene(vertices, cut_faces, cut_colors)
    img3 = _render(scene3, pose3, width, height)
    p3 = Path(f"{out_prefix}_cutopen.png")
    PIL.Image.fromarray(img3).save(p3)
    outputs["cutopen"] = p3

    return outputs


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from geometry.coverage_mesh import load_coverage_mesh

    obj_path, out_prefix = sys.argv[1], sys.argv[2]
    mesh = load_coverage_mesh(obj_path)
    colors = np.tile(DEFAULT_BASE_COLOR, (len(mesh.faces), 1))
    colors[~mesh.face_observed] = [220, 0, 0, 255]
    paths = render_three_views(mesh.vertices, mesh.faces, colors, out_prefix)
    print(paths)
