"""Reusable mesh geometry stats: per-face area and connected components of a
face subset (used for coverage_stats.py Part 1 GT analysis and Part 2's
re-derived coverage under stricter criteria).
"""
from __future__ import annotations

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def face_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """(M,) array of triangle areas via the cross-product formula."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    return 0.5 * np.linalg.norm(cross, axis=1)


def face_adjacency(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """(K, 2) array of face-index pairs that share an edge (mesh topology only)."""
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    return mesh.face_adjacency


def connected_components_of_subset(
    n_faces: int, adjacency: np.ndarray, subset_mask: np.ndarray
) -> list[np.ndarray]:
    """Connected components of the faces where subset_mask is True, using
    mesh face-adjacency edges restricted to that subset.

    Returns a list of arrays of face indices, one per component, unsorted.
    """
    if adjacency.size == 0:
        return [np.array([i]) for i in np.where(subset_mask)[0]]

    both_in_subset = subset_mask[adjacency[:, 0]] & subset_mask[adjacency[:, 1]]
    edges = adjacency[both_in_subset]

    if edges.shape[0] == 0:
        return [np.array([i]) for i in np.where(subset_mask)[0]]

    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    data = np.ones(len(rows), dtype=np.int8)
    graph = coo_matrix((data, (rows, cols)), shape=(n_faces, n_faces))

    n_labels, labels = connected_components(graph, directed=False)

    subset_idx = np.where(subset_mask)[0]
    subset_labels = labels[subset_idx]
    order = np.argsort(subset_labels)
    sorted_idx = subset_idx[order]
    sorted_labels = subset_labels[order]
    boundaries = np.searchsorted(sorted_labels, np.arange(sorted_labels.max() + 1) + 1)

    components = []
    start = 0
    for end in boundaries:
        if end > start:
            components.append(sorted_idx[start:end])
        start = end
    return components


def area_to_diameter(area: float) -> float:
    """Equivalent-circle diameter: treat the component's area as a disc's
    area, diameter = 2*sqrt(area/pi). Stated assumption, not a measured
    physical diameter (components are irregular blobs of triangles)."""
    return 2.0 * np.sqrt(area / np.pi)
