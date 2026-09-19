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


def boundary_edges(faces: np.ndarray) -> np.ndarray:
    """(K, 2) array of undirected vertex-index edges belonging to exactly one
    face -- the mesh boundary (e.g. the phantom's open ends), assuming the
    mesh has no other holes/non-manifold seams elsewhere."""
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges_sorted = np.sort(edges, axis=1)
    uniq, counts = np.unique(edges_sorted, axis=0, return_counts=True)
    return uniq[counts == 1]


def boundary_loops(boundary_edge_array: np.ndarray) -> list[np.ndarray]:
    """Split boundary edges into connected loops (each loop = one hole in
    the mesh, e.g. one of the phantom's two open ends). Returns a list of
    arrays of vertex indices, one per loop."""
    verts = np.unique(boundary_edge_array.ravel())
    remap = {v: i for i, v in enumerate(verts)}
    local_edges = np.array([[remap[a], remap[b]] for a, b in boundary_edge_array])
    n = len(verts)
    rows = np.concatenate([local_edges[:, 0], local_edges[:, 1]])
    cols = np.concatenate([local_edges[:, 1], local_edges[:, 0]])
    data = np.ones(len(rows), dtype=np.int8)
    graph = coo_matrix((data, (rows, cols)), shape=(n, n))
    n_labels, labels = connected_components(graph, directed=False)
    return [verts[labels == lbl] for lbl in range(n_labels)]


def face_boundary_distances(
    vertices: np.ndarray, faces: np.ndarray, boundary_edge_array: np.ndarray | None = None
) -> np.ndarray:
    """(M,) array: Euclidean distance from each face to the nearest boundary
    vertex, approximated as min over the face's 3 vertices of that vertex's
    nearest-boundary-vertex distance (via a KD-tree over boundary vertices).

    This is Euclidean, not geodesic, and a nearest-BOUNDARY-VERTEX distance
    rather than an exact nearest-boundary-EDGE (point-to-segment) distance --
    a reasonable approximation given the mesh's boundary curve is densely
    sampled by triangle edges (this is the "Euclidean" choice explicitly
    permitted by the "geodesic-or-euclidean" task spec).
    """
    from scipy.spatial import cKDTree

    if boundary_edge_array is None:
        boundary_edge_array = boundary_edges(faces)
    boundary_vert_idx = np.unique(boundary_edge_array.ravel())
    tree = cKDTree(vertices[boundary_vert_idx])
    vert_dist, _ = tree.query(vertices)  # (N,) nearest-boundary-vertex distance per vertex
    face_vert_dist = vert_dist[faces]  # (M, 3)
    return face_vert_dist.min(axis=1)
