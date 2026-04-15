from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import numpy as np
import trimesh
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import time

import json
from os.path import join
from typing import List, Dict, Tuple, Optional

from collections import defaultdict
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors
import networkx as nx

from plyfile import PlyData
import open3d as o3d
from partfield.utils import *


#########################
# Auto-selection metrics
#########################

def silhouette_score_sampled(features: np.ndarray, labels: np.ndarray,
                             sample_size: int = 5000, random_state: int = 42) -> float:
    """
    Compute silhouette score on a stratified sample for ~100x speedup on large datasets.

    For datasets with 100k+ points, the full silhouette computation is O(n²) and very slow.
    This uses stratified sampling to preserve cluster proportions while dramatically reducing compute.

    Args:
        features: (N, D) feature array
        labels: (N,) cluster labels
        sample_size: Maximum number of points to sample (default 5000)
        random_state: Random seed for reproducibility

    Returns:
        Silhouette score in range [-1, 1]
    """
    labels = np.squeeze(labels).astype(int)
    n_samples = len(labels)

    # If small enough, compute exact score
    if n_samples <= sample_size:
        try:
            return silhouette_score(features, labels)
        except Exception:
            return -1.0

    np.random.seed(random_state)

    # Stratified sampling to preserve cluster proportions
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return -1.0

    indices = []
    samples_per_cluster = max(10, sample_size // len(unique_labels))

    for label in unique_labels:
        cluster_indices = np.where(labels == label)[0]
        n_take = min(len(cluster_indices), samples_per_cluster)
        if n_take > 0:
            sampled = np.random.choice(cluster_indices, n_take, replace=False)
            indices.extend(sampled)

    indices = np.array(indices)

    # Ensure we have at least 2 clusters represented
    sampled_labels = labels[indices]
    if len(np.unique(sampled_labels)) < 2:
        return -1.0

    try:
        return silhouette_score(features[indices], sampled_labels)
    except Exception:
        return -1.0


def evaluate_clustering(features: np.ndarray, labels: np.ndarray, sample_size: int = 5000) -> Dict[str, float]:
    """
    Evaluate a clustering using multiple metrics.

    Uses sampled silhouette score for large datasets (~100x speedup).

    Returns a dictionary with:
    - silhouette: [-1, 1], higher is better
    - davies_bouldin: [0, inf), lower is better
    - calinski_harabasz: [0, inf), higher is better
    """
    labels = labels.squeeze().astype(int)
    unique_labels = np.unique(labels)

    if len(unique_labels) < 2:
        return {'silhouette': -1.0, 'davies_bouldin': float('inf'), 'calinski_harabasz': 0.0}

    min_samples = min(np.sum(labels == l) for l in unique_labels)
    if min_samples < 2:
        return {'silhouette': -1.0, 'davies_bouldin': float('inf'), 'calinski_harabasz': 0.0}

    # Use sampled silhouette for large datasets (major speedup)
    sil = silhouette_score_sampled(features, labels, sample_size=sample_size)

    try:
        db = davies_bouldin_score(features, labels)
    except Exception:
        db = float('inf')

    try:
        ch = calinski_harabasz_score(features, labels)
    except Exception:
        ch = 0.0

    return {'silhouette': sil, 'davies_bouldin': db, 'calinski_harabasz': ch}


def compute_combined_score(metrics: Dict[str, float], n_clusters: int,
                          min_clusters: int = 3, max_clusters: int = 15,
                          prefer_fewer: bool = True) -> float:
    """
    Compute a combined score from multiple metrics.
    Higher score = better clustering.
    """
    sil = metrics['silhouette']
    db = metrics['davies_bouldin']
    ch = metrics['calinski_harabasz']

    if sil <= -1.0 or db == float('inf'):
        return -float('inf')

    # Normalize silhouette to [0, 1] (from [-1, 1])
    sil_norm = (sil + 1) / 2

    # Normalize Davies-Bouldin (invert, typical values 0-3)
    db_norm = max(0, 1 - db / 3)

    # Normalize Calinski-Harabasz (log scale)
    ch_norm = min(np.log1p(ch) / 10, 1.0)

    # Weighted combination (silhouette most important for part segmentation)
    base_score = 0.5 * sil_norm + 0.3 * db_norm + 0.2 * ch_norm

    # Penalize extreme cluster counts
    if n_clusters < min_clusters:
        base_score -= 0.1 * (min_clusters - n_clusters)
    elif n_clusters > max_clusters:
        base_score -= 0.05 * (n_clusters - max_clusters)

    # Slight preference for fewer clusters (parsimony)
    if prefer_fewer and n_clusters > min_clusters:
        base_score -= 0.01 * (n_clusters - min_clusters)

    return base_score


def select_best_clustering(features: np.ndarray,
                          all_labels: Dict[int, np.ndarray],
                          min_clusters: int = 3,
                          max_clusters: int = 15,
                          prefer_fewer: bool = True,
                          verbose: bool = True,
                          sample_size: int = 5000,
                          early_stop_patience: int = 0) -> Tuple[int, Dict]:
    """
    Select the best clustering from multiple options.

    Args:
        features: (N, D) feature array
        all_labels: Dict mapping n_clusters -> labels array
        min_clusters: Minimum preferred cluster count
        max_clusters: Maximum preferred cluster count
        prefer_fewer: If True, slightly penalize higher cluster counts
        verbose: Print progress
        sample_size: Sample size for silhouette computation (5000 = ~100x speedup)
        early_stop_patience: Stop after N consecutive non-improving k values (0 = disabled)

    Returns: (best_n_clusters, results_dict)
    """
    results = {}
    best_score = -float('inf')
    best_n = None
    no_improve_count = 0
    improvement_threshold = 0.01  # Minimum improvement to reset patience

    for n_clusters, labels in sorted(all_labels.items()):
        metrics = evaluate_clustering(features, labels, sample_size=sample_size)
        score = compute_combined_score(metrics, n_clusters, min_clusters, max_clusters, prefer_fewer)
        results[n_clusters] = {'metrics': metrics, 'combined_score': score}

        if verbose:
            print(f"    k={n_clusters:2d}: silhouette={metrics['silhouette']:.3f}, "
                  f"DB={metrics['davies_bouldin']:.3f}, CH={metrics['calinski_harabasz']:.1f}, "
                  f"score={score:.4f}")

        # Track best and check for early stopping
        if score > best_score + improvement_threshold:
            best_score = score
            best_n = n_clusters
            no_improve_count = 0
        else:
            no_improve_count += 1

        # Early stopping check
        if early_stop_patience > 0 and no_improve_count >= early_stop_patience and best_n is not None:
            if verbose:
                print(f"    Early stopping after {n_clusters} clusters (no improvement for {early_stop_patience} steps)")
            break

    if best_n is None:
        best_n = max(results.keys(), key=lambda k: results[k]['combined_score'])

    if verbose:
        print(f"    -> Best: {best_n} clusters (score={results[best_n]['combined_score']:.4f})")

    return best_n, results
#########################

#### Export to file #####
def export_colored_mesh_ply(V, F, FL, filename='segmented_mesh.ply'):
    """
    Export a mesh with per-face segmentation labels into a colored PLY file.

    Parameters:
    - V (np.ndarray): Vertices array of shape (N, 3)
    - F (np.ndarray): Faces array of shape (M, 3)
    - FL (np.ndarray): Face labels of shape (M,)
    - filename (str): Output filename
    """
    assert V.shape[1] == 3
    assert F.shape[1] == 3
    assert F.shape[0] == FL.shape[0]

    # Generate distinct colors for each unique label
    unique_labels = np.unique(FL)
    colormap = plt.cm.get_cmap("tab20", len(unique_labels))
    label_to_color = {
        label: (np.array(colormap(i)[:3]) * 255).astype(np.uint8)
        for i, label in enumerate(unique_labels)
    }

    mesh = trimesh.Trimesh(vertices=V, faces=F)
    FL = np.squeeze(FL)
    for i, face in enumerate(F):
        label = FL[i]
        color = label_to_color[label]
        color_with_alpha = np.append(color, 255)  # Add alpha value
        mesh.visual.face_colors[i] = color_with_alpha

    mesh.export(filename)
    print(f"Exported mesh to {filename}")

def export_pointcloud_with_labels_to_ply(V, VL, filename='colored_pointcloud.ply'):
    """
    Export a labeled point cloud to a PLY file with vertex colors.
    
    Parameters:
    - V: (N, 3) numpy array of XYZ coordinates
    - VL: (N,) numpy array of integer labels
    - filename: Output PLY file name
    """
    assert V.shape[0] == VL.shape[0], "Number of vertices and labels must match"

    # Generate unique colors for each label
    unique_labels = np.unique(VL)
    colormap = plt.cm.get_cmap("tab20", len(unique_labels))
    label_to_color = {
        label: colormap(i)[:3] for i, label in enumerate(unique_labels)
    }

    VL = np.squeeze(VL)
    # Map labels to RGB colors
    colors = np.array([label_to_color[label] for label in VL])
    
    # Open3D requires colors in float [0, 1]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(V)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Save to .ply
    o3d.io.write_point_cloud(filename, pcd)
    print(f"Point cloud saved to {filename}")
#########################

#########################
def construct_face_adjacency_matrix_ccmst(face_list, vertices, k=10, with_knn=True):
    """
    Given a list of faces (each face is a 3-tuple of vertex indices),
    construct a face-based adjacency matrix of shape (num_faces, num_faces).

    Two faces are adjacent if they share an edge (the "mesh adjacency").
    If multiple connected components remain, we:
      1) Compute the centroid of each connected component as the mean of all face centroids.
      2) Use a KNN graph (k=10) based on centroid distances on each connected component.
      3) Compute MST of that KNN graph.
      4) Add MST edges that connect different components as "dummy" edges
         in the face adjacency matrix, ensuring one connected component. The selected face for 
         each connected component is the face closest to the component centroid.

    Parameters
    ----------
    face_list : list of tuples
        List of faces, each face is a tuple (v0, v1, v2) of vertex indices.
    vertices : np.ndarray of shape (num_vertices, 3)
        Array of vertex coordinates.
    k : int, optional
        Number of neighbors to use in centroid KNN. Default is 10.

    Returns
    -------
    face_adjacency : scipy.sparse.csr_matrix
        A CSR sparse matrix of shape (num_faces, num_faces),
        containing 1s for adjacent faces (shared-edge adjacency)
        plus dummy edges ensuring a single connected component.
    """
    num_faces = len(face_list)
    if num_faces == 0:
        # Return an empty matrix if no faces
        return csr_matrix((0, 0))

    #--------------------------------------------------------------------------
    # 1) Build adjacency based on shared edges.
    #    (Same logic as the original code, plus import statements.)
    #--------------------------------------------------------------------------
    edge_to_faces = defaultdict(list)
    uf = UnionFind(num_faces)
    for f_idx, (v0, v1, v2) in enumerate(face_list):
        # Sort each edge’s endpoints so (i, j) == (j, i)
        edges = [
            tuple(sorted((v0, v1))),
            tuple(sorted((v1, v2))),
            tuple(sorted((v2, v0)))
        ]
        for e in edges:
            edge_to_faces[e].append(f_idx)

    row = []
    col = []
    for edge, face_indices in edge_to_faces.items():
        unique_faces = list(set(face_indices))
        if len(unique_faces) > 1:
            # For every pair of distinct faces that share this edge,
            # mark them as mutually adjacent
            for i in range(len(unique_faces)):
                for j in range(i + 1, len(unique_faces)):
                    fi = unique_faces[i]
                    fj = unique_faces[j]
                    row.append(fi)
                    col.append(fj)
                    row.append(fj)
                    col.append(fi)
                    uf.union(fi, fj)

    data = np.ones(len(row), dtype=np.int8)
    face_adjacency = coo_matrix(
        (data, (row, col)), shape=(num_faces, num_faces)
    ).tocsr()

    #--------------------------------------------------------------------------
    # 2) Check if the graph from shared edges is already connected.
    #--------------------------------------------------------------------------
    n_components = 0
    for i in range(num_faces):
        if uf.find(i) == i:
            n_components += 1
    print("n_components", n_components)

    if n_components == 1:
        # Already a single connected component, no need for dummy edges
        return face_adjacency

    #--------------------------------------------------------------------------
    # 3) Compute centroids of each face for building a KNN graph.
    #--------------------------------------------------------------------------
    face_centroids = []
    for (v0, v1, v2) in face_list:
        centroid = (vertices[v0] + vertices[v1] + vertices[v2]) / 3.0
        face_centroids.append(centroid)
    face_centroids = np.array(face_centroids)

    # #--------------------------------------------------------------------------
    # # 4a) Build a KNN graph (k=10) over face centroids using scikit‐learn
    # #--------------------------------------------------------------------------
    # knn = NearestNeighbors(n_neighbors=k, algorithm='auto')
    # knn.fit(face_centroids)
    # distances, indices = knn.kneighbors(face_centroids)
    # # 'distances[i]' are the distances from face i to each of its 'k' neighbors
    # # 'indices[i]' are the face indices of those neighbors

    #--------------------------------------------------------------------------
    # 4b) Build a KNN graph on connected components
    #--------------------------------------------------------------------------
    # Group faces by their root representative in the Union-Find structure
    component_dict = {}
    for face_idx in range(num_faces):
        root = uf.find(face_idx)
        if root not in component_dict:
            component_dict[root] = set()
        component_dict[root].add(face_idx)

    connected_components = list(component_dict.values())
    
    print("Using connected component MST.")
    component_centroid_face_idx = []
    connected_component_centroids = []
    knn = NearestNeighbors(n_neighbors=1, algorithm='auto')
    for component in connected_components:
        curr_component_faces = list(component)
        curr_component_face_centroids = face_centroids[curr_component_faces]
        component_centroid = np.mean(curr_component_face_centroids, axis=0)

        ### Assign a face closest to the centroid
        face_idx = curr_component_faces[np.argmin(np.linalg.norm(curr_component_face_centroids-component_centroid, axis=-1))]

        connected_component_centroids.append(component_centroid)
        component_centroid_face_idx.append(face_idx)

    component_centroid_face_idx = np.array(component_centroid_face_idx)
    connected_component_centroids = np.array(connected_component_centroids)

    if n_components < k:
        knn = NearestNeighbors(n_neighbors=n_components, algorithm='auto')
    else:
        knn = NearestNeighbors(n_neighbors=k, algorithm='auto')
    knn.fit(connected_component_centroids)
    distances, indices = knn.kneighbors(connected_component_centroids)    

    #--------------------------------------------------------------------------
    # 5) Build a weighted graph in NetworkX using centroid-distances as edges
    #--------------------------------------------------------------------------
    G = nx.Graph()
    # Add each face as a node in the graph
    G.add_nodes_from(range(num_faces))

    # For each face i, add edges (i -> j) for each neighbor j in the KNN
    for idx1 in range(n_components):
        i = component_centroid_face_idx[idx1]
        for idx2, dist in zip(indices[idx1], distances[idx1]):
            j = component_centroid_face_idx[idx2]
            if i == j:
                continue  # skip self-loop
            # Add an undirected edge with 'weight' = distance
            # NetworkX handles parallel edges gracefully via last add_edge,
            # but it typically overwrites the weight if (i, j) already exists.
            G.add_edge(i, j, weight=dist)

    #--------------------------------------------------------------------------
    # 6) Compute MST on that KNN graph
    #--------------------------------------------------------------------------
    mst = nx.minimum_spanning_tree(G, weight='weight')
    # Sort MST edges by ascending weight, so we add the shortest edges first
    mst_edges_sorted = sorted(
        mst.edges(data=True), key=lambda e: e[2]['weight']
    )
    print("mst edges sorted", len(mst_edges_sorted))
    #--------------------------------------------------------------------------
    # 7) Use a union-find structure to add MST edges only if they
    #    connect two currently disconnected components of the adjacency matrix
    #--------------------------------------------------------------------------

    # Convert face_adjacency to LIL format for efficient edge addition
    adjacency_lil = face_adjacency.tolil()

    # Now, step through MST edges in ascending order
    for (u, v, attr) in mst_edges_sorted:
        if uf.find(u) != uf.find(v):
            # These belong to different components, so unify them
            uf.union(u, v)
            # And add a "dummy" edge to our adjacency matrix
            adjacency_lil[u, v] = 1
            adjacency_lil[v, u] = 1

    # Convert back to CSR format and return
    face_adjacency = adjacency_lil.tocsr()

    if with_knn:
        print("Adding KNN edges.")
        ### Add KNN edges graph too
        dummy_row = []
        dummy_col = []
        for idx1 in range(n_components):
            i = component_centroid_face_idx[idx1]
            for idx2 in indices[idx1]:
                j = component_centroid_face_idx[idx2]     
                dummy_row.extend([i, j])
                dummy_col.extend([j, i]) ### duplicates are handled by coo

        dummy_data = np.ones(len(dummy_row), dtype=np.int16)
        dummy_mat = coo_matrix(
            (dummy_data, (dummy_row, dummy_col)),
            shape=(num_faces, num_faces)
        ).tocsr()
        face_adjacency = face_adjacency + dummy_mat
        ###########################

    return face_adjacency
#########################

def construct_face_adjacency_matrix_facemst(face_list, vertices, k=10, with_knn=True):
    """
    Given a list of faces (each face is a 3-tuple of vertex indices),
    construct a face-based adjacency matrix of shape (num_faces, num_faces).

    Two faces are adjacent if they share an edge (the "mesh adjacency").
    If multiple connected components remain, we:
      1) Compute the centroid of each face.
      2) Use a KNN graph (k=10) based on centroid distances.
      3) Compute MST of that KNN graph.
      4) Add MST edges that connect different components as "dummy" edges
         in the face adjacency matrix, ensuring one connected component.

    Parameters
    ----------
    face_list : list of tuples
        List of faces, each face is a tuple (v0, v1, v2) of vertex indices.
    vertices : np.ndarray of shape (num_vertices, 3)
        Array of vertex coordinates.
    k : int, optional
        Number of neighbors to use in centroid KNN. Default is 10.

    Returns
    -------
    face_adjacency : scipy.sparse.csr_matrix
        A CSR sparse matrix of shape (num_faces, num_faces),
        containing 1s for adjacent faces (shared-edge adjacency)
        plus dummy edges ensuring a single connected component.
    """
    num_faces = len(face_list)
    if num_faces == 0:
        # Return an empty matrix if no faces
        return csr_matrix((0, 0))

    #--------------------------------------------------------------------------
    # 1) Build adjacency based on shared edges.
    #    (Same logic as the original code, plus import statements.)
    #--------------------------------------------------------------------------
    edge_to_faces = defaultdict(list)
    uf = UnionFind(num_faces)
    for f_idx, (v0, v1, v2) in enumerate(face_list):
        # Sort each edge’s endpoints so (i, j) == (j, i)
        edges = [
            tuple(sorted((v0, v1))),
            tuple(sorted((v1, v2))),
            tuple(sorted((v2, v0)))
        ]
        for e in edges:
            edge_to_faces[e].append(f_idx)

    row = []
    col = []
    for edge, face_indices in edge_to_faces.items():
        unique_faces = list(set(face_indices))
        if len(unique_faces) > 1:
            # For every pair of distinct faces that share this edge,
            # mark them as mutually adjacent
            for i in range(len(unique_faces)):
                for j in range(i + 1, len(unique_faces)):
                    fi = unique_faces[i]
                    fj = unique_faces[j]
                    row.append(fi)
                    col.append(fj)
                    row.append(fj)
                    col.append(fi)
                    uf.union(fi, fj)

    data = np.ones(len(row), dtype=np.int8)
    face_adjacency = coo_matrix(
        (data, (row, col)), shape=(num_faces, num_faces)
    ).tocsr()

    #--------------------------------------------------------------------------
    # 2) Check if the graph from shared edges is already connected.
    #--------------------------------------------------------------------------
    n_components = 0
    for i in range(num_faces):
        if uf.find(i) == i:
            n_components += 1
    print("n_components", n_components)

    if n_components == 1:
        # Already a single connected component, no need for dummy edges
        return face_adjacency
    #--------------------------------------------------------------------------
    # 3) Compute centroids of each face for building a KNN graph.
    #--------------------------------------------------------------------------
    face_centroids = []
    for (v0, v1, v2) in face_list:
        centroid = (vertices[v0] + vertices[v1] + vertices[v2]) / 3.0
        face_centroids.append(centroid)
    face_centroids = np.array(face_centroids)

    #--------------------------------------------------------------------------
    # 4) Build a KNN graph (k=10) over face centroids using scikit‐learn
    #--------------------------------------------------------------------------
    knn = NearestNeighbors(n_neighbors=k, algorithm='auto')
    knn.fit(face_centroids)
    distances, indices = knn.kneighbors(face_centroids)
    # 'distances[i]' are the distances from face i to each of its 'k' neighbors
    # 'indices[i]' are the face indices of those neighbors

    #--------------------------------------------------------------------------
    # 5) Build a weighted graph in NetworkX using centroid-distances as edges
    #--------------------------------------------------------------------------
    G = nx.Graph()
    # Add each face as a node in the graph
    G.add_nodes_from(range(num_faces))

    # For each face i, add edges (i -> j) for each neighbor j in the KNN
    for i in range(num_faces):
        for j, dist in zip(indices[i], distances[i]):
            if i == j:
                continue  # skip self-loop
            # Add an undirected edge with 'weight' = distance
            # NetworkX handles parallel edges gracefully via last add_edge,
            # but it typically overwrites the weight if (i, j) already exists.
            G.add_edge(i, j, weight=dist)

    #--------------------------------------------------------------------------
    # 6) Compute MST on that KNN graph
    #--------------------------------------------------------------------------
    mst = nx.minimum_spanning_tree(G, weight='weight')
    # Sort MST edges by ascending weight, so we add the shortest edges first
    mst_edges_sorted = sorted(
        mst.edges(data=True), key=lambda e: e[2]['weight']
    )
    print("mst edges sorted", len(mst_edges_sorted))
    #--------------------------------------------------------------------------
    # 7) Use a union-find structure to add MST edges only if they
    #    connect two currently disconnected components of the adjacency matrix
    #--------------------------------------------------------------------------

    # Convert face_adjacency to LIL format for efficient edge addition
    adjacency_lil = face_adjacency.tolil()

    # Now, step through MST edges in ascending order
    for (u, v, attr) in mst_edges_sorted:
        if uf.find(u) != uf.find(v):
            # These belong to different components, so unify them
            uf.union(u, v)
            # And add a "dummy" edge to our adjacency matrix
            adjacency_lil[u, v] = 1
            adjacency_lil[v, u] = 1

    # Convert back to CSR format and return
    face_adjacency = adjacency_lil.tocsr()

    if with_knn:
        print("Adding KNN edges.")
        ### Add KNN edges graph too
        dummy_row = []
        dummy_col = []
        for i in range(num_faces):
            for j in indices[i]:        
                dummy_row.extend([i, j])
                dummy_col.extend([j, i]) ### duplicates are handled by coo

        dummy_data = np.ones(len(dummy_row), dtype=np.int16)
        dummy_mat = coo_matrix(
            (dummy_data, (dummy_row, dummy_col)),
            shape=(num_faces, num_faces)
        ).tocsr()
        face_adjacency = face_adjacency + dummy_mat
        ###########################

    return face_adjacency

def construct_face_adjacency_matrix_naive(face_list):
    """
    Given a list of faces (each face is a 3-tuple of vertex indices),
    construct a face-based adjacency matrix of shape (num_faces, num_faces).
    Two faces are adjacent if they share an edge.

    If multiple connected components exist, dummy edges are added to 
    turn them into a single connected component. Edges are added naively by
    randomly selecting a face and connecting consecutive components -- (comp_i, comp_i+1) ...

    Parameters
    ----------
    face_list : list of tuples
        List of faces, each face is a tuple (v0, v1, v2) of vertex indices.

    Returns
    -------
    face_adjacency : scipy.sparse.csr_matrix
        A CSR sparse matrix of shape (num_faces, num_faces), 
        containing 1s for adjacent faces and 0s otherwise. 
        Additional edges are added if the faces are in multiple components.
    """

    num_faces = len(face_list)
    if num_faces == 0:
        # Return an empty matrix if no faces
        return csr_matrix((0, 0))

    # Step 1: Map each undirected edge -> list of face indices that contain that edge
    edge_to_faces = defaultdict(list)

    # Populate the edge_to_faces dictionary
    for f_idx, (v0, v1, v2) in enumerate(face_list):
        # For an edge, we always store its endpoints in sorted order
        # to avoid duplication (e.g. edge (2,5) is the same as (5,2)).
        edges = [
            tuple(sorted((v0, v1))),
            tuple(sorted((v1, v2))),
            tuple(sorted((v2, v0)))
        ]
        for e in edges:
            edge_to_faces[e].append(f_idx)

    # Step 2: Build the adjacency (row, col) lists among faces
    row = []
    col = []
    for e, faces_sharing_e in edge_to_faces.items():
        # If an edge is shared by multiple faces, make each pair of those faces adjacent
        f_indices = list(set(faces_sharing_e))  # unique face indices for this edge
        if len(f_indices) > 1:
            # For each pair of faces, mark them as adjacent
            for i in range(len(f_indices)):
                for j in range(i + 1, len(f_indices)):
                    f_i = f_indices[i]
                    f_j = f_indices[j]
                    row.append(f_i)
                    col.append(f_j)
                    row.append(f_j)
                    col.append(f_i)

    # Create a COO matrix, then convert it to CSR
    data = np.ones(len(row), dtype=np.int8)
    face_adjacency = coo_matrix(
        (data, (row, col)),
        shape=(num_faces, num_faces)
    ).tocsr()

    # Step 3: Ensure single connected component
    # Use connected_components to see how many components exist
    n_components, labels = connected_components(face_adjacency, directed=False)

    if n_components > 1:
        # We have multiple components; let's "connect" them via dummy edges
        # The simplest approach is to pick one face from each component
        # and connect them sequentially to enforce a single component.
        component_representatives = []

        for comp_id in range(n_components):
            # indices of faces in this component
            faces_in_comp = np.where(labels == comp_id)[0]
            if len(faces_in_comp) > 0:
                # take the first face in this component as a representative
                component_representatives.append(faces_in_comp[0])

        # Now, add edges between consecutive representatives
        dummy_row = []
        dummy_col = []
        for i in range(len(component_representatives) - 1):
            f_i = component_representatives[i]
            f_j = component_representatives[i + 1]
            dummy_row.extend([f_i, f_j])
            dummy_col.extend([f_j, f_i])

        if dummy_row:
            dummy_data = np.ones(len(dummy_row), dtype=np.int8)
            dummy_mat = coo_matrix(
                (dummy_data, (dummy_row, dummy_col)),
                shape=(num_faces, num_faces)
            ).tocsr()
            face_adjacency = face_adjacency + dummy_mat

    return face_adjacency

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [1] * n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)
        
        if rootX != rootY:
            if self.rank[rootX] > self.rank[rootY]:
                self.parent[rootY] = rootX
            elif self.rank[rootX] < self.rank[rootY]:
                self.parent[rootX] = rootY
            else:
                self.parent[rootY] = rootX
                self.rank[rootX] += 1

def hierarchical_clustering_labels(children, n_samples, max_cluster=20):
    # Union-Find structure to maintain cluster merges
    uf = UnionFind(2 * n_samples - 1)  # We may need to store up to 2*n_samples - 1 clusters
    
    current_cluster_count = n_samples
    
    # Process merges from the children array
    hierarchical_labels = []
    for i, (child1, child2) in enumerate(children):
        uf.union(child1, i + n_samples)
        uf.union(child2, i + n_samples)
        #uf.union(child1, child2)
        current_cluster_count -= 1  # After each merge, we reduce the cluster count
        
        if current_cluster_count <= max_cluster:
            labels = [uf.find(i) for i in range(n_samples)]
            hierarchical_labels.append(labels)
    
    return hierarchical_labels

def load_ply_to_numpy(filename):
    """
    Load a PLY file and extract the point cloud as a (N, 3) NumPy array.

    Parameters:
        filename (str): Path to the PLY file.

    Returns:
        numpy.ndarray: Point cloud array of shape (N, 3).
    """
    # Read PLY file
    ply_data = PlyData.read(filename)
    
    # Extract vertex data
    vertex_data = ply_data["vertex"]
    
    # Convert to NumPy array (x, y, z)
    points = np.vstack([vertex_data["x"], vertex_data["y"], vertex_data["z"]]).T
    
    return points

def solve_clustering(input_fname, uid, view_id, save_dir="test_results1", out_render_fol= "test_render_clustering", use_agglo=False, max_num_clusters=18, is_pc=False, option=1, with_knn=True, export_mesh=True, auto_select=False, min_preferred_clusters=3, max_preferred_clusters=15, sample_size=5000, early_stop_patience=3):
    """
    Run clustering on PartField features and optionally auto-select the best clustering.

    Args:
        auto_select: If True, evaluate all clusterings and save only the best one
        min_preferred_clusters: Minimum preferred cluster count for auto-selection
        max_preferred_clusters: Maximum preferred cluster count for auto-selection
        sample_size: Sample size for silhouette score computation (5000 = ~100x speedup)
        early_stop_patience: Stop after N consecutive non-improving k values (0 = disabled)
    """
    print(uid, view_id)

    if not is_pc:
        input_fname = f'{save_dir}/input_{uid}_{view_id}.ply'
        mesh = load_mesh_util(input_fname)

    else:
        pc = load_ply_to_numpy(input_fname)

    ### Load inferred PartField features
    try:
        point_feat = np.load(f'{save_dir}/part_feat_{uid}_{view_id}.npy')
    except:
        try:
            point_feat = np.load(f'{save_dir}/part_feat_{uid}_{view_id}_batch.npy')

        except:
            print()
            print("pointfeat loading error. skipping...")
            print(f'{save_dir}/part_feat_{uid}_{view_id}_batch.npy')
            return

    point_feat = point_feat / np.linalg.norm(point_feat, axis=-1, keepdims=True)

    if not use_agglo:
        # Store all labels for auto-selection
        all_labels_dict = {}

        for num_cluster in range(2, max_num_clusters):
            clustering = KMeans(n_clusters=num_cluster, random_state=0).fit(point_feat)
            labels = clustering.labels_


            pred_labels = np.zeros((len(labels), 1))
            for i, label in enumerate(np.unique(labels)):
                # print(i, label)
                pred_labels[labels == label] = i  # Assign RGB values to each label

            # Store for auto-selection
            all_labels_dict[num_cluster] = pred_labels.copy()

            fname_clustering = os.path.join(out_render_fol, "cluster_out", str(uid) + "_" + str(view_id) + "_" + str(num_cluster).zfill(2))
            np.save(fname_clustering, pred_labels)


            if not is_pc:
                V = mesh.vertices
                F = mesh.faces

                if export_mesh :
                    fname_mesh = os.path.join(out_render_fol, "ply", str(uid) + "_" + str(view_id) + "_" + str(num_cluster).zfill(2) + ".ply")
                    export_colored_mesh_ply(V, F, pred_labels, filename=fname_mesh)


            else:
                if export_mesh:
                    fname_pc = os.path.join(out_render_fol, "ply", str(uid) + "_" + str(view_id) + "_" + str(num_cluster).zfill(2) + ".ply")
                    export_pointcloud_with_labels_to_ply(pc, pred_labels, filename=fname_pc)

        # Auto-select best clustering
        if auto_select and len(all_labels_dict) > 0:
            print(f"  Auto-selecting best clustering for {uid} (sample_size={sample_size}, patience={early_stop_patience})...")
            best_n, results = select_best_clustering(
                point_feat, all_labels_dict,
                min_clusters=min_preferred_clusters,
                max_clusters=max_preferred_clusters,
                prefer_fewer=True,
                verbose=True,
                sample_size=sample_size,
                early_stop_patience=early_stop_patience
            )

            # Save best selection info
            best_info_file = os.path.join(out_render_fol, "cluster_out", f"{uid}_{view_id}_best.txt")
            with open(best_info_file, 'w') as f:
                f.write(f"best_n_clusters: {best_n}\n")
                f.write(f"score: {results[best_n]['combined_score']:.4f}\n")
                f.write(f"silhouette: {results[best_n]['metrics']['silhouette']:.4f}\n")
                f.write(f"davies_bouldin: {results[best_n]['metrics']['davies_bouldin']:.4f}\n")
                f.write(f"calinski_harabasz: {results[best_n]['metrics']['calinski_harabasz']:.4f}\n")

            # Save just the best cluster count as numpy for easy loading
            np.save(os.path.join(out_render_fol, "cluster_out", f"{uid}_{view_id}_best_n.npy"), np.array([best_n]))

            print(f"  Best clustering saved: {best_n} clusters")
            return best_n
        
    else:
        if is_pc:
            print("Not implemented error. Agglomerative clustering only for mesh inputs.")
            exit()

        if option == 0:
            adj_matrix = construct_face_adjacency_matrix_naive(mesh.faces)
        elif option == 1:
            adj_matrix = construct_face_adjacency_matrix_facemst(mesh.faces, mesh.vertices, with_knn=with_knn)
        else:
            adj_matrix = construct_face_adjacency_matrix_ccmst(mesh.faces, mesh.vertices, with_knn=with_knn)

        clustering = AgglomerativeClustering(connectivity=adj_matrix,
                                    n_clusters=1,
                                    ).fit(point_feat)
        hierarchical_labels = hierarchical_clustering_labels(clustering.children_, point_feat.shape[0], max_cluster=max_num_clusters)

        all_FL = []
        for n_cluster in range(max_num_clusters):
            print("Processing cluster: "+str(n_cluster))
            labels = hierarchical_labels[n_cluster]
            all_FL.append(labels)
        
        
        all_FL = np.array(all_FL)
        unique_labels = np.unique(all_FL)

        for n_cluster in range(max_num_clusters):
            FL = all_FL[n_cluster]
            relabel = np.zeros((len(FL), 1))
            for i, label in enumerate(unique_labels):
                relabel[FL == label] = i  # Assign RGB values to each label

            V = mesh.vertices
            F = mesh.faces

            if export_mesh :
                fname_mesh = os.path.join(out_render_fol, "ply", str(uid) + "_" + str(view_id) + "_" + str(max_num_clusters - n_cluster).zfill(2) + ".ply")
                export_colored_mesh_ply(V, F, FL, filename=fname_mesh)

            fname_clustering = os.path.join(out_render_fol, "cluster_out", str(uid) + "_" + str(view_id) + "_" + str(max_num_clusters - n_cluster).zfill(2))
            np.save(fname_clustering, FL)
        
        
            
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', default= "", type=str)
    parser.add_argument('--root', default= "", type=str)
    parser.add_argument('--dump_dir', default= "", type=str)

    parser.add_argument('--max_num_clusters', default= 20, type=int)
    parser.add_argument('--use_agglo', default= False, type=bool)
    parser.add_argument('--is_pc', default= False, type=bool)
    parser.add_argument('--option', default= 1, type=int)
    parser.add_argument('--with_knn', default= False, type=bool)

    parser.add_argument('--export_mesh', default= True, type=bool)

    # Auto-selection arguments
    parser.add_argument('--auto_select', action='store_true',
                       help='Auto-select best clustering using quality metrics (silhouette, Davies-Bouldin, Calinski-Harabasz)')
    parser.add_argument('--min_preferred_clusters', default=3, type=int,
                       help='Minimum preferred cluster count for auto-selection (default: 3)')
    parser.add_argument('--max_preferred_clusters', default=15, type=int,
                       help='Maximum preferred cluster count for auto-selection (default: 15)')

    # Performance optimization arguments
    parser.add_argument('--sample_size', default=40000, type=int,
                       help='Sample size for silhouette computation (default: 40000). Larger = more accurate but slower.')
    parser.add_argument('--early_stop_patience', default=0, type=int,
                       help='Stop evaluation after N consecutive non-improving cluster counts (default: 0 = disabled)')

    FLAGS = parser.parse_args()
    root = FLAGS.root
    OUTPUT_FOL = FLAGS.dump_dir
    SOURCE_DIR = FLAGS.source_dir

    MAX_NUM_CLUSTERS = FLAGS.max_num_clusters
    USE_AGGLO = FLAGS.use_agglo
    IS_PC = FLAGS.is_pc

    OPTION = FLAGS.option
    WITH_KNN = FLAGS.with_knn

    EXPORT_MESH = FLAGS.export_mesh

    AUTO_SELECT = FLAGS.auto_select
    MIN_PREFERRED = FLAGS.min_preferred_clusters
    MAX_PREFERRED = FLAGS.max_preferred_clusters
    SAMPLE_SIZE = FLAGS.sample_size
    EARLY_STOP_PATIENCE = FLAGS.early_stop_patience

    models = os.listdir(root)
    os.makedirs(OUTPUT_FOL, exist_ok=True)

    cluster_fol = os.path.join(OUTPUT_FOL, "cluster_out")
    os.makedirs(cluster_fol, exist_ok=True)

    if EXPORT_MESH:
        ply_fol = os.path.join(OUTPUT_FOL, "ply")
        os.makedirs(ply_fol, exist_ok=True)

    #### Get existing model_ids ###
    # Extract model IDs from existing ply files
    # Format: {uid}_{view_id}_{cluster_num}.ply, e.g., "van_model_0_02.ply" -> "van_model"
    # We extract everything before the last two underscore-separated parts
    ply_files = os.listdir(os.path.join(OUTPUT_FOL, "ply"))

    existing_model_ids = []
    for sample in ply_files:
        # Remove .ply extension and split by underscore
        parts = sample.replace(".ply", "").rsplit("_", 2)
        if len(parts) >= 3:
            # uid is everything except the last two parts (view_id and cluster_num)
            uid = parts[0]
        else:
            # Fallback for unexpected format
            uid = sample.split("_")[0]

        if uid not in existing_model_ids:
            existing_model_ids.append(uid)
    ##############################

    all_files = os.listdir(SOURCE_DIR)
    selected = []
    for f in all_files:
        # Extract model ID the same way solve_clustering does: model.split(".")[-2]
        # This handles filenames like "3dpea.com_assembly1.glb" -> "com_assembly1"
        if ".ply" in f and IS_PC:
            model_id = f.split(".")[0]  # For ply files, take everything before first dot
            if model_id not in existing_model_ids:
                selected.append(f)
        elif (".obj" in f or ".glb" in f) and not IS_PC:
            model_id = f.split(".")[-2]  # For obj/glb, take the part before extension
            if model_id not in existing_model_ids:
                selected.append(f)

    print("Number of models to process: " + str(len(selected)))
    if AUTO_SELECT:
        print(f"Auto-selection enabled: preferring {MIN_PREFERRED}-{MAX_PREFERRED} clusters")
        print(f"Performance: sample_size={SAMPLE_SIZE}, early_stop_patience={EARLY_STOP_PATIENCE}")

    best_selections = {}
    for model in selected:
        fname = os.path.join(SOURCE_DIR, model)
        uid = model.split(".")[-2]
        view_id = 0

        result = solve_clustering(fname, uid, view_id, save_dir=root, out_render_fol=OUTPUT_FOL,
                                 use_agglo=USE_AGGLO, max_num_clusters=MAX_NUM_CLUSTERS,
                                 is_pc=IS_PC, option=OPTION, with_knn=WITH_KNN,
                                 export_mesh=EXPORT_MESH, auto_select=AUTO_SELECT,
                                 min_preferred_clusters=MIN_PREFERRED,
                                 max_preferred_clusters=MAX_PREFERRED,
                                 sample_size=SAMPLE_SIZE,
                                 early_stop_patience=EARLY_STOP_PATIENCE)
        if AUTO_SELECT and result is not None:
            best_selections[uid] = result

    # Print summary if auto-selection was used
    if AUTO_SELECT and best_selections:
        print("\n" + "=" * 50)
        print("AUTO-SELECTION SUMMARY")
        print("=" * 50)
        for uid, best_n in best_selections.items():
            print(f"  {uid}: {best_n} clusters")
        avg = np.mean(list(best_selections.values()))
        print(f"\nAverage best cluster count: {avg:.1f}")
