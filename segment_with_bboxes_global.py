#!/usr/bin/env python3
"""
Segment with Global-Oriented Bounding Boxes - Fit OBBs with consistent global orientation.

This tool generates oriented bounding boxes (OBBs) where ALL boxes share a common
global orientation computed from the entire mesh via PCA. This is different from
segment_with_bboxes.py which orients each box to its own segment's principal axes.

Key difference:
- segment_with_bboxes.py: Each segment's OBB oriented to its own PCA axes
- THIS FILE: All OBBs share a single global rotation from the whole mesh

When to use global orientation:
- CAD models with consistent part orientations
- Architectural models where alignment matters
- Models where visual consistency of boxes is desired

When to use per-segment orientation (segment_with_bboxes.py):
- Organic models with parts at various angles
- When you want the tightest possible fit per segment

Usage:
    python segment_with_bboxes_global.py -i model.glb -l labels.npy -o output.ply
    python segment_with_bboxes_global.py -i model.glb -l labels.npy -o output.ply --style all
"""

import os
import argparse
import numpy as np
from typing import List, Tuple, Optional, Dict
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

import trimesh
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt


class PointCloudFilter:
    """Filters point clouds to remove outliers and low-density regions."""

    def __init__(self,
                 outlier_std_ratio: float = 1.5,
                 density_percentile: float = 15.0,
                 min_points_for_filtering: int = 20,
                 k_neighbors: int = 12):
        """
        Args:
            outlier_std_ratio: Points beyond this many std devs from centroid are outliers (lower = more aggressive)
            density_percentile: Remove points below this percentile in local density (higher = more aggressive)
            min_points_for_filtering: Minimum points required to apply filtering
            k_neighbors: Number of neighbors for density estimation
        """
        self.outlier_std_ratio = outlier_std_ratio
        self.density_percentile = density_percentile
        self.min_points_for_filtering = min_points_for_filtering
        self.k_neighbors = k_neighbors

    def remove_statistical_outliers(self, points: np.ndarray) -> np.ndarray:
        """Remove outliers based on distance from centroid using IQR method."""
        if len(points) < self.min_points_for_filtering:
            return points

        centroid = np.mean(points, axis=0)
        distances = np.linalg.norm(points - centroid, axis=1)

        # Use IQR method for robust outlier detection
        q1, q3 = np.percentile(distances, [25, 75])
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr

        # Also use standard deviation as a secondary check
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        std_upper = mean_dist + self.outlier_std_ratio * std_dist

        # Use the more conservative bound (smaller value)
        threshold = min(upper_bound, std_upper)

        mask = distances <= threshold
        filtered = points[mask]

        # Ensure we keep at least 50% of points
        if len(filtered) < len(points) * 0.5:
            # Fall back to keeping points within std_ratio
            mask = distances <= std_upper
            filtered = points[mask]

        return filtered if len(filtered) >= 3 else points

    def remove_low_density_points(self, points: np.ndarray) -> np.ndarray:
        """Remove points in low-density regions using KNN density estimation."""
        if len(points) < self.min_points_for_filtering:
            return points

        k = min(self.k_neighbors, len(points) - 1)
        if k < 2:
            return points

        # Compute local density using KNN
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(points)
        distances, _ = nbrs.kneighbors(points)

        # Local density = inverse of mean distance to k neighbors
        mean_distances = np.mean(distances[:, 1:], axis=1)  # Exclude self

        # Avoid division by zero
        mean_distances = np.maximum(mean_distances, 1e-10)
        local_density = 1.0 / mean_distances

        # Remove points below the density percentile threshold
        threshold = np.percentile(local_density, self.density_percentile)
        mask = local_density >= threshold

        filtered = points[mask]
        return filtered if len(filtered) >= 3 else points

    def remove_off_angle_protrusions(self, points: np.ndarray) -> np.ndarray:
        """Remove points that form off-angle protrusions from the main body."""
        if len(points) < self.min_points_for_filtering:
            return points

        # Compute initial PCA to find main orientation
        centroid = np.mean(points, axis=0)
        centered = points - centroid

        try:
            pca = PCA(n_components=3)
            pca.fit(centered)
            transformed = pca.transform(centered)

            # For each axis, remove extreme outliers
            # Focus especially on minor axes (2nd and 3rd components)
            masks = []
            for axis in range(3):
                axis_values = transformed[:, axis]
                q1, q3 = np.percentile(axis_values, [10, 90])
                iqr = q3 - q1

                # Tighter bounds for minor axes
                multiplier = 1.5 if axis == 0 else 1.2
                lower = q1 - multiplier * iqr
                upper = q3 + multiplier * iqr

                masks.append((axis_values >= lower) & (axis_values <= upper))

            combined_mask = masks[0] & masks[1] & masks[2]
            filtered = points[combined_mask]

            return filtered if len(filtered) >= 3 else points
        except:
            return points

    def filter_points(self, points: np.ndarray) -> np.ndarray:
        """Apply all filtering steps to clean the point cloud."""
        if len(points) < 3:
            return points

        # Step 1: Remove statistical outliers (distance-based)
        filtered = self.remove_statistical_outliers(points)

        # Step 2: Remove low-density points
        filtered = self.remove_low_density_points(filtered)

        # Step 3: Remove off-angle protrusions
        filtered = self.remove_off_angle_protrusions(filtered)

        return filtered


class GlobalOrientationComputer:
    """Computes global orientation for the entire mesh using PCA."""

    def __init__(self, method: str = 'all_vertices'):
        """
        Args:
            method: 'all_vertices' - PCA on all mesh vertices (after outlier removal)
                    'single_box' - Same as all_vertices (conceptually fitting one OBB)
        """
        self.method = method
        self._global_rotation = None
        self._global_center = None
        self._num_outliers_removed = 0

    def compute_from_mesh(self, mesh: trimesh.Trimesh) -> np.ndarray:
        """Compute global orientation from entire mesh.

        IMPORTANT: Removes extreme outliers BEFORE computing PCA
        to ensure stable orientation estimation.

        Returns:
            3x3 rotation matrix representing global principal directions
        """
        points = mesh.vertices.copy()
        original_count = len(points)

        # Remove extreme outliers before computing global PCA
        points = self._remove_extreme_outliers(points)
        self._num_outliers_removed = original_count - len(points)

        return self._compute_pca_orientation(points)

    def _remove_extreme_outliers(self, points: np.ndarray) -> np.ndarray:
        """Remove extreme outliers using IQR method on distance from centroid.

        Uses a conservative 2x IQR threshold to only remove truly extreme outliers.
        """
        if len(points) < 10:
            return points

        centroid = np.mean(points, axis=0)
        distances = np.linalg.norm(points - centroid, axis=1)

        q1, q3 = np.percentile(distances, [25, 75])
        iqr = q3 - q1
        upper_bound = q3 + 2.0 * iqr  # Use 2x IQR for extreme outliers only

        mask = distances <= upper_bound
        filtered = points[mask]

        # Ensure we keep at least 10 points for valid PCA
        return filtered if len(filtered) >= 10 else points

    def _compute_pca_orientation(self, points: np.ndarray) -> np.ndarray:
        """Compute PCA orientation from filtered vertices."""
        center = np.mean(points, axis=0)
        centered = points - center

        pca = PCA(n_components=3)
        pca.fit(centered)

        # Get rotation matrix (columns are principal axes)
        rotation = self._ensure_right_handed(pca.components_.T)

        self._global_rotation = rotation
        self._global_center = center
        return rotation

    def _ensure_right_handed(self, rotation: np.ndarray) -> np.ndarray:
        """Ensure the rotation matrix forms a right-handed coordinate system."""
        if np.linalg.det(rotation) < 0:
            rotation[:, 2] = -rotation[:, 2]
        return rotation

    @property
    def global_rotation(self) -> np.ndarray:
        """Get the computed global rotation matrix."""
        return self._global_rotation

    @property
    def global_center(self) -> np.ndarray:
        """Get the global center (mean of filtered vertices)."""
        return self._global_center

    @property
    def num_outliers_removed(self) -> int:
        """Get the number of outliers removed during global orientation computation."""
        return self._num_outliers_removed


class OBBFitterGlobal:
    """Fits oriented bounding boxes using a global orientation."""

    def __init__(self, global_rotation: np.ndarray,
                 min_points: int = 3,
                 apply_filtering: bool = True):
        """
        Args:
            global_rotation: 3x3 rotation matrix for global orientation (shared by all boxes)
            min_points: Minimum points required for fitting
            apply_filtering: Whether to filter outliers per segment
        """
        self.global_rotation = global_rotation
        self.min_points = min_points
        self.apply_filtering = apply_filtering
        self.filter = PointCloudFilter()

    def fit_obb(self, points: np.ndarray, filter_points: bool = None) -> dict:
        """Fit oriented bounding box using GLOBAL orientation.

        Algorithm:
        1. Optionally filter points (outliers, density)
        2. Project filtered points onto global PCA axes
        3. Compute axis-aligned bounds in global PCA space
        4. Create box with global rotation but segment-specific center/dimensions

        Args:
            points: Nx3 array of segment points
            filter_points: Override filtering setting

        Returns:
            dict with: center, dimensions, rotation, mesh, filtered_points, original_points
        """
        original_points = points.copy()
        should_filter = filter_points if filter_points is not None else self.apply_filtering

        if len(points) < self.min_points:
            center = np.mean(points, axis=0) if len(points) > 0 else np.zeros(3)
            return {
                'center': center,
                'dimensions': np.array([0.01, 0.01, 0.01]),
                'rotation': self.global_rotation,  # Use global rotation
                'mesh': None,
                'filtered_points': points,
                'original_points': original_points
            }

        # Step 1: Apply filtering to remove outliers
        if should_filter:
            points = self.filter.filter_points(points)

        # Step 2: Project points onto GLOBAL PCA axes (not segment-specific)
        center = np.mean(points, axis=0)
        centered = points - center

        # Transform to global PCA space
        # global_rotation columns are principal axes, so multiply to project
        transformed = centered @ self.global_rotation

        # Step 3: Compute axis-aligned bounds in global PCA space
        # Use robust bounds (percentile-based) to further reduce outlier influence
        percentile_margin = 2  # Use 2nd to 98th percentile for bounds
        min_c = np.percentile(transformed, percentile_margin, axis=0)
        max_c = np.percentile(transformed, 100 - percentile_margin, axis=0)
        dimensions = np.maximum(max_c - min_c, 0.001)

        # Step 4: Compute box center in world coordinates
        # Center of the bounding box in PCA space
        bbox_center_pca = (min_c + max_c) / 2

        # Transform back to world coordinates
        center = center + bbox_center_pca @ self.global_rotation.T

        # Step 5: Create box mesh with GLOBAL rotation
        mesh = self._create_box_mesh(center, dimensions, self.global_rotation)

        return {
            'center': center,
            'dimensions': dimensions,
            'rotation': self.global_rotation,  # All boxes share this rotation
            'mesh': mesh,
            'filtered_points': points,
            'original_points': original_points
        }

    def _create_box_mesh(self, center: np.ndarray, dims: np.ndarray,
                         rotation: np.ndarray) -> trimesh.Trimesh:
        """Create solid box mesh with given center, dimensions, and rotation."""
        box = trimesh.creation.box(extents=dims)
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = center
        box.apply_transform(transform)
        return box

    def _create_wireframe_box(self, center: np.ndarray, dims: np.ndarray,
                              rotation: np.ndarray) -> trimesh.Trimesh:
        """Create wireframe box as thin cylinder edges."""
        # 8 corners of unit box
        corners = np.array([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
        ]) * 0.5

        # Scale by dimensions
        corners = corners * dims

        # Rotate and translate
        corners = corners @ rotation.T + center

        # 12 edges
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # bottom face
            [4, 5], [5, 6], [6, 7], [7, 4],  # top face
            [0, 4], [1, 5], [2, 6], [3, 7]   # vertical edges
        ]

        # Create thin cylinders for each edge
        meshes = []
        edge_radius = min(dims) * 0.02  # 2% of smallest dimension

        for e in edges:
            p1, p2 = corners[e[0]], corners[e[1]]
            # Create cylinder between two points
            direction = p2 - p1
            height = np.linalg.norm(direction)
            if height < 1e-6:
                continue

            cyl = trimesh.creation.cylinder(radius=edge_radius, height=height, sections=8)

            # Align cylinder with edge direction
            z_axis = np.array([0, 0, 1])
            direction_norm = direction / height

            # Rotation to align z-axis with edge direction
            if np.allclose(direction_norm, z_axis):
                rot_matrix = np.eye(3)
            elif np.allclose(direction_norm, -z_axis):
                rot_matrix = np.diag([1, -1, -1])
            else:
                v = np.cross(z_axis, direction_norm)
                s = np.linalg.norm(v)
                c = np.dot(z_axis, direction_norm)
                vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
                rot_matrix = np.eye(3) + vx + vx @ vx * (1 - c) / (s * s)

            # Apply transform
            transform = np.eye(4)
            transform[:3, :3] = rot_matrix
            transform[:3, 3] = (p1 + p2) / 2  # center of edge
            cyl.apply_transform(transform)
            meshes.append(cyl)

        if meshes:
            return trimesh.util.concatenate(meshes)
        return None


class BBoxOverlapResolver:
    """Resolves overlapping bounding boxes by shrinking them only when overlap exceeds threshold.

    The key insight is that clusters from partfield often have fuzzy boundaries where
    parts naturally connect. We allow a small amount of overlap (default 5% of the
    smaller dimension along each axis) to preserve these natural connections while
    preventing excessive overlap.
    """

    def __init__(self, max_overlap_ratio: float = 0.05, min_dimension: float = 0.001):
        """
        Args:
            max_overlap_ratio: Maximum allowed overlap as ratio of smaller dimension (0.05 = 5%)
            min_dimension: Minimum box dimension to prevent collapse
        """
        self.max_overlap_ratio = max_overlap_ratio
        self.min_dimension = min_dimension

    def get_obb_corners(self, bbox: dict) -> np.ndarray:
        """Get the 8 corners of an oriented bounding box."""
        center = bbox['center']
        dims = bbox['dimensions']
        rotation = bbox['rotation']

        # Unit cube corners
        corners = np.array([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
        ]) * 0.5

        # Scale by dimensions, rotate, and translate
        corners = corners * dims
        corners = corners @ rotation.T + center
        return corners

    def compute_axis_overlap(self, bbox1: dict, bbox2: dict, axis: np.ndarray) -> dict:
        """Compute overlap information along a specific axis."""
        center1 = bbox1['center']
        center2 = bbox2['center']
        dims1 = bbox1['dimensions']
        dims2 = bbox2['dimensions']
        rot1 = bbox1['rotation']
        rot2 = bbox2['rotation']

        d = center2 - center1
        dist = abs(np.dot(d, axis))

        proj1 = sum(abs(np.dot(rot1[:, i], axis)) * dims1[i] / 2 for i in range(3))
        proj2 = sum(abs(np.dot(rot2[:, i], axis)) * dims2[i] / 2 for i in range(3))

        combined = proj1 + proj2
        overlap_amount = combined - dist

        min_extent = min(proj1, proj2) * 2
        if min_extent > 1e-10:
            overlap_ratio = overlap_amount / min_extent
        else:
            overlap_ratio = 0.0

        return {
            'overlap_amount': overlap_amount,
            'overlap_ratio': overlap_ratio,
            'proj1': proj1,
            'proj2': proj2,
            'dist': dist,
            'combined': combined
        }

    def boxes_overlap_excessive(self, bbox1: dict, bbox2: dict) -> Tuple[bool, List[dict]]:
        """Check if two OBBs have excessive overlap (beyond allowed threshold)."""
        rot1 = bbox1['rotation']
        rot2 = bbox2['rotation']

        axes = []
        for i in range(3):
            axes.append(('box1', i, rot1[:, i]))
        for i in range(3):
            axes.append(('box2', i, rot2[:, i]))
        for i in range(3):
            for j in range(3):
                cross = np.cross(rot1[:, i], rot2[:, j])
                norm = np.linalg.norm(cross)
                if norm > 1e-10:
                    axes.append(('cross', (i, j), cross / norm))

        axis_info = []
        has_excessive = False

        for source, idx, axis in axes:
            info = self.compute_axis_overlap(bbox1, bbox2, axis)
            info['source'] = source
            info['idx'] = idx

            if info['overlap_amount'] <= 0:
                return False, []

            if info['overlap_ratio'] > self.max_overlap_ratio:
                info['excessive'] = True
                has_excessive = True
            else:
                info['excessive'] = False

            axis_info.append(info)

        return has_excessive, axis_info

    def boxes_overlap(self, bbox1: dict, bbox2: dict) -> bool:
        """Check if two OBBs overlap using separating axis theorem (SAT)."""
        center1 = bbox1['center']
        center2 = bbox2['center']
        dims1 = bbox1['dimensions']
        dims2 = bbox2['dimensions']
        rot1 = bbox1['rotation']
        rot2 = bbox2['rotation']

        d = center2 - center1
        axes = []

        for i in range(3):
            axes.append(rot1[:, i])
        for i in range(3):
            axes.append(rot2[:, i])
        for i in range(3):
            for j in range(3):
                cross = np.cross(rot1[:, i], rot2[:, j])
                if np.linalg.norm(cross) > 1e-10:
                    axes.append(cross / np.linalg.norm(cross))

        for axis in axes:
            proj1 = sum(abs(np.dot(rot1[:, i], axis)) * dims1[i] / 2 for i in range(3))
            proj2 = sum(abs(np.dot(rot2[:, i], axis)) * dims2[i] / 2 for i in range(3))
            dist = abs(np.dot(d, axis))
            if dist > proj1 + proj2:
                return False

        return True

    def compute_overlap_volume_approx(self, bbox1: dict, bbox2: dict) -> float:
        """Approximate overlap volume between two OBBs."""
        if not self.boxes_overlap(bbox1, bbox2):
            return 0.0
        try:
            center_dist = np.linalg.norm(bbox1['center'] - bbox2['center'])
            avg_extent = (np.mean(bbox1['dimensions']) + np.mean(bbox2['dimensions'])) / 2
            if center_dist < avg_extent:
                vol1 = np.prod(bbox1['dimensions'])
                vol2 = np.prod(bbox2['dimensions'])
                overlap_ratio = 1.0 - (center_dist / avg_extent)
                return min(vol1, vol2) * overlap_ratio * 0.5
            return 0.0
        except:
            return 0.0

    def compute_principal_axis_overlap(self, bbox1: dict, bbox2: dict) -> dict:
        """Compute overlap along the direction between box centers."""
        center1 = bbox1['center']
        center2 = bbox2['center']
        dims1 = bbox1['dimensions']
        dims2 = bbox2['dimensions']
        rot1 = bbox1['rotation']
        rot2 = bbox2['rotation']

        direction = center2 - center1
        dist = np.linalg.norm(direction)

        if dist < 1e-10:
            direction = np.array([1, 0, 0])
            dist = 0
        else:
            direction = direction / dist

        proj1 = sum(abs(np.dot(rot1[:, i], direction)) * dims1[i] / 2 for i in range(3))
        proj2 = sum(abs(np.dot(rot2[:, i], direction)) * dims2[i] / 2 for i in range(3))

        combined = proj1 + proj2
        overlap_amount = combined - dist

        min_extent = min(proj1, proj2) * 2
        if min_extent > 1e-10:
            overlap_ratio = overlap_amount / min_extent
        else:
            overlap_ratio = 0.0

        return {
            'direction': direction,
            'dist': dist,
            'proj1': proj1,
            'proj2': proj2,
            'combined': combined,
            'overlap_amount': overlap_amount,
            'overlap_ratio': overlap_ratio,
            'min_extent': min_extent
        }

    def is_containment_scenario(self, bbox1: dict, bbox2: dict,
                                  principal_overlap: dict) -> bool:
        """Check if this is a containment scenario (small box inside/attached to large box)."""
        vol1 = np.prod(bbox1['dimensions'])
        vol2 = np.prod(bbox2['dimensions'])

        vol_ratio = max(vol1, vol2) / max(min(vol1, vol2), 1e-10)

        if principal_overlap['overlap_ratio'] > 1.0 and vol_ratio > 5.0:
            return True

        dist = principal_overlap['dist']
        proj1 = principal_overlap['proj1']
        proj2 = principal_overlap['proj2']

        if dist < max(proj1, proj2) * 0.8:
            if vol_ratio > 3.0:
                return True

        return False

    def compute_shrink_to_target(self, bbox1: dict, bbox2: dict,
                                    axis_info: List[dict]) -> Tuple[dict, dict]:
        """Compute how much to shrink boxes to reduce overlap to target threshold."""
        principal_overlap = self.compute_principal_axis_overlap(bbox1, bbox2)

        if self.is_containment_scenario(bbox1, bbox2, principal_overlap):
            return bbox1, bbox2

        if principal_overlap['overlap_ratio'] <= self.max_overlap_ratio:
            return bbox1, bbox2

        target_ratio = self.max_overlap_ratio
        current_ratio = principal_overlap['overlap_ratio']
        min_extent = principal_overlap['min_extent']

        overlap_to_remove = (current_ratio - target_ratio) * min_extent

        if overlap_to_remove <= 0:
            return bbox1, bbox2

        vol1 = np.prod(bbox1['dimensions'])
        vol2 = np.prod(bbox2['dimensions'])

        if vol1 >= vol2:
            shrink1_ratio = 0.7
            shrink2_ratio = 0.3
        else:
            shrink1_ratio = 0.3
            shrink2_ratio = 0.7

        new_bbox1 = bbox1.copy()
        new_bbox2 = bbox2.copy()

        direction = principal_overlap['direction']

        rot1 = bbox1['rotation']
        alignments1 = [abs(np.dot(direction, rot1[:, i])) for i in range(3)]
        main_axis1 = np.argmax(alignments1)
        alignment1 = alignments1[main_axis1]

        if alignment1 > 0.1:
            shrink_amount1 = (overlap_to_remove * shrink1_ratio) / alignment1
            max_shrink1 = bbox1['dimensions'][main_axis1] * 0.3
            shrink_amount1 = min(shrink_amount1, max_shrink1)

            new_dims1 = bbox1['dimensions'].copy()
            new_dims1[main_axis1] = max(
                new_dims1[main_axis1] - shrink_amount1,
                self.min_dimension
            )
            new_bbox1['dimensions'] = new_dims1

            shift1 = direction * (shrink_amount1 * 0.5)
            new_bbox1['center'] = bbox1['center'] - shift1

        rot2 = bbox2['rotation']
        alignments2 = [abs(np.dot(direction, rot2[:, i])) for i in range(3)]
        main_axis2 = np.argmax(alignments2)
        alignment2 = alignments2[main_axis2]

        if alignment2 > 0.1:
            shrink_amount2 = (overlap_to_remove * shrink2_ratio) / alignment2
            max_shrink2 = bbox2['dimensions'][main_axis2] * 0.3
            shrink_amount2 = min(shrink_amount2, max_shrink2)

            new_dims2 = bbox2['dimensions'].copy()
            new_dims2[main_axis2] = max(
                new_dims2[main_axis2] - shrink_amount2,
                self.min_dimension
            )
            new_bbox2['dimensions'] = new_dims2

            shift2 = -direction * (shrink_amount2 * 0.5)
            new_bbox2['center'] = bbox2['center'] - shift2

        return new_bbox1, new_bbox2

    def resolve_overlaps(self, bboxes: List[dict], segment_to_bbox: Dict[int, int],
                         global_rotation: np.ndarray,
                         max_iterations: int = 20) -> List[dict]:
        """Resolve excessive overlaps between bounding boxes.

        Key principle: We ALLOW up to max_overlap_ratio (default 5%) overlap in any
        dimension. This preserves natural connections between parts while preventing
        boxes from overlapping excessively.
        """
        if len(bboxes) <= 1:
            return bboxes

        resolved_bboxes = [b.copy() for b in bboxes]

        for iteration in range(max_iterations):
            excessive_pairs = []

            n = len(resolved_bboxes)
            for i in range(n):
                for j in range(i + 1, n):
                    has_excessive, axis_info = self.boxes_overlap_excessive(
                        resolved_bboxes[i], resolved_bboxes[j]
                    )
                    if has_excessive:
                        excessive_pairs.append((i, j, axis_info))

            if not excessive_pairs:
                break

            for i, j, axis_info in excessive_pairs:
                new_bbox_i, new_bbox_j = self.compute_shrink_to_target(
                    resolved_bboxes[i], resolved_bboxes[j], axis_info
                )
                resolved_bboxes[i] = new_bbox_i
                resolved_bboxes[j] = new_bbox_j

        # Recreate meshes for modified boxes using global rotation
        fitter = OBBFitterGlobal(global_rotation=global_rotation, apply_filtering=False)
        for i, bbox in enumerate(resolved_bboxes):
            bbox['dimensions'] = np.maximum(bbox['dimensions'], self.min_dimension)
            bbox['mesh'] = fitter._create_box_mesh(
                bbox['center'], bbox['dimensions'], global_rotation
            )

        return resolved_bboxes


class SegmentWithBBoxesGlobal:
    """Main class for segmenting meshes and fitting globally-oriented bounding boxes."""

    def __init__(self, apply_filtering: bool = True, resolve_overlaps: bool = True,
                 orientation_method: str = 'all_vertices'):
        """
        Args:
            apply_filtering: Enable outlier filtering per segment
            resolve_overlaps: Enable overlap resolution between boxes
            orientation_method: 'all_vertices' (PCA on all vertices) or 'single_box'
        """
        self.apply_filtering = apply_filtering
        self.resolve_overlaps = resolve_overlaps
        self.orientation_method = orientation_method

        self.global_orientation_computer = GlobalOrientationComputer(method=orientation_method)
        self.overlap_resolver = BBoxOverlapResolver() if resolve_overlaps else None

        # OBBFitter will be created after computing global orientation
        self.fitter = None
        self._global_rotation = None

    def load_mesh(self, path: str) -> Optional[trimesh.Trimesh]:
        """Load a mesh from file."""
        try:
            mesh = trimesh.load(path, force='mesh', process=False)
            if isinstance(mesh, trimesh.Scene):
                meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
                if meshes:
                    mesh = trimesh.util.concatenate(meshes)
                else:
                    return None
            return mesh
        except Exception as e:
            print(f"Error loading mesh: {e}")
            return None

    def load_labels(self, path: str) -> Optional[np.ndarray]:
        """Load segmentation labels from numpy file."""
        try:
            return np.load(path).flatten()
        except Exception as e:
            print(f"Error loading labels: {e}")
            return None

    def get_segment_points(self, mesh: trimesh.Trimesh, labels: np.ndarray,
                           segment_id: int, is_face_labels: bool = True) -> np.ndarray:
        """Extract points for a specific segment."""
        if is_face_labels:
            face_indices = np.where(labels == segment_id)[0]
            if len(face_indices) == 0:
                return np.array([]).reshape(0, 3)
            vertex_indices = np.unique(mesh.faces[face_indices].flatten())
            return mesh.vertices[vertex_indices]
        else:
            return mesh.vertices[labels == segment_id]

    def process(self, mesh: trimesh.Trimesh, labels: np.ndarray,
                is_face_labels: bool = True) -> Tuple[List[dict], Dict[int, int]]:
        """Process mesh and fit globally-oriented bounding boxes to each segment.

        Key difference from segment_with_bboxes.py:
        - First computes global orientation from entire mesh
        - Then fits each segment's box using that global orientation
        """
        # STEP 1: Compute global orientation from ENTIRE mesh (before processing segments)
        print("Computing global orientation from entire mesh...")
        self._global_rotation = self.global_orientation_computer.compute_from_mesh(mesh)

        det = np.linalg.det(self._global_rotation)
        outliers_removed = self.global_orientation_computer.num_outliers_removed
        print(f"  Global rotation matrix computed (det={det:.4f})")
        print(f"  Vertices used: {len(mesh.vertices) - outliers_removed} "
              f"({outliers_removed} extreme outliers removed)")

        # STEP 2: Create fitter with global rotation
        self.fitter = OBBFitterGlobal(
            global_rotation=self._global_rotation,
            apply_filtering=self.apply_filtering
        )

        # STEP 3: Process each segment using global orientation
        unique_labels = np.unique(labels)
        print(f"\nProcessing {len(unique_labels)} segments with global orientation...")

        bboxes = []
        segment_to_bbox = {}

        for i, segment_id in enumerate(unique_labels):
            points = self.get_segment_points(mesh, labels, segment_id, is_face_labels)
            if len(points) < 3:
                print(f"  Segment {segment_id}: skipped (too few points: {len(points)})")
                continue

            bbox = self.fitter.fit_obb(points)

            # Print filtering stats if available
            orig_count = len(bbox.get('original_points', points))
            filt_count = len(bbox.get('filtered_points', points))
            if orig_count != filt_count:
                print(f"  Segment {segment_id}: {orig_count} -> {filt_count} points (filtered), dims={bbox['dimensions'].round(4)}")
            else:
                print(f"  Segment {segment_id}: {orig_count} points, dims={bbox['dimensions'].round(4)}")

            bboxes.append(bbox)
            segment_to_bbox[segment_id] = len(bboxes) - 1

        # STEP 4: Resolve overlaps if enabled
        if self.resolve_overlaps and self.overlap_resolver and len(bboxes) > 1:
            print(f"\nResolving bounding box overlaps...")
            overlap_count = self._count_overlaps(bboxes)
            print(f"  Initial overlaps: {overlap_count}")

            bboxes = self.overlap_resolver.resolve_overlaps(
                bboxes, segment_to_bbox, self._global_rotation
            )

            overlap_count = self._count_overlaps(bboxes)
            print(f"  Final overlaps: {overlap_count}")

        return bboxes, segment_to_bbox

    def _count_overlaps(self, bboxes: List[dict]) -> int:
        """Count number of overlapping bbox pairs."""
        if not self.overlap_resolver:
            return 0
        count = 0
        n = len(bboxes)
        for i in range(n):
            for j in range(i + 1, n):
                if self.overlap_resolver.boxes_overlap(bboxes[i], bboxes[j]):
                    count += 1
        return count

    def create_bbox_mesh(self, bboxes: List[dict], segment_to_bbox: Dict[int, int],
                         style: str = 'solid', alpha: float = 1.0) -> Optional[trimesh.Trimesh]:
        """Create combined mesh with all colored bounding boxes.

        Args:
            style: 'solid', 'wireframe', or 'transparent'
            alpha: opacity (1.0 for solid, 0.6 for transparent)
        """
        n_segments = len(segment_to_bbox)
        if n_segments == 0:
            return None

        cmap = plt.colormaps.get_cmap("tab20").resampled(max(n_segments, 1))

        meshes = []
        for i, (segment_id, bbox_idx) in enumerate(segment_to_bbox.items()):
            bbox = bboxes[bbox_idx]
            color = np.array(cmap(i % 20)[:3])

            if style == 'wireframe':
                # Create wireframe box
                box_mesh = self.fitter._create_wireframe_box(
                    bbox['center'], bbox['dimensions'], bbox['rotation']
                )
            else:
                # Create solid box
                box_mesh = self.fitter._create_box_mesh(
                    bbox['center'], bbox['dimensions'], bbox['rotation']
                )

            if box_mesh is not None:
                # Apply color with alpha
                face_color = np.append(color, alpha)
                box_mesh.visual.face_colors = (face_color * 255).astype(np.uint8)
                meshes.append(box_mesh)

        if meshes:
            return trimesh.util.concatenate(meshes)
        return None

    def export_results(self, bboxes: List[dict], segment_to_bbox: Dict[int, int],
                       output_path: str, style: str = 'all'):
        """Export results to files.

        Args:
            style: 'solid', 'wireframe', 'transparent', or 'all'
        """
        base, ext = os.path.splitext(output_path)
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        styles_to_export = []
        if style == 'all':
            styles_to_export = [
                ('solid', 1.0, f"{base}_solid{ext}"),
                ('wireframe', 1.0, f"{base}_wireframe{ext}"),
                ('transparent', 0.6, f"{base}_transparent{ext}")
            ]
        else:
            alpha = 0.6 if style == 'transparent' else 1.0
            styles_to_export = [(style, alpha, output_path)]

        # Export each style
        for style_name, alpha, path in styles_to_export:
            mesh = self.create_bbox_mesh(bboxes, segment_to_bbox, style_name, alpha)
            if mesh is not None:
                mesh.export(path)
                print(f"Exported {style_name} bboxes to: {path}")

        # Export info file
        info_path = f"{base}_info.txt"
        with open(info_path, 'w') as f:
            f.write("Global-Oriented Segment Bounding Box Results\n")
            f.write("=" * 50 + "\n")
            f.write("(All boxes share the same global orientation)\n\n")

            # Write global rotation matrix
            f.write("Global Rotation Matrix:\n")
            if self._global_rotation is not None:
                for row in self._global_rotation.T:  # Transpose to show row vectors
                    f.write(f"  [{row[0]:.6f}, {row[1]:.6f}, {row[2]:.6f}]\n")
            f.write("\n" + "-" * 50 + "\n\n")

            for segment_id, bbox_idx in segment_to_bbox.items():
                bbox = bboxes[bbox_idx]
                f.write(f"Segment {segment_id}:\n")
                f.write(f"  Center: [{bbox['center'][0]:.4f}, {bbox['center'][1]:.4f}, {bbox['center'][2]:.4f}]\n")
                f.write(f"  Dimensions: [{bbox['dimensions'][0]:.4f}, {bbox['dimensions'][1]:.4f}, {bbox['dimensions'][2]:.4f}]\n")
                f.write("\n")

        print(f"Exported info to: {info_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Fit globally-oriented bounding boxes to mesh segments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python segment_with_bboxes_global.py -i model.glb -l labels.npy -o output.ply
  python segment_with_bboxes_global.py -i model.glb -l labels.npy -o output.ply --style all
  python segment_with_bboxes_global.py -i model.glb -l labels.npy -o output.ply --method all_vertices

Global vs Local Orientation:
  This tool orients ALL bounding boxes along a single global direction computed
  from the entire mesh (via PCA on all vertices). This is useful when you want
  boxes to be consistently aligned (e.g., for CAD models, architectural parts).

  For per-segment oriented boxes (tighter fit), use segment_with_bboxes.py instead.
        """
    )

    parser.add_argument("--input", "-i", required=True, help="Input mesh file")
    parser.add_argument("--labels", "-l", required=True, help="Segmentation labels (NPY)")
    parser.add_argument("--output", "-o", required=True, help="Output file path")
    parser.add_argument("--style", "-s", choices=['solid', 'wireframe', 'transparent', 'all'],
                        default='all', help="Box style (default: all)")
    parser.add_argument("--method", "-m", choices=['all_vertices', 'single_box'],
                        default='all_vertices',
                        help="Global orientation method: 'all_vertices' (PCA on all vertices) "
                             "or 'single_box' (same as all_vertices) (default: all_vertices)")
    parser.add_argument("--vertex-labels", action="store_true",
                        help="Treat labels as per-vertex (default: per-face)")
    parser.add_argument("--no-filter", action="store_true",
                        help="Disable outlier/density filtering (for raw bounding boxes)")
    parser.add_argument("--no-overlap-resolution", action="store_true",
                        help="Disable automatic overlap resolution between boxes")
    parser.add_argument("--brep", action="store_true",
                        help="Generate BREP STEP file instead of PLY mesh output")
    parser.add_argument("--all-formats", action="store_true",
                        help="Generate both STEP (BREP) and PLY (mesh) output")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return 1
    if not os.path.exists(args.labels):
        print(f"Error: Labels file not found: {args.labels}")
        return 1

    # Create processor with filtering and overlap options
    apply_filtering = not args.no_filter
    resolve_overlaps = not args.no_overlap_resolution
    processor = SegmentWithBBoxesGlobal(
        apply_filtering=apply_filtering,
        resolve_overlaps=resolve_overlaps,
        orientation_method=args.method
    )

    print(f"Options: filtering={'ON' if apply_filtering else 'OFF'}, "
          f"overlap_resolution={'ON' if resolve_overlaps else 'OFF'}, "
          f"orientation={args.method}")

    print(f"\nLoading mesh: {args.input}")
    mesh = processor.load_mesh(args.input)
    if mesh is None:
        return 1

    print(f"Loading labels: {args.labels}")
    labels = processor.load_labels(args.labels)
    if labels is None:
        return 1

    is_face_labels = not args.vertex_labels
    expected_count = len(mesh.faces) if is_face_labels else len(mesh.vertices)

    # Auto-detect label type if mismatch
    if len(labels) != expected_count:
        if len(labels) == len(mesh.faces):
            is_face_labels = True
        elif len(labels) == len(mesh.vertices):
            is_face_labels = False

    print(f"\nMesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    print(f"Labels: {len(labels)} ({'face' if is_face_labels else 'vertex'})")

    # Process
    bboxes, segment_to_bbox = processor.process(mesh, labels, is_face_labels)

    print(f"\n{'=' * 50}")
    print(f"Fitted {len(bboxes)} globally-oriented bounding boxes")

    # Export
    generate_brep = args.brep or args.all_formats
    generate_mesh = not args.brep or args.all_formats

    if generate_mesh:
        processor.export_results(bboxes, segment_to_bbox, args.output, args.style)

    if generate_brep:
        try:
            from brep_generator import BRepShapeBuilder, BRepExporter
            import matplotlib.pyplot as _plt

            builder = BRepShapeBuilder()
            exporter = BRepExporter()
            cmap = _plt.colormaps.get_cmap("tab20").resampled(max(len(segment_to_bbox), 1))

            shapes_and_colors = []
            for i, (seg_id, bbox_idx) in enumerate(segment_to_bbox.items()):
                bbox = bboxes[bbox_idx]
                color = cmap(i % 20)[:3]
                try:
                    shape = builder.shape_from_bbox(bbox)
                    shapes_and_colors.append((shape, tuple(color)))
                except Exception as e:
                    print(f"  Warning: BREP failed for segment {seg_id}: {e}")

            if shapes_and_colors:
                base, _ = os.path.splitext(args.output)
                step_path = base + ".step"
                exporter.export_colored_step(shapes_and_colors, step_path)
        except ImportError:
            print("Warning: pythonocc-core not available. Skipping BREP output.")
            print("Install with: conda install -c conda-forge pythonocc-core")

    print("\nDone!")
    return 0


if __name__ == "__main__":
    exit(main())
