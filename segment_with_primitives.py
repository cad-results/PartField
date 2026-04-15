#!/usr/bin/env python3
"""
Segment with Primitives - Segment mesh into parts and fit geometric primitives to each part.

This tool:
1. Loads a mesh/point cloud file (PLY, OBJ, GLB, etc.)
2. Uses existing segmentation or runs PartField clustering
3. Fits geometric primitives to each segment:
   - Box, Cylinder, Sphere, Hemisphere, Quarter/Eighth spheres
   - Cone, Capsule, Triangular prism
   - Tetrahedron, Octahedron (Platonic solids)
4. Falls back to tight oriented bounding box if no good primitive fit is found
5. Exports the result with primitives visualized

Usage:
    python segment_with_primitives.py --input model.glb --labels clustering.npy --output output.ply
"""

import os
import argparse
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

import trimesh
import open3d as o3d
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


class PrimitiveType(Enum):
    """Types of geometric primitives."""
    BOX = "box"
    CYLINDER = "cylinder"
    SPHERE = "sphere"
    ELLIPSOID = "ellipsoid"
    HEMISPHERE = "hemisphere"
    QUARTER_SPHERE = "quarter_sphere"
    EIGHTH_SPHERE = "eighth_sphere"
    CONE = "cone"
    CAPSULE = "capsule"
    TRIANGULAR_PRISM = "triangular_prism"
    TETRAHEDRON = "tetrahedron"
    OCTAHEDRON = "octahedron"
    ORIENTED_BBOX = "oriented_bbox"


@dataclass
class Primitive:
    """Represents a fitted geometric primitive."""
    type: PrimitiveType
    center: np.ndarray
    dimensions: np.ndarray
    rotation: np.ndarray
    score: float
    mesh: Optional[trimesh.Trimesh] = None
    params: Dict[str, Any] = field(default_factory=dict)


class PrimitiveFitter:
    """Fits various geometric primitives to point clouds."""

    def __init__(self, min_points: int = 10, tight_fit: bool = True):
        """
        Args:
            min_points: Minimum number of points required for fitting
            tight_fit: If True, fit primitives tightly to convex hull
        """
        self.min_points = min_points
        self.tight_fit = tight_fit

    def fit_all(self, points: np.ndarray) -> List[Primitive]:
        """Fit all primitive types and return sorted by score (best first)."""
        if len(points) < self.min_points:
            return [self.fit_oriented_bbox(points)]

        primitives = []
        fitters = [
            ('oriented_bbox', self.fit_oriented_bbox),
            ('box', self.fit_box),
            ('cylinder', self.fit_cylinder),
            ('sphere', self.fit_sphere),
            ('ellipsoid', self.fit_ellipsoid),
            ('hemisphere', self.fit_hemisphere),
            ('quarter_sphere', self.fit_quarter_sphere),
            ('eighth_sphere', self.fit_eighth_sphere),
            ('cone', self.fit_cone),
            ('capsule', self.fit_capsule),
            ('triangular_prism', self.fit_triangular_prism),
            ('tetrahedron', self.fit_tetrahedron),
            ('octahedron', self.fit_octahedron),
        ]

        for name, fitter in fitters:
            try:
                primitives.append(fitter(points))
            except Exception as e:
                pass  # Silent fail, continue with other primitives

        primitives.sort(key=lambda p: p.score)
        return primitives

    def fit_best(self, points: np.ndarray, threshold: float = 0.3) -> Primitive:
        """Fit all primitives and return the best one."""
        primitives = self.fit_all(points)
        if not primitives:
            return self.fit_oriented_bbox(points)

        best = primitives[0]
        if best.score > threshold and best.type != PrimitiveType.ORIENTED_BBOX:
            for p in primitives:
                if p.type == PrimitiveType.ORIENTED_BBOX:
                    return p
        return best

    def _get_convex_hull_volume(self, points: np.ndarray) -> float:
        """Get convex hull volume safely."""
        try:
            if len(points) >= 4:
                hull = ConvexHull(points)
                return hull.volume
        except:
            pass
        return np.prod(np.ptp(points, axis=0))

    def _compute_pca(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, PCA]:
        """Compute PCA for points, return center, centered points, and PCA object."""
        center = np.mean(points, axis=0)
        centered = points - center
        pca = PCA(n_components=3)
        pca.fit(centered)
        return center, centered, pca

    def _rotation_from_axis(self, axis: np.ndarray) -> np.ndarray:
        """Create rotation matrix aligning Z-axis with given axis."""
        axis = axis / np.linalg.norm(axis)
        z = np.array([0, 0, 1])
        if np.allclose(axis, z):
            return np.eye(3)
        if np.allclose(axis, -z):
            return np.diag([1, -1, -1])
        v = np.cross(z, axis)
        s = np.linalg.norm(v)
        c = np.dot(z, axis)
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + vx + vx @ vx * (1 - c) / (s * s)

    # === ORIENTED BOUNDING BOX ===
    def fit_oriented_bbox(self, points: np.ndarray) -> Primitive:
        """Fit tight oriented bounding box using PCA."""
        if len(points) < 3:
            center = np.mean(points, axis=0) if len(points) > 0 else np.zeros(3)
            return Primitive(PrimitiveType.ORIENTED_BBOX, center,
                           np.array([0.01, 0.01, 0.01]), np.eye(3), 1.0)

        center, centered, pca = self._compute_pca(points)
        transformed = pca.transform(centered)

        min_c, max_c = np.min(transformed, axis=0), np.max(transformed, axis=0)
        dimensions = np.maximum(max_c - min_c, 0.001)

        bbox_center_pca = (min_c + max_c) / 2
        center = center + pca.inverse_transform(bbox_center_pca) - pca.inverse_transform(np.zeros(3))
        rotation = pca.components_.T

        hull_vol = self._get_convex_hull_volume(points)
        bbox_vol = np.prod(dimensions)
        score = 1.0 - min(hull_vol / bbox_vol, 1.0) if bbox_vol > 0 else 1.0

        mesh = self._create_box_mesh(center, dimensions, rotation)
        return Primitive(PrimitiveType.ORIENTED_BBOX, center, dimensions, rotation, score, mesh)

    # === BOX ===
    def fit_box(self, points: np.ndarray) -> Primitive:
        """Fit rectangular prism (box)."""
        obb = self.fit_oriented_bbox(points)
        center, centered, pca = self._compute_pca(points)
        transformed = pca.transform(centered)

        score = self._compute_box_score(transformed, obb.dimensions)
        return Primitive(PrimitiveType.BOX, obb.center, obb.dimensions,
                        obb.rotation, score, obb.mesh)

    def _compute_box_score(self, transformed: np.ndarray, dims: np.ndarray) -> float:
        """Score how well points fit a box."""
        if len(transformed) < 10:
            return 1.0

        half_dims = np.maximum(dims / 2, 0.001)
        normalized = transformed / half_dims

        # Check surface coverage
        surface_score = 0
        for axis in range(3):
            for sign in [-1, 1]:
                if np.sum(np.abs(normalized[:, axis] - sign) < 0.3) > 0:
                    surface_score += 1
        surface_score /= 6.0

        # Check volume fill
        n_bins = 3
        occupancy = np.zeros((n_bins, n_bins, n_bins))
        for p in normalized:
            p_clamped = np.clip(p, -0.999, 0.999)
            idx = np.clip(((p_clamped + 1) / 2 * n_bins).astype(int), 0, n_bins - 1)
            occupancy[idx[0], idx[1], idx[2]] = 1
        fill_ratio = np.sum(occupancy) / (n_bins ** 3)

        return 1.0 - (0.5 * fill_ratio + 0.5 * surface_score)

    # === CYLINDER ===
    def fit_cylinder(self, points: np.ndarray) -> Primitive:
        """Fit cylinder using PCA for axis."""
        center, centered, pca = self._compute_pca(points)
        axis = pca.components_[0]

        proj = np.dot(centered, axis)
        height = np.max(proj) - np.min(proj)
        height = max(height, 0.001)

        axis_center = (np.max(proj) + np.min(proj)) / 2
        center = center + axis_center * axis

        perp = centered - np.outer(proj, axis)
        dists = np.linalg.norm(perp, axis=1)
        radius = np.percentile(dists, 90)  # Tighter fit
        radius = max(radius, 0.001)

        rotation = self._rotation_from_axis(axis)
        score = self._compute_cylinder_score(centered, axis, radius, height)
        mesh = self._create_cylinder_mesh(center, radius, height, rotation)

        return Primitive(PrimitiveType.CYLINDER, center,
                        np.array([radius * 2, radius * 2, height]),
                        rotation, score, mesh, {'radius': radius, 'height': height})

    def _compute_cylinder_score(self, centered: np.ndarray, axis: np.ndarray,
                                 radius: float, height: float) -> float:
        """Score cylinder fit."""
        proj = np.dot(centered, axis)
        perp = centered - np.outer(proj, axis)
        dists = np.linalg.norm(perp, axis=1)

        dist_err = np.mean(np.abs(dists - radius) / radius)
        height_cov = min((np.max(proj) - np.min(proj)) / height, 1.0) if height > 0 else 0

        return np.clip(0.7 * dist_err + 0.3 * (1 - height_cov), 0, 1)

    # === SPHERE ===
    def fit_sphere(self, points: np.ndarray) -> Primitive:
        """Fit sphere."""
        center = np.mean(points, axis=0)

        def obj(c):
            return np.var(np.linalg.norm(points - c, axis=1))

        result = minimize(obj, center, method='Powell')
        center = result.x

        dists = np.linalg.norm(points - center, axis=1)
        radius = np.percentile(dists, 90)  # Tighter fit
        radius = max(radius, 0.001)

        score = self._compute_sphere_score(points, center, radius)
        mesh = self._create_sphere_mesh(center, radius)

        return Primitive(PrimitiveType.SPHERE, center,
                        np.array([radius * 2] * 3), np.eye(3),
                        score, mesh, {'radius': radius})

    def _compute_sphere_score(self, points: np.ndarray, center: np.ndarray, radius: float) -> float:
        """Score sphere fit."""
        dists = np.linalg.norm(points - center, axis=1)
        rel_err = np.mean(np.abs(dists - radius) / radius)

        dirs = (points - center) / (dists[:, np.newaxis] + 1e-8)
        dir_var = np.var(dirs, axis=0).sum()
        coverage = min(dir_var / 0.5, 1.0)

        return np.clip(0.7 * rel_err + 0.3 * (1 - coverage), 0, 1)

    # === ELLIPSOID ===
    def fit_ellipsoid(self, points: np.ndarray) -> Primitive:
        """Fit ellipsoid using PCA to determine principal axes and radii."""
        center, centered, pca = self._compute_pca(points)

        # Transform to PCA space
        transformed = pca.transform(centered)

        # Get semi-axes lengths (radii along each principal direction)
        # Use percentile for tighter fit
        radii = np.percentile(np.abs(transformed), 90, axis=0)
        radii = np.maximum(radii, 0.001)

        rotation = pca.components_.T

        score = self._compute_ellipsoid_score(transformed, radii)
        mesh = self._create_ellipsoid_mesh(center, radii, rotation)

        return Primitive(PrimitiveType.ELLIPSOID, center,
                        radii * 2, rotation, score, mesh,
                        {'radii': radii, 'axes': pca.components_})

    def _compute_ellipsoid_score(self, transformed: np.ndarray, radii: np.ndarray) -> float:
        """Score ellipsoid fit."""
        # Normalized coordinates (should be on unit ellipsoid surface)
        normalized = transformed / radii

        # Distance from ellipsoid surface (should be ~1)
        ellipsoid_dists = np.linalg.norm(normalized, axis=1)
        dist_err = np.mean(np.abs(ellipsoid_dists - 1.0))

        # Check coverage (points should be distributed around the ellipsoid)
        dirs = normalized / (ellipsoid_dists[:, np.newaxis] + 1e-8)
        dir_var = np.var(dirs, axis=0).sum()
        coverage = min(dir_var / 0.5, 1.0)

        return np.clip(0.7 * dist_err + 0.3 * (1 - coverage), 0, 1)

    # === HEMISPHERE ===
    def fit_hemisphere(self, points: np.ndarray) -> Primitive:
        """Fit hemisphere (half sphere)."""
        center, centered, pca = self._compute_pca(points)

        # Find the best hemisphere orientation
        best_score = 1.0
        best_axis = pca.components_[2]  # Try smallest variance direction first

        for axis in pca.components_:
            for sign in [1, -1]:
                test_axis = axis * sign
                proj = np.dot(centered, test_axis)

                # Check if most points are on one side
                positive = np.sum(proj > 0)
                negative = np.sum(proj <= 0)
                ratio = max(positive, negative) / len(proj)

                if ratio > 0.7:  # Good hemisphere candidate
                    dists = np.linalg.norm(centered, axis=1)
                    radius = np.percentile(dists, 90)

                    # Score based on fit to hemisphere
                    score = self._compute_hemisphere_score(centered, test_axis, radius)
                    if score < best_score:
                        best_score = score
                        best_axis = test_axis if positive > negative else -test_axis

        dists = np.linalg.norm(centered, axis=1)
        radius = np.percentile(dists, 90)
        radius = max(radius, 0.001)

        rotation = self._rotation_from_axis(best_axis)
        mesh = self._create_hemisphere_mesh(center, radius, rotation)

        return Primitive(PrimitiveType.HEMISPHERE, center,
                        np.array([radius * 2, radius * 2, radius]),
                        rotation, best_score, mesh, {'radius': radius, 'axis': best_axis})

    def _compute_hemisphere_score(self, centered: np.ndarray, axis: np.ndarray, radius: float) -> float:
        """Score hemisphere fit."""
        proj = np.dot(centered, axis)
        dists = np.linalg.norm(centered, axis=1)

        # Points should be on the positive side and at distance ~radius
        on_positive = np.sum(proj >= -0.1 * radius) / len(proj)
        dist_err = np.mean(np.abs(dists - radius) / radius)

        return np.clip(0.5 * dist_err + 0.5 * (1 - on_positive), 0, 1)

    # === QUARTER SPHERE ===
    def fit_quarter_sphere(self, points: np.ndarray) -> Primitive:
        """Fit quarter sphere (1/4 of sphere)."""
        center, centered, pca = self._compute_pca(points)

        # Quarter sphere has two cut planes
        best_score = 1.0
        best_axes = [pca.components_[0], pca.components_[1]]

        for i in range(3):
            for j in range(i + 1, 3):
                axis1, axis2 = pca.components_[i], pca.components_[j]

                for s1 in [1, -1]:
                    for s2 in [1, -1]:
                        a1, a2 = axis1 * s1, axis2 * s2
                        proj1 = np.dot(centered, a1)
                        proj2 = np.dot(centered, a2)

                        in_quarter = (proj1 >= 0) & (proj2 >= 0)
                        ratio = np.sum(in_quarter) / len(proj1)

                        if ratio > 0.6:
                            dists = np.linalg.norm(centered, axis=1)
                            radius = np.percentile(dists, 90)
                            score = 1.0 - ratio

                            if score < best_score:
                                best_score = score
                                best_axes = [a1, a2]

        dists = np.linalg.norm(centered, axis=1)
        radius = np.percentile(dists, 90)
        radius = max(radius, 0.001)

        # Create rotation from first axis
        rotation = self._rotation_from_axis(best_axes[0])
        mesh = self._create_quarter_sphere_mesh(center, radius, rotation)

        return Primitive(PrimitiveType.QUARTER_SPHERE, center,
                        np.array([radius, radius, radius]),
                        rotation, best_score, mesh, {'radius': radius})

    # === EIGHTH SPHERE ===
    def fit_eighth_sphere(self, points: np.ndarray) -> Primitive:
        """Fit eighth sphere (1/8 of sphere, one octant)."""
        center, centered, pca = self._compute_pca(points)

        # Try all 8 octants
        best_score = 1.0
        best_signs = [1, 1, 1]

        for s0 in [1, -1]:
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    axes = [pca.components_[k] * [s0, s1, s2][k] for k in range(3)]
                    projs = [np.dot(centered, a) for a in axes]

                    in_octant = (projs[0] >= 0) & (projs[1] >= 0) & (projs[2] >= 0)
                    ratio = np.sum(in_octant) / len(centered)

                    if ratio > 0.5:
                        score = 1.0 - ratio
                        if score < best_score:
                            best_score = score
                            best_signs = [s0, s1, s2]

        dists = np.linalg.norm(centered, axis=1)
        radius = np.percentile(dists, 90)
        radius = max(radius, 0.001)

        rotation = pca.components_.T
        mesh = self._create_eighth_sphere_mesh(center, radius, rotation, best_signs)

        return Primitive(PrimitiveType.EIGHTH_SPHERE, center,
                        np.array([radius, radius, radius]),
                        rotation, best_score, mesh, {'radius': radius})

    # === CONE ===
    def fit_cone(self, points: np.ndarray) -> Primitive:
        """Fit cone."""
        center, centered, pca = self._compute_pca(points)
        axis = pca.components_[0]

        proj = np.dot(centered, axis)
        perp = centered - np.outer(proj, axis)
        dists = np.linalg.norm(perp, axis=1)

        # Find cone apex and base
        min_proj, max_proj = np.min(proj), np.max(proj)
        height = max_proj - min_proj
        height = max(height, 0.001)

        # Estimate radius at base (where proj is max)
        base_mask = proj > (max_proj - 0.2 * height)
        if np.sum(base_mask) > 0:
            base_radius = np.percentile(dists[base_mask], 90)
        else:
            base_radius = np.percentile(dists, 90)
        base_radius = max(base_radius, 0.001)

        # Cone center is at the middle
        cone_center = center + (min_proj + max_proj) / 2 * axis

        rotation = self._rotation_from_axis(axis)
        score = self._compute_cone_score(centered, axis, base_radius, height, min_proj, max_proj)
        mesh = self._create_cone_mesh(cone_center, base_radius, height, rotation)

        return Primitive(PrimitiveType.CONE, cone_center,
                        np.array([base_radius * 2, base_radius * 2, height]),
                        rotation, score, mesh, {'radius': base_radius, 'height': height})

    def _compute_cone_score(self, centered: np.ndarray, axis: np.ndarray,
                            radius: float, height: float, min_p: float, max_p: float) -> float:
        """Score cone fit."""
        proj = np.dot(centered, axis)
        perp = centered - np.outer(proj, axis)
        dists = np.linalg.norm(perp, axis=1)

        # Expected distance linearly increases from 0 at apex to radius at base
        t = (proj - min_p) / height  # 0 at apex, 1 at base
        expected_r = t * radius

        err = np.mean(np.abs(dists - expected_r) / (expected_r + 0.001))
        return np.clip(err, 0, 1)

    # === CAPSULE ===
    def fit_capsule(self, points: np.ndarray) -> Primitive:
        """Fit capsule (cylinder with hemispherical caps)."""
        center, centered, pca = self._compute_pca(points)
        axis = pca.components_[0]

        proj = np.dot(centered, axis)
        perp = centered - np.outer(proj, axis)
        dists = np.linalg.norm(perp, axis=1)

        radius = np.percentile(dists, 85)  # Tighter fit
        radius = max(radius, 0.001)

        # Height is total length minus the two hemisphere caps
        min_proj, max_proj = np.min(proj), np.max(proj)
        total_length = max_proj - min_proj
        cylinder_height = max(total_length - 2 * radius, 0.001)

        cap_center = center + (min_proj + max_proj) / 2 * axis

        rotation = self._rotation_from_axis(axis)
        score = self._compute_capsule_score(centered, axis, radius, total_length)
        mesh = self._create_capsule_mesh(cap_center, radius, cylinder_height, rotation)

        return Primitive(PrimitiveType.CAPSULE, cap_center,
                        np.array([radius * 2, radius * 2, total_length]),
                        rotation, score, mesh,
                        {'radius': radius, 'cylinder_height': cylinder_height})

    def _compute_capsule_score(self, centered: np.ndarray, axis: np.ndarray,
                                radius: float, length: float) -> float:
        """Score capsule fit."""
        proj = np.dot(centered, axis)
        perp = centered - np.outer(proj, axis)
        dists = np.linalg.norm(perp, axis=1)

        dist_err = np.mean(np.abs(dists - radius) / radius)
        return np.clip(dist_err, 0, 1)

    # === TRIANGULAR PRISM ===
    def fit_triangular_prism(self, points: np.ndarray) -> Primitive:
        """Fit triangular prism."""
        center, centered, pca = self._compute_pca(points)
        axis = pca.components_[0]

        proj = np.dot(centered, axis)
        height = np.max(proj) - np.min(proj)
        height = max(height, 0.001)

        axis_center = (np.max(proj) + np.min(proj)) / 2
        prism_center = center + axis_center * axis

        plane_basis = pca.components_[1:3]
        pts_2d = np.dot(centered, plane_basis.T)

        triangle_2d, tri_score = self._fit_triangle_2d(pts_2d)
        rotation = self._rotation_from_axis(axis)

        mesh = self._create_triangular_prism_mesh(prism_center, triangle_2d, height,
                                                   plane_basis, axis)

        w = np.max(triangle_2d[:, 0]) - np.min(triangle_2d[:, 0])
        d = np.max(triangle_2d[:, 1]) - np.min(triangle_2d[:, 1])

        return Primitive(PrimitiveType.TRIANGULAR_PRISM, prism_center,
                        np.array([w, d, height]), rotation, tri_score, mesh,
                        {'triangle_2d': triangle_2d, 'height': height})

    def _fit_triangle_2d(self, pts_2d: np.ndarray) -> Tuple[np.ndarray, float]:
        """Fit triangle to 2D points."""
        if len(pts_2d) < 3:
            return np.array([[0, 0], [1, 0], [0.5, 1]]), 1.0

        try:
            hull = ConvexHull(pts_2d)
            hull_pts = pts_2d[hull.vertices]

            if len(hull_pts) == 3:
                return hull_pts, 0.0

            # Find minimum bounding triangle
            min_pt = np.min(pts_2d, axis=0)
            max_pt = np.max(pts_2d, axis=0)
            mid_x = (min_pt[0] + max_pt[0]) / 2

            triangle = np.array([
                [min_pt[0] - 0.1 * (max_pt[0] - min_pt[0]), min_pt[1]],
                [max_pt[0] + 0.1 * (max_pt[0] - min_pt[0]), min_pt[1]],
                [mid_x, max_pt[1] + 0.1 * (max_pt[1] - min_pt[1])]
            ])

            tri_area = 0.5 * abs((triangle[1, 0] - triangle[0, 0]) *
                                 (triangle[2, 1] - triangle[0, 1]) -
                                 (triangle[2, 0] - triangle[0, 0]) *
                                 (triangle[1, 1] - triangle[0, 1]))

            score = 1.0 - min(hull.volume / tri_area, 1.0) if tri_area > 0 else 1.0
            return triangle, score
        except:
            min_pt = np.min(pts_2d, axis=0)
            max_pt = np.max(pts_2d, axis=0)
            mid_x = (min_pt[0] + max_pt[0]) / 2
            return np.array([[min_pt[0], min_pt[1]], [max_pt[0], min_pt[1]],
                            [mid_x, max_pt[1]]]), 0.7

    # === TETRAHEDRON ===
    def fit_tetrahedron(self, points: np.ndarray) -> Primitive:
        """Fit tetrahedron (4-faced Platonic solid)."""
        center, centered, pca = self._compute_pca(points)

        # Find bounding tetrahedron
        dists = np.linalg.norm(centered, axis=1)
        radius = np.percentile(dists, 95)
        radius = max(radius, 0.001)

        rotation = pca.components_.T
        score = self._compute_tetrahedron_score(centered, radius)
        mesh = self._create_tetrahedron_mesh(center, radius, rotation)

        return Primitive(PrimitiveType.TETRAHEDRON, center,
                        np.array([radius * 2] * 3), rotation, score, mesh,
                        {'radius': radius})

    def _compute_tetrahedron_score(self, centered: np.ndarray, radius: float) -> float:
        """Score tetrahedron fit."""
        dists = np.linalg.norm(centered, axis=1)
        rel_err = np.mean(np.abs(dists - radius * 0.7) / radius)  # Inscribed sphere radius
        return np.clip(rel_err, 0, 1)

    # === OCTAHEDRON ===
    def fit_octahedron(self, points: np.ndarray) -> Primitive:
        """Fit octahedron (8-faced Platonic solid)."""
        center, centered, pca = self._compute_pca(points)

        transformed = pca.transform(centered)
        extents = np.max(np.abs(transformed), axis=0)
        radius = np.max(extents) * 1.1  # Slight padding
        radius = max(radius, 0.001)

        rotation = pca.components_.T
        score = self._compute_octahedron_score(transformed, radius)
        mesh = self._create_octahedron_mesh(center, radius, rotation)

        return Primitive(PrimitiveType.OCTAHEDRON, center,
                        np.array([radius * 2] * 3), rotation, score, mesh,
                        {'radius': radius})

    def _compute_octahedron_score(self, transformed: np.ndarray, radius: float) -> float:
        """Score octahedron fit."""
        # Octahedron: |x| + |y| + |z| <= radius
        l1_norms = np.sum(np.abs(transformed), axis=1)
        in_octa = np.sum(l1_norms <= radius * 1.1) / len(transformed)
        return 1.0 - in_octa

    # === MESH CREATION ===
    def _create_box_mesh(self, center: np.ndarray, dims: np.ndarray,
                          rotation: np.ndarray) -> trimesh.Trimesh:
        """Create box mesh."""
        box = trimesh.creation.box(extents=dims)
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = center
        box.apply_transform(transform)
        return box

    def _create_cylinder_mesh(self, center: np.ndarray, radius: float,
                               height: float, rotation: np.ndarray) -> trimesh.Trimesh:
        """Create cylinder mesh."""
        cyl = trimesh.creation.cylinder(radius=radius, height=height, sections=24)
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = center
        cyl.apply_transform(transform)
        return cyl

    def _create_sphere_mesh(self, center: np.ndarray, radius: float) -> trimesh.Trimesh:
        """Create sphere mesh."""
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=radius)
        sphere.apply_translation(center)
        return sphere

    def _create_ellipsoid_mesh(self, center: np.ndarray, radii: np.ndarray,
                                rotation: np.ndarray) -> trimesh.Trimesh:
        """Create ellipsoid mesh by scaling a unit sphere."""
        # Create unit sphere
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=1.0)

        # Scale vertices by radii
        vertices = np.asarray(sphere.vertices) * radii

        # Apply rotation and translation
        vertices = vertices @ rotation.T + center

        mesh = trimesh.Trimesh(vertices=vertices, faces=sphere.faces)
        mesh.fix_normals()
        return mesh

    def _create_hemisphere_mesh(self, center: np.ndarray, radius: float,
                                 rotation: np.ndarray) -> trimesh.Trimesh:
        """Create hemisphere mesh."""
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=radius)

        # Keep only vertices with z >= 0
        verts = np.asarray(sphere.vertices)
        mask = verts[:, 2] >= -0.01

        # Create hemisphere by cutting
        hemisphere = sphere.slice_plane([0, 0, 0], [0, 0, -1])
        if hemisphere is None or len(hemisphere.vertices) == 0:
            # Fallback: create manually
            hemisphere = self._create_partial_sphere(radius, 0.5)

        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = center
        hemisphere.apply_transform(transform)
        return hemisphere

    def _create_quarter_sphere_mesh(self, center: np.ndarray, radius: float,
                                     rotation: np.ndarray) -> trimesh.Trimesh:
        """Create quarter sphere mesh."""
        mesh = self._create_partial_sphere(radius, 0.25)
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = center
        mesh.apply_transform(transform)
        return mesh

    def _create_eighth_sphere_mesh(self, center: np.ndarray, radius: float,
                                    rotation: np.ndarray, signs: List[int]) -> trimesh.Trimesh:
        """Create eighth sphere mesh (one octant)."""
        mesh = self._create_partial_sphere(radius, 0.125)
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = center
        mesh.apply_transform(transform)
        return mesh

    def _create_partial_sphere(self, radius: float, fraction: float) -> trimesh.Trimesh:
        """Create a partial sphere mesh."""
        n_lat = 16
        n_lon = 16

        if fraction >= 0.5:
            theta_max = np.pi / 2
        elif fraction >= 0.25:
            theta_max = np.pi / 2
            n_lon = 8
        else:
            theta_max = np.pi / 2
            n_lon = 4

        phi_max = np.pi / 2 if fraction < 0.5 else np.pi

        verts = []
        for i in range(n_lat + 1):
            theta = i * theta_max / n_lat
            for j in range(n_lon + 1):
                phi = j * phi_max / n_lon
                x = radius * np.sin(theta) * np.cos(phi)
                y = radius * np.sin(theta) * np.sin(phi)
                z = radius * np.cos(theta)
                verts.append([x, y, z])

        verts = np.array(verts)

        faces = []
        for i in range(n_lat):
            for j in range(n_lon):
                v0 = i * (n_lon + 1) + j
                v1 = v0 + 1
                v2 = v0 + n_lon + 1
                v3 = v2 + 1
                faces.append([v0, v2, v1])
                faces.append([v1, v2, v3])

        faces = np.array(faces)
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        mesh.fix_normals()
        return mesh

    def _create_cone_mesh(self, center: np.ndarray, radius: float,
                           height: float, rotation: np.ndarray) -> trimesh.Trimesh:
        """Create cone mesh."""
        cone = trimesh.creation.cone(radius=radius, height=height, sections=24)
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = center
        cone.apply_transform(transform)
        return cone

    def _create_capsule_mesh(self, center: np.ndarray, radius: float,
                              height: float, rotation: np.ndarray) -> trimesh.Trimesh:
        """Create capsule mesh."""
        capsule = trimesh.creation.capsule(radius=radius, height=height)
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = center
        capsule.apply_transform(transform)
        return capsule

    def _create_triangular_prism_mesh(self, center: np.ndarray, tri_2d: np.ndarray,
                                       height: float, plane_basis: np.ndarray,
                                       axis: np.ndarray) -> trimesh.Trimesh:
        """Create triangular prism mesh."""
        tri_3d = np.zeros((3, 3))
        for i, pt2d in enumerate(tri_2d):
            tri_3d[i] = pt2d[0] * plane_basis[0] + pt2d[1] * plane_basis[1]

        bottom = tri_3d - (height / 2) * axis + center
        top = tri_3d + (height / 2) * axis + center

        verts = np.vstack([bottom, top])
        faces = np.array([
            [0, 2, 1], [3, 4, 5],
            [0, 1, 4], [0, 4, 3],
            [1, 2, 5], [1, 5, 4],
            [2, 0, 3], [2, 3, 5],
        ])

        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        mesh.fix_normals()
        return mesh

    def _create_tetrahedron_mesh(self, center: np.ndarray, radius: float,
                                  rotation: np.ndarray) -> trimesh.Trimesh:
        """Create tetrahedron mesh."""
        # Regular tetrahedron vertices
        a = radius
        verts = np.array([
            [a, a, a],
            [a, -a, -a],
            [-a, a, -a],
            [-a, -a, a]
        ]) / np.sqrt(3)

        faces = np.array([
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 1],
            [1, 3, 2]
        ])

        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        mesh.fix_normals()

        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = center
        mesh.apply_transform(transform)
        return mesh

    def _create_octahedron_mesh(self, center: np.ndarray, radius: float,
                                 rotation: np.ndarray) -> trimesh.Trimesh:
        """Create octahedron mesh."""
        # Regular octahedron vertices
        verts = np.array([
            [radius, 0, 0],
            [-radius, 0, 0],
            [0, radius, 0],
            [0, -radius, 0],
            [0, 0, radius],
            [0, 0, -radius]
        ])

        faces = np.array([
            [0, 2, 4], [0, 4, 3], [0, 3, 5], [0, 5, 2],
            [1, 4, 2], [1, 3, 4], [1, 5, 3], [1, 2, 5]
        ])

        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        mesh.fix_normals()

        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = center
        mesh.apply_transform(transform)
        return mesh


class SegmentWithPrimitives:
    """Main class for segmenting meshes and fitting primitives."""

    def __init__(self, score_threshold: float = 0.35, tight_fit: bool = True):
        self.fitter = PrimitiveFitter(tight_fit=tight_fit)
        self.score_threshold = score_threshold

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
                is_face_labels: bool = True) -> Tuple[List[Primitive], Dict[int, int]]:
        """Process mesh and fit primitives to each segment."""
        unique_labels = np.unique(labels)
        print(f"Processing {len(unique_labels)} segments...")

        primitives = []
        segment_to_primitive = {}

        for i, segment_id in enumerate(unique_labels):
            print(f"\nSegment {i+1}/{len(unique_labels)} (ID: {segment_id}):")

            points = self.get_segment_points(mesh, labels, segment_id, is_face_labels)
            if len(points) < 3:
                print(f"  Skipping: too few points ({len(points)})")
                continue

            print(f"  Points: {len(points)}")
            primitive = self.fitter.fit_best(points, self.score_threshold)
            print(f"  Best fit: {primitive.type.value} (score: {primitive.score:.3f})")

            primitives.append(primitive)
            segment_to_primitive[segment_id] = len(primitives) - 1

        return primitives, segment_to_primitive

    def create_output_mesh(self, original_mesh: trimesh.Trimesh, labels: np.ndarray,
                           primitives: List[Primitive], segment_to_primitive: Dict[int, int],
                           is_face_labels: bool = True,
                           primitive_opacity: float = 0.6) -> trimesh.Trimesh:
        """Create output mesh with segments and primitive overlays."""
        unique_labels = np.unique(labels)
        n_segments = len(unique_labels)

        cmap = plt.colormaps.get_cmap("tab20").resampled(n_segments)
        label_to_color = {label: np.array(cmap(i)[:3]) for i, label in enumerate(unique_labels)}

        colored_mesh = original_mesh.copy()
        if is_face_labels:
            face_colors = np.zeros((len(labels), 4))
            for i, label in enumerate(labels):
                color = label_to_color.get(label, np.array([0.5, 0.5, 0.5]))
                face_colors[i] = np.append(color, 1.0)
            colored_mesh.visual.face_colors = (face_colors * 255).astype(np.uint8)
        else:
            vertex_colors = np.zeros((len(labels), 4))
            for i, label in enumerate(labels):
                color = label_to_color.get(label, np.array([0.5, 0.5, 0.5]))
                vertex_colors[i] = np.append(color, 1.0)
            colored_mesh.visual.vertex_colors = (vertex_colors * 255).astype(np.uint8)

        meshes_to_combine = [colored_mesh]

        for segment_id, prim_idx in segment_to_primitive.items():
            primitive = primitives[prim_idx]
            if primitive.mesh is not None:
                prim_mesh = primitive.mesh.copy()
                segment_color = label_to_color.get(segment_id, np.array([0.5, 0.5, 0.5]))
                prim_color = np.append(segment_color, primitive_opacity)
                prim_mesh.visual.face_colors = (np.array([prim_color] * len(prim_mesh.faces)) * 255).astype(np.uint8)
                meshes_to_combine.append(prim_mesh)

        return trimesh.util.concatenate(meshes_to_combine)

    def export_primitives_only(self, primitives: List[Primitive],
                               segment_to_primitive: Dict[int, int],
                               output_path: str):
        """Export only the primitive meshes."""
        n_segments = len(segment_to_primitive)
        cmap = plt.colormaps.get_cmap("tab20").resampled(n_segments)

        meshes = []
        for i, (segment_id, prim_idx) in enumerate(segment_to_primitive.items()):
            primitive = primitives[prim_idx]
            if primitive.mesh is not None:
                prim_mesh = primitive.mesh.copy()
                color = np.array(cmap(i)[:3])
                prim_mesh.visual.face_colors = (np.append(color, 0.9) * 255).astype(np.uint8)
                meshes.append(prim_mesh)

        if meshes:
            combined = trimesh.util.concatenate(meshes)
            combined.export(output_path)
            print(f"Exported primitives to: {output_path}")

    def export_results(self, original_mesh: trimesh.Trimesh, labels: np.ndarray,
                       primitives: List[Primitive], segment_to_primitive: Dict[int, int],
                       output_path: str, is_face_labels: bool = True,
                       export_primitives_separately: bool = True):
        """Export results to files."""
        combined_mesh = self.create_output_mesh(
            original_mesh, labels, primitives, segment_to_primitive, is_face_labels
        )
        combined_mesh.export(output_path)
        print(f"Exported combined mesh to: {output_path}")

        if export_primitives_separately:
            base, ext = os.path.splitext(output_path)
            primitives_path = f"{base}_primitives_only{ext}"
            self.export_primitives_only(primitives, segment_to_primitive, primitives_path)

        info_path = os.path.splitext(output_path)[0] + "_info.txt"
        with open(info_path, 'w') as f:
            f.write("Segment Primitive Fitting Results\n")
            f.write("=" * 50 + "\n\n")

            for segment_id, prim_idx in segment_to_primitive.items():
                primitive = primitives[prim_idx]
                f.write(f"Segment {segment_id}:\n")
                f.write(f"  Primitive Type: {primitive.type.value}\n")
                f.write(f"  Fitting Score: {primitive.score:.4f}\n")
                f.write(f"  Center: [{primitive.center[0]:.4f}, {primitive.center[1]:.4f}, {primitive.center[2]:.4f}]\n")
                f.write(f"  Dimensions: [{primitive.dimensions[0]:.4f}, {primitive.dimensions[1]:.4f}, {primitive.dimensions[2]:.4f}]\n")
                if primitive.params:
                    f.write(f"  Parameters:\n")
                    for key, val in primitive.params.items():
                        if isinstance(val, np.ndarray) and val.size <= 6:
                            f.write(f"    {key}: {val.tolist()}\n")
                        elif not isinstance(val, np.ndarray):
                            f.write(f"    {key}: {val}\n")
                f.write("\n")

        print(f"Exported info to: {info_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Segment mesh and fit geometric primitives to each segment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Supported primitives:
  - box, cylinder, sphere, hemisphere, quarter_sphere, eighth_sphere
  - cone, capsule, triangular_prism, tetrahedron, octahedron

Examples:
  python segment_with_primitives.py -i model.glb -l labels.npy -o output.ply
  python segment_with_primitives.py -i mesh.ply -l clustering.npy -o result.ply --threshold 0.4
        """
    )

    parser.add_argument("--input", "-i", required=True, help="Input mesh file")
    parser.add_argument("--labels", "-l", required=True, help="Segmentation labels (NPY)")
    parser.add_argument("--output", "-o", required=True, help="Output file path")
    parser.add_argument("--threshold", "-t", type=float, default=0.35,
                        help="Score threshold (default: 0.35)")
    parser.add_argument("--vertex-labels", action="store_true",
                        help="Treat labels as per-vertex")
    parser.add_argument("--no-primitives-separate", action="store_true",
                        help="Don't export primitives-only file")
    parser.add_argument("--brep", action="store_true",
                        help="Generate BREP STEP file instead of PLY mesh output")
    parser.add_argument("--all-formats", action="store_true",
                        help="Generate both STEP (BREP) and PLY (mesh) output")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return
    if not os.path.exists(args.labels):
        print(f"Error: Labels file not found: {args.labels}")
        return

    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    processor = SegmentWithPrimitives(score_threshold=args.threshold)

    print(f"Loading mesh: {args.input}")
    mesh = processor.load_mesh(args.input)
    if mesh is None:
        return

    print(f"Loading labels: {args.labels}")
    labels = processor.load_labels(args.labels)
    if labels is None:
        return

    is_face_labels = not args.vertex_labels
    expected_count = len(mesh.faces) if is_face_labels else len(mesh.vertices)

    if len(labels) != expected_count:
        if len(labels) == len(mesh.faces):
            is_face_labels = True
        elif len(labels) == len(mesh.vertices):
            is_face_labels = False

    print(f"\nMesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    print(f"Labels: {len(labels)} ({'face' if is_face_labels else 'vertex'})")

    primitives, segment_to_primitive = processor.process(mesh, labels, is_face_labels)

    print(f"\n{'=' * 50}")
    print(f"Fitted {len(primitives)} primitives")

    type_counts = {}
    for p in primitives:
        type_counts[p.type.value] = type_counts.get(p.type.value, 0) + 1

    print("\nPrimitive distribution:")
    for ptype, count in sorted(type_counts.items()):
        print(f"  {ptype}: {count}")

    print(f"\n{'=' * 50}")

    generate_brep = args.brep or args.all_formats
    generate_mesh = not args.brep or args.all_formats

    if generate_mesh:
        processor.export_results(
            mesh, labels, primitives, segment_to_primitive,
            args.output, is_face_labels,
            export_primitives_separately=not args.no_primitives_separate
        )

    if generate_brep:
        try:
            from brep_generator import BRepShapeBuilder, BRepExporter
            import matplotlib.pyplot as _plt

            builder = BRepShapeBuilder()
            exporter = BRepExporter()
            cmap = _plt.colormaps.get_cmap("tab20").resampled(max(len(segment_to_primitive), 1))

            shapes_and_colors = []
            for i, (seg_id, prim_idx) in enumerate(segment_to_primitive.items()):
                prim = primitives[prim_idx]
                color = cmap(i % 20)[:3]
                try:
                    shape = builder.shape_from_primitive(prim)
                    shapes_and_colors.append((shape, tuple(color)))
                except Exception as e:
                    print(f"  Warning: BREP failed for segment {seg_id} ({prim.type.value}): {e}")

            if shapes_and_colors:
                base, _ = os.path.splitext(args.output)
                step_path = base + ".step"
                exporter.export_colored_step(shapes_and_colors, step_path)
        except ImportError:
            print("Warning: pythonocc-core not available. Skipping BREP output.")
            print("Install with: conda install -c conda-forge pythonocc-core")

    print("\nDone!")


if __name__ == "__main__":
    main()
