#!/usr/bin/env python3
"""
PartField Viewer - Interactive 3D visualization for PartField segmentation results.

Supports viewing GLB, PLY, OBJ files with their part segmentation overlays.
Toggle between original and segmented views, browse multiple files, and cycle
through different clustering results.

Keyboard Controls:
    TAB/T   - Cycle views: Original → Segmented → Bounding Boxes → PCA Features
    C       - Next clustering result (more parts)
    V       - Previous clustering result (fewer parts)
    LEFT/A  - Previous file (browse mode)
    RIGHT/D - Next file (browse mode)
    S       - Save screenshot
    ESC/Q   - Exit viewer
    R       - Reset camera view
    H       - Show help
"""

# Configure environment for WSL2/software rendering BEFORE importing Open3D
import os
os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'
os.environ['MESA_GLSL_VERSION_OVERRIDE'] = '330'
os.environ['GALLIUM_DRIVER'] = 'llvmpipe'

import argparse
import glob
import re
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import numpy as np

try:
    import open3d as o3d
except ImportError:
    print("Error: open3d is required. Install with: pip install open3d")
    exit(1)

try:
    import trimesh
except ImportError:
    print("Error: trimesh is required. Install with: pip install trimesh")
    exit(1)

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Error: matplotlib is required. Install with: pip install matplotlib")
    exit(1)

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("Error: Pillow is required. Install with: pip install Pillow")
    exit(1)


class DataLoader:
    """Handles loading of 3D meshes and clustering labels."""

    SUPPORTED_MESH_FORMATS = {'.glb', '.gltf', '.obj', '.ply', '.stl', '.off'}

    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)

    def load_mesh(self, path: str) -> Optional[o3d.geometry.TriangleMesh]:
        """Load a 3D mesh file and convert to Open3D format."""
        path = Path(path)
        if not path.exists():
            print(f"Error: File not found: {path}")
            return None

        suffix = path.suffix.lower()

        try:
            if suffix == '.ply':
                # Try Open3D first for PLY
                mesh = o3d.io.read_triangle_mesh(str(path))
                if len(mesh.vertices) == 0:
                    # Might be a point cloud
                    pcd = o3d.io.read_point_cloud(str(path))
                    if len(pcd.points) > 0:
                        mesh = o3d.geometry.TriangleMesh()
                        mesh.vertices = pcd.points
                        if pcd.has_colors():
                            mesh.vertex_colors = pcd.colors
                        return mesh

                # If Open3D didn't load colors, try trimesh
                if not mesh.has_vertex_colors():
                    try:
                        tm = trimesh.load(str(path))
                        if hasattr(tm, 'visual') and hasattr(tm.visual, 'vertex_colors'):
                            if tm.visual.vertex_colors is not None:
                                colors = tm.visual.vertex_colors[:, :3] / 255.0
                                # Handle vertex count mismatch
                                if len(colors) == len(mesh.vertices):
                                    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
                                elif len(colors) < len(mesh.vertices):
                                    # Pad with gray
                                    full_colors = np.ones((len(mesh.vertices), 3)) * 0.5
                                    full_colors[:len(colors)] = colors
                                    mesh.vertex_colors = o3d.utility.Vector3dVector(full_colors)
                    except Exception:
                        pass  # Keep mesh without colors

                return mesh

            elif suffix in {'.glb', '.gltf', '.obj', '.stl', '.off'}:
                # Use trimesh for other formats
                scene_or_mesh = trimesh.load(str(path), force='mesh')

                if isinstance(scene_or_mesh, trimesh.Scene):
                    # Combine all geometries in scene
                    meshes = []
                    for name, geom in scene_or_mesh.geometry.items():
                        if isinstance(geom, trimesh.Trimesh):
                            meshes.append(geom)
                    if meshes:
                        combined = trimesh.util.concatenate(meshes)
                    else:
                        print(f"No mesh geometry found in {path}")
                        return None
                else:
                    combined = scene_or_mesh

                # Convert trimesh to Open3D
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(combined.vertices)
                mesh.triangles = o3d.utility.Vector3iVector(combined.faces)

                # Transfer vertex colors if present
                if combined.visual is not None:
                    if hasattr(combined.visual, 'vertex_colors') and combined.visual.vertex_colors is not None:
                        colors = combined.visual.vertex_colors[:, :3] / 255.0
                        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

                return mesh
            else:
                print(f"Unsupported file format: {suffix}")
                return None

        except Exception as e:
            print(f"Error loading mesh {path}: {e}")
            return None

    def load_labels(self, path: str) -> Optional[np.ndarray]:
        """Load clustering labels from NPY file."""
        path = Path(path)
        if not path.exists():
            print(f"Warning: Labels file not found: {path}")
            return None

        try:
            labels = np.load(str(path))
            return labels
        except Exception as e:
            print(f"Error loading labels {path}: {e}")
            return None

    def extract_mesh_id(self, path: str) -> Optional[str]:
        """Extract the mesh ID (hash) from a filename."""
        path = Path(path)
        name = path.stem

        # Pattern: 32-character hex hash, optionally followed by _0, _1, etc.
        match = re.search(r'([a-f0-9]{32})', name)
        if match:
            return match.group(1)

        # Try to extract from input_XXX_0 pattern
        match = re.search(r'input_([a-f0-9]{32})_\d+', name)
        if match:
            return match.group(1)

        return None

    def find_matching_results(self, input_path: str) -> Dict[str, List[str]]:
        """Find matching clustering results for an input file."""
        mesh_id = self.extract_mesh_id(input_path)
        results = {
            'input_meshes': [],
            'clustering_labels': [],
            'pca_features': [],
            'bbox_meshes': []
        }

        if not mesh_id:
            return results

        # Search in exp_results directory
        exp_results = self.base_path / "exp_results"

        # Find input meshes
        input_pattern = exp_results / "partfield_features" / "**" / f"input_{mesh_id}*.ply"
        results['input_meshes'] = sorted(glob.glob(str(input_pattern), recursive=True))

        # Find clustering labels
        cluster_pattern = exp_results / "clustering" / "**" / "cluster_out" / f"{mesh_id}*.npy"
        results['clustering_labels'] = sorted(glob.glob(str(cluster_pattern), recursive=True))

        # Also try alternative patterns
        if not results['clustering_labels']:
            cluster_pattern = exp_results / "clustering" / "**" / f"*{mesh_id}*.npy"
            results['clustering_labels'] = sorted(glob.glob(str(cluster_pattern), recursive=True))

        # Find PCA feature meshes
        pca_pattern = exp_results / "partfield_features" / "**" / f"feat_pca_{mesh_id}*.ply"
        results['pca_features'] = sorted(glob.glob(str(pca_pattern), recursive=True))

        # Find bounding box meshes (from segment_with_bboxes.py output)
        bbox_dir = exp_results / "bboxes"
        for bbox_style in ['solid', 'wireframe', 'transparent']:
            bbox_pattern = bbox_dir / "**" / f"*{mesh_id}*_{bbox_style}.*"
            results['bbox_meshes'].extend(glob.glob(str(bbox_pattern), recursive=True))
        # Also check for bbox files without style suffix
        bbox_pattern = bbox_dir / "**" / f"*{mesh_id}*.ply"
        for f in glob.glob(str(bbox_pattern), recursive=True):
            if f not in results['bbox_meshes']:
                results['bbox_meshes'].append(f)
        bbox_pattern = bbox_dir / "**" / f"*{mesh_id}*.glb"
        for f in glob.glob(str(bbox_pattern), recursive=True):
            if f not in results['bbox_meshes']:
                results['bbox_meshes'].append(f)
        results['bbox_meshes'] = sorted(results['bbox_meshes'])

        return results

    def get_files_in_directory(self, directory: str) -> List[str]:
        """Get all viewable files in a directory."""
        directory = Path(directory)
        files = []

        for ext in self.SUPPORTED_MESH_FORMATS:
            files.extend(glob.glob(str(directory / f"*{ext}")))
            files.extend(glob.glob(str(directory / f"**/*{ext}"), recursive=True))

        return sorted(set(files))


class PartColorizer:
    """Handles colorization of meshes based on part labels."""

    def __init__(self, colormap: str = "tab20"):
        self.colormap_name = colormap

    def get_color_palette(self, n_parts: int) -> np.ndarray:
        """Generate a color palette for n parts."""
        if n_parts <= 20:
            cmap = plt.colormaps.get_cmap("tab20")
        elif n_parts <= 40:
            # Combine tab20 and tab20b
            cmap1 = plt.colormaps.get_cmap("tab20")
            cmap2 = plt.colormaps.get_cmap("tab20b")
            colors1 = [cmap1(i)[:3] for i in range(20)]
            colors2 = [cmap2(i)[:3] for i in range(20)]
            colors = np.array(colors1 + colors2)
            return colors[:n_parts]
        else:
            # Use HSV for many parts
            cmap = plt.colormaps.get_cmap("hsv")

        colors = np.array([cmap(i)[:3] for i in range(n_parts)])
        return colors

    def colorize_mesh(self, mesh: o3d.geometry.TriangleMesh,
                      labels: np.ndarray) -> o3d.geometry.TriangleMesh:
        """Apply colors to mesh vertices based on labels."""
        mesh_colored = o3d.geometry.TriangleMesh(mesh)

        n_vertices = len(mesh_colored.vertices)
        n_labels = len(labels)

        # Handle size mismatch
        if n_labels != n_vertices:
            # Check if labels are for faces instead of vertices
            n_faces = len(mesh_colored.triangles)
            if n_labels == n_faces:
                # Convert face labels to vertex labels
                labels = self._face_labels_to_vertex_labels(mesh_colored, labels)
            else:
                print(f"Warning: Label count ({n_labels}) doesn't match vertex ({n_vertices}) or face ({n_faces}) count")
                # Resample labels to match vertices
                if n_labels > n_vertices:
                    indices = np.linspace(0, n_labels - 1, n_vertices, dtype=int)
                    labels = labels[indices]
                else:
                    # Repeat labels
                    labels = np.resize(labels, n_vertices)

        # Get unique labels and create color mapping
        unique_labels = np.unique(labels)
        n_parts = len(unique_labels)
        colors = self.get_color_palette(n_parts)

        # Create label to color mapping
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}

        # Apply colors to vertices
        vertex_colors = np.zeros((len(labels), 3))
        for i, label in enumerate(labels):
            vertex_colors[i] = colors[label_to_idx[label]]

        mesh_colored.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

        return mesh_colored, n_parts

    def _face_labels_to_vertex_labels(self, mesh: o3d.geometry.TriangleMesh,
                                       face_labels: np.ndarray) -> np.ndarray:
        """Convert face labels to vertex labels by majority voting."""
        n_vertices = len(mesh.vertices)
        triangles = np.asarray(mesh.triangles)

        # Count label occurrences for each vertex
        vertex_label_counts = {}
        for face_idx, face in enumerate(triangles):
            label = face_labels[face_idx]
            for vertex_idx in face:
                if vertex_idx not in vertex_label_counts:
                    vertex_label_counts[vertex_idx] = {}
                if label not in vertex_label_counts[vertex_idx]:
                    vertex_label_counts[vertex_idx][label] = 0
                vertex_label_counts[vertex_idx][label] += 1

        # Assign most common label to each vertex
        vertex_labels = np.zeros(n_vertices, dtype=np.int64)
        for vertex_idx in range(n_vertices):
            if vertex_idx in vertex_label_counts:
                counts = vertex_label_counts[vertex_idx]
                vertex_labels[vertex_idx] = max(counts, key=counts.get)

        return vertex_labels


class PartFieldViewer:
    """Interactive 3D viewer for PartField visualization."""

    # View modes
    VIEW_ORIGINAL = 0
    VIEW_SEGMENTED = 1
    VIEW_BBOXES = 2
    VIEW_PCA = 3
    VIEW_NAMES = ["Original", "Segmented", "Bounding Boxes", "PCA Features"]

    def __init__(self):
        self.loader = DataLoader()
        self.colorizer = PartColorizer()

        # Current state
        self.current_mesh_original: Optional[o3d.geometry.TriangleMesh] = None
        self.current_mesh_segmented: Optional[o3d.geometry.TriangleMesh] = None
        self.current_mesh_bboxes: Optional[o3d.geometry.TriangleMesh] = None
        self.current_mesh_pca: Optional[o3d.geometry.TriangleMesh] = None
        self.current_file_path: str = ""
        self.current_file_index: int = 0
        self.file_list: List[str] = []
        self.clustering_files: List[str] = []
        self.bbox_files: List[str] = []
        self.pca_files: List[str] = []
        self.clustering_index: int = 0
        self.n_parts: int = 0

        # View state (0=Original, 1=Segmented, 2=PCA)
        self.view_mode: int = self.VIEW_SEGMENTED
        self.browse_mode: bool = False

        # Help overlay state
        self.show_help_overlay: bool = True  # Show by default
        self.help_text_geometry = None

        # Visualizer
        self.vis: Optional[o3d.visualization.VisualizerWithKeyCallback] = None
        self.should_exit: bool = False

    def load_file(self, file_path: str, labels_path: Optional[str] = None) -> bool:
        """Load a mesh file and its associated clustering results."""
        self.current_file_path = file_path

        # Load original mesh
        self.current_mesh_original = self.loader.load_mesh(file_path)
        if self.current_mesh_original is None:
            return False

        # Compute normals for better visualization
        self.current_mesh_original.compute_vertex_normals()

        # Set default color if no vertex colors
        if not self.current_mesh_original.has_vertex_colors():
            n_vertices = len(self.current_mesh_original.vertices)
            default_color = np.ones((n_vertices, 3)) * 0.7  # Light gray
            self.current_mesh_original.vertex_colors = o3d.utility.Vector3dVector(default_color)

        # Find or use provided clustering labels
        if labels_path:
            self.clustering_files = [labels_path]
            self.pca_files = []
            self.bbox_files = []
        else:
            results = self.loader.find_matching_results(file_path)
            self.clustering_files = results['clustering_labels']
            self.pca_files = results['pca_features']
            self.bbox_files = results['bbox_meshes']

        self.clustering_index = 0

        # Load segmented mesh
        self._load_current_clustering()

        # Load bounding box mesh if available
        self._load_bbox_mesh()

        # Load PCA feature mesh if available
        self._load_pca_mesh()

        return True

    def _load_bbox_mesh(self):
        """Load the bounding box visualization mesh."""
        self.current_mesh_bboxes = None

        if not self.bbox_files:
            return

        # Prefer wireframe style for clearer visualization, fall back to solid
        bbox_path = None
        for f in self.bbox_files:
            if 'wireframe' in f.lower():
                bbox_path = f
                break
        if bbox_path is None and self.bbox_files:
            bbox_path = self.bbox_files[0]

        if bbox_path:
            self.current_mesh_bboxes = self.loader.load_mesh(bbox_path)

            if self.current_mesh_bboxes is not None:
                self.current_mesh_bboxes.compute_vertex_normals()

    def _load_pca_mesh(self):
        """Load the PCA feature visualization mesh."""
        self.current_mesh_pca = None

        if not self.pca_files:
            return

        pca_path = self.pca_files[0]  # Usually just one PCA file per mesh
        self.current_mesh_pca = self.loader.load_mesh(pca_path)

        if self.current_mesh_pca is not None:
            self.current_mesh_pca.compute_vertex_normals()

    def _load_current_clustering(self):
        """Load the current clustering result."""
        if not self.clustering_files:
            # No clustering available, use original
            self.current_mesh_segmented = o3d.geometry.TriangleMesh(self.current_mesh_original)
            self.n_parts = 0
            return

        labels_path = self.clustering_files[self.clustering_index]
        labels = self.loader.load_labels(labels_path)

        if labels is None:
            self.current_mesh_segmented = o3d.geometry.TriangleMesh(self.current_mesh_original)
            self.n_parts = 0
            return

        self.current_mesh_segmented, self.n_parts = self.colorizer.colorize_mesh(
            self.current_mesh_original, labels
        )
        self.current_mesh_segmented.compute_vertex_normals()

    def _get_window_title(self) -> str:
        """Generate window title with current state."""
        mode = self.VIEW_NAMES[self.view_mode]
        filename = Path(self.current_file_path).name

        parts = [f"PartField Viewer | {mode} | {filename}"]

        if self.view_mode == self.VIEW_SEGMENTED and self.clustering_files:
            parts.append(f"Clustering {self.clustering_index + 1}/{len(self.clustering_files)}")
            if self.n_parts > 0:
                parts.append(f"Parts: {self.n_parts}")

        if self.browse_mode:
            parts.append(f"File {self.current_file_index + 1}/{len(self.file_list)}")

        return " | ".join(parts)

    def _get_current_mesh(self) -> Optional[o3d.geometry.TriangleMesh]:
        """Get the mesh for the current view mode."""
        if self.view_mode == self.VIEW_ORIGINAL:
            return self.current_mesh_original
        elif self.view_mode == self.VIEW_SEGMENTED:
            return self.current_mesh_segmented
        elif self.view_mode == self.VIEW_BBOXES:
            return self.current_mesh_bboxes if self.current_mesh_bboxes else self.current_mesh_original
        elif self.view_mode == self.VIEW_PCA:
            return self.current_mesh_pca if self.current_mesh_pca else self.current_mesh_original
        return self.current_mesh_original

    def _create_help_overlay_image(self) -> np.ndarray:
        """Create a help overlay image with keyboard shortcuts."""
        # Create image with transparent background
        width, height = 350, 280
        img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Try to use a nice font, fall back to default if not available
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 14)
            font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 16)
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf", 14)
                font_title = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf", 16)
            except:
                font = ImageFont.load_default()
                font_title = ImageFont.load_default()

        # Semi-transparent background
        bg_color = (30, 30, 30, 220)
        border_color = (100, 100, 100, 255)
        text_color = (255, 255, 255, 255)
        highlight_color = (100, 200, 255, 255)

        # Draw background with border
        draw.rectangle([0, 0, width-1, height-1], fill=bg_color, outline=border_color, width=2)

        # Draw title
        title = "KEYBOARD SHORTCUTS"
        draw.text((10, 10), title, fill=highlight_color, font=font_title)

        # Draw separator line
        draw.line([(10, 35), (width-10, 35)], fill=border_color, width=1)

        # Draw shortcuts
        shortcuts = [
            ("T/TAB", "Cycle views"),
            ("C", "Next clustering"),
            ("V", "Prev clustering"),
            ("A/LEFT", "Previous file"),
            ("D/RIGHT", "Next file"),
            ("S", "Screenshot"),
            ("R", "Reset view"),
            ("H", "Toggle help"),
            ("ESC/Q", "Exit")
        ]

        y_offset = 45
        for key, action in shortcuts:
            # Draw key in highlight color
            draw.text((15, y_offset), f"{key:10s}", fill=highlight_color, font=font)
            # Draw action in white
            draw.text((130, y_offset), action, fill=text_color, font=font)
            y_offset += 24

        return np.array(img)

    def _update_visualization(self):
        """Update the visualization with current mesh."""
        if self.vis is None:
            return

        self.vis.clear_geometries()

        mesh = self._get_current_mesh()
        if mesh is not None:
            self.vis.add_geometry(mesh)

        # Add help overlay if enabled
        if self.show_help_overlay:
            self._add_help_text_overlay()

        # Update window title - this doesn't work directly in Open3D,
        # so we'll print status to console
        print(f"\r{self._get_window_title()}", end="", flush=True)

    def _create_help_billboard(self) -> o3d.geometry.TriangleMesh:
        """Create a 3D billboard mesh with help text texture."""
        try:
            # Create the help overlay image
            help_img = self._create_help_overlay_image()

            # Create a rectangular mesh for the billboard
            # Size ratio based on image dimensions
            aspect_ratio = help_img.shape[1] / help_img.shape[0]  # width / height
            billboard_height = 0.5  # Scale factor for viewport
            billboard_width = billboard_height * aspect_ratio

            # Create a simple quad mesh
            vertices = np.array([
                [0, 0, 0],
                [billboard_width, 0, 0],
                [billboard_width, billboard_height, 0],
                [0, billboard_height, 0]
            ])

            # Create two triangles for the quad
            triangles = np.array([
                [0, 1, 2],
                [0, 2, 3]
            ])

            # Create UV coordinates for texture mapping
            uvs = np.array([
                [0, 1],  # bottom-left
                [1, 1],  # bottom-right
                [1, 0],  # top-right
                [0, 0]   # top-left
            ])

            # Create the mesh
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(vertices)
            mesh.triangles = o3d.utility.Vector3iVector(triangles)

            # Apply a simple solid color (since texture mapping is complex in Open3D)
            # We'll use vertex colors instead
            # For now, just make it semi-transparent dark gray
            colors = np.array([
                [0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1]
            ])
            mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
            mesh.compute_vertex_normals()

            return mesh
        except Exception as e:
            print(f"Could not create help billboard: {e}")
            return None

    def _add_help_text_overlay(self):
        """Add help text overlay to the visualization."""
        # Note: Open3D's VisualizerWithKeyCallback has very limited overlay support
        # The help is primarily displayed in the console
        # Advanced overlay would require using o3d.visualization.gui which requires
        # a complete rewrite of the viewer
        pass

    def _toggle_view(self, vis):
        """Cycle through view modes: Original → Segmented → Bounding Boxes → PCA Features."""
        # Determine available modes
        available_modes = [self.VIEW_ORIGINAL, self.VIEW_SEGMENTED]
        if self.current_mesh_bboxes is not None:
            available_modes.append(self.VIEW_BBOXES)
        if self.current_mesh_pca is not None:
            available_modes.append(self.VIEW_PCA)

        # Find next mode
        try:
            current_idx = available_modes.index(self.view_mode)
            next_idx = (current_idx + 1) % len(available_modes)
            self.view_mode = available_modes[next_idx]
        except ValueError:
            self.view_mode = self.VIEW_ORIGINAL

        self._update_visualization()
        return False

    def _next_clustering(self, vis):
        """Switch to next clustering result."""
        if len(self.clustering_files) > 1:
            self.clustering_index = (self.clustering_index + 1) % len(self.clustering_files)
            self._load_current_clustering()
            self._update_visualization()
        return False

    def _prev_clustering(self, vis):
        """Switch to previous clustering result."""
        if len(self.clustering_files) > 1:
            self.clustering_index = (self.clustering_index - 1) % len(self.clustering_files)
            self._load_current_clustering()
            self._update_visualization()
        return False

    def _next_file(self, vis):
        """Switch to next file in browse mode."""
        if self.browse_mode and len(self.file_list) > 1:
            self.current_file_index = (self.current_file_index + 1) % len(self.file_list)
            self.load_file(self.file_list[self.current_file_index])
            self._update_visualization()
            vis.reset_view_point(True)
        return False

    def _prev_file(self, vis):
        """Switch to previous file in browse mode."""
        if self.browse_mode and len(self.file_list) > 1:
            self.current_file_index = (self.current_file_index - 1) % len(self.file_list)
            self.load_file(self.file_list[self.current_file_index])
            self._update_visualization()
            vis.reset_view_point(True)
        return False

    def _save_screenshot(self, vis):
        """Save current view as screenshot."""
        filename = f"screenshot_{Path(self.current_file_path).stem}_{self.clustering_index}.png"
        vis.capture_screen_image(filename)
        print(f"\nScreenshot saved: {filename}")
        return False

    def _exit_viewer(self, vis):
        """Exit the viewer."""
        self.should_exit = True
        vis.destroy_window()
        return True

    def _reset_view(self, vis):
        """Reset camera view."""
        vis.reset_view_point(True)
        return False

    def _toggle_help_overlay(self, vis):
        """Toggle the help overlay visibility."""
        self.show_help_overlay = not self.show_help_overlay

        if self.show_help_overlay:
            print("\n" + "╔" + "═" * 53 + "╗")
            print("║" + " " * 10 + "KEYBOARD SHORTCUTS" + " " * 25 + "║")
            print("╠" + "═" * 53 + "╣")
            print("║  T/TAB     Cycle views: Original → Seg → BBox → PCA  ║")
            print("║  C         Next clustering (more parts)             ║")
            print("║  V         Previous clustering (fewer parts)        ║")
            print("║  A/LEFT    Previous file (browse mode)              ║")
            print("║  D/RIGHT   Next file (browse mode)                  ║")
            print("║  S         Save screenshot                          ║")
            print("║  R         Reset camera view                        ║")
            print("║  H         Toggle help panel                        ║")
            print("║  ESC/Q     Exit viewer                              ║")
            print("╚" + "═" * 53 + "╝")
        else:
            print("\n[Help overlay hidden - Press H to show]")

        return False

    def _show_help(self, vis):
        """Show help information (legacy, calls toggle)."""
        return self._toggle_help_overlay(vis)

    def run(self, file_path: str = None, labels_path: str = None,
            browse_dir: str = None):
        """Run the interactive viewer."""

        # Setup browse mode if directory specified
        if browse_dir:
            self.browse_mode = True
            self.file_list = self.loader.get_files_in_directory(browse_dir)
            if not self.file_list:
                print(f"No viewable files found in {browse_dir}")
                return
            print(f"Browse mode: Found {len(self.file_list)} files")
            file_path = self.file_list[0]
        elif file_path:
            self.file_list = [file_path]
        else:
            print("Error: No file or directory specified")
            return

        # Load initial file
        if not self.load_file(file_path, labels_path):
            print("Failed to load file")
            return

        # Create visualizer with key callbacks
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        try:
            success = self.vis.create_window(window_name="PartField Viewer", width=1280, height=720)
            if not success:
                print("\nError: Failed to create visualization window.")
                print("This may be due to:")
                print("  - Missing display server (run with DISPLAY variable set)")
                print("  - WSL2 without X11 forwarding (install VcXsrv/X410 and set DISPLAY)")
                print("  - Wayland compatibility issues (try X11 backend)")
                print("\nFor WSL2, try:")
                print("  1. Install X server (VcXsrv, X410, or WSLg)")
                print("  2. export DISPLAY=:0")
                print("  3. Run the viewer again")
                return
        except Exception as e:
            print(f"\nError creating visualization window: {e}")
            print("Open3D visualization requires a display server with OpenGL support.")
            return

        # Register key callbacks
        # TAB key
        self.vis.register_key_callback(ord('\t'), self._toggle_view)
        self.vis.register_key_callback(9, self._toggle_view)  # TAB key code

        # T as alternative for TAB (since TAB might not work in all systems)
        self.vis.register_key_callback(ord('T'), self._toggle_view)
        self.vis.register_key_callback(ord('t'), self._toggle_view)

        # C/V for clustering navigation
        self.vis.register_key_callback(ord('C'), self._next_clustering)
        self.vis.register_key_callback(ord('c'), self._next_clustering)
        self.vis.register_key_callback(ord('V'), self._prev_clustering)
        self.vis.register_key_callback(ord('v'), self._prev_clustering)

        # Arrow keys and A/D for file navigation
        self.vis.register_key_callback(ord('D'), self._next_file)
        self.vis.register_key_callback(ord('d'), self._next_file)
        self.vis.register_key_callback(ord('A'), self._prev_file)
        self.vis.register_key_callback(ord('a'), self._prev_file)
        self.vis.register_key_callback(262, self._next_file)  # RIGHT arrow
        self.vis.register_key_callback(263, self._prev_file)  # LEFT arrow

        # S for screenshot
        self.vis.register_key_callback(ord('S'), self._save_screenshot)
        self.vis.register_key_callback(ord('s'), self._save_screenshot)

        # R for reset view
        self.vis.register_key_callback(ord('R'), self._reset_view)
        self.vis.register_key_callback(ord('r'), self._reset_view)

        # H for help toggle
        self.vis.register_key_callback(ord('H'), self._toggle_help_overlay)
        self.vis.register_key_callback(ord('h'), self._toggle_help_overlay)

        # ESC and Q for exit
        self.vis.register_key_callback(256, self._exit_viewer)  # ESC
        self.vis.register_key_callback(ord('Q'), self._exit_viewer)
        self.vis.register_key_callback(ord('q'), self._exit_viewer)

        # Add initial mesh
        mesh = self._get_current_mesh()
        if mesh is not None:
            self.vis.add_geometry(mesh)

        # Set render options
        render_option = self.vis.get_render_option()
        if render_option is not None:
            render_option.background_color = np.array([0.1, 0.1, 0.1])  # Dark background
            render_option.point_size = 2.0
            render_option.mesh_show_back_face = True
        else:
            print("\nWarning: Could not get render options. This may be due to display/OpenGL issues.")
            print("The viewer may not display correctly in headless or WSL environments.")
            print("Consider running with X11 forwarding or using a native display.")

        # Print initial status and help
        print("\n" + "╔" + "═" * 53 + "╗")
        print("║" + " " * 16 + "PARTFIELD VIEWER" + " " * 21 + "║")
        print("╠" + "═" * 53 + "╣")
        print(f"║  File: {Path(file_path).name:<44} ║")
        if self.clustering_files:
            print(f"║  Clustering results: {len(self.clustering_files):<31} ║")
            print(f"║  Current parts: {self.n_parts:<36} ║")
        else:
            print("║  No clustering results found                        ║")
        if self.bbox_files:
            print("║  Bounding boxes: available (press T to view)        ║")
        if self.pca_files:
            print("║  PCA features: available (press T to view)          ║")
        print("╠" + "═" * 53 + "╣")
        print("║" + " " * 10 + "KEYBOARD SHORTCUTS" + " " * 25 + "║")
        print("╠" + "═" * 53 + "╣")
        print("║  T/TAB     Cycle views: Original → Seg → BBox → PCA  ║")
        print("║  C         Next clustering (more parts)             ║")
        print("║  V         Previous clustering (fewer parts)        ║")
        print("║  A/LEFT    Previous file (browse mode)              ║")
        print("║  D/RIGHT   Next file (browse mode)                  ║")
        print("║  S         Save screenshot                          ║")
        print("║  R         Reset camera view                        ║")
        print("║  H         Toggle help panel                        ║")
        print("║  ESC/Q     Exit viewer                              ║")
        print("╚" + "═" * 53 + "╝")
        print(f"\n{self._get_window_title()}")

        # Reset view to fit mesh
        self.vis.reset_view_point(True)

        # Run visualization loop
        self.vis.run()

        # Cleanup
        self.vis.destroy_window()
        print("\nViewer closed.")


def main():
    parser = argparse.ArgumentParser(
        description="PartField Viewer - Interactive 3D visualization for part segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # View a specific file with auto-detected clustering
  python viewer.py data/objaverse_samples/model.glb

  # View with specific clustering labels
  python viewer.py mesh.ply --labels clustering.npy

  # Browse all files in a directory
  python viewer.py --browse data/objaverse_samples/

  # Browse exp_results
  python viewer.py --browse exp_results/partfield_features/objaverse/

Keyboard Controls:
  TAB/T   - Cycle views: Original → Segmented → BBoxes → PCA
  C/V     - Next/Previous clustering result
  A/D     - Previous/Next file (browse mode)
  S       - Save screenshot
  R       - Reset camera view
  H       - Show help
  ESC/Q   - Exit viewer
        """
    )

    parser.add_argument(
        "file",
        nargs="?",
        help="Path to mesh file (GLB, PLY, OBJ, etc.)"
    )
    parser.add_argument(
        "--labels", "-l",
        help="Path to clustering labels NPY file (auto-detected if not specified)"
    )
    parser.add_argument(
        "--browse", "-b",
        metavar="DIR",
        help="Browse mode: view all files in directory"
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.file and not args.browse:
        parser.print_help()
        print("\nError: Please specify a file or use --browse for directory browsing")
        return

    # Create and run viewer
    viewer = PartFieldViewer()
    viewer.run(
        file_path=args.file,
        labels_path=args.labels,
        browse_dir=args.browse
    )


if __name__ == "__main__":
    main()
