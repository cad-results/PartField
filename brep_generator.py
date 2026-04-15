#!/usr/bin/env python3
"""
BREP CAD Output for PartField Segmentation - Generate STEP files from segmentation results.

Converts PartField segmentation results (bounding boxes, primitives) into parametric
BREP solid geometry (STEP format) viewable in any CAD tool (FreeCAD, SolidWorks, CATIA).

Three classes:
  BRepShapeBuilder  - Converts bbox/primitive metadata to TopoDS_Shape
  BRepExporter      - STEP file export (plain and colored XCAF)
  BRepFromSegmentation - Pipeline orchestration reusing existing fitters

Usage:
    # From bounding box segmentation (self-aligned)
    python brep_generator.py --input model.glb --labels labels.npy --output result.step --mode bbox --alignment self

    # From bounding box segmentation (global-aligned)
    python brep_generator.py --input model.glb --labels labels.npy --output result.step --mode bbox --alignment global

    # From primitive segmentation
    python brep_generator.py --input model.glb --labels labels.npy --output result.step --mode primitive
"""

import os
import argparse
import math
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

try:
    from OCC.Core.gp import (
        gp_Pnt, gp_Vec, gp_Dir, gp_Ax1, gp_Ax2, gp_Trsf, gp_GTrsf, gp_XYZ, gp_Mat,
    )
    from OCC.Core.BRepPrimAPI import (
        BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakeSphere,
        BRepPrimAPI_MakeCone,
    )
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform, BRepBuilderAPI_GTransform
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse
    from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Compound
    from OCC.Core.BRep import BRep_Builder
    from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
    from OCC.Core.STEPCAFControl import STEPCAFControl_Writer
    from OCC.Core.XCAFDoc import XCAFDoc_DocumentTool, XCAFDoc_ColorSurf
    from OCC.Core.TDocStd import TDocStd_Document
    from OCC.Core.TCollection import TCollection_ExtendedString
    from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.Interface import Interface_Static
    HAS_OCC = True
except ImportError:
    HAS_OCC = False


def _check_occ():
    """Check that pythonOCC is available."""
    if not HAS_OCC:
        raise ImportError(
            "pythonocc-core is required for BREP generation.\n"
            "Install with: conda install -c conda-forge pythonocc-core"
        )


class BRepShapeBuilder:
    """Converts bbox/primitive metadata to TopoDS_Shape using pythonOCC BREP API."""

    def _make_transform(self, center: np.ndarray, rotation: np.ndarray) -> gp_Trsf:
        """Create gp_Trsf from a 3x3 rotation matrix and translation vector.

        Args:
            center: 3D translation vector
            rotation: 3x3 rotation matrix (columns are local axes)

        Returns:
            gp_Trsf with rotation and translation applied
        """
        trsf = gp_Trsf()
        # Build the rotation part as a 3x3 gp_Mat (row-major)
        # rotation columns are the local axes, so rotation matrix R has:
        #   R[:,0] = local X, R[:,1] = local Y, R[:,2] = local Z
        # gp_Mat takes (row, col) values
        mat = gp_Mat(
            rotation[0, 0], rotation[0, 1], rotation[0, 2],
            rotation[1, 0], rotation[1, 1], rotation[1, 2],
            rotation[2, 0], rotation[2, 1], rotation[2, 2],
        )
        trsf.SetValues(
            mat.Value(1, 1), mat.Value(1, 2), mat.Value(1, 3), center[0],
            mat.Value(2, 1), mat.Value(2, 2), mat.Value(2, 3), center[1],
            mat.Value(3, 1), mat.Value(3, 2), mat.Value(3, 3), center[2],
        )
        return trsf

    def make_oriented_box(self, center: np.ndarray, dims: np.ndarray,
                          rotation: np.ndarray) -> TopoDS_Shape:
        """Create an oriented box from center, dimensions, and rotation.

        Args:
            center: Box center [x, y, z]
            dims: Box dimensions [dx, dy, dz]
            rotation: 3x3 rotation matrix

        Returns:
            Transformed TopoDS_Shape
        """
        _check_occ()
        dx, dy, dz = float(dims[0]), float(dims[1]), float(dims[2])

        # Create axis-aligned box centered at origin
        box = BRepPrimAPI_MakeBox(
            gp_Pnt(-dx / 2, -dy / 2, -dz / 2),
            dx, dy, dz,
        ).Shape()

        # Apply rotation + translation
        trsf = self._make_transform(center, rotation)
        return BRepBuilderAPI_Transform(box, trsf, True).Shape()

    def make_cylinder(self, center: np.ndarray, radius: float, height: float,
                      rotation: np.ndarray) -> TopoDS_Shape:
        """Create an oriented cylinder.

        The cylinder's axis is along local Z. It is centered at `center`.

        Args:
            center: Cylinder center
            radius: Cylinder radius
            height: Cylinder height
            rotation: 3x3 rotation matrix
        """
        _check_occ()
        r, h = float(radius), float(height)

        # Create cylinder centered at origin along Z
        cyl = BRepPrimAPI_MakeCylinder(r, h).Shape()

        # Shift so center is at origin (cylinder starts at z=0)
        shift = gp_Trsf()
        shift.SetTranslation(gp_Vec(0, 0, -h / 2))
        cyl = BRepBuilderAPI_Transform(cyl, shift, True).Shape()

        # Apply rotation + translation
        trsf = self._make_transform(center, rotation)
        return BRepBuilderAPI_Transform(cyl, trsf, True).Shape()

    def make_sphere(self, center: np.ndarray, radius: float) -> TopoDS_Shape:
        """Create a sphere at the given center."""
        _check_occ()
        return BRepPrimAPI_MakeSphere(gp_Pnt(float(center[0]), float(center[1]), float(center[2])),
                                      float(radius)).Shape()

    def make_cone(self, center: np.ndarray, base_radius: float, height: float,
                  rotation: np.ndarray, top_radius: float = 0.0) -> TopoDS_Shape:
        """Create an oriented cone.

        Args:
            center: Cone center (midpoint of axis)
            base_radius: Base radius
            height: Cone height
            rotation: 3x3 rotation matrix
            top_radius: Top radius (0 for pointed cone)
        """
        _check_occ()
        r1 = float(base_radius)
        r2 = float(top_radius)
        h = float(height)

        if r2 < 1e-10:
            r2 = 0.0
        # BRepPrimAPI_MakeCone(R1, R2, H) with R2=0 makes a pointed cone
        # R2 must be >= 0
        cone = BRepPrimAPI_MakeCone(r1, r2, h).Shape()

        # Center it (cone starts at z=0)
        shift = gp_Trsf()
        shift.SetTranslation(gp_Vec(0, 0, -h / 2))
        cone = BRepBuilderAPI_Transform(cone, shift, True).Shape()

        trsf = self._make_transform(center, rotation)
        return BRepBuilderAPI_Transform(cone, trsf, True).Shape()

    def make_capsule(self, center: np.ndarray, radius: float, height: float,
                     rotation: np.ndarray) -> TopoDS_Shape:
        """Create an oriented capsule (cylinder + 2 hemisphere caps).

        The total length is height (cylinder body) + 2*radius (caps).

        Args:
            center: Capsule center
            radius: Cap/cylinder radius
            height: Cylinder body height (excluding caps)
            rotation: 3x3 rotation matrix
        """
        _check_occ()
        r = float(radius)
        h = float(max(height, 0.001))

        # Cylinder body centered at origin
        cyl = BRepPrimAPI_MakeCylinder(r, h).Shape()
        shift_cyl = gp_Trsf()
        shift_cyl.SetTranslation(gp_Vec(0, 0, -h / 2))
        cyl = BRepBuilderAPI_Transform(cyl, shift_cyl, True).Shape()

        # Top hemisphere (at z = +h/2)
        top_sphere = BRepPrimAPI_MakeSphere(
            gp_Pnt(0, 0, h / 2), r,
            0.0, math.pi / 2,  # upper hemisphere: angle1=0, angle2=pi/2
        ).Shape()

        # Bottom hemisphere (at z = -h/2)
        bot_sphere = BRepPrimAPI_MakeSphere(
            gp_Pnt(0, 0, -h / 2), r,
            -math.pi / 2, 0.0,  # lower hemisphere
        ).Shape()

        # Fuse into compound
        builder = BRep_Builder()
        compound = TopoDS_Compound()
        builder.MakeCompound(compound)
        builder.Add(compound, cyl)
        builder.Add(compound, top_sphere)
        builder.Add(compound, bot_sphere)

        trsf = self._make_transform(center, rotation)
        return BRepBuilderAPI_Transform(compound, trsf, True).Shape()

    def make_ellipsoid(self, center: np.ndarray, radii: np.ndarray,
                       rotation: np.ndarray) -> TopoDS_Shape:
        """Create an oriented ellipsoid using non-uniform scaling of a sphere.

        Args:
            center: Ellipsoid center
            radii: [rx, ry, rz] radii along local axes
            rotation: 3x3 rotation matrix
        """
        _check_occ()
        rx, ry, rz = float(radii[0]), float(radii[1]), float(radii[2])

        # Create unit sphere at origin
        sphere = BRepPrimAPI_MakeSphere(1.0).Shape()

        # Non-uniform scale via gp_GTrsf
        gtrsf = gp_GTrsf()
        mat = gp_Mat(
            rx, 0, 0,
            0, ry, 0,
            0, 0, rz,
        )
        gtrsf.SetVectorialPart(mat)
        scaled = BRepBuilderAPI_GTransform(sphere, gtrsf, True).Shape()

        # Apply rotation + translation
        trsf = self._make_transform(center, rotation)
        return BRepBuilderAPI_Transform(scaled, trsf, True).Shape()

    def make_hemisphere(self, center: np.ndarray, radius: float,
                        rotation: np.ndarray) -> TopoDS_Shape:
        """Create an oriented hemisphere (upper half of sphere along local Z).

        Args:
            center: Hemisphere center (base center)
            radius: Sphere radius
            rotation: 3x3 rotation matrix
        """
        _check_occ()
        r = float(radius)
        # Make hemisphere: angle limits 0 to pi/2 (upper half along Z)
        hemi = BRepPrimAPI_MakeSphere(r, 0.0, math.pi / 2).Shape()

        trsf = self._make_transform(center, rotation)
        return BRepBuilderAPI_Transform(hemi, trsf, True).Shape()

    def shape_from_bbox(self, bbox: dict) -> TopoDS_Shape:
        """Convert a bbox dict (from OBBFitter) to a TopoDS_Shape.

        Args:
            bbox: dict with 'center', 'dimensions', 'rotation' keys
        """
        return self.make_oriented_box(
            center=np.asarray(bbox['center']),
            dims=np.asarray(bbox['dimensions']),
            rotation=np.asarray(bbox['rotation']),
        )

    def shape_from_primitive(self, primitive) -> TopoDS_Shape:
        """Convert a Primitive dataclass to a TopoDS_Shape.

        Dispatches to the correct make_* method based on PrimitiveType.

        Args:
            primitive: Primitive instance from segment_with_primitives.py
        """
        from segment_with_primitives import PrimitiveType

        ptype = primitive.type
        center = np.asarray(primitive.center)
        dims = np.asarray(primitive.dimensions)
        rotation = np.asarray(primitive.rotation)
        params = primitive.params

        if ptype in (PrimitiveType.BOX, PrimitiveType.ORIENTED_BBOX):
            return self.make_oriented_box(center, dims, rotation)

        elif ptype == PrimitiveType.CYLINDER:
            radius = params.get('radius', max(dims[0], dims[1]) / 2)
            height = params.get('height', dims[2])
            return self.make_cylinder(center, float(radius), float(height), rotation)

        elif ptype == PrimitiveType.SPHERE:
            radius = params.get('radius', np.mean(dims) / 2)
            return self.make_sphere(center, float(radius))

        elif ptype == PrimitiveType.ELLIPSOID:
            radii = np.array([dims[0] / 2, dims[1] / 2, dims[2] / 2])
            return self.make_ellipsoid(center, radii, rotation)

        elif ptype == PrimitiveType.HEMISPHERE:
            radius = params.get('radius', np.mean(dims[:2]) / 2)
            return self.make_hemisphere(center, float(radius), rotation)

        elif ptype == PrimitiveType.CONE:
            base_radius = params.get('base_radius', max(dims[0], dims[1]) / 2)
            height = params.get('height', dims[2])
            return self.make_cone(center, float(base_radius), float(height), rotation)

        elif ptype == PrimitiveType.CAPSULE:
            radius = params.get('radius', max(dims[0], dims[1]) / 2)
            height = params.get('height', dims[2])
            return self.make_capsule(center, float(radius), float(height), rotation)

        elif ptype in (PrimitiveType.QUARTER_SPHERE, PrimitiveType.EIGHTH_SPHERE):
            # Approximate as sphere (BREP quarter/eighth sphere is complex)
            radius = params.get('radius', np.mean(dims) / 2)
            return self.make_sphere(center, float(radius))

        elif ptype == PrimitiveType.TRIANGULAR_PRISM:
            # Approximate as oriented box
            return self.make_oriented_box(center, dims, rotation)

        elif ptype in (PrimitiveType.TETRAHEDRON, PrimitiveType.OCTAHEDRON):
            # Approximate as sphere bounding the solid
            radius = np.max(dims) / 2
            return self.make_sphere(center, float(radius))

        else:
            # Fallback: oriented box
            return self.make_oriented_box(center, dims, rotation)


class BRepExporter:
    """Export TopoDS_Shapes to STEP files, optionally with per-shape colors."""

    def export_step(self, shapes: List[TopoDS_Shape], output_path: str) -> bool:
        """Export shapes to a plain STEP file (no colors).

        Args:
            shapes: List of TopoDS_Shape objects
            output_path: Output .step file path

        Returns:
            True if successful
        """
        _check_occ()
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

        writer = STEPControl_Writer()
        Interface_Static.SetCVal("write.step.schema", "AP214")

        for shape in shapes:
            writer.Transfer(shape, STEPControl_AsIs)

        status = writer.Write(output_path)
        if status == IFSelect_RetDone:
            print(f"STEP file saved: {output_path}")
            return True
        else:
            print(f"Failed to write STEP file: {output_path}")
            return False

    def export_colored_step(self, shapes_and_colors: List[Tuple[TopoDS_Shape, Tuple[float, float, float]]],
                            output_path: str) -> bool:
        """Export shapes with per-shape colors to a STEP file using XCAF.

        Pattern from BrepMFR/brepformer/export_freecad.py.

        Args:
            shapes_and_colors: List of (shape, (r, g, b)) tuples where r,g,b in [0,1]
            output_path: Output .step file path

        Returns:
            True if successful
        """
        _check_occ()
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

        # Create XCAF document (simple string ctor works with pythonocc 7.x)
        doc = TDocStd_Document("pythonocc-doc")
        shape_tool = XCAFDoc_DocumentTool.ShapeTool(doc.Main())
        color_tool = XCAFDoc_DocumentTool.ColorTool(doc.Main())

        for shape, (r, g, b) in shapes_and_colors:
            label = shape_tool.AddShape(shape, False)
            color = Quantity_Color(float(r), float(g), float(b), Quantity_TOC_RGB)
            color_tool.SetColor(label, color, XCAFDoc_ColorSurf)

        # Write colored STEP
        writer = STEPCAFControl_Writer()
        writer.SetColorMode(True)
        writer.SetNameMode(True)
        writer.Transfer(doc)

        status = writer.Write(output_path)
        if status == IFSelect_RetDone:
            print(f"Colored STEP file saved: {output_path}")
            return True
        else:
            print(f"Failed to write colored STEP file: {output_path}")
            return False


class BRepFromSegmentation:
    """Pipeline orchestration: loads mesh + labels, runs fitters, converts to BREP, exports STEP."""

    def __init__(self):
        self.builder = BRepShapeBuilder()
        self.exporter = BRepExporter()

    def _get_tab20_color(self, index: int, total: int) -> Tuple[float, float, float]:
        """Get a color from matplotlib's tab20 colormap."""
        import matplotlib.pyplot as plt
        cmap = plt.colormaps.get_cmap("tab20").resampled(max(total, 1))
        rgba = cmap(index % 20)
        return (rgba[0], rgba[1], rgba[2])

    def process_bboxes(self, mesh_path: str, labels_path: str, output_path: str,
                       alignment: str = 'self',
                       apply_filtering: bool = True,
                       resolve_overlaps: bool = True) -> bool:
        """Process mesh + labels into BREP bounding boxes and export STEP.

        Reuses OBBFitter/OBBFitterGlobal from existing scripts.

        Args:
            mesh_path: Path to input mesh (GLB, PLY, OBJ, etc.)
            labels_path: Path to segmentation labels (.npy)
            output_path: Output STEP file path
            alignment: 'self' for per-segment PCA, 'global' for shared global PCA
            apply_filtering: Enable outlier filtering
            resolve_overlaps: Enable overlap resolution

        Returns:
            True if successful
        """
        _check_occ()

        if alignment == 'global':
            from segment_with_bboxes_global import SegmentWithBBoxesGlobal
            processor = SegmentWithBBoxesGlobal(
                apply_filtering=apply_filtering,
                resolve_overlaps=resolve_overlaps,
            )
        else:
            from segment_with_bboxes import SegmentWithBBoxes
            processor = SegmentWithBBoxes(
                apply_filtering=apply_filtering,
                resolve_overlaps=resolve_overlaps,
            )

        print(f"Loading mesh: {mesh_path}")
        mesh = processor.load_mesh(mesh_path)
        if mesh is None:
            print(f"Error: Failed to load mesh: {mesh_path}")
            return False

        print(f"Loading labels: {labels_path}")
        labels = processor.load_labels(labels_path)
        if labels is None:
            print(f"Error: Failed to load labels: {labels_path}")
            return False

        # Auto-detect label type
        is_face_labels = True
        if len(labels) == len(mesh.vertices):
            is_face_labels = False
        elif len(labels) == len(mesh.faces):
            is_face_labels = True

        print(f"Mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        print(f"Labels: {len(labels)} ({'face' if is_face_labels else 'vertex'})")

        # Run the fitter pipeline
        bboxes, segment_to_bbox = processor.process(mesh, labels, is_face_labels)

        if not bboxes:
            print("Error: No bounding boxes generated")
            return False

        # Convert each bbox to BREP shape with color
        n_segments = len(segment_to_bbox)
        shapes_and_colors = []

        for i, (segment_id, bbox_idx) in enumerate(segment_to_bbox.items()):
            bbox = bboxes[bbox_idx]
            color = self._get_tab20_color(i, n_segments)

            try:
                shape = self.builder.shape_from_bbox(bbox)
                shapes_and_colors.append((shape, color))
            except Exception as e:
                print(f"  Warning: Failed to create BREP for segment {segment_id}: {e}")

        if not shapes_and_colors:
            print("Error: No BREP shapes created")
            return False

        print(f"\nGenerated {len(shapes_and_colors)} BREP shapes")

        # Export colored STEP
        return self.exporter.export_colored_step(shapes_and_colors, output_path)

    def process_primitives(self, mesh_path: str, labels_path: str, output_path: str,
                           threshold: float = 0.35,
                           apply_filtering: bool = True) -> bool:
        """Process mesh + labels into BREP primitives and export STEP.

        Reuses PrimitiveFitter from segment_with_primitives.py.

        Args:
            mesh_path: Path to input mesh
            labels_path: Path to segmentation labels (.npy)
            output_path: Output STEP file path
            threshold: Primitive fit score threshold
            apply_filtering: Enable point filtering

        Returns:
            True if successful
        """
        _check_occ()

        from segment_with_primitives import PrimitiveFitter, SegmentWithPrimitives

        processor = SegmentWithPrimitives(threshold=threshold)

        print(f"Loading mesh: {mesh_path}")
        mesh = processor.load_mesh(mesh_path)
        if mesh is None:
            print(f"Error: Failed to load mesh: {mesh_path}")
            return False

        print(f"Loading labels: {labels_path}")
        labels = processor.load_labels(labels_path)
        if labels is None:
            print(f"Error: Failed to load labels: {labels_path}")
            return False

        # Auto-detect label type
        is_face_labels = True
        if len(labels) == len(mesh.vertices):
            is_face_labels = False
        elif len(labels) == len(mesh.faces):
            is_face_labels = True

        print(f"Mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        print(f"Labels: {len(labels)} ({'face' if is_face_labels else 'vertex'})")

        # Run the primitive fitter pipeline
        primitives, segment_to_prim = processor.process(mesh, labels, is_face_labels)

        if not primitives:
            print("Error: No primitives generated")
            return False

        # Convert each primitive to BREP shape with color
        n_segments = len(segment_to_prim)
        shapes_and_colors = []

        for i, (segment_id, prim_idx) in enumerate(segment_to_prim.items()):
            prim = primitives[prim_idx]
            color = self._get_tab20_color(i, n_segments)

            try:
                shape = self.builder.shape_from_primitive(prim)
                shapes_and_colors.append((shape, color))
                print(f"  Segment {segment_id}: {prim.type.value} -> BREP")
            except Exception as e:
                print(f"  Warning: Failed to create BREP for segment {segment_id} ({prim.type.value}): {e}")

        if not shapes_and_colors:
            print("Error: No BREP shapes created")
            return False

        print(f"\nGenerated {len(shapes_and_colors)} BREP primitives")

        # Export colored STEP
        return self.exporter.export_colored_step(shapes_and_colors, output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Generate BREP CAD files (STEP) from PartField segmentation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Bounding boxes with self-alignment
  python brep_generator.py -i model.glb -l labels.npy -o result.step --mode bbox --alignment self

  # Bounding boxes with global alignment
  python brep_generator.py -i model.glb -l labels.npy -o result.step --mode bbox --alignment global

  # Primitives
  python brep_generator.py -i model.glb -l labels.npy -o result.step --mode primitive

  # Disable filtering
  python brep_generator.py -i model.glb -l labels.npy -o result.step --mode bbox --no-filter
        """
    )

    parser.add_argument("--input", "-i", required=True, help="Input mesh file (GLB, PLY, OBJ, etc.)")
    parser.add_argument("--labels", "-l", required=True, help="Segmentation labels (.npy)")
    parser.add_argument("--output", "-o", required=True, help="Output STEP file path")
    parser.add_argument("--mode", "-m", choices=['bbox', 'primitive'], default='bbox',
                        help="Generation mode: 'bbox' or 'primitive' (default: bbox)")
    parser.add_argument("--alignment", "-a", choices=['self', 'global'], default='self',
                        help="BBox alignment: 'self' (per-segment PCA) or 'global' (shared PCA). "
                             "Only used with --mode bbox (default: self)")
    parser.add_argument("--threshold", "-t", type=float, default=0.35,
                        help="Primitive fit threshold (default: 0.35). Only used with --mode primitive")
    parser.add_argument("--no-filter", action="store_true",
                        help="Disable outlier/density filtering")
    parser.add_argument("--no-overlap-resolution", action="store_true",
                        help="Disable automatic overlap resolution (bbox mode only)")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return 1
    if not os.path.exists(args.labels):
        print(f"Error: Labels file not found: {args.labels}")
        return 1

    _check_occ()

    pipeline = BRepFromSegmentation()

    apply_filtering = not args.no_filter
    resolve_overlaps = not args.no_overlap_resolution

    print(f"Mode: {args.mode}, Alignment: {args.alignment}")
    print(f"Options: filtering={'ON' if apply_filtering else 'OFF'}, "
          f"overlap_resolution={'ON' if resolve_overlaps else 'OFF'}")

    if args.mode == 'bbox':
        success = pipeline.process_bboxes(
            mesh_path=args.input,
            labels_path=args.labels,
            output_path=args.output,
            alignment=args.alignment,
            apply_filtering=apply_filtering,
            resolve_overlaps=resolve_overlaps,
        )
    else:
        success = pipeline.process_primitives(
            mesh_path=args.input,
            labels_path=args.labels,
            output_path=args.output,
            threshold=args.threshold,
            apply_filtering=apply_filtering,
        )

    if success:
        print(f"\nDone! Output: {args.output}")
        return 0
    else:
        print("\nFailed to generate BREP output")
        return 1


if __name__ == "__main__":
    exit(main())
