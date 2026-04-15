#!/usr/bin/env python3
"""
OBJ+MTL to GLB Converter

Converts OBJ files (with optional MTL materials) to GLB format for use with
PartField segmenter and viewer.

Usage:
    python obj2glb.py input.obj output.glb
    python obj2glb.py --batch data/mtlfiles/ data/glb_output/
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, List
import numpy as np

try:
    import trimesh
except ImportError:
    print("Error: trimesh is required. Install with: pip install trimesh")
    sys.exit(1)


def convert_obj_to_glb(input_path: str, output_path: str, verbose: bool = True) -> bool:
    """
    Convert an OBJ file (with MTL if present) to GLB format.

    Args:
        input_path: Path to the input OBJ file
        output_path: Path for the output GLB file
        verbose: Whether to print progress messages

    Returns:
        True if conversion succeeded, False otherwise
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return False

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if verbose:
            print(f"Loading: {input_path}")

        # Load with trimesh - it automatically finds MTL files in the same directory
        mesh_or_scene = trimesh.load(
            str(input_path),
            force=None,  # Let trimesh decide if it's a scene or mesh
            process=False  # Don't process to preserve original structure
        )

        # Handle scene vs single mesh
        if isinstance(mesh_or_scene, trimesh.Scene):
            # Get all meshes from scene
            meshes = []
            for name, geom in mesh_or_scene.geometry.items():
                if isinstance(geom, trimesh.Trimesh):
                    meshes.append(geom)
                    if verbose:
                        print(f"  - Object: {name} ({len(geom.vertices)} vertices, {len(geom.faces)} faces)")

            if not meshes:
                print(f"Error: No mesh geometry found in {input_path}")
                return False

            # Combine all meshes
            if len(meshes) == 1:
                combined = meshes[0]
            else:
                combined = trimesh.util.concatenate(meshes)

            if verbose:
                print(f"  Combined: {len(combined.vertices)} vertices, {len(combined.faces)} faces")
        else:
            combined = mesh_or_scene
            if verbose:
                print(f"  Mesh: {len(combined.vertices)} vertices, {len(combined.faces)} faces")

        # Ensure the mesh has valid vertex colors (helps with visualization)
        if not hasattr(combined.visual, 'vertex_colors') or combined.visual.vertex_colors is None:
            # Check for face colors
            if hasattr(combined.visual, 'face_colors') and combined.visual.face_colors is not None:
                if verbose:
                    print("  Converting face colors to vertex colors...")
            else:
                # Set default gray color if no colors present
                if verbose:
                    print("  No colors found, setting default gray")
                combined.visual.vertex_colors = np.ones((len(combined.vertices), 4), dtype=np.uint8) * 180
                combined.visual.vertex_colors[:, 3] = 255  # Full opacity

        # Export to GLB
        if verbose:
            print(f"Exporting to: {output_path}")

        combined.export(str(output_path), file_type='glb')

        if verbose:
            file_size = output_path.stat().st_size
            print(f"  Done! ({file_size / 1024:.1f} KB)")

        return True

    except Exception as e:
        print(f"Error converting {input_path}: {e}")
        import traceback
        traceback.print_exc()
        return False


def find_obj_files(directory: str, recursive: bool = True) -> List[Path]:
    """Find all OBJ files in a directory."""
    directory = Path(directory)

    if recursive:
        pattern = "**/*.obj"
    else:
        pattern = "*.obj"

    return sorted(directory.glob(pattern))


def batch_convert(input_dir: str, output_dir: str,
                  recursive: bool = True, verbose: bool = True) -> tuple:
    """
    Batch convert all OBJ files in a directory to GLB.

    Args:
        input_dir: Directory containing OBJ files
        output_dir: Directory for output GLB files
        recursive: Whether to search recursively
        verbose: Whether to print progress

    Returns:
        Tuple of (successful_count, failed_count, failed_files)
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    obj_files = find_obj_files(input_dir, recursive)

    if not obj_files:
        print(f"No OBJ files found in {input_dir}")
        return 0, 0, []

    print(f"Found {len(obj_files)} OBJ files to convert")
    print("=" * 60)

    success_count = 0
    fail_count = 0
    failed_files = []

    for i, obj_path in enumerate(obj_files, 1):
        # Compute relative path for output
        try:
            rel_path = obj_path.relative_to(input_dir)
        except ValueError:
            rel_path = obj_path.name

        # Change extension to .glb
        glb_name = rel_path.with_suffix('.glb')
        output_path = output_dir / glb_name

        print(f"\n[{i}/{len(obj_files)}] Converting: {obj_path.name}")

        if convert_obj_to_glb(str(obj_path), str(output_path), verbose=verbose):
            success_count += 1
        else:
            fail_count += 1
            failed_files.append(str(obj_path))

    print("\n" + "=" * 60)
    print(f"Conversion complete: {success_count} succeeded, {fail_count} failed")

    if failed_files:
        print("\nFailed files:")
        for f in failed_files:
            print(f"  - {f}")

    return success_count, fail_count, failed_files


def main():
    parser = argparse.ArgumentParser(
        description="Convert OBJ+MTL files to GLB format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert single file
  python obj2glb.py model.obj model.glb

  # Batch convert directory
  python obj2glb.py --batch data/mtlfiles/ data/glb_output/

  # Batch convert with flat output (no subdirectories)
  python obj2glb.py --batch --no-recursive data/obj_files/ data/glb_output/
"""
    )

    parser.add_argument(
        "input",
        help="Input OBJ file or directory (with --batch)"
    )
    parser.add_argument(
        "output",
        help="Output GLB file or directory (with --batch)"
    )
    parser.add_argument(
        "--batch", "-b",
        action="store_true",
        help="Batch mode: convert all OBJ files in input directory"
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="In batch mode, don't search subdirectories"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output"
    )

    args = parser.parse_args()

    if args.batch:
        # Batch mode
        if not os.path.isdir(args.input):
            print(f"Error: In batch mode, input must be a directory: {args.input}")
            sys.exit(1)

        success, fail, _ = batch_convert(
            args.input,
            args.output,
            recursive=not args.no_recursive,
            verbose=not args.quiet
        )

        sys.exit(0 if fail == 0 else 1)
    else:
        # Single file mode
        if not args.input.lower().endswith('.obj'):
            print(f"Warning: Input file doesn't have .obj extension: {args.input}")

        # Auto-add .glb extension if not present
        output = args.output
        if not output.lower().endswith('.glb'):
            output += '.glb'

        success = convert_obj_to_glb(
            args.input,
            output,
            verbose=not args.quiet
        )

        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
