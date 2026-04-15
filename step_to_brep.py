#!/usr/bin/env python3
"""
STEP → BREP BBox Pipeline

End-to-end script that takes a STEP CAD file and produces a new STEP file
where the original geometry is replaced with colored oriented bounding boxes
(one per detected part segment).

Pipeline:
    1. Read STEP file with pythonocc
    2. Tessellate to a temporary PLY mesh
    3. Run PartField feature extraction (GPU)
    4. Run part clustering
    5. Fit oriented bounding boxes to each segment
    6. Convert bboxes to BREP solids
    7. Export colored STEP file

Usage:
    # Basic (auto-selects best cluster count)
    python step_to_brep.py -i model.step -o model_bboxes.step

    # Fixed number of clusters
    python step_to_brep.py -i model.step -o model_bboxes.step --clusters 5

    # Global alignment (all boxes share same orientation)
    python step_to_brep.py -i model.step -o model_bboxes.step --alignment global

    # With primitives instead of boxes
    python step_to_brep.py -i model.step -o model_bboxes.step --mode primitive

    # Batch: process all STEP files in a directory
    python step_to_brep.py --input-dir steps/ --output-dir brep_out/

    # Keep intermediate files for debugging
    python step_to_brep.py -i model.step -o model_bboxes.step --keep-intermediates
"""

import os
import sys
import argparse
import tempfile
import shutil
import subprocess
import glob
import numpy as np
from pathlib import Path


def step_to_mesh(step_path: str, output_path: str,
                 linear_deflection: float = 0.1,
                 angular_deflection: float = 0.5) -> bool:
    """Convert a STEP file to a triangulated mesh (STL/PLY).

    Uses pythonocc to read the STEP and tessellate it.

    Args:
        step_path: Input STEP file
        output_path: Output mesh file (.stl or .ply)
        linear_deflection: Tessellation linear deflection (smaller = finer mesh)
        angular_deflection: Tessellation angular deflection in radians

    Returns:
        True if successful
    """
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.StlAPI import StlAPI_Writer

    reader = STEPControl_Reader()
    status = reader.ReadFile(step_path)
    if status != IFSelect_RetDone:
        print(f"Error: Failed to read STEP file: {step_path}")
        return False

    reader.TransferRoots()
    shape = reader.OneShape()

    # Tessellate
    mesh = BRepMesh_IncrementalMesh(shape, linear_deflection, False, angular_deflection)
    mesh.Perform()

    if not mesh.IsDone():
        print(f"Error: Tessellation failed for: {step_path}")
        return False

    ext = Path(output_path).suffix.lower()

    if ext == '.stl':
        writer = StlAPI_Writer()
        writer.Write(shape, output_path)
    else:
        # Use OCC's STL writer to a temp file, then convert with trimesh
        import trimesh
        tmp_stl = output_path + '.tmp.stl'
        writer = StlAPI_Writer()
        writer.Write(shape, tmp_stl)
        try:
            tm = trimesh.load(tmp_stl)
            tm.export(output_path)
        finally:
            if os.path.exists(tmp_stl):
                os.unlink(tmp_stl)

    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return True
    else:
        print(f"Error: Output mesh is empty: {output_path}")
        return False


def run_partfield_inference(data_dir: str, features_name: str, exp_dir: str,
                            ckpt_path: str = "model/model_objaverse.ckpt") -> bool:
    """Run PartField feature extraction.

    Args:
        data_dir: Directory containing input mesh files
        features_name: Feature output subdirectory name
        exp_dir: Experiment results root directory
        ckpt_path: Path to PartField checkpoint

    Returns:
        True if successful
    """
    cmd = [
        sys.executable, "partfield_inference.py",
        "--config-file", "configs/final/demo.yaml",
        "--opts",
        f"dataset.data_path", data_dir,
        f"output_dir", exp_dir,
        f"result_name", f"partfield_features/{features_name}",
        f"continue_ckpt", ckpt_path,
    ]

    print(f"  Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def run_clustering(data_dir: str, features_dir: str, clustering_dir: str,
                   max_clusters: int = 20, auto_select: bool = True,
                   fixed_clusters: int = None) -> bool:
    """Run part clustering on extracted features.

    Args:
        data_dir: Original input data directory
        features_dir: PartField features directory
        clustering_dir: Output directory for clustering results
        max_clusters: Maximum number of clusters to try
        auto_select: Use metric-based auto selection
        fixed_clusters: If set, use exactly this many clusters

    Returns:
        True if successful
    """
    cmd = [
        sys.executable, "run_part_clustering.py",
        "--source_dir", data_dir,
        "--root", features_dir,
        "--dump_dir", clustering_dir,
        "--max_num_clusters", str(max_clusters),
        "--use_agglo", "False",
        "--export_mesh", "True",
    ]

    if auto_select and fixed_clusters is None:
        cmd.extend(["--auto_select",
                     "--min_preferred_clusters", "3",
                     "--max_preferred_clusters", "15"])

    print(f"  Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def find_best_labels(clustering_dir: str, model_id: str,
                     fixed_clusters: int = None) -> tuple:
    """Find the best clustering labels file for a model.

    Args:
        clustering_dir: Clustering results directory
        model_id: Model identifier
        fixed_clusters: If set, use this exact cluster count

    Returns:
        (labels_path, n_clusters) or (None, 0)
    """
    cluster_out = os.path.join(clustering_dir, "cluster_out")

    if fixed_clusters is not None:
        label_file = os.path.join(cluster_out, f"{model_id}_0_{fixed_clusters:02d}.npy")
        if os.path.exists(label_file):
            return label_file, fixed_clusters
        # Try without leading zero
        label_file = os.path.join(cluster_out, f"{model_id}_0_{fixed_clusters}.npy")
        if os.path.exists(label_file):
            return label_file, fixed_clusters
        return None, 0

    # Check for auto-selected best
    best_n_file = os.path.join(cluster_out, f"{model_id}_0_best_n.npy")
    if os.path.exists(best_n_file):
        best_n = int(np.load(best_n_file)[0])
        label_file = os.path.join(cluster_out, f"{model_id}_0_{best_n:02d}.npy")
        if os.path.exists(label_file):
            return label_file, best_n

    # Fallback: find any label file and pick the one with most clusters
    pattern = os.path.join(cluster_out, f"{model_id}_0_*.npy")
    files = sorted(glob.glob(pattern))
    # Exclude best_n files
    files = [f for f in files if 'best_n' not in f]

    if not files:
        return None, 0

    # Pick the file with the highest cluster count
    best_file = files[-1]
    # Extract cluster count from filename
    try:
        n = int(Path(best_file).stem.split('_')[-1])
    except ValueError:
        n = 0

    return best_file, n


def generate_brep_bboxes(mesh_path: str, labels_path: str, output_path: str,
                         alignment: str = 'self', mode: str = 'bbox',
                         threshold: float = 0.35) -> bool:
    """Generate BREP bounding boxes from mesh + labels.

    Args:
        mesh_path: Input mesh file
        labels_path: Clustering labels (.npy)
        output_path: Output STEP file
        alignment: 'self' or 'global'
        mode: 'bbox' or 'primitive'
        threshold: Primitive fit threshold

    Returns:
        True if successful
    """
    from brep_generator import BRepFromSegmentation

    pipeline = BRepFromSegmentation()

    if mode == 'primitive':
        return pipeline.process_primitives(mesh_path, labels_path, output_path,
                                           threshold=threshold)
    else:
        return pipeline.process_bboxes(mesh_path, labels_path, output_path,
                                       alignment=alignment)


def process_single(step_input: str, step_output: str,
                   alignment: str = 'self',
                   mode: str = 'bbox',
                   fixed_clusters: int = None,
                   max_clusters: int = 20,
                   threshold: float = 0.35,
                   tessellation_quality: float = 0.1,
                   keep_intermediates: bool = False,
                   ckpt_path: str = "model/model_objaverse.ckpt") -> bool:
    """Process a single STEP file end-to-end.

    Args:
        step_input: Input STEP file path
        step_output: Output STEP file path (bounding boxes)
        alignment: 'self' or 'global' bbox alignment
        mode: 'bbox' or 'primitive'
        fixed_clusters: Exact number of clusters (None = auto)
        max_clusters: Max clusters to try when auto-selecting
        threshold: Primitive fit threshold
        tessellation_quality: Linear deflection for tessellation (smaller = finer)
        keep_intermediates: Keep intermediate files for debugging
        ckpt_path: Path to PartField model checkpoint

    Returns:
        True if successful
    """
    step_name = Path(step_input).stem
    print(f"\n{'='*60}")
    print(f"Processing: {step_name}")
    print(f"  Input:  {step_input}")
    print(f"  Output: {step_output}")
    print(f"  Mode: {mode}, Alignment: {alignment}")
    if fixed_clusters:
        print(f"  Clusters: {fixed_clusters} (fixed)")
    else:
        print(f"  Clusters: auto (2-{max_clusters})")
    print(f"{'='*60}")

    # Create working directory
    work_dir = tempfile.mkdtemp(prefix=f"step2brep_{step_name}_")
    data_dir = os.path.join(work_dir, "data")
    exp_dir = os.path.join(work_dir, "exp_results")
    features_name = "step_features"
    features_dir = os.path.join(exp_dir, "partfield_features", features_name)
    clustering_dir = os.path.join(exp_dir, "clustering", "step_clustering")

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(exp_dir, exist_ok=True)

    try:
        # ====================================================================
        # Step 1: STEP → mesh (tessellate)
        # ====================================================================
        print(f"\n[1/4] Tessellating STEP → mesh...")
        mesh_path = os.path.join(data_dir, f"{step_name}.obj")

        if not step_to_mesh(step_input, mesh_path,
                            linear_deflection=tessellation_quality):
            print("Error: STEP tessellation failed")
            return False

        mesh_size = os.path.getsize(mesh_path)
        print(f"  Tessellated mesh: {mesh_path} ({mesh_size:,} bytes)")

        # Verify mesh is valid
        import trimesh
        check = trimesh.load(mesh_path)
        print(f"  Mesh: {len(check.vertices)} vertices, {len(check.faces)} faces")
        if len(check.faces) < 10:
            print("Error: Tessellated mesh has too few faces. Try smaller --tessellation-quality value.")
            return False

        # ====================================================================
        # Step 2: PartField feature extraction
        # ====================================================================
        print(f"\n[2/4] Extracting PartField features...")

        if not run_partfield_inference(data_dir, features_name, exp_dir,
                                       ckpt_path=ckpt_path):
            print("Error: PartField inference failed")
            return False

        # PartField's config appends {name}/{datetime} to output_dir,
        # so features end up in exp_dir/{name}/{datetime}/partfield_features/{features_name}
        # Search for the actual features directory.
        if not os.path.exists(features_dir):
            found = glob.glob(os.path.join(exp_dir, "**", "partfield_features", features_name), recursive=True)
            if found:
                features_dir = found[0]
                print(f"  Found features at: {features_dir}")
            else:
                print(f"Error: Features directory not found: {features_dir}")
                return False

        feat_files = glob.glob(os.path.join(features_dir, "*.npy"))
        print(f"  Feature files: {len(feat_files)}")

        # ====================================================================
        # Step 3: Clustering
        # ====================================================================
        print(f"\n[3/4] Running part clustering...")

        actual_max = fixed_clusters if fixed_clusters else max_clusters
        if not run_clustering(data_dir, features_dir, clustering_dir,
                              max_clusters=actual_max,
                              auto_select=(fixed_clusters is None),
                              fixed_clusters=fixed_clusters):
            print("Error: Clustering failed")
            return False

        # Find the model_id used by PartField (derived from filename)
        # PartField strips the extension and may modify the name
        model_id = step_name
        labels_path, n_clusters = find_best_labels(clustering_dir, model_id,
                                                    fixed_clusters=fixed_clusters)

        # Try alternative model IDs if first attempt fails
        if labels_path is None:
            cluster_out = os.path.join(clustering_dir, "cluster_out")
            if os.path.exists(cluster_out):
                npy_files = glob.glob(os.path.join(cluster_out, "*.npy"))
                npy_files = [f for f in npy_files if 'best_n' not in f]
                if npy_files:
                    # Extract model_id from the first available file
                    first = Path(npy_files[0]).stem
                    # Pattern: {model_id}_0_{nn}
                    parts = first.rsplit('_', 2)
                    if len(parts) >= 3:
                        alt_model_id = parts[0]
                        labels_path, n_clusters = find_best_labels(
                            clustering_dir, alt_model_id,
                            fixed_clusters=fixed_clusters)

        if labels_path is None:
            print("Error: No clustering labels found")
            # List what's in cluster_out for debugging
            cluster_out = os.path.join(clustering_dir, "cluster_out")
            if os.path.exists(cluster_out):
                files = os.listdir(cluster_out)
                print(f"  Available files in cluster_out: {files[:10]}")
            return False

        print(f"  Using {n_clusters} clusters from: {Path(labels_path).name}")

        # ====================================================================
        # Step 4: Generate BREP bounding boxes
        # ====================================================================
        print(f"\n[4/4] Generating BREP {mode}s...")

        os.makedirs(os.path.dirname(step_output) or '.', exist_ok=True)

        if not generate_brep_bboxes(mesh_path, labels_path, step_output,
                                    alignment=alignment, mode=mode,
                                    threshold=threshold):
            print("Error: BREP generation failed")
            return False

        output_size = os.path.getsize(step_output)
        print(f"\n{'='*60}")
        print(f"SUCCESS: {step_output} ({output_size:,} bytes)")
        print(f"  {n_clusters} segments as BREP {mode}s")
        print(f"{'='*60}")
        return True

    finally:
        # Cleanup
        if keep_intermediates:
            print(f"\nIntermediate files kept at: {work_dir}")
        else:
            shutil.rmtree(work_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(
        description="STEP → BREP BBox Pipeline: Convert a STEP CAD file to a STEP file of colored bounding boxes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file (auto cluster selection)
  python step_to_brep.py -i model.step -o model_bboxes.step

  # Fixed 5 clusters
  python step_to_brep.py -i model.step -o model_bboxes.step --clusters 5

  # Global alignment
  python step_to_brep.py -i model.step -o model_bboxes.step --alignment global

  # Primitives instead of boxes
  python step_to_brep.py -i model.step -o model_bboxes.step --mode primitive

  # Finer tessellation for detailed models
  python step_to_brep.py -i model.step -o model_bboxes.step --tessellation-quality 0.01

  # Batch mode
  python step_to_brep.py --input-dir steps/ --output-dir brep_out/

  # Keep intermediate files (mesh, features, labels) for debugging
  python step_to_brep.py -i model.step -o model_bboxes.step --keep-intermediates
        """
    )

    # Input/output
    parser.add_argument("--input", "-i", help="Input STEP file")
    parser.add_argument("--output", "-o", help="Output STEP file (bounding boxes)")
    parser.add_argument("--input-dir", help="Input directory of STEP files (batch mode)")
    parser.add_argument("--output-dir", help="Output directory (batch mode)")

    # Segmentation options
    parser.add_argument("--clusters", type=int, default=None,
                        help="Fixed number of clusters (default: auto-select 2-20)")
    parser.add_argument("--max-clusters", type=int, default=20,
                        help="Maximum clusters for auto-selection (default: 20)")
    parser.add_argument("--alignment", "-a", choices=['self', 'global'], default='self',
                        help="BBox alignment: 'self' (per-segment) or 'global' (shared). Default: self")
    parser.add_argument("--mode", "-m", choices=['bbox', 'primitive'], default='bbox',
                        help="Output geometry: 'bbox' or 'primitive'. Default: bbox")
    parser.add_argument("--threshold", "-t", type=float, default=0.35,
                        help="Primitive fit threshold (default: 0.35, only for --mode primitive)")

    # Quality
    parser.add_argument("--tessellation-quality", type=float, default=0.1,
                        help="Tessellation linear deflection — smaller = finer mesh. Default: 0.1")

    # Other
    parser.add_argument("--checkpoint", default="model/model_objaverse.ckpt",
                        help="PartField model checkpoint path")
    parser.add_argument("--keep-intermediates", action="store_true",
                        help="Keep intermediate files (tessellated mesh, features, labels)")

    args = parser.parse_args()

    # Validate arguments
    if args.input and args.input_dir:
        print("Error: Specify either --input or --input-dir, not both")
        return 1

    if not args.input and not args.input_dir:
        parser.print_help()
        print("\nError: Specify --input FILE or --input-dir DIR")
        return 1

    # Single file mode
    if args.input:
        if not os.path.exists(args.input):
            print(f"Error: Input file not found: {args.input}")
            return 1

        output = args.output
        if not output:
            base = Path(args.input).stem
            output = f"{base}_bboxes.step"

        success = process_single(
            step_input=args.input,
            step_output=output,
            alignment=args.alignment,
            mode=args.mode,
            fixed_clusters=args.clusters,
            max_clusters=args.max_clusters,
            threshold=args.threshold,
            tessellation_quality=args.tessellation_quality,
            keep_intermediates=args.keep_intermediates,
            ckpt_path=args.checkpoint,
        )
        return 0 if success else 1

    # Batch mode
    if args.input_dir:
        input_dir = Path(args.input_dir)
        if not input_dir.is_dir():
            print(f"Error: Input directory not found: {args.input_dir}")
            return 1

        output_dir = Path(args.output_dir) if args.output_dir else input_dir / "brep_output"
        output_dir.mkdir(parents=True, exist_ok=True)

        step_files = sorted(
            list(input_dir.glob("*.step")) + list(input_dir.glob("*.stp")) +
            list(input_dir.glob("*.STEP")) + list(input_dir.glob("*.STP"))
        )

        if not step_files:
            print(f"No STEP files found in {input_dir}")
            return 1

        print(f"Found {len(step_files)} STEP files in {input_dir}")
        print(f"Output directory: {output_dir}")

        results = []
        for step_file in step_files:
            out_file = output_dir / f"{step_file.stem}_bboxes.step"
            success = process_single(
                step_input=str(step_file),
                step_output=str(out_file),
                alignment=args.alignment,
                mode=args.mode,
                fixed_clusters=args.clusters,
                max_clusters=args.max_clusters,
                threshold=args.threshold,
                tessellation_quality=args.tessellation_quality,
                keep_intermediates=args.keep_intermediates,
                ckpt_path=args.checkpoint,
            )
            results.append((step_file.name, success))

        # Summary
        print(f"\n{'='*60}")
        print("BATCH SUMMARY")
        print(f"{'='*60}")
        for name, ok in results:
            status = "OK" if ok else "FAILED"
            print(f"  [{status}] {name}")

        n_ok = sum(1 for _, ok in results if ok)
        print(f"\n{n_ok}/{len(results)} files processed successfully")
        print(f"Output: {output_dir}")
        return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    exit(main())
