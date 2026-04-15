#!/usr/bin/env python3
"""
Batch OBJ Processing Pipeline - Complete pipeline for processing OBJ/MTL files with PartField.

This script automates the entire pipeline:
1. Organizes OBJ/MTL files into flat structure
2. Extracts PartField features using pretrained model
3. Runs part clustering (agglomerative)
4. Optionally fits geometric primitives to segments
5. Provides viewer instructions for visualization

Usage:
    # Process directory with OBJ files
    python process_obj_batch.py --input-dir data/mtlfiles --name my_models

    # Process with primitive fitting
    python process_obj_batch.py --input-dir data/mtlfiles --name my_models --fit-primitives

    # Process with custom parameters
    python process_obj_batch.py --input-dir data/mtlfiles --name my_models \\
        --max-clusters 15 --num-cluster-targets 8 --primitive-threshold 0.3
"""

import os
import sys
import argparse
import subprocess
import shutil
from pathlib import Path
from typing import List, Tuple


class OBJBatchProcessor:
    """Automates the full PartField processing pipeline for OBJ files."""

    def __init__(self, input_dir: str, name: str, verbose: bool = True):
        """
        Args:
            input_dir: Directory containing OBJ files (can be nested)
            name: Name for this processing run
            verbose: Print progress messages
        """
        self.input_dir = Path(input_dir)
        self.name = name
        self.verbose = verbose

        # Setup paths
        self.data_dir = Path("data") / f"{name}_flat"
        self.features_dir = Path("exp_results/partfield_features") / f"{name}_flat"
        self.clustering_dir = Path("exp_results/clustering") / f"{name}_flat"
        self.primitives_dir = self.clustering_dir / "primitives"

        # Model checkpoint
        self.model_ckpt = "model/model_objaverse.ckpt"

    def log(self, message: str, prefix: str = "INFO"):
        """Print message if verbose enabled."""
        if self.verbose:
            print(f"[{prefix}] {message}")

    def find_obj_files(self) -> List[Path]:
        """Find all OBJ files in input directory (recursive)."""
        obj_files = list(self.input_dir.rglob("*.obj"))
        self.log(f"Found {len(obj_files)} OBJ files")
        return obj_files

    def find_mtl_files(self) -> List[Path]:
        """Find all MTL files in input directory (recursive)."""
        mtl_files = list(self.input_dir.rglob("*.mtl"))
        self.log(f"Found {len(mtl_files)} MTL files")
        return mtl_files

    def step1_organize_files(self) -> int:
        """
        Step 1: Copy OBJ and MTL files to flat structure.
        Returns number of files copied.
        """
        self.log("="*60, "STEP 1")
        self.log("Organizing OBJ/MTL files into flat structure")

        # Create data directory
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Copy OBJ files
        obj_files = self.find_obj_files()
        for obj_file in obj_files:
            dest = self.data_dir / obj_file.name
            shutil.copy2(obj_file, dest)
            self.log(f"  Copied: {obj_file.name}")

        # Copy MTL files
        mtl_files = self.find_mtl_files()
        for mtl_file in mtl_files:
            dest = self.data_dir / mtl_file.name
            shutil.copy2(mtl_file, dest)
            self.log(f"  Copied: {mtl_file.name}")

        total_files = len(obj_files) + len(mtl_files)
        self.log(f"Copied {total_files} files to {self.data_dir}")
        return len(obj_files)

    def step2_extract_features(self) -> bool:
        """
        Step 2: Extract PartField features using pretrained model.
        Returns True if successful.
        """
        self.log("="*60, "STEP 2")
        self.log("Extracting PartField features")

        if not Path(self.model_ckpt).exists():
            self.log(f"ERROR: Model checkpoint not found: {self.model_ckpt}", "ERROR")
            self.log("Please download the model from:", "ERROR")
            self.log("https://huggingface.co/mikaelaangel/partfield-ckpt/blob/main/model_objaverse.ckpt", "ERROR")
            return False

        cmd = [
            "python3", "partfield_inference.py",
            "-c", "configs/final/demo.yaml",
            "--opts",
            "continue_ckpt", self.model_ckpt,
            "result_name", f"partfield_features/{self.name}_flat",
            "dataset.data_path", str(self.data_dir)
        ]

        self.log(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=False)

        if result.returncode != 0:
            self.log("Feature extraction failed", "ERROR")
            return False

        self.log("Feature extraction completed")
        return True

    def step3_run_clustering(self, max_clusters: int = 20,
                            option: int = 1, with_knn: bool = True) -> bool:
        """
        Step 3: Run part clustering on extracted features.
        Returns True if successful.
        """
        self.log("="*60, "STEP 3")
        self.log("Running part clustering")

        cmd = [
            "python3", "run_part_clustering.py",
            "--root", str(self.features_dir),
            "--dump_dir", str(self.clustering_dir),
            "--source_dir", str(self.data_dir),
            "--use_agglo", "True",
            "--max_num_clusters", str(max_clusters),
            "--option", str(option)
        ]

        if with_knn:
            cmd.extend(["--with_knn", "True"])

        self.log(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=False)

        if result.returncode != 0:
            self.log("Clustering failed", "ERROR")
            return False

        self.log("Clustering completed")
        return True

    def step4_fit_primitives(self, cluster_count: int = 10,
                            threshold: float = 0.35) -> bool:
        """
        Step 4: Fit geometric primitives to segmented parts.
        Returns True if successful.
        """
        self.log("="*60, "STEP 4")
        self.log("Fitting geometric primitives to segments")

        # Create primitives directory
        self.primitives_dir.mkdir(parents=True, exist_ok=True)

        # Find all OBJ files
        obj_files = list(self.data_dir.glob("*.obj"))
        cluster_out = self.clustering_dir / "cluster_out"

        success_count = 0
        for obj_file in obj_files:
            # Find corresponding label file
            # Try different naming patterns
            base_name = obj_file.stem
            label_patterns = [
                f"{base_name}_0_{cluster_count:02d}.npy",
                f"{base_name.split()[0]}_0_{cluster_count:02d}.npy",  # Handle spaces
            ]

            label_file = None
            for pattern in label_patterns:
                test_path = cluster_out / pattern
                if test_path.exists():
                    label_file = test_path
                    break

            if not label_file:
                self.log(f"Warning: No labels found for {obj_file.name}", "WARN")
                continue

            # Generate output name
            out_name = obj_file.stem.replace(" ", "_") + "_primitives.ply"
            out_path = self.primitives_dir / out_name

            cmd = [
                "python3", "segment_with_primitives.py",
                "--input", str(obj_file),
                "--labels", str(label_file),
                "--output", str(out_path),
                "--threshold", str(threshold)
            ]

            self.log(f"Processing: {obj_file.name}")
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                success_count += 1
                self.log(f"  ✓ Primitives fitted: {out_name}")
            else:
                self.log(f"  ✗ Failed: {obj_file.name}", "WARN")

        self.log(f"Fitted primitives for {success_count}/{len(obj_files)} models")
        return success_count > 0

    def print_summary(self, num_models: int):
        """Print summary and instructions."""
        self.log("="*60, "SUMMARY")
        self.log(f"Processing complete for {num_models} models!")
        self.log("")
        self.log("Generated files:")
        self.log(f"  Features:    {self.features_dir}")
        self.log(f"  Clustering:  {self.clustering_dir}")
        self.log(f"  Primitives:  {self.primitives_dir}")
        self.log("")
        self.log("Visualization commands:")
        self.log("")
        self.log("  # View segmented models in browse mode:")
        self.log(f"  ./run_viewer.sh --browse {self.clustering_dir}/ply/")
        self.log("")
        self.log("  # View specific model with original OBJ:")
        obj_files = list(self.data_dir.glob("*.obj"))
        if obj_files:
            example_obj = obj_files[0]
            self.log(f'  ./run_viewer.sh "{example_obj}"')
        self.log("")
        self.log("  # View primitive-fitted models:")
        if self.primitives_dir.exists():
            self.log(f"  ./run_viewer.sh --browse {self.primitives_dir}/")
        self.log("")
        self.log("Keyboard shortcuts in viewer:")
        self.log("  TAB/T - Cycle views (Original → Segmented → PCA)")
        self.log("  C/V   - Next/Previous clustering (more/fewer parts)")
        self.log("  A/D   - Previous/Next file (browse mode)")
        self.log("  S     - Save screenshot")
        self.log("  Q/ESC - Exit")
        self.log("")

    def run(self, max_clusters: int = 20, fit_primitives: bool = False,
            primitive_cluster_count: int = 10, primitive_threshold: float = 0.35) -> bool:
        """
        Run the complete pipeline.

        Args:
            max_clusters: Maximum number of clusters for hierarchical clustering
            fit_primitives: Whether to fit geometric primitives
            primitive_cluster_count: Which clustering level to use for primitives
            primitive_threshold: Score threshold for primitive fitting

        Returns:
            True if pipeline completed successfully
        """
        try:
            # Step 1: Organize files
            num_models = self.step1_organize_files()
            if num_models == 0:
                self.log("No OBJ files found in input directory", "ERROR")
                return False

            # Step 2: Extract features
            if not self.step2_extract_features():
                return False

            # Step 3: Run clustering
            if not self.step3_run_clustering(max_clusters=max_clusters):
                return False

            # Step 4: Fit primitives (optional)
            if fit_primitives:
                self.step4_fit_primitives(
                    cluster_count=primitive_cluster_count,
                    threshold=primitive_threshold
                )

            # Print summary
            self.print_summary(num_models)
            return True

        except Exception as e:
            self.log(f"Pipeline failed with error: {e}", "ERROR")
            import traceback
            traceback.print_exc()
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Batch process OBJ files with PartField pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic processing
  python process_obj_batch.py --input-dir data/mtlfiles --name my_models

  # With primitive fitting
  python process_obj_batch.py --input-dir data/mtlfiles --name my_models --fit-primitives

  # Custom parameters
  python process_obj_batch.py --input-dir data/mtlfiles --name my_models \\
      --max-clusters 15 --num-cluster-targets 8 --primitive-threshold 0.3

Output:
  - Features: exp_results/partfield_features/{name}_flat/
  - Clustering: exp_results/clustering/{name}_flat/
  - Primitives: exp_results/clustering/{name}_flat/primitives/
        """
    )

    parser.add_argument("--input-dir", "-i", required=True,
                       help="Input directory containing OBJ files (can have subdirectories)")
    parser.add_argument("--name", "-n", required=True,
                       help="Name for this processing run")
    parser.add_argument("--max-clusters", type=int, default=20,
                       help="Maximum number of clusters (default: 20)")
    parser.add_argument("--fit-primitives", action="store_true",
                       help="Fit geometric primitives to segments")
    parser.add_argument("--num-cluster-targets", type=int, default=10,
                       help="Number of clusters to use for primitive fitting (default: 10)")
    parser.add_argument("--primitive-threshold", type=float, default=0.35,
                       help="Score threshold for primitive fitting (default: 0.35)")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Suppress progress messages")

    args = parser.parse_args()

    # Validate input directory
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory not found: {args.input_dir}")
        return 1

    # Create processor
    processor = OBJBatchProcessor(
        args.input_dir,
        args.name,
        verbose=not args.quiet
    )

    # Run pipeline
    success = processor.run(
        max_clusters=args.max_clusters,
        fit_primitives=args.fit_primitives,
        primitive_cluster_count=args.num_cluster_targets,
        primitive_threshold=args.primitive_threshold
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
