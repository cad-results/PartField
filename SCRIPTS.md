# PartField Pipeline Scripts Reference

## Pipeline Variants Overview

All pipeline scripts follow three steps:
1. **Feature Extraction** - Run `partfield_inference.py` to extract PartField features
2. **Clustering** - Run `run_part_clustering.py` to generate part segmentation
3. **Geometry Export** - Generate bounding boxes or primitives from segments

### Complete Variant Table

| # | Script | Clusters | Alignment | Output | Description |
|---|--------|----------|-----------|--------|-------------|
| 1 | `pipeline_variant1_auto_self.sh` | Auto 2-20 | Self (per-segment PCA) | PLY mesh | Tightest fit, auto-selected cluster count |
| 2 | `pipeline_variant2_auto_global.sh` | Auto 2-20 | Global (shared PCA) | PLY mesh | Consistent alignment, auto-selected |
| 3 | `pipeline_variant3_fixed5_self.sh` | Fixed 5 | Self | PLY mesh | Quick 5-part decomposition, tight fit |
| 4 | `pipeline_variant4_fixed5_global.sh` | Fixed 5 | Global | PLY mesh | Quick 5-part, consistent alignment |
| **5** | `pipeline_variant5_auto_self_brep.sh` | Auto 2-20 | Self | **STEP CAD** | Parametric BREP, tight fit |
| **6** | `pipeline_variant6_auto_global_brep.sh` | Auto 2-20 | Global | **STEP CAD** | Parametric BREP, consistent alignment |
| **7** | `pipeline_variant7_fixed5_self_brep.sh` | Fixed 5 | Self | **STEP CAD** | Quick BREP, tight fit |
| **8** | `pipeline_variant8_fixed5_global_brep.sh` | Fixed 5 | Global | **STEP CAD** | Quick BREP, consistent alignment |

### Key Differences

**Cluster Selection:**
- **Auto (variants 1-2, 5-6):** Generates clusters from 2 to 20, auto-selects best using silhouette score
- **Fixed 5 (variants 3-4, 7-8):** Always produces exactly 5 segments

**Alignment:**
- **Self (variants 1, 3, 5, 7):** Each segment's bounding box oriented to its own PCA axes (tighter fit)
- **Global (variants 2, 4, 6, 8):** All boxes share the mesh's global PCA orientation (consistent alignment)

**Output Format:**
- **PLY (variants 1-4):** Triangulated mesh approximations (solid, wireframe, transparent styles)
- **STEP (variants 5-8):** Parametric BREP solid geometry, viewable in any CAD tool

---

## Output Directories

```
exp_results/
├── partfield_features/glb_pipeline/    # Step 1: Features (shared)
├── clustering/
│   ├── variant1_auto_self/             # Step 2: Clustering results
│   ├── variant2_auto_global/
│   ├── ...
│   └── variant8_fixed5_global_brep/
├── bboxes/
│   ├── variant1_auto_self/             # Step 3: PLY bbox output
│   ├── variant2_auto_global/
│   ├── variant3_fixed5_self/
│   └── variant4_fixed5_global/
└── brep/
    ├── variant5_auto_self_brep/        # Step 3: STEP CAD output
    ├── variant6_auto_global_brep/
    ├── variant7_fixed5_self_brep/
    └── variant8_fixed5_global_brep/
```

---

## Usage Examples

### PLY Pipeline (Variants 1-4)

```bash
# Variant 1: Auto clusters, self-aligned, PLY output
./pipeline_variant1_auto_self.sh

# View results
./run_viewer.sh exp_results/bboxes/variant1_auto_self/model_10_bbox_solid.ply
./run_viewer.sh --browse exp_results/bboxes/variant1_auto_self/
```

### BREP Pipeline (Variants 5-8)

```bash
# Variant 5: Auto clusters, self-aligned, STEP output
./pipeline_variant5_auto_self_brep.sh

# View results in BREP viewer
./run_brep_viewer.sh exp_results/brep/variant5_auto_self_brep/model_10_brep.step
./run_brep_viewer.sh --browse exp_results/brep/variant5_auto_self_brep/

# Or open in FreeCAD / SolidWorks
freecad exp_results/brep/variant5_auto_self_brep/model_10_brep.step
```

### Standalone Tools

```bash
# Generate BREP directly (without pipeline)
python brep_generator.py \
    -i data/glb_output/model.glb \
    -l exp_results/clustering/objaverse/cluster_out/model_0_10.npy \
    -o result.step --mode bbox --alignment self

# Generate primitives as BREP
python brep_generator.py \
    -i model.glb -l labels.npy -o result.step --mode primitive

# Add BREP to existing PLY scripts
python segment_with_bboxes.py -i model.glb -l labels.npy -o output.ply --brep
python segment_with_bboxes.py -i model.glb -l labels.npy -o output.ply --all-formats
python segment_with_primitives.py -i model.glb -l labels.npy -o output.ply --brep
```

---

## Script Details

### step_to_brep.py

End-to-end STEP → STEP pipeline. Reads a CAD STEP file, tessellates it to a mesh, runs PartField segmentation, and outputs a new STEP file where the geometry is replaced by colored BREP bounding boxes.

```bash
# Auto cluster selection
python step_to_brep.py -i model.step -o model_bboxes.step

# Fixed clusters
python step_to_brep.py -i model.step -o model_bboxes.step --clusters 5

# Batch mode
python step_to_brep.py --input-dir steps/ --output-dir brep_out/
```

| Option | Short | Description |
|--------|-------|-------------|
| `--input` | `-i` | Input STEP file |
| `--output` | `-o` | Output STEP file (bboxes) |
| `--input-dir` | | Directory of STEP files (batch) |
| `--output-dir` | | Output directory (batch) |
| `--clusters` | | Fixed cluster count (default: auto 2-20) |
| `--max-clusters` | | Max clusters for auto-selection (default: 20) |
| `--alignment` | `-a` | `self` or `global` (default: self) |
| `--mode` | `-m` | `bbox` or `primitive` (default: bbox) |
| `--tessellation-quality` | | Linear deflection for mesh (default: 0.1, smaller = finer) |
| `--keep-intermediates` | | Keep temp mesh, features, labels for debugging |
| `--checkpoint` | | PartField model checkpoint path |

**Pipeline steps:**
1. STEP → mesh (pythonocc tessellation)
2. Mesh → PartField features (GPU inference)
3. Features → clustering labels
4. Labels + mesh → BREP bounding boxes → colored STEP

---

### brep_generator.py

Core BREP generation module. Converts segmentation results to STEP CAD files.

| Option | Short | Description |
|--------|-------|-------------|
| `--input` | `-i` | Input mesh file (GLB, PLY, OBJ, etc.) |
| `--labels` | `-l` | Segmentation labels (.npy) |
| `--output` | `-o` | Output STEP file path |
| `--mode` | `-m` | `bbox` or `primitive` (default: bbox) |
| `--alignment` | `-a` | `self` or `global` (default: self, bbox only) |
| `--threshold` | `-t` | Primitive fit threshold (default: 0.35) |
| `--no-filter` | | Disable outlier filtering |
| `--no-overlap-resolution` | | Disable overlap resolution |

### brep_viewer.py

Interactive Qt+pythonOCC 3D viewer for STEP files.

| Option | Short | Description |
|--------|-------|-------------|
| `file` | | STEP file path |
| `--mesh` | `-m` | Mesh for on-the-fly generation |
| `--labels` | `-l` | Labels for on-the-fly generation |
| `--alignment` | `-a` | `self` or `global` |
| `--mode` | | `bbox` or `primitive` |
| `--browse` | `-b` | Browse STEP files in directory |

---

## Requirements

### PLY Pipelines (Variants 1-4)
- `trimesh`, `numpy`, `scikit-learn`, `scipy`, `matplotlib`
- `open3d` (for viewer only)

### BREP Pipelines (Variants 5-8)
All PLY requirements plus:
- `pythonocc-core` (`conda install -c conda-forge pythonocc-core`)
- `PyQt5` or `PySide2` (for BREP viewer only)
