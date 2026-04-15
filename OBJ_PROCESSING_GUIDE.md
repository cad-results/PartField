# OBJ File Processing Guide for PartField

## Commands to Segment, Generate Bounding Boxes, and Visualize

Step 1: Extract PartField Features

python partfield_inference.py -c configs/final/demo.yaml \
--opts continue_ckpt model/model_objaverse.ckpt \
result_name partfield_features/glb_output \
dataset.data_path data/glb_output

Step 2: Run Part Clustering (Multiple Varieties)

Option A - Standard Agglomerative Clustering:
python run_part_clustering.py \
--root exp_results/partfield_features/glb_output \
--dump_dir exp_results/clustering/glb_output_agglo \
--source_dir data/glb_output \
--use_agglo True \
--max_num_clusters 20 \
--option 0

Option B - KMeans Clustering (no adjacency required):
python run_part_clustering.py \
--root exp_results/partfield_features/glb_output \
--dump_dir exp_results/clustering/glb_output_kmeans \
--source_dir data/glb_output \
--max_num_clusters 20

Option C - MST-based Adjacency (for messy meshes):
python run_part_clustering.py \
--root exp_results/partfield_features/glb_output \
--dump_dir exp_results/clustering/glb_output_mst \
--source_dir data/glb_output \
--use_agglo True \
--max_num_clusters 20 \
--option 1 \
--with_knn True

Step 3: Generate Bounding Boxes (for each mesh and clustering)

#### Option A - Per-Segment Oriented Bounding Boxes (tighter fit)

# For each mesh file and label file, run:
python segment_with_bboxes.py \
-i data/glb_output/partmodel.glb \
-l exp_results/clustering/glb_output_agglo/cluster_out/partmodel_0_10.npy \
-o exp_results/bboxes/glb_output/partmodel_bboxes.ply \
--style all

#### Option B - Global-Oriented Bounding Boxes (consistent alignment)

If you want all bounding boxes aligned to the model's global principal axes:

python segment_with_bboxes_global.py \
-i data/glb_output/partmodel.glb \
-l exp_results/clustering/glb_output_agglo/cluster_out/partmodel_0_10.npy \
-o exp_results/bboxes/glb_output/partmodel_bboxes_global.ply \
--style all

**When to use global orientation:**
- CAD models with consistent part orientations
- Architectural models where alignment matters
- Models where visual consistency of boxes is desired

**When to use per-segment orientation:**
- Organic models with parts at various angles
- When you want the tightest possible fit per segment

Repeat for other files (van_model.glb, 3dpea.com_assembly1.glb) and different cluster counts (the _XX.npy suffix
indicates number of clusters, e.g., _05, _10, _15).

Step 4: Visualize Results

Browse all files with auto-detected clustering:
./run_viewer.sh --browse data/glb_output/

View specific file:
./run_viewer.sh data/glb_output/partmodel.glb

View with specific labels:
./run_viewer.sh data/glb_output/partmodel.glb \
-l exp_results/clustering/glb_output_agglo/cluster_out/partmodel_0_10.npy

View generated bounding boxes:
./run_viewer.sh exp_results/bboxes/glb_output/partmodel_bboxes_solid.ply

Keyboard Controls in Viewer

- T/TAB - Cycle views: Original → Segmented → BBoxes → PCA
- C/V - Next/Previous clustering result (different granularities)
- A/D - Previous/Next file (browse mode)
- S - Save screenshot
- Q/ESC - Exit


## Quick Start

### Option 1: Batch Processing (Recommended)
Process all OBJ files in a directory with one command:

```bash
# Basic processing
python process_obj_batch.py --input-dir data/mtlfiles --name my_project

# With primitive fitting
python process_obj_batch.py --input-dir data/mtlfiles --name my_project --fit-primitives

# Custom parameters
python process_obj_batch.py --input-dir data/mtlfiles --name my_project \
    --max-clusters 15 --num-cluster-targets 8 --fit-primitives
```

### Option 2: Manual Step-by-Step
For more control over each step:

```bash
# 1. Organize files (flatten directory structure)
mkdir -p data/my_models_flat
cp data/my_models/*/*.obj data/my_models_flat/
cp data/my_models/*/*.mtl data/my_models_flat/

# 2. Extract PartField features
python partfield_inference.py -c configs/final/demo.yaml \
    --opts continue_ckpt model/model_objaverse.ckpt \
    result_name partfield_features/my_models_flat \
    dataset.data_path data/my_models_flat

# 3. Run part clustering
python run_part_clustering.py \
    --root exp_results/partfield_features/my_models_flat \
    --dump_dir exp_results/clustering/my_models_flat \
    --source_dir data/my_models_flat \
    --use_agglo True --max_num_clusters 20 \
    --option 1 --with_knn True

# 4. (Optional) Fit geometric primitives
python segment_with_primitives.py \
    --input data/my_models_flat/model.obj \
    --labels exp_results/clustering/my_models_flat/cluster_out/model_0_10.npy \
    --output results/model_primitives.ply
```

## Visualization

### Interactive Viewer

```bash
# Browse all segmented models
./run_viewer.sh --browse exp_results/clustering/my_models_flat/ply/

# View specific OBJ file with auto-detected clustering
./run_viewer.sh data/my_models_flat/model.obj

# View primitive results
./run_viewer.sh --browse exp_results/clustering/my_models_flat/primitives/
```

### Viewer Controls
| Key | Action |
|-----|--------|
| `TAB` / `T` | Cycle views: Original → Segmented → PCA |
| `C` | Next clustering (more parts) |
| `V` | Previous clustering (fewer parts) |
| `A` / `LEFT` | Previous file (browse mode) |
| `D` / `RIGHT` | Next file (browse mode) |
| `S` | Save screenshot |
| `R` | Reset camera |
| `H` | Toggle help |
| `ESC` / `Q` | Exit |

## File Structure

### Input Requirements
- **OBJ files**: Standard Wavefront OBJ format
- **MTL files**: Optional material definitions (automatically loaded if present)
- **Directory structure**: Can be nested (will be flattened during processing)

### Output Structure
```
exp_results/
├── partfield_features/{name}_flat/
│   ├── part_feat_{model}_0.npy      # Feature vectors
│   └── input_{model}_0.ply          # Preprocessed mesh
├── clustering/{name}_flat/
│   ├── ply/
│   │   ├── {model}_0_01.ply         # 1-part clustering
│   │   ├── {model}_0_02.ply         # 2-part clustering
│   │   └── ...                      # Up to max_num_clusters
│   ├── cluster_out/
│   │   ├── {model}_0_01.npy         # Label arrays
│   │   └── ...
│   └── primitives/                  # (if --fit-primitives used)
│       ├── {model}_primitives.ply
│       ├── {model}_primitives_info.txt
│       └── {model}_primitives_primitives_only.ply
```

## Parameters

### Clustering Parameters
- `--max_num_clusters` (default: 20): Maximum number of parts to segment
- `--option` (default: 1): Adjacency matrix construction method
  - 0: Naive (edge-based only)
  - 1: Face MST with KNN (recommended)
  - 2: Connected components MST
- `--with_knn`: Enable KNN for better connectivity

### Primitive Fitting Parameters
- `--threshold` (default: 0.35): Score threshold for primitive selection
- `--num-cluster-targets`: Which clustering level to use (default: 10)

Supported primitives:
- Box, Cylinder, Sphere, Ellipsoid
- Hemisphere, Quarter-sphere, Eighth-sphere
- Cone, Capsule, Triangular prism
- Tetrahedron, Octahedron
- Oriented bounding box (fallback)

### Bounding Box Parameters
The `segment_with_bboxes.py` script now includes intelligent filtering and overlap resolution:
- **Outlier removal**: Automatically removes statistical outliers and low-density points
- **PCA orientation**: Boxes are oriented along principal components for optimal fit
- **Overlap resolution**: Automatically shrinks overlapping boxes to eliminate intersections

Options:
- `--no-filter`: Disable outlier/density filtering (for raw bounding boxes)
- `--no-overlap-resolution`: Disable automatic overlap resolution

## Troubleshooting

### No files found during inference
**Problem**: "val dataset len: 0"

**Solution**: OBJ files must be directly in the data directory (no subdirectories). Use the batch processor or flatten the structure manually.

### Clustering connectivity warnings
**Problem**: "connectivity matrix has N > 1 components"

**Solution**: This is expected for meshes with disconnected parts. The pipeline automatically handles this with KNN connections. Use `--option 1 --with_knn True` for best results.

### MTL file not loading
**Problem**: Materials not appearing

**Solution**:
1. Ensure MTL file is in the same directory as OBJ
2. Check that OBJ file references MTL with `mtllib` directive
3. MTL filename must match the reference in OBJ

### Viewer not opening
**Problem**: Viewer fails to start

**Solution** (WSL2/software rendering):
```bash
# Use the provided wrapper script
./run_viewer.sh [args]

# Or set environment variables manually
export LIBGL_ALWAYS_SOFTWARE=1
export MESA_GL_VERSION_OVERRIDE=3.3
python viewer.py [args]
```

## Examples

### Example 1: Process a directory of OBJ files
```bash
python process_obj_batch.py \
    --input-dir data/furniture_models \
    --name furniture \
    --fit-primitives
```

### Example 2: Process with custom clustering
```bash
python process_obj_batch.py \
    --input-dir data/mechanical_parts \
    --name mechanical \
    --max-clusters 25 \
    --num-cluster-targets 12 \
    --fit-primitives
```

### Example 3: View results
```bash
# Browse all clustering levels
./run_viewer.sh --browse exp_results/clustering/furniture/ply/

# View primitives
./run_viewer.sh --browse exp_results/clustering/furniture/primitives/
```

### Example 4: Generate bounding boxes
```bash
# Generate per-segment oriented bounding boxes with filtering and overlap resolution
python segment_with_bboxes.py \
    -i data/my_models_flat/model.obj \
    -l exp_results/clustering/my_models_flat/cluster_out/model_0_10.npy \
    -o exp_results/bboxes/model_bboxes.ply \
    --style all

# Generate globally-oriented bounding boxes (all boxes share same alignment)
python segment_with_bboxes_global.py \
    -i data/my_models_flat/model.obj \
    -l exp_results/clustering/my_models_flat/cluster_out/model_0_10.npy \
    -o exp_results/bboxes/model_bboxes_global.ply \
    --style all

# View and compare both results
./run_viewer.sh exp_results/bboxes/model_bboxes_wireframe.ply
./run_viewer.sh exp_results/bboxes/model_bboxes_global_wireframe.ply

# Generate raw boxes without filtering (captures all geometry)
python segment_with_bboxes.py \
    -i data/my_models_flat/model.obj \
    -l exp_results/clustering/my_models_flat/cluster_out/model_0_10.npy \
    -o exp_results/bboxes/model_bboxes_raw.ply \
    --style wireframe --no-filter
```

## Notes

- **GPU Required**: Feature extraction requires a CUDA-capable GPU
- **Processing Time**: ~5-10 seconds per model for feature extraction
- **Memory**: Ensure sufficient RAM for large meshes (>1M vertices)
- **Formats**: OBJ files are converted to PLY internally; original files are preserved
- **Clustering Levels**: All levels from 1 to max_num_clusters are generated
- **Primitives**: Fitting quality depends on part geometry and threshold setting

## Created Files

- `process_obj_batch.py` - Automated batch processing pipeline
- `segment_with_primitives.py` - Geometric primitive fitting tool
- `segment_with_bboxes.py` - Per-segment oriented bounding boxes
- `segment_with_bboxes_global.py` - Globally-oriented bounding boxes (consistent alignment)
- `viewer.py` - Interactive 3D visualization
- `run_viewer.sh` - Viewer wrapper for WSL2/software rendering

## Links

- Original PartField: [GitHub](https://github.com/nv-research/partfield)
- Paper: [arXiv](https://arxiv.org/pdf/2504.11451)
- Model: [HuggingFace](https://huggingface.co/mikaelaangel/partfield-ckpt/blob/main/model_objaverse.ckpt)
