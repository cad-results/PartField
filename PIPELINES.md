# PartField Processing Pipelines

This document describes the 4 pipeline variations for processing GLB files through PartField to generate segmented bounding boxes.

## Overview

The pipelines take `.glb` files as input and produce:
1. **PartField Features** - Neural feature embeddings per face/vertex
2. **Clustering/Segmentation** - Part segmentation labels
3. **Bounding Boxes** - Oriented bounding boxes (OBBs) for each segment

Each pipeline variation is a combination of:
- **Clustering Mode**: Auto-segmented (2-20 clusters) vs Fixed (exactly 5 clusters)
- **BBox Alignment**: Self-aligned (per-segment PCA) vs Globally-aligned (shared global PCA)

## Pipeline Variants Summary

| Variant | Script | Clusters | BBox Alignment | Best For |
|---------|--------|----------|----------------|----------|
| 1 | `pipeline_variant1_auto_self.sh` | Auto (2-20) | Self-aligned | Organic models, exploration |
| 2 | `pipeline_variant2_auto_global.sh` | Auto (2-20) | Globally-aligned | CAD models, exploration |
| 3 | `pipeline_variant3_fixed5_self.sh` | Fixed (5) | Self-aligned | Known part count, organic |
| 4 | `pipeline_variant4_fixed5_global.sh` | Fixed (5) | Globally-aligned | Known part count, CAD |

---

## Detailed Pipeline Steps

### Step 1: PartField Feature Extraction

**Script**: `partfield_inference.py`

Extracts neural feature embeddings from the input mesh using the PartField model.

**Outputs**:
- `input_{id}_{view}.ply` - Preprocessed input mesh
- `feat_pca_{id}_{view}.ply` - Mesh with PCA-colored features (visualization)
- `part_feat_{id}_{view}.npy` - Raw feature vectors per face

**Visualization**:
```bash
./run_viewer.sh exp_results/partfield_features/glb_pipeline/feat_pca_*.ply
```

The PCA visualization shows the 3D feature space mapped to RGB colors - similar colors indicate similar parts.

---

### Step 2: Clustering/Segmentation

**Script**: `run_part_clustering.py`

Clusters the feature vectors to produce part segmentation.

**Clustering Modes**:

| Mode | `--max_num_clusters` | Description |
|------|---------------------|-------------|
| Auto | 20 | Generates clusters for counts 2, 3, 4, ... up to max |
| Fixed (5) | 6 | Only generates up to 5 clusters |

**Algorithm Options**:
- `--use_agglo False` - K-Means clustering (faster, simpler)
- `--use_agglo True` - Agglomerative clustering with mesh connectivity (better for complex meshes)

**Outputs**:
- `cluster_out/{id}_{view}_{n}.npy` - Cluster labels for n clusters
- `ply/{id}_{view}_{n}.ply` - Colored segmentation mesh

**Visualization**:
```bash
# View specific cluster count
./run_viewer.sh exp_results/clustering/variant1_auto_self/ply/*_10.ply

# Interactive view (cycle with C/V keys)
./run_viewer.sh data/glb_output/model.glb
```

---

### Step 3: Bounding Box Generation

**Scripts**:
- `segment_with_bboxes.py` - Self-aligned (per-segment PCA)
- `segment_with_bboxes_global.py` - Globally-aligned (shared PCA)

#### Self-Aligned Bounding Boxes

Each segment's bounding box is oriented according to its own principal axes (PCA).

**Pros**: Tighter fit, better for irregularly oriented parts
**Cons**: Boxes may have inconsistent orientations

```bash
python segment_with_bboxes.py \
    --input model.glb \
    --labels labels.npy \
    --output output.ply \
    --style all
```

#### Globally-Aligned Bounding Boxes

All bounding boxes share a single global orientation computed from the entire mesh.

**Pros**: Consistent alignment, better for CAD/architectural models
**Cons**: May not fit individual parts as tightly

```bash
python segment_with_bboxes_global.py \
    --input model.glb \
    --labels labels.npy \
    --output output.ply \
    --style all \
    --method all_vertices
```

**Output Styles**:
- `*_solid.ply` - Solid colored boxes
- `*_wireframe.ply` - Wireframe edges only
- `*_transparent.ply` - Semi-transparent boxes
- `*_info.txt` - Box parameters (center, dimensions, rotation)

**Visualization**:
```bash
./run_viewer.sh exp_results/bboxes/variant1_auto_self/*_bbox_solid.ply
```

---

## Usage

### Quick Start

```bash
# Make scripts executable (first time only)
chmod +x pipeline_variant*.sh

# Run your preferred variant
./pipeline_variant1_auto_self.sh
```

### Input Requirements

Place your `.glb` files in:
```
data/glb_output/
├── model1.glb
├── model2.glb
└── ...
```

### Output Structure

```
exp_results/
├── partfield_features/glb_pipeline/
│   ├── input_{id}_0.ply              # Preprocessed mesh
│   ├── feat_pca_{id}_0.ply           # PCA feature visualization
│   └── part_feat_{id}_0.npy          # Raw features
│
├── clustering/
│   ├── variant1_auto_self/
│   │   ├── cluster_out/
│   │   │   ├── {id}_0_02.npy         # 2 clusters
│   │   │   ├── {id}_0_03.npy         # 3 clusters
│   │   │   └── ...                   # up to 20 clusters
│   │   └── ply/
│   │       ├── {id}_0_02.ply         # Colored mesh (2 clusters)
│   │       └── ...
│   ├── variant2_auto_global/
│   ├── variant3_fixed5_self/
│   └── variant4_fixed5_global/
│
└── bboxes/
    ├── variant1_auto_self/
    │   ├── {id}_10_bbox_solid.ply
    │   ├── {id}_10_bbox_wireframe.ply
    │   ├── {id}_10_bbox_transparent.ply
    │   └── {id}_10_bbox_info.txt
    ├── variant2_auto_global/
    ├── variant3_fixed5_self/
    └── variant4_fixed5_global/
```

---

## Interactive Viewer

The viewer (`viewer.py`) supports interactive visualization of all pipeline outputs.

### Launch Methods

```bash
# Via wrapper script (recommended for WSL2)
./run_viewer.sh <file.glb|file.ply>

# Direct Python
python viewer.py <file.glb|file.ply>

# With specific labels
python viewer.py model.glb --labels clustering.npy

# Browse directory
python viewer.py --browse exp_results/bboxes/variant1_auto_self/
```

### Keyboard Controls

| Key | Action |
|-----|--------|
| `T` / `TAB` | Cycle views: Original → Segmented → BBoxes → PCA |
| `C` | Next clustering (more parts) |
| `V` | Previous clustering (fewer parts) |
| `A` / `LEFT` | Previous file (browse mode) |
| `D` / `RIGHT` | Next file (browse mode) |
| `S` | Save screenshot |
| `R` | Reset camera view |
| `H` | Toggle help |
| `ESC` / `Q` | Exit |

---

## Choosing a Variant

### When to use Auto-segmented (Variants 1, 2)

- You don't know how many parts the model has
- You want to explore different segmentation granularities
- You want to pick the best cluster count after seeing results

### When to use Fixed clusters (Variants 3, 4)

- You know the expected number of parts
- You need consistent part counts across models
- You're building a processing pipeline with fixed expectations

### When to use Self-aligned BBoxes (Variants 1, 3)

- Parts are oriented in different directions
- You need the tightest possible bounding boxes
- Working with organic or artistic models

### When to use Globally-aligned BBoxes (Variants 2, 4)

- Parts should have consistent orientation
- Working with CAD models or architectural parts
- Visual consistency is more important than tight fit

---

## Advanced Options

### Bounding Box Script Options

```bash
# segment_with_bboxes.py (self-aligned)
python segment_with_bboxes.py \
    --input model.glb \
    --labels labels.npy \
    --output output.ply \
    --style all \              # solid, wireframe, transparent, or all
    --vertex-labels \          # Use vertex labels instead of face labels
    --no-filter \              # Disable outlier filtering
    --no-overlap-resolution    # Disable overlap resolution

# segment_with_bboxes_global.py (globally-aligned)
python segment_with_bboxes_global.py \
    --input model.glb \
    --labels labels.npy \
    --output output.ply \
    --style all \
    --method all_vertices \    # Global orientation method
    --no-filter \
    --no-overlap-resolution
```

### Clustering Script Options

```bash
python run_part_clustering.py \
    --source_dir data/glb_output \
    --root exp_results/partfield_features/glb_pipeline \
    --dump_dir exp_results/clustering/output \
    --max_num_clusters 20 \    # Maximum cluster count to generate
    --use_agglo False \        # True for agglomerative, False for K-Means
    --export_mesh True         # Export colored PLY files
```

---

## Troubleshooting

### Viewer won't start (WSL2)

```bash
# Use the wrapper script which sets up display
./run_viewer.sh <file>

# Or manually set display
export DISPLAY=:0
export LIBGL_ALWAYS_SOFTWARE=1
python viewer.py <file>
```

### No clustering labels found

Ensure PartField inference completed successfully and the feature files exist:
```bash
ls exp_results/partfield_features/glb_pipeline/part_feat_*.npy
```

### Bounding boxes look wrong

Try disabling filtering for raw boxes:
```bash
python segment_with_bboxes.py -i model.glb -l labels.npy -o output.ply --no-filter
```

### Memory issues with large models

Process models one at a time or reduce `--n_point_per_face` in inference config.

---

## File Reference

| File | Description |
|------|-------------|
| `partfield_inference.py` | Main inference script for feature extraction |
| `run_part_clustering.py` | Clustering script for segmentation |
| `segment_with_bboxes.py` | Self-aligned bounding box generation |
| `segment_with_bboxes_global.py` | Globally-aligned bounding box generation |
| `viewer.py` | Interactive 3D visualization |
| `run_viewer.sh` | Wrapper script for viewer (WSL2 compatible) |
| `pipeline_variant1_auto_self.sh` | Auto clusters + Self-aligned |
| `pipeline_variant2_auto_global.sh` | Auto clusters + Globally-aligned |
| `pipeline_variant3_fixed5_self.sh` | Fixed 5 clusters + Self-aligned |
| `pipeline_variant4_fixed5_global.sh` | Fixed 5 clusters + Globally-aligned |

---

## Shell Script Details

This section provides a line-by-line explanation of what each pipeline script does.

### Common Structure

All 4 variants follow the same structure with these differences:

| Variant | `max_num_clusters` | BBox Script | Output Suffix |
|---------|-------------------|-------------|---------------|
| 1 | 20 | `segment_with_bboxes.py` | `_bbox` |
| 2 | 20 | `segment_with_bboxes_global.py` | `_bbox_global` |
| 3 | 6 (for 5 clusters) | `segment_with_bboxes.py` | `_bbox` |
| 4 | 6 (for 5 clusters) | `segment_with_bboxes_global.py` | `_bbox_global` |

---

### Variant 1: `pipeline_variant1_auto_self.sh`

**Configuration**: Auto-segmented (2-20 clusters) + Self-aligned bounding boxes

```bash
#!/bin/bash
set -e  # Exit immediately if any command fails
```

#### Directory Setup
```bash
INPUT_DIR="data/glb_output"                                    # Where your .glb files are
EXP_DIR="exp_results"                                          # Base output directory
FEATURES_DIR="${EXP_DIR}/partfield_features/glb_pipeline"      # PartField features output
CLUSTERING_DIR="${EXP_DIR}/clustering/variant1_auto_self"      # Clustering results
BBOX_DIR="${EXP_DIR}/bboxes/variant1_auto_self"                # Bounding boxes output

# Create all required directories
mkdir -p "${FEATURES_DIR}"
mkdir -p "${CLUSTERING_DIR}/cluster_out"   # For .npy label files
mkdir -p "${CLUSTERING_DIR}/ply"           # For colored mesh visualizations
mkdir -p "${BBOX_DIR}"
```

#### Step 1: PartField Feature Extraction
```bash
python partfield_inference.py \
    --config-name demo \                              # Use demo configuration
    dataset.data_path="${INPUT_DIR}" \                # Input directory with .glb files
    output_dir="${EXP_DIR}" \                         # Base output directory
    result_name="partfield_features/glb_pipeline" \   # Subdirectory for results
    continue_ckpt="model/ckpt/epoch=00.ckpt"          # Pre-trained model checkpoint
```

**What this does**:
- Loads each `.glb` file from `data/glb_output/`
- Runs the PartField neural network to extract per-face feature embeddings
- Saves preprocessed mesh as `input_{hash}_0.ply`
- Saves PCA-colored visualization as `feat_pca_{hash}_0.ply`
- Saves raw features as `part_feat_{hash}_0.npy`

#### Step 2: Clustering
```bash
python run_part_clustering.py \
    --source_dir "${INPUT_DIR}" \         # Original .glb files location
    --root "${FEATURES_DIR}" \            # Where features were saved
    --dump_dir "${CLUSTERING_DIR}" \      # Where to save clustering results
    --max_num_clusters 20 \               # Generate clusters 2, 3, 4, ... 20
    --use_agglo False \                   # Use K-Means (faster than agglomerative)
    --export_mesh True                    # Export colored .ply visualizations
```

**What this does**:
- Loads feature vectors from `part_feat_{hash}_0.npy`
- Runs K-Means clustering for each cluster count from 2 to 20
- Saves cluster labels as `{hash}_0_{n}.npy` (n = cluster count, zero-padded)
- Saves colored segmentation mesh as `{hash}_0_{n}.ply`

#### Step 3: Bounding Box Generation
```bash
for glb_file in "${INPUT_DIR}"/*.glb; do
    if [ -f "$glb_file" ]; then
        basename=$(basename "$glb_file" .glb)

        # Find clustering label file (prefer 10 clusters as middle ground)
        label_file=$(find "${CLUSTERING_DIR}/cluster_out" -name "${basename}*_10.npy" | head -1)

        if [ -n "$label_file" ]; then
            python segment_with_bboxes.py \
                --input "$glb_file" \           # Original mesh
                --labels "$label_file" \        # Cluster labels
                --output "$output_file" \       # Output path
                --style all                     # Generate solid, wireframe, transparent
        fi
    fi
done
```

**What this does**:
- Iterates through each `.glb` file in the input directory
- Finds the corresponding 10-cluster label file (fallback to any available)
- Runs `segment_with_bboxes.py` which:
  - Loads mesh and labels
  - For each segment, extracts vertices belonging to that segment
  - Applies outlier filtering (removes sparse/distant points)
  - Computes PCA on segment points to find principal axes
  - Creates oriented bounding box aligned to segment's own PCA axes
  - Resolves overlapping boxes by shrinking
  - Exports solid, wireframe, and transparent visualizations

---

### Variant 2: `pipeline_variant2_auto_global.sh`

**Configuration**: Auto-segmented (2-20 clusters) + Globally-aligned bounding boxes

**Key difference from Variant 1** - Step 3 uses `segment_with_bboxes_global.py`:

```bash
python segment_with_bboxes_global.py \
    --input "$glb_file" \
    --labels "$label_file" \
    --output "$output_file" \
    --style all \
    --method all_vertices    # Compute global orientation from all mesh vertices
```

**What `segment_with_bboxes_global.py` does differently**:
1. **First**: Computes a single global PCA orientation from ALL mesh vertices
2. **Then**: For each segment:
   - Projects segment points onto the GLOBAL PCA axes (not segment-specific)
   - Computes axis-aligned bounds in global PCA space
   - Creates box using the shared global rotation matrix
3. All boxes end up with the same orientation, just different positions/sizes

---

### Variant 3: `pipeline_variant3_fixed5_self.sh`

**Configuration**: Fixed 5 clusters + Self-aligned bounding boxes

**Key differences**:

```bash
FIXED_CLUSTERS=5

# Step 2: Only generate up to 5+1=6 clusters (gives us 2,3,4,5)
python run_part_clustering.py \
    --max_num_clusters $((FIXED_CLUSTERS + 1)) \   # = 6
    ...

# Step 3: Only use the 5-cluster result
label_file=$(find "${CLUSTERING_DIR}/cluster_out" -name "${basename}*_05.npy" | head -1)
```

**What this does**:
- Clustering only generates 2, 3, 4, 5 cluster results (not 2-20)
- Bounding box step specifically looks for `*_05.npy` (5 clusters)
- Uses `segment_with_bboxes.py` for self-aligned boxes (same as Variant 1)

---

### Variant 4: `pipeline_variant4_fixed5_global.sh`

**Configuration**: Fixed 5 clusters + Globally-aligned bounding boxes

**Combines**:
- Fixed cluster count from Variant 3 (`max_num_clusters=6`, use `*_05.npy`)
- Global alignment from Variant 2 (`segment_with_bboxes_global.py`)

---

### Key Script Parameters Explained

#### `partfield_inference.py`

| Parameter | Description |
|-----------|-------------|
| `--config-name demo` | Load demo configuration from `partfield/config/` |
| `dataset.data_path` | Directory containing input meshes |
| `output_dir` | Base directory for all outputs |
| `result_name` | Subdirectory name for this run's results |
| `continue_ckpt` | Path to pre-trained model weights |

#### `run_part_clustering.py`

| Parameter | Description |
|-----------|-------------|
| `--source_dir` | Directory with original mesh files (for ID extraction) |
| `--root` | Directory containing PartField feature files |
| `--dump_dir` | Output directory for clustering results |
| `--max_num_clusters` | Maximum cluster count to generate (generates 2 to N) |
| `--use_agglo` | `True`=Agglomerative clustering, `False`=K-Means |
| `--export_mesh` | Whether to export colored PLY visualizations |

#### `segment_with_bboxes.py` (Self-aligned)

| Parameter | Description |
|-----------|-------------|
| `--input`, `-i` | Input mesh file (GLB, PLY, OBJ, etc.) |
| `--labels`, `-l` | NPY file with cluster labels |
| `--output`, `-o` | Output file path |
| `--style`, `-s` | `solid`, `wireframe`, `transparent`, or `all` |
| `--vertex-labels` | Treat labels as per-vertex (default: per-face) |
| `--no-filter` | Disable outlier filtering |
| `--no-overlap-resolution` | Disable box overlap resolution |

#### `segment_with_bboxes_global.py` (Globally-aligned)

Same as above, plus:

| Parameter | Description |
|-----------|-------------|
| `--method`, `-m` | `all_vertices` - PCA on all mesh vertices |

---

### Output File Naming Convention

#### Features (Step 1)
```
input_{hash}_0.ply        # Preprocessed input mesh
feat_pca_{hash}_0.ply     # PCA-colored feature visualization
part_feat_{hash}_0.npy    # Raw feature vectors (Nx64 or similar)
```

#### Clustering (Step 2)
```
cluster_out/{hash}_0_02.npy   # Labels for 2 clusters
cluster_out/{hash}_0_05.npy   # Labels for 5 clusters
cluster_out/{hash}_0_10.npy   # Labels for 10 clusters
ply/{hash}_0_02.ply           # Colored mesh (2 clusters)
ply/{hash}_0_10.ply           # Colored mesh (10 clusters)
```

#### Bounding Boxes (Step 3)
```
{hash}_10_bbox_solid.ply        # Solid boxes (Variant 1, 3)
{hash}_10_bbox_wireframe.ply    # Wireframe boxes
{hash}_10_bbox_transparent.ply  # Transparent boxes
{hash}_10_bbox_info.txt         # Box parameters (center, dims, rotation)

{hash}_10_bbox_global_solid.ply # Global-aligned solid boxes (Variant 2, 4)
```

---

### Visualization Commands in Scripts

Each script includes visualization hints after each step:

```bash
# After Step 1 (Features)
./run_viewer.sh exp_results/partfield_features/glb_pipeline/feat_pca_*.ply

# After Step 2 (Clustering)
./run_viewer.sh data/glb_output/model.glb  # Press C/V to cycle clusters

# After Step 3 (Bounding Boxes)
./run_viewer.sh exp_results/bboxes/variant1_auto_self/*_bbox_solid.ply
```

---

### Error Handling

```bash
set -e  # At top of script - exit on any error
```

Each step checks for file existence before processing:
```bash
if [ -f "$glb_file" ]; then        # Check file exists
    ...
    if [ -n "$label_file" ]; then  # Check label file was found
        ...
    else
        echo "WARNING: No clustering labels found for ${basename}"
    fi
fi
```
