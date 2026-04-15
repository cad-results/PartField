#!/bin/bash
# ==============================================================================
# VARIANT 1: Auto-segmented clusters + Self-aligned bounding boxes
# ==============================================================================
# - Clustering: Generates multiple cluster counts (2 to 20), user can pick best
# - BBox: Each segment gets its own PCA orientation (tighter fit)
# ==============================================================================

set -e

INPUT_DIR="data/glb_output"
EXP_DIR="exp_results"
FEATURES_DIR="${EXP_DIR}/partfield_features/glb_pipeline"
CLUSTERING_DIR="${EXP_DIR}/clustering/variant1_auto_self"
BBOX_DIR="${EXP_DIR}/bboxes/variant1_auto_self"

echo "=============================================="
echo "VARIANT 1: Auto-segmented + Self-aligned BBox"
echo "=============================================="

# Create output directories
mkdir -p "${FEATURES_DIR}"
mkdir -p "${CLUSTERING_DIR}/cluster_out"
mkdir -p "${CLUSTERING_DIR}/ply"
mkdir -p "${BBOX_DIR}"

# ==============================================================================
# STEP 1: Run PartField inference to extract features
# ==============================================================================
echo ""
echo "STEP 1: Running PartField inference..."
echo "--------------------------------------"

python partfield_inference.py \
    --config-file configs/final/demo.yaml \
    --opts \
    dataset.data_path "${INPUT_DIR}" \
    output_dir "${EXP_DIR}" \
    result_name "partfield_features/glb_pipeline" \
    continue_ckpt "model/model_objaverse.ckpt"

echo "Features extracted to: ${FEATURES_DIR}"

# ==============================================================================
# STEP 1b: Visualize PartField features (PCA colors)
# ==============================================================================
echo ""
echo "STEP 1b: Visualizing PartField features..."
echo "------------------------------------------"

# Find all PCA feature files and show first one
PCA_FILES=$(find "${FEATURES_DIR}" -name "feat_pca_*.ply" 2>/dev/null | head -5)
if [ -n "${PCA_FILES}" ]; then
    echo "Generated PCA feature visualizations:"
    echo "${PCA_FILES}"
    echo ""
    echo "To view interactively, run:"
    FIRST_PCA=$(echo "${PCA_FILES}" | head -1)
    echo "  ./run_viewer.sh ${FIRST_PCA}"
fi

# ==============================================================================
# STEP 2: Run clustering (AUTO mode - generates 2 to 20 clusters + auto-select best)
# ==============================================================================
echo ""
echo "STEP 2: Running clustering with auto-selection..."
echo "--------------------------------------------------"

python run_part_clustering.py \
    --source_dir "${INPUT_DIR}" \
    --root "${FEATURES_DIR}" \
    --dump_dir "${CLUSTERING_DIR}" \
    --max_num_clusters 20 \
    --use_agglo False \
    --export_mesh True \
    --auto_select \
    --min_preferred_clusters 3 \
    --max_preferred_clusters 15

echo "Clustering results saved to: ${CLUSTERING_DIR}"
echo "Best clustering auto-selected using silhouette/Davies-Bouldin/Calinski-Harabasz metrics"

# ==============================================================================
# STEP 2b: Visualize clustering results
# ==============================================================================
echo ""
echo "STEP 2b: Visualizing clustering results..."
echo "------------------------------------------"

CLUSTER_PLY=$(find "${CLUSTERING_DIR}/ply" -name "*.ply" 2>/dev/null | head -5)
if [ -n "${CLUSTER_PLY}" ]; then
    echo "Generated clustering visualizations:"
    ls -la "${CLUSTERING_DIR}/ply/" | head -10
    echo ""
    echo "To view interactively and cycle through cluster counts:"
    FIRST_INPUT=$(find "${INPUT_DIR}" -name "*.glb" | head -1)
    if [ -n "${FIRST_INPUT}" ]; then
        echo "  ./run_viewer.sh ${FIRST_INPUT}"
        echo "  Press 'C'/'V' to cycle through different cluster counts"
    fi
fi

# ==============================================================================
# STEP 3: Generate SELF-ALIGNED bounding boxes (per-segment PCA)
# ==============================================================================
echo ""
echo "STEP 3: Generating self-aligned bounding boxes..."
echo "--------------------------------------------------"

# Arrays to track processed files for summary
declare -a PROCESSED_MODELS=()
declare -a PROCESSED_GLB_FILES=()
declare -a PROCESSED_CLUSTER_COUNTS=()
declare -a PROCESSED_CLUSTER_PLYS=()
declare -a PROCESSED_BBOX_FILES=()

# Process each GLB file with its clustering results
for glb_file in "${INPUT_DIR}"/*.glb; do
    if [ -f "$glb_file" ]; then
        glb_basename=$(basename "$glb_file" .glb)
        # Extract model ID the same way Python does: take everything after the last '.' (excluding extension)
        # This handles filenames like "3dpea.com_assembly1.glb" -> "com_assembly1"
        basename=$(echo "$glb_basename" | rev | cut -d'.' -f1 | rev)
        # If no dots in the name, use the full basename
        if [ "$basename" = "$glb_basename" ]; then
            basename="$glb_basename"
        fi
        echo "Processing: ${glb_basename} (model_id: ${basename})"

        # First, try to use the auto-selected best clustering from _best_n.npy
        best_n_file="${CLUSTERING_DIR}/cluster_out/${basename}_0_best_n.npy"
        if [ -f "$best_n_file" ]; then
            # Read the best cluster count from the numpy file
            best_n=$(python3 -c "import numpy as np; print(int(np.load('${best_n_file}')[0]))")
            echo "  Found saved best: ${best_n} clusters"
        else
            # Compute best cluster count on-the-fly using FAST silhouette score
            # Uses sampling (~5000 points) + early stopping for ~100x speedup
            echo "  Computing optimal cluster count using fast silhouette score..."
            best_n=$(python3 << PYEOF
import numpy as np
import os
from sklearn.metrics import silhouette_score

# Performance settings for ~100x speedup
SAMPLE_SIZE = 5000
EARLY_STOP_PATIENCE = 3
IMPROVEMENT_THRESHOLD = 0.01

# Load features
features_file = "${FEATURES_DIR}/part_feat_${basename}_0.npy"
if not os.path.exists(features_file):
    features_file = "${FEATURES_DIR}/part_feat_${basename}_0_batch.npy"

if not os.path.exists(features_file):
    print(10)  # fallback
    exit()

features = np.load(features_file)
features = features / np.linalg.norm(features, axis=-1, keepdims=True)
n_samples = len(features)

# Create sample indices once (stratified sampling done per-k)
np.random.seed(42)

def silhouette_sampled(features, labels, sample_size=5000):
    """Fast silhouette score using stratified sampling."""
    n = len(labels)
    if n <= sample_size:
        return silhouette_score(features, labels)

    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return -1.0

    indices = []
    samples_per_cluster = max(10, sample_size // len(unique_labels))
    for label in unique_labels:
        cluster_idx = np.where(labels == label)[0]
        n_take = min(len(cluster_idx), samples_per_cluster)
        if n_take > 0:
            indices.extend(np.random.choice(cluster_idx, n_take, replace=False))

    indices = np.array(indices)
    return silhouette_score(features[indices], labels[indices])

best_score = -2
best_k = 10
no_improve = 0

for k in range(2, 20):
    label_file = f"${CLUSTERING_DIR}/cluster_out/${basename}_0_{k:02d}.npy"
    if os.path.exists(label_file):
        labels = np.load(label_file).flatten().astype(int)
        unique_labels = np.unique(labels)
        if len(unique_labels) >= 2:
            try:
                score = silhouette_sampled(features, labels, SAMPLE_SIZE)
                if score > best_score + IMPROVEMENT_THRESHOLD:
                    best_score = score
                    best_k = k
                    no_improve = 0
                else:
                    no_improve += 1

                # Early stopping
                if no_improve >= EARLY_STOP_PATIENCE:
                    break
            except:
                pass

print(best_k)
PYEOF
)
        fi

        echo "  Selected ${best_n} clusters for ${basename}"

        # Get the label file for the selected cluster count
        label_file="${CLUSTERING_DIR}/cluster_out/${basename}_0_$(printf '%02d' ${best_n}).npy"

        if [ -f "$label_file" ]; then
            # Use actual best_n in filename (not extracted from filename)
            output_file="${BBOX_DIR}/${basename}_${best_n}_bbox.ply"
            cluster_ply="${CLUSTERING_DIR}/ply/${basename}_0_$(printf '%02d' ${best_n}).ply"

            echo "  Using labels: $(basename $label_file)"
            echo "  Output: ${output_file}"

            python segment_with_bboxes.py \
                --input "$glb_file" \
                --labels "$label_file" \
                --output "$output_file" \
                --style all

            # Track for summary
            PROCESSED_MODELS+=("${basename}")
            PROCESSED_GLB_FILES+=("${glb_file}")
            PROCESSED_CLUSTER_COUNTS+=("${best_n}")
            PROCESSED_CLUSTER_PLYS+=("${cluster_ply}")
            PROCESSED_BBOX_FILES+=("${output_file}")
        else
            echo "  WARNING: No clustering labels found for ${basename} with ${best_n} clusters"
        fi
    fi
done

echo "Bounding boxes saved to: ${BBOX_DIR}"

# ==============================================================================
# SUMMARY WITH PERSONALIZED VIEWER COMMANDS
# ==============================================================================
echo ""
echo "=============================================="
echo "VARIANT 1 COMPLETE"
echo "=============================================="
echo ""
echo "Output locations:"
echo "  Features:   ${FEATURES_DIR}"
echo "  Clustering: ${CLUSTERING_DIR}"
echo "  BBoxes:     ${BBOX_DIR}"
echo ""

# Generate personalized viewer commands for each processed model
if [ ${#PROCESSED_MODELS[@]} -gt 0 ]; then
    echo "=============================================="
    echo "VIEWER COMMANDS (copy & paste to visualize)"
    echo "=============================================="

    for i in "${!PROCESSED_MODELS[@]}"; do
        model="${PROCESSED_MODELS[$i]}"
        glb_file="${PROCESSED_GLB_FILES[$i]}"
        cluster_count="${PROCESSED_CLUSTER_COUNTS[$i]}"
        cluster_ply="${PROCESSED_CLUSTER_PLYS[$i]}"
        bbox_file="${PROCESSED_BBOX_FILES[$i]}"

        # Find the PCA feature visualization file
        pca_file="${FEATURES_DIR}/feat_pca_${model}_0.ply"
        if [ ! -f "$pca_file" ]; then
            pca_file=$(find "${FEATURES_DIR}" -name "feat_pca_${model}*.ply" 2>/dev/null | head -1)
        fi

        echo ""
        echo "--- ${model} (${cluster_count} clusters) ---"
        echo ""
        echo "# Step 1: View original model"
        echo "./run_viewer.sh ${glb_file}"
        echo ""
        echo "# Step 1b: View PartField features (PCA colors)"
        if [ -n "$pca_file" ] && [ -f "$pca_file" ]; then
            echo "./run_viewer.sh ${pca_file}"
        else
            echo "./run_viewer.sh ${FEATURES_DIR}/feat_pca_${model}_0.ply"
        fi
        echo ""
        echo "# Step 2: View clustering result (${cluster_count} clusters)"
        echo "./run_viewer.sh ${cluster_ply}"
        echo ""
        echo "# Step 3: View bounding boxes (solid)"
        echo "./run_viewer.sh ${bbox_file%.ply}_solid.ply"
        echo ""
        echo "# Step 3: View bounding boxes (wireframe)"
        echo "./run_viewer.sh ${bbox_file%.ply}_wireframe.ply"
        echo ""
        echo "# Step 3: View bounding boxes (transparent)"
        echo "./run_viewer.sh ${bbox_file%.ply}_transparent.ply"
    done

    echo ""
    echo "=============================================="
else
    echo "No models were processed."
fi

echo ""
echo "=============================================="
