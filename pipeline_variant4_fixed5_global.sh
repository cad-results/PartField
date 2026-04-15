#!/bin/bash
# ==============================================================================
# VARIANT 4: Fixed 5 clusters + Globally-aligned bounding boxes
# ==============================================================================
# - Clustering: Exactly 5 clusters (fixed)
# - BBox: All segments share a single global PCA orientation (consistent alignment)
# ==============================================================================

set -e

INPUT_DIR="data/glb_output"
EXP_DIR="exp_results"
FEATURES_DIR="${EXP_DIR}/partfield_features/glb_pipeline"
CLUSTERING_DIR="${EXP_DIR}/clustering/variant4_fixed5_global"
BBOX_DIR="${EXP_DIR}/bboxes/variant4_fixed5_global"

FIXED_CLUSTERS=5

echo "=================================================="
echo "VARIANT 4: Fixed 5 clusters + Globally-aligned BBox"
echo "=================================================="

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
# STEP 2: Run clustering (FIXED mode - exactly 5 clusters)
# ==============================================================================
echo ""
echo "STEP 2: Running clustering (FIXED mode: exactly ${FIXED_CLUSTERS} clusters)..."
echo "-------------------------------------------------------------------------------"

python run_part_clustering.py \
    --source_dir "${INPUT_DIR}" \
    --root "${FEATURES_DIR}" \
    --dump_dir "${CLUSTERING_DIR}" \
    --max_num_clusters $((FIXED_CLUSTERS + 1)) \
    --use_agglo False \
    --export_mesh True

echo "Clustering results saved to: ${CLUSTERING_DIR}"

# ==============================================================================
# STEP 2b: Visualize clustering results
# ==============================================================================
echo ""
echo "STEP 2b: Visualizing clustering results (${FIXED_CLUSTERS} clusters)..."
echo "------------------------------------------------------------------------"

CLUSTER_PLY=$(find "${CLUSTERING_DIR}/ply" -name "*_05.ply" 2>/dev/null | head -5)
if [ -n "${CLUSTER_PLY}" ]; then
    echo "Generated ${FIXED_CLUSTERS}-cluster visualizations:"
    for f in ${CLUSTER_PLY}; do
        echo "  $(basename $f)"
    done
    echo ""
    echo "To view interactively:"
    FIRST_PLY=$(echo "${CLUSTER_PLY}" | head -1)
    echo "  ./run_viewer.sh ${FIRST_PLY}"
fi

# ==============================================================================
# STEP 3: Generate GLOBALLY-ALIGNED bounding boxes (shared global PCA)
# ==============================================================================
echo ""
echo "STEP 3: Generating globally-aligned bounding boxes for ${FIXED_CLUSTERS} clusters..."
echo "--------------------------------------------------------------------------------------"

# Arrays to track processed files for summary
declare -a PROCESSED_MODELS=()
declare -a PROCESSED_GLB_FILES=()
declare -a PROCESSED_CLUSTER_PLYS=()
declare -a PROCESSED_BBOX_FILES=()

# Process each GLB file with its 5-cluster result
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

        # Find the FIXED cluster label file (exactly 5 clusters)
        label_file=$(find "${CLUSTERING_DIR}/cluster_out" -name "${basename}*_05.npy" 2>/dev/null | head -1)

        if [ -z "$label_file" ]; then
            label_file=$(find "${CLUSTERING_DIR}/cluster_out" -name "${basename}*_5.npy" 2>/dev/null | head -1)
        fi

        if [ -n "$label_file" ]; then
            output_file="${BBOX_DIR}/${basename}_${FIXED_CLUSTERS}clusters_bbox_global.ply"
            cluster_ply="${CLUSTERING_DIR}/ply/${basename}_0_05.ply"

            echo "  Using labels: $(basename $label_file)"
            echo "  Output: ${output_file}"

            # Use segment_with_bboxes_global.py for globally-aligned boxes
            python segment_with_bboxes_global.py \
                --input "$glb_file" \
                --labels "$label_file" \
                --output "$output_file" \
                --style all \
                --method all_vertices

            # Track for summary
            PROCESSED_MODELS+=("${basename}")
            PROCESSED_GLB_FILES+=("${glb_file}")
            PROCESSED_CLUSTER_PLYS+=("${cluster_ply}")
            PROCESSED_BBOX_FILES+=("${output_file}")
        else
            echo "  WARNING: No ${FIXED_CLUSTERS}-cluster labels found for ${basename}"
        fi
    fi
done

echo "Bounding boxes saved to: ${BBOX_DIR}"

# ==============================================================================
# SUMMARY WITH PERSONALIZED VIEWER COMMANDS
# ==============================================================================
echo ""
echo "=================================================="
echo "VARIANT 4 COMPLETE"
echo "=================================================="
echo ""
echo "Output locations:"
echo "  Features:   ${FEATURES_DIR}"
echo "  Clustering: ${CLUSTERING_DIR}"
echo "  BBoxes:     ${BBOX_DIR}"
echo ""
echo "Configuration:"
echo "  Clusters: FIXED at ${FIXED_CLUSTERS}"
echo "  BBox:     Globally-aligned (all boxes share same PCA orientation)"
echo ""
echo "Key difference from Variant 3:"
echo "  All bounding boxes share a GLOBAL orientation computed from the"
echo "  entire mesh via PCA. This is useful for CAD models where you want"
echo "  consistent part alignment."
echo ""

# Generate personalized viewer commands for each processed model
if [ ${#PROCESSED_MODELS[@]} -gt 0 ]; then
    echo "=================================================="
    echo "VIEWER COMMANDS (copy & paste to visualize)"
    echo "=================================================="

    for i in "${!PROCESSED_MODELS[@]}"; do
        model="${PROCESSED_MODELS[$i]}"
        glb_file="${PROCESSED_GLB_FILES[$i]}"
        cluster_ply="${PROCESSED_CLUSTER_PLYS[$i]}"
        bbox_file="${PROCESSED_BBOX_FILES[$i]}"

        # Find the PCA feature visualization file
        pca_file="${FEATURES_DIR}/feat_pca_${model}_0.ply"
        if [ ! -f "$pca_file" ]; then
            pca_file=$(find "${FEATURES_DIR}" -name "feat_pca_${model}*.ply" 2>/dev/null | head -1)
        fi

        echo ""
        echo "--- ${model} (${FIXED_CLUSTERS} clusters) ---"
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
        echo "# Step 2: View clustering result (${FIXED_CLUSTERS} clusters)"
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
    echo "=================================================="
else
    echo "No models were processed."
fi

echo ""
echo "=================================================="
