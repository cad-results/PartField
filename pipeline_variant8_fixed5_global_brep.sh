#!/bin/bash
# ==============================================================================
# VARIANT 8: Fixed 5 clusters + Globally-aligned BREP bounding boxes (STEP)
# ==============================================================================
# - Clustering: Exactly 5 clusters (fixed)
# - BBox: All segments share a single global PCA orientation (consistent alignment)
# - Output: STEP CAD files (BREP solid geometry)
# ==============================================================================

set -e

INPUT_DIR="data/glb_output"
EXP_DIR="exp_results"
FEATURES_DIR="${EXP_DIR}/partfield_features/glb_pipeline"
CLUSTERING_DIR="${EXP_DIR}/clustering/variant8_fixed5_global_brep"
BREP_DIR="${EXP_DIR}/brep/variant8_fixed5_global_brep"

FIXED_CLUSTERS=5

echo "============================================================"
echo "VARIANT 8: Fixed ${FIXED_CLUSTERS} clusters + Globally-aligned BREP (STEP)"
echo "============================================================"

mkdir -p "${FEATURES_DIR}"
mkdir -p "${CLUSTERING_DIR}/cluster_out"
mkdir -p "${CLUSTERING_DIR}/ply"
mkdir -p "${BREP_DIR}"

# ==============================================================================
# STEP 1: Run PartField inference
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
# STEP 2: Run clustering (FIXED 5 clusters)
# ==============================================================================
echo ""
echo "STEP 2: Running clustering with fixed ${FIXED_CLUSTERS} clusters..."
echo "------------------------------------------------------------"

python run_part_clustering.py \
    --source_dir "${INPUT_DIR}" \
    --root "${FEATURES_DIR}" \
    --dump_dir "${CLUSTERING_DIR}" \
    --max_num_clusters ${FIXED_CLUSTERS} \
    --use_agglo False \
    --export_mesh True

echo "Clustering results saved to: ${CLUSTERING_DIR}"

# ==============================================================================
# STEP 3: Generate globally-aligned BREP bounding boxes (STEP output)
# ==============================================================================
echo ""
echo "STEP 3: Generating globally-aligned BREP bounding boxes..."
echo "-----------------------------------------------------------"

declare -a PROCESSED_MODELS=()
declare -a PROCESSED_BREP_FILES=()

for glb_file in "${INPUT_DIR}"/*.glb; do
    if [ -f "$glb_file" ]; then
        glb_basename=$(basename "$glb_file" .glb)
        basename=$(echo "$glb_basename" | rev | cut -d'.' -f1 | rev)
        if [ "$basename" = "$glb_basename" ]; then
            basename="$glb_basename"
        fi
        echo "Processing: ${glb_basename}"

        label_file="${CLUSTERING_DIR}/cluster_out/${basename}_0_$(printf '%02d' ${FIXED_CLUSTERS}).npy"

        if [ -f "$label_file" ]; then
            output_file="${BREP_DIR}/${basename}_${FIXED_CLUSTERS}_brep_global.step"

            python brep_generator.py \
                --input "$glb_file" \
                --labels "$label_file" \
                --output "$output_file" \
                --mode bbox --alignment global

            PROCESSED_MODELS+=("${basename}")
            PROCESSED_BREP_FILES+=("${output_file}")
        else
            echo "  WARNING: No labels found for ${basename}"
        fi
    fi
done

echo "BREP files saved to: ${BREP_DIR}"

# ==============================================================================
# STEP 4: Visualize original STEP + generated BREP STEP in viewer
# ==============================================================================
echo ""
echo "STEP 4: Launching BREP viewer for comparison..."
echo "-------------------------------------------------"

STEP_DIR="data/stepfiles"

if [ ${#PROCESSED_MODELS[@]} -gt 0 ]; then
    for i in "${!PROCESSED_MODELS[@]}"; do
        model="${PROCESSED_MODELS[$i]}"
        brep_file="${PROCESSED_BREP_FILES[$i]}"

        echo ""
        echo "--- ${model} (${FIXED_CLUSTERS} clusters) ---"

        # Find original STEP file for this model
        ORIGINAL_STEP=""
        if [ -d "${STEP_DIR}" ]; then
            for ext in step stp STEP STP; do
                for step_file in "${STEP_DIR}"/*.${ext}; do
                    if [ -f "$step_file" ]; then
                        step_basename=$(basename "$step_file" ".${ext}")
                        step_norm=$(echo "$step_basename" | tr -d ' ')
                        model_norm=$(echo "$model" | tr -d ' ')
                        if [[ "$step_norm" == *"$model_norm"* ]] || [[ "$model_norm" == *"$step_norm"* ]]; then
                            ORIGINAL_STEP="$step_file"
                            break 2
                        fi
                    fi
                done
            done
        fi

        if [ -n "$ORIGINAL_STEP" ] && [ -f "$brep_file" ]; then
            echo "  Original STEP: ${ORIGINAL_STEP}"
            echo "  Generated BREP: ${brep_file}"
            ./run_brep_viewer.sh --visualize "$ORIGINAL_STEP" "$brep_file"
        elif [ -f "$brep_file" ]; then
            echo "  No original STEP found in ${STEP_DIR}, viewing generated only"
            echo "  Generated BREP: ${brep_file}"
            ./run_brep_viewer.sh --visualize "$brep_file"
        else
            echo "  WARNING: Generated BREP file not found: ${brep_file}"
        fi
    done
else
    echo "  No models were processed, skipping visualization."
fi

# ==============================================================================
# SUMMARY
# ==============================================================================
echo ""
echo "============================================================"
echo "VARIANT 8 COMPLETE"
echo "============================================================"
echo ""
echo "Output: ${BREP_DIR}"
echo ""
echo "============================================================"
