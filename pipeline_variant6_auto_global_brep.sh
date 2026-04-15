#!/bin/bash
# ==============================================================================
# VARIANT 6: Auto-segmented clusters + Globally-aligned BREP bounding boxes (STEP)
# ==============================================================================
# - Clustering: Generates multiple cluster counts (2 to 20), auto-selects best
# - BBox: All segments share a single global PCA orientation (consistent alignment)
# - Output: STEP CAD files (BREP solid geometry)
# ==============================================================================

set -e

INPUT_DIR="data/glb_output"
EXP_DIR="exp_results"
FEATURES_DIR="${EXP_DIR}/partfield_features/glb_pipeline"
CLUSTERING_DIR="${EXP_DIR}/clustering/variant6_auto_global_brep"
BREP_DIR="${EXP_DIR}/brep/variant6_auto_global_brep"

echo "============================================================"
echo "VARIANT 6: Auto-segmented + Globally-aligned BREP (STEP)"
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
# STEP 2: Run clustering (AUTO mode)
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

# ==============================================================================
# STEP 3: Generate BREP bounding boxes with GLOBAL alignment (STEP output)
# ==============================================================================
echo ""
echo "STEP 3: Generating globally-aligned BREP bounding boxes..."
echo "-----------------------------------------------------------"

declare -a PROCESSED_MODELS=()
declare -a PROCESSED_BREP_FILES=()
declare -a PROCESSED_CLUSTER_COUNTS=()

for glb_file in "${INPUT_DIR}"/*.glb; do
    if [ -f "$glb_file" ]; then
        glb_basename=$(basename "$glb_file" .glb)
        basename=$(echo "$glb_basename" | rev | cut -d'.' -f1 | rev)
        if [ "$basename" = "$glb_basename" ]; then
            basename="$glb_basename"
        fi
        echo "Processing: ${glb_basename} (model_id: ${basename})"

        best_n_file="${CLUSTERING_DIR}/cluster_out/${basename}_0_best_n.npy"
        if [ -f "$best_n_file" ]; then
            best_n=$(python3 -c "import numpy as np; print(int(np.load('${best_n_file}')[0]))")
        else
            best_n=10
        fi
        echo "  Selected ${best_n} clusters"

        label_file="${CLUSTERING_DIR}/cluster_out/${basename}_0_$(printf '%02d' ${best_n}).npy"

        if [ -f "$label_file" ]; then
            output_file="${BREP_DIR}/${basename}_${best_n}_brep_global.step"

            python brep_generator.py \
                --input "$glb_file" \
                --labels "$label_file" \
                --output "$output_file" \
                --mode bbox --alignment global

            PROCESSED_MODELS+=("${basename}")
            PROCESSED_BREP_FILES+=("${output_file}")
            PROCESSED_CLUSTER_COUNTS+=("${best_n}")
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
        cluster_count="${PROCESSED_CLUSTER_COUNTS[$i]}"

        echo ""
        echo "--- ${model} (${cluster_count} clusters) ---"

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
echo "VARIANT 6 COMPLETE"
echo "============================================================"
echo ""
echo "Output: ${BREP_DIR}"
echo ""
echo "============================================================"
