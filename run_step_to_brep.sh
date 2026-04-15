#!/bin/bash
# ==============================================================================
# STEP → BREP BBox Pipeline
# ==============================================================================
#
# End-to-end: takes a STEP CAD file, runs PartField segmentation, and outputs
# a new STEP file filled with colored oriented bounding boxes.
#
#   Input STEP → tessellate → PartField features → clustering → BREP bboxes → Output STEP
#
# Usage:
#   ./run_step_to_brep.sh -i model.step -o model_bboxes.step
#   ./run_step_to_brep.sh -i model.step -o model_bboxes.step --clusters 5
#   ./run_step_to_brep.sh -i model.step                      # output: model_bboxes.step
#   ./run_step_to_brep.sh --input-dir steps/ --output-dir brep_out/
#
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Dependency checks ────────────────────────────────────────────────────────

check_dep() {
    local name="$1" cmd="$2" hint="$3"
    if ! python3 -c "$cmd" 2>/dev/null; then
        echo "ERROR: $name is not installed"
        echo "  Install with: $hint"
        exit 1
    fi
}

echo "Checking dependencies..."

check_dep "pythonocc-core" \
    "from OCC.Core.STEPControl import STEPControl_Reader" \
    "conda install -c conda-forge pythonocc-core"

check_dep "trimesh" \
    "import trimesh" \
    "pip install trimesh"

check_dep "numpy" \
    "import numpy" \
    "pip install numpy"

# PartField model checkpoint
CKPT="model/model_objaverse.ckpt"
if [ ! -f "$CKPT" ]; then
    echo "WARNING: PartField checkpoint not found at $CKPT"
    echo "  Download from: https://huggingface.co/mikaelaangel/partfield-ckpt/blob/main/model_objaverse.ckpt"
    echo "  Place in: model/"
fi

# Quick GPU check (non-fatal)
if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
    echo "  GPU: $GPU_NAME"
else
    echo "  WARNING: No CUDA GPU detected – PartField inference will be very slow or fail"
fi

echo "  All dependencies OK"
echo ""

# ── Run pipeline ─────────────────────────────────────────────────────────────

python3 "$SCRIPT_DIR/step_to_brep.py" "$@"
PIPELINE_EXIT=$?

if [ $PIPELINE_EXIT -ne 0 ]; then
    echo "Pipeline failed with exit code $PIPELINE_EXIT"
    exit $PIPELINE_EXIT
fi

# ── Visualize original + generated STEP files ────────────────────────────────

# Parse -i/--input and -o/--output from the arguments
INPUT_STEP=""
OUTPUT_STEP=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        -i|--input)
            INPUT_STEP="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_STEP="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# Derive default output name if not specified
if [ -n "$INPUT_STEP" ] && [ -z "$OUTPUT_STEP" ]; then
    base=$(basename "$INPUT_STEP")
    OUTPUT_STEP="${base%.*}_bboxes.step"
fi

if [ -n "$INPUT_STEP" ] && [ -f "$INPUT_STEP" ]; then
    echo ""
    echo "========================================================="
    echo "Launching BREP viewer for comparison..."
    echo "========================================================="
    VISUALIZE_ARGS=()
    echo "  Original STEP: ${INPUT_STEP}"
    VISUALIZE_ARGS+=("$INPUT_STEP")
    if [ -n "$OUTPUT_STEP" ] && [ -f "$OUTPUT_STEP" ]; then
        echo "  Generated BREP: ${OUTPUT_STEP}"
        VISUALIZE_ARGS+=("$OUTPUT_STEP")
    fi
    "$SCRIPT_DIR/run_brep_viewer.sh" --visualize "${VISUALIZE_ARGS[@]}"
else
    echo ""
    echo "Skipping visualization (no input STEP file found)."
fi
