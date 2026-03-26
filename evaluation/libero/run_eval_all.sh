#!/bin/bash
# =============================================================================
# SimVLA LIBERO Evaluation Script (parallel 4 task suites)
# =============================================================================

# Don't use set -e with parallel background jobs
# set -e

# =============================================================================
# LIBERO Environment Setup
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LIBERO_ROOT="${SCRIPT_DIR}/LIBERO"
export PYTHONPATH="${LIBERO_ROOT}:${PYTHONPATH}"

echo "LIBERO Environment:"
echo "   LIBERO_ROOT: $LIBERO_ROOT"
echo "   PYTHONPATH: $PYTHONPATH"
echo ""

# Default arguments
PORT=${1:-8089}
NUM_TRIALS=${2:-10}
OUTPUT_PREFIX=${3:-"eval_simvla"}

# GPU assignment per suite: spatial=2, object=2, goal=1, 10=3
GPU_SPATIAL=${GPU_SPATIAL:-2}
GPU_OBJECT=${GPU_OBJECT:-2}
GPU_GOAL=${GPU_GOAL:-1}
GPU_10=${GPU_10:-3}

# Output directory (with timestamp to preserve previous runs)
TIMESTAMP=$(date +%m%d_%H%M%S)
OUTPUT_DIR="./eval_simvla_${PORT}_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

echo "Starting LIBERO evaluation (PARALLEL)..."
echo "   Server Port  : $PORT"
echo "   Num Trials   : $NUM_TRIALS"
echo "   Output Prefix: $OUTPUT_PREFIX"
echo "   Output Dir   : $OUTPUT_DIR"
echo "   GPU mapping  : spatial=$GPU_SPATIAL, object=$GPU_OBJECT, goal=$GPU_GOAL, 10=$GPU_10"
echo ""

# Suites and their GPU assignments
SUITES=("libero_spatial" "libero_object" "libero_goal" "libero_10")
GPUS=("$GPU_SPATIAL" "$GPU_OBJECT" "$GPU_GOAL" "$GPU_10")
PIDS=()

echo "Launching all 4 suites in parallel..."
echo ""

for i in "${!SUITES[@]}"; do
    SUITE="${SUITES[$i]}"
    GPU="${GPUS[$i]}"
    SUITE_SHORT="${SUITE#libero_}"
    LOG_FILE="${OUTPUT_PREFIX}_${SUITE_SHORT}.txt"

    echo "[$(( i + 1 ))/4] Launching ${SUITE} on GPU ${GPU} → ${LOG_FILE}"

    CUDA_VISIBLE_DEVICES=$GPU python -u libero_client.py \
        --host 127.0.0.1 \
        --port $PORT \
        --client_type websocket \
        --task_suite "$SUITE" \
        --num_trials $NUM_TRIALS \
        --video_out "$OUTPUT_DIR" \
        > >(tee "${LOG_FILE}") 2>&1 &

    PIDS+=($!)
done

echo ""
echo "All 4 suites launched. PIDs: ${PIDS[*]}"
echo "Waiting for all to finish..."
echo ""

# Wait for all and track results
FAILED=0
for i in "${!SUITES[@]}"; do
    SUITE="${SUITES[$i]}"
    PID="${PIDS[$i]}"
    if wait "$PID"; then
        echo "✓ ${SUITE} (PID $PID) finished successfully."
    else
        echo "✗ ${SUITE} (PID $PID) failed with exit code $?."
        FAILED=$((FAILED + 1))
    fi
done

echo ""
if [ $FAILED -eq 0 ]; then
    echo "All evaluations completed successfully!"
else
    echo "WARNING: $FAILED suite(s) failed."
fi

echo ""
echo "Results summary:"
echo "=========================================="
for suite in spatial object goal 10; do
    file="${OUTPUT_PREFIX}_${suite}.txt"
    if [ -f "$file" ]; then
        echo "--- $suite ---"
        grep -E "Success Rate|Average" "$file" 2>/dev/null || echo "  (see $file)"
    fi
done
echo "=========================================="
