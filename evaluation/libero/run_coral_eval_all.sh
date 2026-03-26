#!/bin/bash
# =============================================================================
# CORAL LIBERO Evaluation Script (4 servers, 4 parallel evaluations)
#
# Deploys one dedicated MoE server per suite to avoid LoRA thrashing,
# then runs all 4 evaluations in parallel.
#
# Server uses simvla conda env, eval client uses libero conda env.
#
# Usage:
#   # Auto-start 4 servers on GPU 0-3, ports 8089-8092, then evaluate
#   bash run_coral_eval_all.sh
#
#   # Custom GPUs & ports
#   SERVER_GPUS="0 1 2 3" \
#   EVAL_GPUS="4 5 6 7" \
#   BASE_PORT=8089 \
#   bash run_coral_eval_all.sh
#
#   # Servers already running on ports 8089-8092
#   SKIP_SERVER=1 BASE_PORT=8089 bash run_coral_eval_all.sh
#
#   # Specific LoRA checkpoint step (e.g. step500)
#   LORA_DIR=./lora_adapters/coral_libero_r16 LORA_STEP=500 bash run_coral_eval_all.sh
#
#   # Custom conda envs
#   SERVER_CONDA=simvla EVAL_CONDA=libero bash run_coral_eval_all.sh
# =============================================================================

set -e

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export LIBERO_ROOT="${SCRIPT_DIR}/LIBERO"
export PYTHONPATH="${LIBERO_ROOT}:${PROJECT_ROOT}:${PYTHONPATH}"

# Model (defaults to HuggingFace repo; override with local path if needed)
MODEL_PATH=${MODEL_PATH:-"YuankaiLuo/SimVLA-LIBERO"}
LORA_DIR=${LORA_DIR:-"${PROJECT_ROOT}/lora_adapters/coral_libero"}
NORM_STATS=${NORM_STATS:-"${PROJECT_ROOT}/norm_stats/libero_norm.json"}
TASK_INDEX=${TASK_INDEX:-"${PROJECT_ROOT}/datasets/coral_metas/task_index.json"}

# LoRA step selection (empty = latest)
LORA_STEP=${LORA_STEP:-""}

# Server config
BASE_PORT=${BASE_PORT:-8089}
SERVER_GPUS=${SERVER_GPUS:-"0 1 2 3"}
SKIP_SERVER=${SKIP_SERVER:-0}

# Conda environments
SERVER_CONDA=${SERVER_CONDA:-"simvla"}
EVAL_CONDA=${EVAL_CONDA:-"libero"}

# Evaluation config
NUM_TRIALS=${NUM_TRIALS:-50}
OUTPUT_PREFIX=${OUTPUT_PREFIX:-"eval_coral"}
EVAL_GPUS=${EVAL_GPUS:-"4 5 6 7"}

# Suite list
SUITES=("libero_spatial" "libero_object" "libero_goal" "libero_10")
SUITE_SHORTS=("spatial" "object" "goal" "10")

# Parse GPU lists
read -ra SRV_GPU_ARRAY <<< "$SERVER_GPUS"
read -ra EVAL_GPU_ARRAY <<< "$EVAL_GPUS"

# Assign ports: BASE_PORT, BASE_PORT+1, BASE_PORT+2, BASE_PORT+3
PORTS=()
for i in 0 1 2 3; do
    PORTS+=($((BASE_PORT + i)))
done

# Output (with timestamp + LoRA info to preserve previous runs)
TIMESTAMP=$(date +%m%d_%H%M%S)
LORA_TAG=$(basename "$LORA_DIR" | sed 's/^coral_libero_//')
STEP_TAG=${LORA_STEP:+_step${LORA_STEP}}
OUTPUT_DIR="./eval_coral_${LORA_TAG}${STEP_TAG}_${BASE_PORT}_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

# Helper: get conda base path
CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")

echo "=============================================="
echo "CORAL LIBERO Evaluation (4-server parallel)"
echo "=============================================="
echo "  Model       : $MODEL_PATH"
echo "  LoRA dir    : $LORA_DIR"
echo "  LoRA step   : ${LORA_STEP:-latest}"
echo "  Servers     : ports ${PORTS[0]},${PORTS[1]},${PORTS[2]},${PORTS[3]}"
echo "  Server GPUs : ${SERVER_GPUS}"
echo "  Eval GPUs   : ${EVAL_GPUS}"
echo "  Server env  : ${SERVER_CONDA}"
echo "  Eval env    : ${EVAL_CONDA}"
echo "  Num trials  : $NUM_TRIALS"
echo "  Output dir  : $OUTPUT_DIR"
echo "=============================================="
echo ""

# =============================================================================
# Step 1: Start 4 CORAL MoE Servers (one per suite)
# =============================================================================

SERVER_PIDS=()

if [ "$SKIP_SERVER" = "0" ]; then
    echo "Starting 4 CORAL MoE servers..."
    echo ""

    # Build optional --lora-step flag
    LORA_STEP_FLAG=""
    if [ -n "$LORA_STEP" ]; then
        LORA_STEP_FLAG="--lora-step ${LORA_STEP}"
    fi

    for i in 0 1 2 3; do
        S_GPU="${SRV_GPU_ARRAY[$i]:-0}"
        S_PORT="${PORTS[$i]}"
        S_SUITE="${SUITES[$i]}"
        S_LOG="${OUTPUT_DIR}/server_${SUITE_SHORTS[$i]}.log"

        echo "  [${i}] ${S_SUITE} -> GPU ${S_GPU}, port ${S_PORT}"

        CUDA_VISIBLE_DEVICES=$S_GPU \
        bash -c "
            source '${CONDA_BASE}/etc/profile.d/conda.sh'
            conda activate '${SERVER_CONDA}'
            export LIBERO_ROOT='${LIBERO_ROOT}'
            export PYTHONPATH='${LIBERO_ROOT}:${PROJECT_ROOT}:\${PYTHONPATH}'
            python -u '${PROJECT_ROOT}/serve_smolvlm_coral.py' \
                --model '${MODEL_PATH}' \
                --lora-dir '${LORA_DIR}' \
                --task-index '${TASK_INDEX}' \
                --norm-stats '${NORM_STATS}' \
                --port ${S_PORT} \
                --steps 10 \
                ${LORA_STEP_FLAG}
        " > "$S_LOG" 2>&1 &
        SERVER_PIDS+=($!)
    done

    echo ""
    echo "  Server PIDs: ${SERVER_PIDS[*]}"
    echo ""

    # Wait for all servers to be ready
    echo "Waiting for servers to start..."
    for i in 0 1 2 3; do
        S_LOG="${OUTPUT_DIR}/server_${SUITE_SHORTS[$i]}.log"
        S_PID="${SERVER_PIDS[$i]}"
        for attempt in $(seq 1 90); do
            if grep -q "Starting\|listening\|Loaded" "$S_LOG" 2>/dev/null; then
                echo "  [${i}] ${SUITES[$i]} server ready (port ${PORTS[$i]})"
                break
            fi
            if ! kill -0 $S_PID 2>/dev/null; then
                echo "ERROR: Server ${i} died. Check $S_LOG"
                cat "$S_LOG"
                # Kill other servers
                for pid in "${SERVER_PIDS[@]}"; do
                    kill $pid 2>/dev/null || true
                done
                exit 1
            fi
            sleep 2
        done
    done

    # Extra wait for model loading
    echo "  Waiting 10s for model loading to finish..."
    sleep 10
    echo ""
fi

# =============================================================================
# Step 2: Run 4 Evaluations in Parallel (each hits its own server)
# =============================================================================

echo "Launching 4 evaluations in parallel (env: $EVAL_CONDA)..."
echo ""

cd "$SCRIPT_DIR"

EVAL_PIDS=()
for i in 0 1 2 3; do
    E_GPU="${EVAL_GPU_ARRAY[$i]:-0}"
    E_PORT="${PORTS[$i]}"
    E_SUITE="${SUITES[$i]}"
    E_SHORT="${SUITE_SHORTS[$i]}"
    E_LOG="${OUTPUT_DIR}/${OUTPUT_PREFIX}_${E_SHORT}.txt"

    echo "  [${i}] ${E_SUITE} -> GPU ${E_GPU}, server port ${E_PORT}"

    bash -c "
        source '${CONDA_BASE}/etc/profile.d/conda.sh'
        conda activate '${EVAL_CONDA}'
        export LIBERO_ROOT='${LIBERO_ROOT}'
        export PYTHONPATH='${LIBERO_ROOT}:${PROJECT_ROOT}:\${PYTHONPATH}'
        CUDA_VISIBLE_DEVICES=${E_GPU} python -u libero_client.py \
            --host 127.0.0.1 \
            --port ${E_PORT} \
            --client_type websocket \
            --task_suite '${E_SUITE}' \
            --num_trials ${NUM_TRIALS} \
            --video_out '${OUTPUT_DIR}'
    " 2>&1 | tee "$E_LOG" &
    EVAL_PIDS+=($!)
done

echo ""
echo "  Eval PIDs: ${EVAL_PIDS[*]}"
echo "  Monitor:   tail -f ${OUTPUT_DIR}/${OUTPUT_PREFIX}_*.txt"
echo ""
echo "Waiting for all evaluations to complete..."

# Wait for all evaluations
FAILED=0
for i in 0 1 2 3; do
    wait ${EVAL_PIDS[$i]} || true
    echo "  [${i}] ${SUITES[$i]} finished."
done

# =============================================================================
# Step 3: Summary
# =============================================================================

echo ""
echo "=============================================="
echo "CORAL Evaluation Results"
echo "=============================================="
for i in 0 1 2 3; do
    file="${OUTPUT_DIR}/${OUTPUT_PREFIX}_${SUITE_SHORTS[$i]}.txt"
    if [ -f "$file" ]; then
        echo "--- ${SUITES[$i]} ---"
        grep -E "Total success rate|Success Rate|success rate" "$file" 2>/dev/null || echo "  (see $file)"
    fi
done
echo "=============================================="

# Cleanup servers
if [ ${#SERVER_PIDS[@]} -gt 0 ]; then
    echo ""
    echo "Stopping CORAL servers..."
    for pid in "${SERVER_PIDS[@]}"; do
        if kill -0 $pid 2>/dev/null; then
            kill $pid 2>/dev/null || true
        fi
    done
    # Wait for all to exit
    for pid in "${SERVER_PIDS[@]}"; do
        wait $pid 2>/dev/null || true
    done
    echo "All servers stopped."
fi

echo ""
echo "Full results saved to: $OUTPUT_DIR"
echo "Done!"
