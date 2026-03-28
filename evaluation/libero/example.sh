#!/bin/bash
# =============================================================================
# CORAL LIBERO Evaluation Example
#
# Evaluates 4 suites sequentially using 2 GPUs:
#   GPU 0 = server, GPU 1 = eval client
#
# Suite order: libero_10 -> libero_goal -> libero_spatial -> libero_object
#
# Requirements:
#   - conda env "simvla" for the policy server
#   - conda env "libero" for the evaluation client
#
# Usage:
#   bash example.sh
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SERVER_GPU=${SERVER_GPU:-0}
EVAL_GPU=${EVAL_GPU:-1}
PORT=${PORT:-8089}
NUM_TRIALS=${NUM_TRIALS:-50}
SERVER_CONDA=${SERVER_CONDA:-"simvla"}
EVAL_CONDA=${EVAL_CONDA:-"libero"}

NORM_STATS="${PROJECT_ROOT}/norm_stats/libero_norm.json"
CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")

TIMESTAMP=$(date +%m%d_%H%M%S)
OUTPUT_DIR="${SCRIPT_DIR}/eval_coral_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "CORAL LIBERO Evaluation (sequential, 2 GPUs)"
echo "=============================================="
echo "  Server GPU  : ${SERVER_GPU}"
echo "  Eval GPU    : ${EVAL_GPU}"
echo "  Port        : ${PORT}"
echo "  Num trials  : ${NUM_TRIALS}"
echo "  Output dir  : ${OUTPUT_DIR}"
echo "=============================================="
echo ""

wait_for_server() {
    local log_file=$1
    local pid=$2
    for attempt in $(seq 1 90); do
        if grep -q "Starting\|listening\|Loaded\|CORAL MoE WebSocket Server" "$log_file" 2>/dev/null; then
            echo "  Server ready."
            sleep 5
            return 0
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "ERROR: Server died. Check $log_file"
            return 1
        fi
        sleep 2
    done
    echo "ERROR: Server timed out."
    return 1
}

run_eval() {
    local suite=$1
    local log_file="${OUTPUT_DIR}/eval_${suite}.txt"
    echo "  Running eval: ${suite} ..."
    bash -c "
        source '${CONDA_BASE}/etc/profile.d/conda.sh'
        conda activate '${EVAL_CONDA}'
        export LIBERO_ROOT='${SCRIPT_DIR}/LIBERO'
        export PYTHONPATH='${SCRIPT_DIR}/LIBERO:${PROJECT_ROOT}:\${PYTHONPATH}'
        CUDA_VISIBLE_DEVICES=${EVAL_GPU} python -u '${SCRIPT_DIR}/libero_client.py' \
            --host 127.0.0.1 \
            --port ${PORT} \
            --client_type websocket \
            --task_suite '${suite}' \
            --num_trials ${NUM_TRIALS} \
            --video_out '${OUTPUT_DIR}'
    " > "$log_file" 2>&1
    echo "  Done. Results: ${log_file}"
}

# =============================================
# 1. libero_10  (LoRA: coral_libero_r16)
# =============================================
echo "[1/4] libero_10 (LoRA: coral_libero_r16)"
SERVER_LOG="${OUTPUT_DIR}/server_10.log"
CUDA_VISIBLE_DEVICES=$SERVER_GPU \
bash -c "
    source '${CONDA_BASE}/etc/profile.d/conda.sh'
    conda activate '${SERVER_CONDA}'
    export PYTHONPATH='${PROJECT_ROOT}:\${PYTHONPATH}'
    python -u '${PROJECT_ROOT}/serve_smolvlm_coral.py' \
        --lora-dir '${PROJECT_ROOT}/lora_adapters/coral_libero_r16' \
        --norm-stats '${NORM_STATS}' \
        --port ${PORT}
" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
wait_for_server "$SERVER_LOG" $SERVER_PID
run_eval "libero_10"
kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null || true
echo ""

# =============================================
# 2. libero_goal  (LoRA: coral_libero_r16)
# =============================================
echo "[2/4] libero_goal (base SimVLA)"
SERVER_LOG="${OUTPUT_DIR}/server_goal.log"
CUDA_VISIBLE_DEVICES=$SERVER_GPU \
bash -c "
    source '${CONDA_BASE}/etc/profile.d/conda.sh'
    conda activate '${SERVER_CONDA}'
    export PYTHONPATH='${PROJECT_ROOT}:\${PYTHONPATH}'
    python -u '${SCRIPT_DIR}/serve_smolvlm_libero.py' \
        --checkpoint YuankaiLuo/SimVLA-LIBERO \
        --norm_stats '${NORM_STATS}' \
        --port ${PORT}
" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
wait_for_server "$SERVER_LOG" $SERVER_PID
run_eval "libero_goal"
kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null || true
echo ""

# =============================================
# 3. libero_spatial  (base SimVLA, no LoRA)
# =============================================
echo "[3/4] libero_spatial (base SimVLA)"
SERVER_LOG="${OUTPUT_DIR}/server_spatial.log"
CUDA_VISIBLE_DEVICES=$SERVER_GPU \
bash -c "
    source '${CONDA_BASE}/etc/profile.d/conda.sh'
    conda activate '${SERVER_CONDA}'
    export PYTHONPATH='${PROJECT_ROOT}:\${PYTHONPATH}'
    python -u '${SCRIPT_DIR}/serve_smolvlm_libero.py' \
        --checkpoint YuankaiLuo/SimVLA-LIBERO \
        --norm_stats '${NORM_STATS}' \
        --port ${PORT}
" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
wait_for_server "$SERVER_LOG" $SERVER_PID
run_eval "libero_spatial"
kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null || true
echo ""

# =============================================
# 4. libero_object  (base SimVLA, no LoRA)
# =============================================
echo "[4/4] libero_object (base SimVLA)"
SERVER_LOG="${OUTPUT_DIR}/server_object.log"
CUDA_VISIBLE_DEVICES=$SERVER_GPU \
bash -c "
    source '${CONDA_BASE}/etc/profile.d/conda.sh'
    conda activate '${SERVER_CONDA}'
    export PYTHONPATH='${PROJECT_ROOT}:\${PYTHONPATH}'
    python -u '${SCRIPT_DIR}/serve_smolvlm_libero.py' \
        --checkpoint YuankaiLuo/SimVLA-LIBERO \
        --norm_stats '${NORM_STATS}' \
        --port ${PORT}
" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
wait_for_server "$SERVER_LOG" $SERVER_PID
run_eval "libero_object"
kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null || true
echo ""

echo "=============================================="
echo "All 4 suites completed!"
echo "Results in: ${OUTPUT_DIR}"
echo "=============================================="
