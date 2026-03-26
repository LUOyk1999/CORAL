#!/bin/bash
# =============================================================================
# CORAL: Train Per-Task LoRA Experts for ALL LIBERO Evaluation Tasks (Rank-16)
# =============================================================================
#
# This script trains 40 LoRA experts (one per LIBERO evaluation task):
#   - libero_spatial  (10 tasks)
#   - libero_object   (10 tasks)
#   - libero_goal     (10 tasks)
#   - libero_10       (10 tasks)
#
# Each expert uses rank-16 LoRA (~7MB) for lightweight task adaptation.
#
# Usage:
#   bash train_coral_libero_all_r16.sh
#
#   # Override GPU and other settings
#   CUDA_VISIBLE_DEVICES=4,5,6,7 TOTAL_STEPS=500 bash train_coral_libero_all_r16.sh
#
#   # Train only a specific suite
#   SUITES="libero_10" bash train_coral_libero_all_r16.sh
#
# =============================================================================

set -e

# =============================================================================
# Configuration (override via environment variables)
# =============================================================================

# GPU (default: 4 GPUs)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

# TensorFlow settings
export TF_CPP_MIN_LOG_LEVEL=3
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Base model (SimVLA pretrained on LIBERO, downloaded from HuggingFace)
BASE_MODEL=${BASE_MODEL:-"YuankaiLuo/SimVLA-LIBERO"}

# LoRA hyperparameters (rank-16 for CORAL)
LORA_RANK=${LORA_RANK:-16}
LORA_ALPHA=${LORA_ALPHA:-32}
LEARNING_RATE=${LEARNING_RATE:-5e-5}

# Training schedule
TOTAL_STEPS=${TOTAL_STEPS:-50}
WARMUP_STEPS=${WARMUP_STEPS:-10}
BATCH_SIZE=${BATCH_SIZE:-64}
SAVE_INTERVAL=${SAVE_INTERVAL:-50}

# Normalization statistics
NORM_STATS_PATH=${NORM_STATS_PATH:-"./norm_stats/libero_norm.json"}

# Source metadata (full LIBERO training data)
LIBERO_TRAIN_META=${LIBERO_TRAIN_META:-"./datasets/metas/libero_train.json"}

# Output (separate from rank-64)
OUTPUT_ROOT=${OUTPUT_ROOT:-"./lora_adapters/coral_libero_r16"}

# Which suites to train (space-separated)
SUITES=${SUITES:-"libero_spatial libero_object libero_goal libero_10"}

# Temp directory for per-task metadata files (shared, same tasks)
CORAL_METAS_DIR=${CORAL_METAS_DIR:-"./datasets/coral_metas"}

# =============================================================================
# Validation
# =============================================================================

if [ ! -f "$LIBERO_TRAIN_META" ]; then
    echo "Error: LIBERO train meta not found: $LIBERO_TRAIN_META"
    exit 1
fi

# =============================================================================
# Step 1: Generate per-task metadata JSON files
# =============================================================================

echo "=============================================="
echo "CORAL (R16): Generating per-task metadata files..."
echo "=============================================="

mkdir -p "$CORAL_METAS_DIR"

python3 -c "
import json, os, re

with open('${LIBERO_TRAIN_META}') as f:
    meta = json.load(f)

suites = '${SUITES}'.split()
metas_dir = '${CORAL_METAS_DIR}'
os.makedirs(metas_dir, exist_ok=True)

tasks_generated = []
for item in meta['datalist']:
    subset = item['subset']
    if subset not in suites:
        continue

    task_desc = item['task']
    # Create a slug from the task description
    slug = re.sub(r'[^a-z0-9]+', '_', task_desc.lower()).strip('_')
    if len(slug) > 80:
        slug = slug[:80]

    # Create per-task metadata
    task_meta = {
        'dataset_name': meta.get('dataset_name', 'libero_hdf5'),
        'data_dir': meta.get('data_dir', './datasets/metas'),
        'datalist': [item],
        'num_files': 1,
        'num_episodes': item.get('num_demos', 50),
        'subsets': [subset],
        'observation_key': meta.get('observation_key', ['obs/agentview_rgb', 'obs/eye_in_hand_rgb']),
        'action_key': meta.get('action_key', 'actions'),
        'state_dim': meta.get('state_dim', 8),
        'action_dim': meta.get('action_dim', 7),
        'fps': meta.get('fps', 10),
    }

    out_path = os.path.join(metas_dir, f'{subset}_{slug}.json')
    with open(out_path, 'w') as f:
        json.dump(task_meta, f, indent=2)

    tasks_generated.append((subset, slug, out_path, task_desc))
    print(f'  [{subset}] {slug}')

# Write task index
index_path = os.path.join(metas_dir, 'task_index.json')
with open(index_path, 'w') as f:
    json.dump([
        {'suite': s, 'slug': sl, 'meta_path': p, 'task': t}
        for s, sl, p, t in tasks_generated
    ], f, indent=2)

print(f'\nGenerated {len(tasks_generated)} per-task metadata files')
print(f'Task index: {index_path}')
"

if [ ! -f "$CORAL_METAS_DIR/task_index.json" ]; then
    echo "Error: Failed to generate task metadata"
    exit 1
fi

TOTAL_TASKS=$(python3 -c "import json; print(len(json.load(open('${CORAL_METAS_DIR}/task_index.json'))))")

echo ""
echo "=============================================="
echo "CORAL (R16): Training $TOTAL_TASKS LoRA Experts"
echo "=============================================="
echo "  Base model    : ${BASE_MODEL}"
echo "  LoRA rank     : ${LORA_RANK}, alpha: ${LORA_ALPHA}"
echo "  Learning rate : ${LEARNING_RATE}"
echo "  Steps/task    : ${TOTAL_STEPS}"
echo "  Batch size    : ${BATCH_SIZE}"
echo "  Output root   : ${OUTPUT_ROOT}"
echo "  Suites        : ${SUITES}"
echo "=============================================="
echo ""

# =============================================================================
# Step 2: Train LoRA experts
# =============================================================================

TASK_IDX=0
FAILED=0

# Auto-detect number of GPUs
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

python3 -c "
import json
tasks = json.load(open('${CORAL_METAS_DIR}/task_index.json'))
for t in tasks:
    print(f\"{t['suite']}|{t['slug']}|{t['meta_path']}|{t['task']}\")
" | while IFS='|' read -r SUITE SLUG META_PATH TASK_DESC; do
    TASK_IDX=$((TASK_IDX + 1))
    LORA_NAME="${SUITE}_${SLUG}"
    TASK_OUTPUT_DIR="${OUTPUT_ROOT}/${SUITE}/${SLUG}"

    echo ""
    echo "----------------------------------------------"
    echo "[$TASK_IDX/$TOTAL_TASKS] Training: $LORA_NAME"
    echo "  Suite: $SUITE"
    echo "  Task:  $TASK_DESC"
    echo "  Meta:  $META_PATH"
    echo "  Out:   $TASK_OUTPUT_DIR"
    echo "----------------------------------------------"

    mkdir -p "$TASK_OUTPUT_DIR"

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    accelerate launch \
        --num_processes=$NUM_GPUS \
        --main_process_port 29521 \
        --mixed_precision bf16 \
        train_coral_smolvlm.py \
        --base_model "$BASE_MODEL" \
        --lora_name "$LORA_NAME" \
        --lora_rank $LORA_RANK \
        --lora_alpha $LORA_ALPHA \
        --train_metas_path "$META_PATH" \
        --output_dir "$TASK_OUTPUT_DIR" \
        --learning_rate $LEARNING_RATE \
        --iters $TOTAL_STEPS \
        --warmup_steps $WARMUP_STEPS \
        --batch_size $BATCH_SIZE \
        --action_mode libero_joint \
        --norm_stats_path "$NORM_STATS_PATH" \
        --save_interval $SAVE_INTERVAL \
        --log_interval 20 \
        --num_workers 4 \
        --num_actions 10 \
        --image_size 384 \
    2>&1 | tee "${TASK_OUTPUT_DIR}/train.log"

    if [ $? -ne 0 ]; then
        echo "WARNING: Training failed for $LORA_NAME"
        FAILED=$((FAILED + 1))
    fi

done

echo ""
echo "=============================================="
echo "CORAL (R16) Training Complete!"
echo "=============================================="
echo "  Total tasks : $TOTAL_TASKS"
echo "  Failed      : $FAILED"
echo "  Output root : $OUTPUT_ROOT"
echo ""
echo "Directory structure:"
echo "  $OUTPUT_ROOT/"
echo "    libero_spatial/"
echo "      <task_slug>/"
echo "        lora-<name>-step${TOTAL_STEPS}/"
echo "          adapter_config.json"
echo "          adapter_model.safetensors"
echo "    libero_object/..."
echo "    libero_goal/..."
echo "    libero_10/..."
echo "=============================================="
