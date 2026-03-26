# LIBERO Evaluation

You need **two separate conda environments**: one for the policy server (`simvla`) and one for the LIBERO evaluation client (`libero`).

## 1. Environment Setup

### SimVLA Server Environment

```bash
conda activate simvla
```

### LIBERO Client Environment

```bash
conda create -n libero python=3.8.13
conda activate libero
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -r requirements.txt
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -e .
```

## 2. SimVLA Baseline Evaluation

Start server (`simvla` env), then run eval (`libero` env) in a separate terminal:

```bash
# Terminal 1
conda activate simvla
CUDA_VISIBLE_DEVICES=0 python serve_smolvlm_libero.py \
    --checkpoint YuankaiLuo/SimVLA-LIBERO \
    --norm_stats ../../norm_stats/libero_norm.json \
    --port 8102

# Terminal 2
conda activate libero
bash run_eval_all.sh 8102 50 "eval_simvla" "0 1 2 3"
```

## 3. CORAL Evaluation


```bash
# Default: GPU 0 = server, GPU 2 = eval
SERVER_GPU=2 EVAL_GPU=2 bash example.sh
```
or

```
conda activate simvla
SERVER_GPUS="0 1 2 3" \
EVAL_GPUS="4 5 6 7" \
LORA_DIR=../../lora_adapters/coral_libero_r16 \
bash run_coral_eval_all.sh
```