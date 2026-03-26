# CORAL: Scalable Multi-Task Robot Learning via LoRA Experts

| **Paper** | **Website** |
| :------------------: | :-----------------------: | 
| [![Paper](https://img.shields.io/badge/Paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2603.09298) | [![Website](https://img.shields.io/badge/Project%20Page-181717?style=for-the-badge&logo=githubpages&logoColor=white)](https://frontierrobo.github.io/CORAL/) | 

A backbone- and embodiment-agnostic framework that resolves multi-task interference in VLA deployment through strict parameter isolation via LoRA experts.

<img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/35098236-33ec-4f3e-9c66-74731975c685" />

## Installation

```bash
conda create -n simvla python=3.10 -y
conda activate simvla

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install transformers>=4.57.0
pip install peft accelerate fastapi tensorboard uvicorn json_numpy safetensors scipy einops timm mmengine pyarrow h5py mediapy num2words av wandb websockets msgpack_numpy
pip install flash-attn==2.5.6 --no-build-isolation
pip install tensorflow tensorflow-datasets
```

> Important: Use `transformers>=4.57.0`.

## Training (LIBERO Dataset)

### 1. Prepare LIBERO Dataset

Download [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) dataset, and place it in `./datasets/metas/`.

### 2. Train CORAL LoRA Experts

Train 40 per-task LoRA experts (one per LIBERO evaluation task). The [SimVLA](https://huggingface.co/YuankaiLuo/SimVLA-LIBERO) base model is automatically downloaded from HuggingFace.

**Lora A** (For Libero 10, lr=5e-5, 50 steps):
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 SUITES="libero_10" bash train_coral_libero_all_r16.sh
```

### 3. Evaluation

```bash
cd evaluation/libero
```

See [evaluation/libero/README.md](evaluation/libero/README.md) for detailed instructions.

### 4. Results

CORAL achieves a **99.3%** overall average success rate on the LIBERO benchmark.

| Method | Spatial | Object | Goal | Long | **Average** |
|--------|---------|--------|------|------|-------------|
| SimVLA (base) | 96.0 | 98.0 | 96.0 | 84.0 | 93.5 |
| **CORAL (SimVLA)** | **99.6** | **99.8** | **99.0** | **98.8** | **99.3** |

## Reference

If you find our codes useful, please consider citing our work

```
@article{luo2026coral,
  title={CORAL: Scalable Multi-Task Robot Learning via LoRA Experts},
  author={Luo, Yuankai and Chen, Woping and Liang, Tong and Li, Zhenguo},
  journal={arXiv preprint arXiv:2603.09298},
  year={2026}
}
```
