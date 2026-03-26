"""
CORAL: Continual Robot Skill Learning via LoRA Experts (SmolVLM-VLA)

Per-task LoRA fine-tuning for SmolVLM-VLA using PEFT.
Injects LoRA adapters into VLM attention and ActionTransformer attention,
while keeping action encoder/decoder fully trainable per task.

Usage:
    python train_coral_smolvlm.py \
        --base_model YuankaiLuo/SimVLA-LIBERO \
        --train_metas_path ./datasets/coral_metas/task_0.json \
        --output_dir ./lora_adapters/coral_libero/task_0 \
        --lora_name task_0 \
        --lora_rank 16 \
        --learning_rate 5e-5 \
        --iters 500
"""

import os
import math
import time
import json
import random
import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.optim import AdamW

from accelerate import Accelerator, DistributedDataParallelKwargs
from datasets import create_smolvlm_dataloader
from models.modeling_smolvlm_vla import SmolVLMVLA
from models.processing_smolvlm_vla import SmolVLMVLAProcessor

from peft import LoraConfig, get_peft_model

import logging
import sys


# ============================================================
# Logger
# ============================================================
def get_logger(name="coral", output_dir=None, accelerator=None, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    if logger.handlers:
        return logger
    is_main = accelerator is None or accelerator.is_main_process
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    datefmt = "%H:%M:%S"
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
    if is_main:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        ch.setLevel(level)
        logger.addHandler(ch)
    if output_dir and is_main:
        os.makedirs(output_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(output_dir, "coral_train.log"), mode="a")
        fh.setFormatter(formatter)
        fh.setLevel(level)
        logger.addHandler(fh)
    return logger


# ============================================================
# Argument Parser
# ============================================================
def get_args_parser():
    parser = argparse.ArgumentParser("CORAL SmolVLM-VLA LoRA Training", add_help=False)

    # Model
    parser.add_argument("--base_model", type=str, default="YuankaiLuo/SimVLA-LIBERO",
                        help="Path or HuggingFace repo ID of pretrained SimVLA checkpoint")

    # LoRA config
    parser.add_argument("--lora_rank", type=int, default=16,
                        help="LoRA rank (r). Higher = more capacity but more params")
    parser.add_argument("--lora_alpha", type=float, default=32.0,
                        help="LoRA alpha (scaling factor, typically 2x rank)")
    parser.add_argument("--lora_dropout", type=float, default=0.0,
                        help="LoRA dropout rate")
    parser.add_argument("--lora_target_modules", type=str, nargs="+", default=None,
                        help="Target modules for LoRA injection. Default: all attention layers")
    parser.add_argument("--modules_to_save", type=str, nargs="+", default=None,
                        help="Modules to fully train (not LoRA). e.g., action_encoder, action_decoder")

    # I/O
    parser.add_argument("--output_dir", type=str, default="lora_output",
                        help="Directory to save LoRA checkpoints")
    parser.add_argument("--lora_name", type=str, default="my_task",
                        help="Name for this LoRA adapter")

    # Data
    parser.add_argument("--train_metas_path", type=str, required=True,
                        help="Path to training metadata JSON")
    parser.add_argument("--batch_size", type=int, default=32)

    # Optimizer
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate for LoRA (lower than full fine-tuning)")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.999))
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Schedule
    parser.add_argument("--iters", type=int, default=100,
                        help="Total training iterations")
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--use_cosine_decay", action="store_true", default=True)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)

    # Logging / saving
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--log_interval", type=int, default=20)

    # System
    parser.add_argument("--seed", type=int, default=0)

    # Action mode
    parser.add_argument("--action_mode", type=str, default="libero_joint")

    # Data loading
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")

    # Normalization
    parser.add_argument("--norm_stats_path", type=str, default=None)

    # Image size (SmolVLM)
    parser.add_argument("--image_size", type=int, default=384,
                        help="Image size for SmolVLM (384 or 512)")

    # Action horizon
    parser.add_argument("--num_actions", type=int, default=10,
                        help="Action horizon (number of future actions to predict)")

    return parser


# ============================================================
# Utilities
# ============================================================
def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True


def linear_warmup_cosine(step, warmup, total, base_lr, min_ratio):
    if step < warmup:
        return base_lr * (step / max(1, warmup))
    remain = max(1, total - warmup)
    ratio = 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / remain)))
    return base_lr * (min_ratio + (1 - min_ratio) * ratio)


# ============================================================
# Main Training
# ============================================================
def main(args):
    output_dir = Path(args.output_dir)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        log_with=["tensorboard"],
        project_dir=output_dir,
        kwargs_handlers=[ddp_kwargs]
    )

    tracker_config = {
        "lora_name": args.lora_name,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "iters": args.iters,
        "base_model": args.base_model,
        "action_mode": args.action_mode,
    }
    accelerator.init_trackers(f"CORAL-{args.lora_name}", config=tracker_config)

    accelerator.wait_for_everyone()
    logger = get_logger(__name__, output_dir=output_dir, accelerator=accelerator)

    set_seed(args.seed + accelerator.process_index)
    logger.info(f"CORAL LoRA Fine-tuning: {args.lora_name}")
    logger.info(f"Args: {args}")

    # ========== Load base model ==========
    from models.configuration_smolvlm_vla import SmolVLMVLAConfig
    from models.action_hub import build_action_space

    action_space_kwargs = {}
    if args.norm_stats_path:
        action_space_kwargs["norm_stats_path"] = args.norm_stats_path

    logger.info(f"Loading base model from: {args.base_model}")
    model = SmolVLMVLA.from_pretrained(args.base_model, trust_remote_code=True)

    if args.action_mode != model.action_mode:
        logger.warning(f"Overriding action_mode: {model.action_mode} -> {args.action_mode}")
        model.action_mode = args.action_mode
        model.action_space = build_action_space(args.action_mode, **action_space_kwargs)
    elif action_space_kwargs:
        model.action_space = build_action_space(args.action_mode, **action_space_kwargs)

    if args.num_actions != model.num_actions:
        logger.warning(f"Overriding num_actions: {model.num_actions} -> {args.num_actions}")
        model.config.num_actions = args.num_actions
        model.num_actions = args.num_actions

    # ========== Inject LoRA via PEFT ==========
    logger.info(f"Injecting LoRA: rank={args.lora_rank}, alpha={args.lora_alpha}")

    target_modules = args.lora_target_modules
    if target_modules is None:
        target_modules = [
            # SmolVLM VLM Attention (Idefics3/SigLIP)
            "q_proj", "k_proj", "v_proj", "o_proj",
            # SmolVLM ActionTransformer Attention
            "qkv", "proj",
        ]

    # Action encoder/decoder: fully trained for task adaptation
    modules_to_save = args.modules_to_save
    if modules_to_save is None:
        modules_to_save = [
            "transformer.action_encoder",
            "transformer.action_decoder",
        ]
        logger.info(f"Using default modules_to_save: {modules_to_save}")

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=target_modules,
        modules_to_save=modules_to_save,
    )

    model = get_peft_model(model, lora_config)

    if accelerator.is_main_process:
        model.print_trainable_parameters()

    # Processor
    smolvlm_path = getattr(model, "vlm", None)
    smolvlm_model_path = "HuggingFaceTB/SmolVLM-500M-Instruct"
    if hasattr(model, "config") and hasattr(model.config, "smolvlm_model_path"):
        smolvlm_model_path = model.config.smolvlm_model_path
    processor = SmolVLMVLAProcessor.from_pretrained(smolvlm_model_path)

    # Sanity check: verify data paths before starting training
    if accelerator.is_main_process:
        import json as _json
        with open(args.train_metas_path) as _f:
            _meta = _json.load(_f)
        for _item in _meta.get("datalist", []):
            _p = _item.get("path", "")
            if not os.path.exists(_p):
                logger.error(f"HDF5 NOT FOUND: {_p}")
            else:
                import h5py
                with h5py.File(_p, "r") as _hf:
                    _demos = list(_hf.get("data", {}).keys())
                    logger.info(f"HDF5 OK: {os.path.basename(_p)} -> {len(_demos)} demos")

    # Dataloader (SmolVLM-specific)
    train_dataloader = create_smolvlm_dataloader(
        batch_size=args.batch_size,
        metas_path=args.train_metas_path,
        num_actions=model.num_actions if hasattr(model, "num_actions") else args.num_actions,
        action_mode=model.action_mode if hasattr(model, "action_mode") else args.action_mode,
        training=True,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )

    # ========== Optimizer ==========
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optim = AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=tuple(args.betas),
    )

    model, optim = accelerator.prepare(model, optim)

    # ========== Training loop ==========
    model.train()
    global_step, t0 = 0, time.time()

    logger.info(f"Start CORAL training for {args.iters} iterations | world_size={accelerator.num_processes}")

    for batch in train_dataloader:
        lang = processor.encode_language(batch["language_instruction"])
        batch.pop("language_instruction", None)
        inputs = {**batch, **lang}
        inputs = {k: v.cuda(non_blocking=True) for k, v in inputs.items()}

        # Learning rate schedule
        if args.use_cosine_decay:
            lr = linear_warmup_cosine(global_step, args.warmup_steps, args.iters, args.learning_rate, args.min_lr_ratio)
        else:
            lr = args.learning_rate if global_step >= args.warmup_steps else args.learning_rate * (global_step / max(1, args.warmup_steps))

        for g in optim.param_groups:
            g["lr"] = lr

        # Forward & backward
        loss_dict: Dict[str, torch.Tensor] = model(**inputs)
        loss = sum(loss_dict.values())
        accelerator.backward(loss)

        if args.max_grad_norm:
            accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
        optim.step()
        optim.zero_grad()

        # Logging
        if global_step % args.log_interval == 0:
            logs = {k: v.detach().float().item() for k, v in loss_dict.items()}
            logs["loss_total"] = float(loss.detach().item())
            logs["lr"] = lr
            accelerator.log(logs, step=global_step)

            if accelerator.is_main_process:
                dt = (time.time() - t0) / args.log_interval
                t0 = time.time()
                logger.info(
                    f"[{global_step}/{args.iters}] "
                    f"CORAL={args.lora_name} "
                    f"loss={logs['loss_total']:.4f} "
                    f"lr={lr:.2e} ({dt:.2f}s/it)"
                )

        # Checkpointing (only saves LoRA weights via PEFT)
        global_step += 1
        if accelerator.is_main_process:
            if global_step == args.iters or global_step % args.save_interval == 0:
                save_dir = os.path.join(output_dir, f"lora-{args.lora_name}-step{global_step}")

                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(save_dir, save_embedding_layers=False, create_model_card=False)

                config_dict = {
                    "lora_name": args.lora_name,
                    "global_step": global_step,
                    "base_model": args.base_model,
                    "action_mode": args.action_mode,
                    "framework": "coral",
                }
                with open(os.path.join(save_dir, "training_config.json"), "w") as f:
                    json.dump(config_dict, f, indent=2)

                adapter_path = os.path.join(save_dir, "adapter_model.safetensors")
                if os.path.exists(adapter_path):
                    lora_size = os.path.getsize(adapter_path) / 1024 / 1024
                else:
                    adapter_path = os.path.join(save_dir, "adapter_model.bin")
                    lora_size = os.path.getsize(adapter_path) / 1024 / 1024 if os.path.exists(adapter_path) else 0

                accelerator.print(f"Saved CORAL LoRA '{args.lora_name}' to {save_dir} ({lora_size:.2f} MB)")

        if global_step >= args.iters:
            break

    accelerator.end_training()
    logger.info(f"CORAL LoRA '{args.lora_name}' training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("CORAL SmolVLM-VLA LoRA Training", parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
