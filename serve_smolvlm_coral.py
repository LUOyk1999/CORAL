#!/usr/bin/env python
"""
CORAL (Continual Robot Skill Learning via LoRA Experts): SmolVLM-VLA MoE Inference Service

Loads one frozen SmolVLM-VLA base model and dynamically switches between
multiple LoRA experts based on the task instruction (prompt).

This server is compatible with the LIBERO evaluation client (libero_client.py)
via the openpi_client WebSocket protocol (msgpack serialization).

Usage:
    python serve_smolvlm_coral.py \
        --model YuankaiLuo/SimVLA-LIBERO \
        --lora-dir ./lora_adapters/coral_libero \
        --norm-stats ./norm_stats/libero_norm.json \
        --port 8089

    # Client usage (libero_client.py):
    #   python libero_client.py --host 127.0.0.1 --port 8089 --client_type websocket

Protocol:
    1. On connect: server sends metadata (msgpack dict)
    2. Client sends observation dict (msgpack)
    3. Server replies with {"actions": np.ndarray} (msgpack)
    4. The server auto-routes to the correct LoRA expert based on "prompt"
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import os
import re
import logging
import socket
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import torch
from PIL import Image

try:
    import websockets
    import websockets.server
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

try:
    import msgpack
    MSGPACK_AVAILABLE = True
except ImportError:
    MSGPACK_AVAILABLE = False


# ============================================================
# Msgpack numpy serialization
# ============================================================
def _pack_numpy_array(obj):
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def _unpack_numpy_array(obj):
    if isinstance(obj, dict):
        if b"__ndarray__" in obj:
            return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
        if b"__npgeneric__" in obj:
            return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


def msgpack_packb(obj):
    return msgpack.packb(obj, default=_pack_numpy_array)


def msgpack_unpackb(data):
    return msgpack.unpackb(data, object_hook=_unpack_numpy_array)


from models.modeling_smolvlm_vla import SmolVLMVLA
from models.processing_smolvlm_vla import SmolVLMVLAProcessor


# ============================================================
# Utilities
# ============================================================
def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _resolve_dtype(dtype_arg: str, device: torch.device) -> torch.dtype:
    if dtype_arg == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    mapping = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = mapping[dtype_arg]
    if device.type == "cpu" and dtype in (torch.bfloat16, torch.float16):
        return torch.float32
    return dtype


def _decode_images_from_obs(obs: Dict[str, Any]) -> List[Image.Image]:
    """Decode images from LIBERO observation dict.
    
    Expected keys: observation/image, observation/wrist_image
    """
    images: List[Image.Image] = []
    for key in ["image", "wrist_image", "image_camera_head",
                 "image_camera_wrist_left", "image_camera_wrist_right"]:
        flat_key = f"observation/{key}"
        if flat_key in obs:
            v = obs[flat_key]
        elif "observation" in obs and isinstance(obs["observation"], dict) and key in obs["observation"]:
            v = obs["observation"][key]
        elif key in obs:
            v = obs[key]
        else:
            continue
        if isinstance(v, np.ndarray):
            if v.ndim == 3 and v.shape[2] == 3:
                images.append(Image.fromarray(v.astype(np.uint8)))
            elif v.ndim == 2:
                images.append(Image.fromarray(np.stack([v, v, v], axis=-1).astype(np.uint8)))
    return images


def _extract_state_from_obs(obs: Dict[str, Any]) -> Optional[np.ndarray]:
    """Extract proprioceptive state from LIBERO observation.
    
    LIBERO state: [eef_pos(3), axis_angle(3), gripper_qpos(2)] = 8D
    """
    for key in ["observation/state", "state"]:
        if key in obs:
            return np.asarray(obs[key], dtype=np.float32).reshape(-1)
    if "observation" in obs and isinstance(obs["observation"], dict):
        if "state" in obs["observation"]:
            return np.asarray(obs["observation"]["state"], dtype=np.float32).reshape(-1)
    return None


# ============================================================
# LoRA Scanner for CORAL
# ============================================================
def scan_coral_adapters(lora_dir: str, target_step: Optional[int] = None) -> Dict[str, str]:
    """Scan directory tree for CORAL LoRA adapters.
    
    Expected layout:
      lora_dir/
        libero_spatial/
          <task_slug>/
            lora-<name>-step<N>/
              adapter_config.json
              adapter_model.safetensors
        libero_object/...
        libero_goal/...
        libero_10/...
    
    Args:
        lora_dir: Root directory containing CORAL LoRA adapters.
        target_step: If specified, select the checkpoint at exactly this step number.
                     If None, select the latest (highest step) checkpoint.
    
    Returns: {task_key: adapter_path}
      e.g., {"libero_spatial/pick_up_the_black_bowl_...": "/abs/path/lora-xxx-step500/"}
    """
    adapters = {}
    lora_dir = Path(lora_dir)
    if not lora_dir.exists():
        logging.warning(f"LoRA directory does not exist: {lora_dir}")
        return adapters

    for suite_dir in sorted(lora_dir.iterdir()):
        if not suite_dir.is_dir():
            continue
        suite_name = suite_dir.name

        for task_dir in sorted(suite_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            task_slug = task_dir.name

            best_ckpt = None
            best_step = -1
            for ckpt_dir in sorted(task_dir.iterdir()):
                if not ckpt_dir.is_dir():
                    continue
                if not (ckpt_dir / "adapter_config.json").exists():
                    continue
                match = re.search(r"step(\d+)", ckpt_dir.name)
                step = int(match.group(1)) if match else 0

                if target_step is not None:
                    # Select exact step match
                    if step == target_step:
                        best_ckpt = ckpt_dir
                        best_step = step
                        break
                else:
                    # Select latest (highest step)
                    if step > best_step:
                        best_step = step
                        best_ckpt = ckpt_dir

            if best_ckpt:
                key = f"{suite_name}/{task_slug}"
                adapters[key] = str(best_ckpt.absolute())

    if target_step is not None:
        logging.info(f"CORAL adapter scan: target_step={target_step}, found {len(adapters)} adapters")
    return adapters


def _build_prompt_to_lora_map(adapters: Dict[str, str], task_index_path: str = None) -> Dict[str, str]:
    """Build a mapping from task descriptions (prompts) to LoRA adapter keys.
    
    Uses the task_index.json if available, otherwise derives from slug names.
    """
    prompt_map = {}

    if task_index_path and os.path.exists(task_index_path):
        import json
        with open(task_index_path) as f:
            tasks = json.load(f)
        for task in tasks:
            suite = task["suite"]
            slug = task["slug"]
            key = f"{suite}/{slug}"
            if key in adapters:
                prompt_map[task["task"].lower().strip()] = key
    else:
        # Derive from slugs
        for key in adapters:
            parts = key.split("/", 1)
            if len(parts) == 2:
                slug = parts[1]
                prompt = slug.replace("_", " ").strip()
                prompt_map[prompt] = key

    return prompt_map


def _match_prompt_to_lora(prompt: str, prompt_map: Dict[str, str],
                          threshold: float = 0.8) -> Optional[str]:
    """Match a prompt to the closest LoRA adapter using fuzzy matching."""
    prompt_lower = prompt.lower().strip()

    # Exact match
    if prompt_lower in prompt_map:
        return prompt_map[prompt_lower]

    # Fuzzy match
    best_key = None
    best_score = 0.0
    for known_prompt, adapter_key in prompt_map.items():
        score = SequenceMatcher(None, prompt_lower, known_prompt).ratio()
        if score > best_score:
            best_score = score
            best_key = adapter_key

    if best_score >= threshold:
        return best_key

    return None


# ============================================================
# CORAL MoE Manager
# ============================================================
class CORALManager:
    """Manages loading, switching, and unloading of CORAL LoRA experts."""

    def __init__(self, base_model: SmolVLMVLA, lora_adapters: Dict[str, str],
                 prompt_map: Dict[str, str], device: torch.device,
                 merge_default: bool = True):
        self.base_model = base_model
        self.lora_adapters = lora_adapters
        self.prompt_map = prompt_map
        self.device = device
        self.merge_default = merge_default

        self.current_lora_key: Optional[str] = None
        self.current_model: Any = base_model
        self._base_state_dict: Optional[Dict[str, torch.Tensor]] = None

        logging.info(f"CORAL Manager initialized with {len(lora_adapters)} experts")

    def list_loras(self) -> List[str]:
        return sorted(self.lora_adapters.keys())

    def get_current_lora(self) -> Optional[str]:
        return self.current_lora_key

    def _save_base_state(self):
        if self._base_state_dict is None:
            logging.info("Caching base model state...")
            self._base_state_dict = {k: v.clone() for k, v in self.base_model.state_dict().items()}

    def _restore_base_state(self):
        if self._base_state_dict is not None:
            logging.info("Restoring base model state...")
            self.base_model.load_state_dict(self._base_state_dict)
            self.current_lora_key = None
            self.current_model = self.base_model

    def route_prompt(self, prompt: str) -> Optional[str]:
        """Route a task prompt to the best matching LoRA adapter."""
        return _match_prompt_to_lora(prompt, self.prompt_map)

    def load_lora(self, lora_key: str, merge: bool = None) -> bool:
        """Load and activate a LoRA adapter by key."""
        if merge is None:
            merge = self.merge_default
        if lora_key not in self.lora_adapters:
            logging.error(f"LoRA not found: {lora_key}")
            return False
        if lora_key == self.current_lora_key:
            return True

        lora_path = self.lora_adapters[lora_key]
        logging.info(f"CORAL switching: {self.current_lora_key} -> {lora_key}")

        try:
            from peft import PeftModel

            if self.current_lora_key is not None and self._base_state_dict is not None:
                self._restore_base_state()
            self._save_base_state()

            peft_model = PeftModel.from_pretrained(self.base_model, lora_path)
            if merge:
                self.current_model = peft_model.merge_and_unload()
            else:
                self.current_model = peft_model
            self.current_lora_key = lora_key
            self.current_model.eval()
            logging.info(f"CORAL expert activated: {lora_key}")
            return True
        except Exception as e:
            logging.error(f"Failed to load LoRA: {e}")
            import traceback
            traceback.print_exc()
            return False

    def unload_lora(self) -> bool:
        """Unload current LoRA, restoring the base model."""
        if self.current_lora_key is None:
            return True
        try:
            self._restore_base_state()
            logging.info("CORAL expert unloaded, base model restored.")
            return True
        except Exception as e:
            logging.error(f"Failed to unload: {e}")
            return False

    def get_model(self) -> Any:
        return self.current_model


# ============================================================
# CORAL MoE Server
# ============================================================
class SmolVLMCORALServer:
    """SmolVLM-VLA CORAL MoE WebSocket Server for LIBERO evaluation.
    
    Features:
      - Auto-routes task prompts to the correct LoRA expert
      - Compatible with openpi_client WebSocket protocol (msgpack)
      - Supports dynamic LoRA switching per request
    """

    def __init__(self, coral_manager: CORALManager, base_model: SmolVLMVLA,
                 processor: SmolVLMVLAProcessor, host: str = "0.0.0.0", port: int = 8089,
                 default_steps: int = 10, default_lora: Optional[str] = None):
        self.coral_manager = coral_manager
        self.processor = processor
        self.host = host
        self.port = port
        self.default_steps = default_steps
        self.default_lora = default_lora

        self.device = next(base_model.parameters()).device
        self.dtype = next(base_model.parameters()).dtype

        dim_action = getattr(getattr(base_model, "action_space", None), "dim_action", 7)
        num_actions = getattr(base_model, "num_actions", 10)

        self.metadata = {
            "action_mode": getattr(base_model, "action_mode", "libero_joint"),
            "dim_action": dim_action,
            "num_actions": num_actions,
            "server_type": "smolvlm_coral_moe",
            "has_norm_stats": getattr(getattr(base_model, "action_space", None), "action_norm_stats", None) is not None,
            "available_loras": coral_manager.list_loras(),
            "current_lora": coral_manager.get_current_lora(),
        }

        if default_lora:
            coral_manager.load_lora(default_lora)

    def _to_model(self, t: torch.Tensor) -> torch.Tensor:
        if not isinstance(t, torch.Tensor):
            t = torch.as_tensor(t)
        return t.to(device=self.device, dtype=self.dtype) if t.is_floating_point() else t.to(device=self.device)

    def infer(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # Handle LoRA management requests
            if obs.get("list_loras") or obs.get("get_loras"):
                return {
                    "available_loras": self.coral_manager.list_loras(),
                    "current_lora": self.coral_manager.get_current_lora(),
                }

            if obs.get("unload_lora"):
                success = self.coral_manager.unload_lora()
                return {"success": success, "current_lora": None}

            # Auto-route prompt to LoRA expert
            prompt = obs.get("prompt") or obs.get("language_instruction") or ""
            if prompt:
                lora_key = self.coral_manager.route_prompt(prompt)
                if lora_key:
                    if lora_key != self.coral_manager.get_current_lora():
                        logging.info(f"Auto-routing '{prompt[:50]}...' -> {lora_key}")
                        self.coral_manager.load_lora(lora_key)
                elif self.coral_manager.get_current_lora() is not None:
                    logging.info(f"No LoRA match for '{prompt[:60]}...', falling back to base model")
                    self.coral_manager.unload_lora()

            # Manual LoRA override
            explicit_lora = obs.get("lora_name") or obs.get("lora")
            if explicit_lora:
                self.coral_manager.load_lora(explicit_lora)

            model = self.coral_manager.get_model()

            # Decode images
            images = _decode_images_from_obs(obs)
            if not images:
                num_actions = self.metadata["num_actions"]
                dim_action = self.metadata["dim_action"]
                return {"error": "No valid images", "actions": np.zeros((num_actions, dim_action))}

            # Process inputs through SmolVLM processor
            inputs = self.processor(images, prompt)

            # Extract proprioceptive state
            proprio_np = _extract_state_from_obs(obs)
            if proprio_np is None:
                proprio_np = np.zeros(8, dtype=np.float32)  # LIBERO: 8D state

            # Move to model device
            inputs = {k: self._to_model(v) for k, v in inputs.items()}
            inputs["proprio"] = self._to_model(torch.as_tensor(proprio_np).unsqueeze(0))

            # Ensure image_input is float32 (not bf16)
            if isinstance(inputs.get("image_input"), torch.Tensor):
                inputs["image_input"] = inputs["image_input"].to(device=self.device, dtype=torch.float32)

            # Generate actions
            steps = int(obs.get("steps", self.default_steps))
            with torch.no_grad():
                action_raw = model.generate_actions(**inputs, steps=steps).squeeze(0).float().cpu().numpy()

            result = {
                "actions": action_raw,
                "action": action_raw,
                "current_lora": self.coral_manager.get_current_lora(),
            }
            return result

        except Exception as exc:
            import traceback
            traceback.print_exc()
            num_actions = self.metadata.get("num_actions", 10)
            dim_action = self.metadata.get("dim_action", 7)
            return {"error": str(exc), "actions": np.zeros((num_actions, dim_action))}

    async def handle_connection(self, websocket):
        client_addr = websocket.remote_address
        logging.info(f"Client connected: {client_addr}")

        try:
            self.metadata["current_lora"] = self.coral_manager.get_current_lora()
            await websocket.send(msgpack_packb(self.metadata))
        except Exception as e:
            logging.error(f"Failed to send metadata: {e}")
            return

        try:
            async for message in websocket:
                try:
                    obs = msgpack_unpackb(message)
                    result = self.infer(obs)
                    await websocket.send(msgpack_packb(result))
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    num_actions = self.metadata.get("num_actions", 10)
                    dim_action = self.metadata.get("dim_action", 7)
                    await websocket.send(msgpack_packb({
                        "error": str(e),
                        "actions": np.zeros((num_actions, dim_action))
                    }))
        except websockets.exceptions.ConnectionClosed:
            logging.info(f"Client disconnected: {client_addr}")

    async def serve(self):
        logging.info(f"Starting CORAL MoE server: ws://{self.host}:{self.port}")
        async with websockets.serve(
            self.handle_connection, self.host, self.port,
            compression=None, max_size=None
        ):
            await asyncio.Future()

    def serve_forever(self):
        asyncio.run(self.serve())


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="CORAL (Continual Robot Skill Learning via LoRA Experts) SmolVLM-VLA MoE Server")
    parser.add_argument("--model", default="YuankaiLuo/SimVLA-LIBERO",
                        help="Base SimVLA checkpoint path or HuggingFace repo ID")
    parser.add_argument("--lora-dir", type=str, required=True,
                        help="Root directory containing CORAL LoRA adapters")
    parser.add_argument("--lora-step", type=int, default=None,
                        help="Select checkpoint at this exact step number (default: latest)")
    parser.add_argument("--task-index", type=str, default=None,
                        help="Path to datasets/coral_metas/task_index.json for prompt routing")
    parser.add_argument("--default-lora", type=str, default=None,
                        help="Default LoRA to load at startup")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8089)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--steps", type=int, default=10, help="Flow matching sampling steps")
    parser.add_argument("--norm-stats", type=str, default="./norm_stats/libero_norm.json")
    parser.add_argument("--no-merge", action="store_true",
                        help="Keep LoRA in adapter mode (faster switch, slower inference)")
    args = parser.parse_args()

    if not WEBSOCKETS_AVAILABLE:
        print("Error: websockets required. Install: pip install websockets"); sys.exit(1)
    if not MSGPACK_AVAILABLE:
        print("Error: msgpack required. Install: pip install msgpack"); sys.exit(1)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")

    device = _resolve_device(args.device)
    dtype = _resolve_dtype(args.dtype, device)

    # Scan LoRA adapters
    step_msg = f" (target step: {args.lora_step})" if args.lora_step else " (latest step)"
    print(f"Scanning CORAL LoRA adapters...{step_msg}")
    lora_adapters = scan_coral_adapters(args.lora_dir, target_step=args.lora_step)
    if not lora_adapters:
        print(f"Warning: No LoRA adapters found in: {args.lora_dir}")
        print("Running in base model mode (no CORAL routing)")

    print(f"Found {len(lora_adapters)} CORAL experts:")
    for name in sorted(lora_adapters.keys()):
        print(f"  - {name}")
    print()

    # Build prompt-to-LoRA map
    task_index_path = args.task_index
    if task_index_path is None:
        # Auto-detect
        candidate = os.path.join(os.path.dirname(args.lora_dir), "datasets/coral_metas", "task_index.json")
        if os.path.exists(candidate):
            task_index_path = candidate
        else:
            candidate = "./datasets/coral_metas/task_index.json"
            if os.path.exists(candidate):
                task_index_path = candidate

    prompt_map = _build_prompt_to_lora_map(lora_adapters, task_index_path)
    print(f"Prompt routing table ({len(prompt_map)} entries):")
    for prompt, key in sorted(prompt_map.items())[:10]:
        print(f"  '{prompt[:60]}' -> {key}")
    if len(prompt_map) > 10:
        print(f"  ... and {len(prompt_map) - 10} more")
    print()

    # Load base model
    print("=" * 70)
    print("CORAL: SmolVLM-VLA Mixture-of-Experts Server")
    print("=" * 70)
    print(f"  Base model    : {args.model}")
    print(f"  LoRA dir      : {args.lora_dir}")
    print(f"  LoRA step     : {args.lora_step or 'latest'}")
    print(f"  Num experts   : {len(lora_adapters)}")
    print(f"  Default LoRA  : {args.default_lora or 'none (auto-route)'}")
    print(f"  Listen        : ws://0.0.0.0:{args.port}")
    print(f"  Device        : {device}")
    print(f"  Dtype         : {dtype}")
    print(f"  LoRA mode     : {'adapter (no merge)' if args.no_merge else 'merged (faster inference)'}")
    print()

    print("Loading base model...")
    model = SmolVLMVLA.from_pretrained(
        args.model, trust_remote_code=True,
        torch_dtype=dtype if device.type == "cuda" else torch.float32,
    )
    model = model.to(device=device, dtype=torch.float32)
    print("Base model loaded (SmolVLMVLA).")

    model.eval()

    # Load normalization stats
    if args.norm_stats and os.path.exists(args.norm_stats):
        from models.action_hub import build_action_space
        print(f"Loading normalization stats: {args.norm_stats}")
        action_mode = getattr(model, "action_mode", "libero_joint")
        model.action_space = build_action_space(action_mode, norm_stats_path=args.norm_stats)
        model.action_space.to(device)
        print("Normalization stats loaded.")

    # Create CORAL manager
    coral_manager = CORALManager(
        base_model=model, lora_adapters=lora_adapters,
        prompt_map=prompt_map, device=device,
        merge_default=not args.no_merge,
    )

    # Load processor
    smolvlm_model_path = getattr(model.config, "smolvlm_model_path", "HuggingFaceTB/SmolVLM-500M-Instruct")
    try:
        processor = SmolVLMVLAProcessor.from_pretrained(smolvlm_model_path)
    except Exception:
        processor = SmolVLMVLAProcessor.from_pretrained("HuggingFaceTB/SmolVLM-500M-Instruct")

    # Create server
    server = SmolVLMCORALServer(
        coral_manager=coral_manager, base_model=model,
        processor=processor, host=args.host, port=args.port,
        default_steps=args.steps, default_lora=args.default_lora,
    )

    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except Exception:
        local_ip = "0.0.0.0"

    print()
    print("=" * 70)
    print("CORAL MoE WebSocket Server Starting...")
    print("=" * 70)
    print(f"  Connect at: ws://{local_ip}:{args.port}")
    print()
    print("  Auto-routing: prompt -> LoRA expert (fuzzy matching)")
    print()
    print("  Client API:")
    print("    List LoRAs     : {\"list_loras\": true}")
    print("    Switch LoRA    : {\"lora_name\": \"suite/task_slug\"}")
    print("    Auto-route     : {\"prompt\": \"task description\", ...} (default)")
    print("    Unload LoRA    : {\"unload_lora\": true}")
    print()
    print("  Available CORAL experts:")
    for name in sorted(lora_adapters.keys()):
        print(f"    - {name}")
    print("=" * 70)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    except Exception as exc:
        print(f"Server error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
