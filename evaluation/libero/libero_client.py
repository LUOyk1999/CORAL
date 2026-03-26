#!/usr/bin/env python3
"""
SimVLA LIBERO Evaluation Client

Observation format:
1. State: [eef_pos(3), axis_angle(3), gripper_qpos(2)] = 8D
2. Action: delta action (7D)
3. Default delta control mode
4. Images rotated 180 degrees
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Deque, Dict, List, Optional


import imageio
import json_numpy
import numpy as np
import requests
from tqdm import tqdm

try:
    from openpi_client import image_tools
    from openpi_client import websocket_client_policy as ws_client
    HAS_WS_CLIENT = True
except ImportError:
    HAS_WS_CLIENT = False

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256

# Max steps per task suite (based on longest demo + buffer)
MAX_STEPS = {
    "libero_spatial": 800,   # longest demo: 193
    "libero_object": 800,    # longest demo: 254
    "libero_goal": 800,      # longest demo: 270
    "libero_10": 900,        # longest demo: 505
    "libero_90": 900,        # longest demo: 373
}

NUM_STEPS_WAIT = 10  # Wait for objects to stabilize

benchmark_dict = benchmark.get_benchmark_dict()


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """
    Convert quaternion [x, y, z, w] to axis-angle representation.
    
    Uses the same convention as robosuite for consistency with training data.
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


# -----------------------------------------------------------------------------
# Client Policy Classes
# -----------------------------------------------------------------------------

class WebSocketClient:
    """
    WebSocket client for SimVLA server.
    
    Requires: pip install openpi-client
    """
    def __init__(self, host: str, port: int, replan_steps: int = 5, resize_size: int = 224):
        if not HAS_WS_CLIENT:
            raise ImportError("openpi_client not installed. Run: pip install openpi-client")
        self.client = ws_client.WebsocketClientPolicy(host, port)
        self.replan_steps = replan_steps
        self.resize_size = resize_size
        self.reset()

    def reset(self) -> None:
        self.action_plan: Deque[np.ndarray] = collections.deque()

    def step(self, obs: Dict, goal: str) -> np.ndarray:
        if not self.action_plan:
            # Preprocess images
            img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(obs["image"], self.resize_size, self.resize_size)
            )
            wrist_img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(obs["wrist_image"], self.resize_size, self.resize_size)
            )
            
            # Build observation dict
            element = {
                "observation/image": img,
                "observation/wrist_image": wrist_img,
                "observation/state": obs["state"],
                "prompt": goal,
            }
            
            # Query server
            result = self.client.infer(element)
            action_chunk = result["actions"]
            
            # Ensure numpy array
            if not isinstance(action_chunk, np.ndarray):
                action_chunk = np.array(action_chunk)
            
            assert len(action_chunk) >= self.replan_steps, \
                f"Need {self.replan_steps} steps but got {len(action_chunk)}"
            
            for i in range(min(self.replan_steps, len(action_chunk))):
                self.action_plan.append(action_chunk[i])

        return self.action_plan.popleft()


class HTTPClient:
    """
    HTTP client for SimVLA server.
    """
    def __init__(self, host: str, port: int, replan_steps: int = 5):
        self.url = f"http://{host}:{port}/act"
        self.replan_steps = replan_steps
        self.reset()

    def reset(self) -> None:
        self.action_plan: Deque[np.ndarray] = collections.deque()

    def infer(self, element: Dict) -> Dict:
        try:
            payload = {
                "image0": json_numpy.dumps(element["observation/image"]),
                "image1": json_numpy.dumps(element["observation/wrist_image"]),
                "proprio": json_numpy.dumps(element["observation/state"]),
                "language_instruction": element["prompt"],
                "steps": 10,
            }
            
            resp = requests.post(self.url, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            
            actions = np.array(data["action"])
            return {"actions": actions}
            
        except Exception as e:
            raise RuntimeError(f"Policy server request failed: {e}") from e

    def step(self, obs: Dict, goal: str) -> np.ndarray:
        if not self.action_plan:
            element = {
                "observation/image": obs["image"],
                "observation/wrist_image": obs["wrist_image"],
                "observation/state": obs["state"],
                "prompt": goal,
            }
            
            result = self.infer(element)
            action_chunk = result["actions"]
            
            for action in action_chunk[:self.replan_steps]:
                self.action_plan.append(action)

        return self.action_plan.popleft()


# -----------------------------------------------------------------------------
# Evaluator
# -----------------------------------------------------------------------------
def get_libero_env(task, resolution: int, seed: int):
    """Initialize a LIBERO environment."""
    task_description = task.language
    task_bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": str(task_bddl_file), "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def eval_libero(
    client,
    task_suite_name: str,
    num_trials: int = 50,
    seed: int = 7,
    video_out_path: str = "data/libero/videos",
    save_video: bool = True,
) -> float:
    """
    Run LIBERO evaluation across all tasks in a suite.
    """
    np.random.seed(seed)
    
    # Initialize task suite
    task_suite = benchmark_dict[task_suite_name]()
    num_tasks = task_suite.n_tasks
    max_steps = MAX_STEPS.get(task_suite_name, 400)
    
    Path(video_out_path).mkdir(parents=True, exist_ok=True)
    
    print(f"Task suite: {task_suite_name}")
    print(f"   Tasks: {num_tasks}, Trials per task: {num_trials}")
    print(f"   Max steps: {max_steps}")
    
    total_episodes, total_successes = 0, 0
    
    for task_id in tqdm(range(num_tasks - 1, -1, -1), desc="Tasks"):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, LIBERO_ENV_RESOLUTION, seed)
        
        task_successes = 0
        for ep in tqdm(range(num_trials), desc=f"{task_description[:30]}...", leave=False):
            # Reset
            env.reset()
            client.reset()
            obs = env.set_init_state(initial_states[ep % len(initial_states)])
            
            replay_images = []
            t = 0
            done = False
            
            while t < max_steps + NUM_STEPS_WAIT:
                try:
                    # Wait for objects to stabilize
                    if t < NUM_STEPS_WAIT:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue
                    
                    # Get images (rotated 180 degrees)
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    
                    if save_video:
                        replay_images.append(img)
                    
                    # Build state vector
                    # [eef_pos(3), axis_angle(3), gripper_qpos(2)] = 8D
                    state = np.concatenate([
                        obs["robot0_eef_pos"],
                        _quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],
                    ])
                    
                    # Pack observation
                    obs_dict = {
                        "image": img,
                        "wrist_image": wrist_img,
                        "state": state,
                    }
                    
                    # Get action (7D delta action)
                    action = client.step(obs_dict, task_description)
                    
                    # Execute (send delta action directly)
                    obs, reward, done, info = env.step(action.tolist())
                    
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    
                    t += 1
                    
                except Exception as e:
                    print(f"Error in rollout: {e}")
                    break

            total_episodes += 1
            
            # Save video
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")[:50]
            video_path = Path(video_out_path) / f"{task_segment}_ep{ep}_{suffix}.mp4"
            if replay_images and save_video:
                imageio.mimwrite(str(video_path), replay_images, fps=10)
            
            # Print episode result
            status_icon = "[OK]" if done else "[FAIL]"
            print(f"  {status_icon} Task {task_id} Ep {ep}: {suffix.upper()} (steps={t})")

        env.close()
        print(f"   Task {task_id}: {task_successes}/{num_trials} ({task_successes/num_trials*100:.1f}%)")
    
    success_rate = total_successes / max(total_episodes, 1)
    print(f"\nTotal success rate: {total_successes}/{total_episodes} ({success_rate*100:.1f}%)")
    
    return success_rate


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser("LIBERO Evaluation Client")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--connection_info", type=str, default=None,
                        help="Path to server connection info JSON")
    parser.add_argument("--client_type", type=str, default="websocket",
                        choices=["websocket", "http"],
                        help="Client type: websocket or http")
    parser.add_argument("--task_suite", type=str, default="libero_spatial",
                        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"])
    parser.add_argument("--num_trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--replan_steps", type=int, default=5)
    parser.add_argument("--video_out", type=str, default="./eval_results")
    parser.add_argument("--no_video", action="store_true", help="Disable video recording for faster evaluation")

    args = parser.parse_args()

    # Load connection info
    if args.connection_info:
        print(f"Loading connection info from: {args.connection_info}")
        while not Path(args.connection_info).exists():
            sys.stdout.write("\rWaiting for server...")
            sys.stdout.flush()
            time.sleep(0.5)
        print()
        with open(args.connection_info) as f:
            info = json.load(f)
            args.host = info["host"]
            args.port = info["port"]
    
    protocol = "ws" if args.client_type == "websocket" else "http"
    print(f"Starting LIBERO evaluation client")
    print(f"   Client type: {args.client_type}")
    print(f"   Server: {protocol}://{args.host}:{args.port}")
    print(f"   Task suite: {args.task_suite}")
    print(f"   Replan steps: {args.replan_steps}")
    print()
    
    # Initialize client
    if args.client_type == "websocket":
        client = WebSocketClient(args.host, args.port, replan_steps=args.replan_steps)
    else:
        client = HTTPClient(args.host, args.port, replan_steps=args.replan_steps)
    
    # Run evaluation
    video_path = Path(args.video_out) / args.task_suite
    eval_libero(
        client=client,
        task_suite_name=args.task_suite,
        num_trials=args.num_trials,
        seed=args.seed,
        video_out_path=str(video_path),
        save_video=not args.no_video,
    )


if __name__ == "__main__":
    main()
