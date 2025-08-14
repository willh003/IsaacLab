# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module with utilities for parsing and loading configurations."""

import importlib
import json
import os
import yaml
import time
import datetime
import robomimic
import torch
import ast


# Global registry cache
_TASK_REGISTRY_CACHE = None


def _extract_gym_register_from_file(file_path: str) -> list[dict]:
    """Extract gym.register calls from a Python file without importing it."""
    registrations = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse the AST
        tree = ast.parse(content)
        
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and 
                isinstance(node.func, ast.Attribute) and
                isinstance(node.func.value, ast.Name) and
                node.func.value.id == 'gym' and
                node.func.attr == 'register'):
                
                # Extract the registration details
                registration = {}
                
                # Extract keyword arguments
                for keyword in node.keywords:
                    if keyword.arg == 'id':
                        if isinstance(keyword.value, ast.Constant):
                            registration['id'] = keyword.value.value
                    elif keyword.arg == 'kwargs':
                        if isinstance(keyword.value, ast.Dict):
                            kwargs = {}
                            for k, v in zip(keyword.value.keys, keyword.value.values):
                                if isinstance(k, ast.Constant) and isinstance(v, ast.JoinedStr):
                                    # Handle f-strings - this is more complex, we'll approximate
                                    kwargs[k.value] = _reconstruct_fstring(v, file_path)
                                elif isinstance(k, ast.Constant) and isinstance(v, ast.Constant):
                                    kwargs[k.value] = v.value
                            registration['kwargs'] = kwargs
                
                if 'id' in registration and 'kwargs' in registration:
                    registrations.append(registration)
                    
    except Exception as e:
        # Skip files that can't be parsed
        pass
    
    return registrations


def _reconstruct_fstring(node: ast.JoinedStr, file_path: str) -> str:
    """Reconstruct f-string value by replacing common patterns."""
    result = ""
    for value in node.values:
        if isinstance(value, ast.Constant):
            result += value.value
        elif isinstance(value, ast.FormattedValue):
            if isinstance(value.value, ast.Attribute):
                # Handle patterns like {agents.__name__}
                if (isinstance(value.value.value, ast.Name) and 
                    value.value.value.id == "agents" and
                    value.value.attr == "__name__"):
                    # This is {agents.__name__} - convert to module path
                    module_dir = os.path.dirname(file_path)
                    agents_path = os.path.join(module_dir, "agents")
                    if os.path.exists(agents_path):
                        # Get relative module path for agents from isaaclab_tasks root
                        path_parts = file_path.split(os.sep)
                        if "isaaclab_tasks" in path_parts:
                            idx = path_parts.index("isaaclab_tasks")
                            # Build module path to agents folder
                            module_parts = path_parts[idx:-1] + ["agents"]  # exclude __init__.py, add agents
                            result += ".".join(module_parts)
                        else:
                            result += "agents"
                    else:
                        result += "agents"
            elif isinstance(value.value, ast.Name) and value.value.id == "__name__":
                # Handle {__name__} - convert file path to module name
                path_parts = file_path.split(os.sep)
                if "isaaclab_tasks" in path_parts:
                    idx = path_parts.index("isaaclab_tasks")
                    # Take everything from isaaclab_tasks to the directory containing __init__.py
                    module_parts = path_parts[idx:-1]  # exclude __init__.py filename
                    result += ".".join(module_parts)
    return result


def build_task_registry_from_files() -> dict:
    """Build task registry by scanning isaaclab_tasks files."""
    global _TASK_REGISTRY_CACHE
    
    if _TASK_REGISTRY_CACHE is not None:
        return _TASK_REGISTRY_CACHE
    
    print("[INFO]: Building task registry from isaaclab_tasks files...")
    
    registry = {}
    
    # Find isaaclab_tasks directory
    current_dir = os.path.dirname(__file__)  # dexterous/dp/
    repo_root = os.path.dirname(os.path.dirname(current_dir))  # IsaacLab/
    isaaclab_tasks_dir = os.path.join(repo_root, "source", "isaaclab_tasks", "isaaclab_tasks")
    
    if not os.path.exists(isaaclab_tasks_dir):
        print(f"[WARNING]: Could not find isaaclab_tasks directory at {isaaclab_tasks_dir}")
        return registry
    
    # Recursively scan for __init__.py files
    for root, dirs, files in os.walk(isaaclab_tasks_dir):
        if "__init__.py" in files:
            init_file = os.path.join(root, "__init__.py")
            registrations = _extract_gym_register_from_file(init_file)
            
            for reg in registrations:
                task_id = reg['id']
                registry[task_id] = reg['kwargs']
    
    _TASK_REGISTRY_CACHE = registry
    print(f"[INFO]: Found {len(registry)} task registrations")
    return registry


def load_cfg_from_registry_no_gym(task_name: str, entry_point_key: str) -> dict | object:
    """Load configuration without gym registry dependency.
    
    This function provides the same functionality as load_cfg_from_registry but without
    requiring gym registration, making it suitable for cluster environments without IsaacLab.
    """
    # Build registry from files
    registry = build_task_registry_from_files()
    
    # Clean task name
    clean_task_name = task_name.split(":")[-1]
    
    # Get task config mapping
    if clean_task_name not in registry:
        raise ValueError(
            f"Task '{clean_task_name}' not found in task registry. "
            f"Available tasks: {list(registry.keys())}"
        )
    
    task_mapping = registry[clean_task_name]
    
    # Get config entry point
    if entry_point_key not in task_mapping:
        available_keys = list(task_mapping.keys())
        raise ValueError(
            f"Entry point '{entry_point_key}' not found for task '{clean_task_name}'. "
            f"Available entry points: {available_keys}"
        )
    
    cfg_entry_point = task_mapping[entry_point_key]
    
    # Parse the configuration (same logic as original function but avoid module imports for JSON/YAML)
    if isinstance(cfg_entry_point, str) and (cfg_entry_point.endswith(".yaml") or cfg_entry_point.endswith(".json")):
        if os.path.exists(cfg_entry_point):
            # absolute path for the config file
            config_file = cfg_entry_point
        else:
            # resolve path without importing modules (to avoid isaaclab dependencies)
            if ":" in cfg_entry_point:
                mod_name, file_name = cfg_entry_point.split(":")
                # Convert module name to file path
                repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # go up from dexterous/dp/
                isaaclab_tasks_root = os.path.join(repo_root, "source")
                mod_path = os.path.join(isaaclab_tasks_root, mod_name.replace(".", os.sep))
                config_file = os.path.join(mod_path, file_name)
            else:
                # assume relative to repo root
                repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # go up from dexterous/dp/
                config_file = os.path.join(repo_root, cfg_entry_point)
        
        # load the configuration
        print(f"[INFO]: Parsing configuration from: {config_file}")
        with open(config_file, encoding="utf-8") as f:
            if cfg_entry_point.endswith(".yaml"):
                cfg = yaml.full_load(f)
            else:  # .json file
                cfg = json.load(f)
    else:
        if callable(cfg_entry_point):
            cfg_cls = cfg_entry_point()
        elif isinstance(cfg_entry_point, str):
            # resolve path to the module location
            mod_name, attr_name = cfg_entry_point.split(":")
            mod = importlib.import_module(mod_name)
            cfg_cls = getattr(mod, attr_name)
        else:
            cfg_cls = cfg_entry_point
        
        # load the configuration
        print(f"[INFO]: Parsing configuration from: {cfg_entry_point}")
        if callable(cfg_cls):
            cfg = cfg_cls()
        else:
            cfg = cfg_cls
    
    return cfg


def detect_z_rotation_direction_batch(quaternions):
    """
    Determine rotation direction about z-axis for multiple environments.
    
    Args:
        quaternions: torch.Tensor of shape (n_steps, n_envs, 4)
    
    Returns:
        torch.Tensor of shape (n_envs) with 1 for CCW, -1 for CW, 0 for no rotation
    """
    # Stack deque into tensor: (n_observations, n_envs, 4)
    
    # Normalize quaternions
    q = quaternions / torch.norm(quaternions, dim=2, keepdim=True)
    
    # Compute derivatives
    q_dot = q[1:] - q[:-1]  # (n_obs-1, n_envs, 4)
    
    # Quaternion conjugate
    q_conj = torch.cat([q[:-1, :, :1], -q[:-1, :, 1:]], dim=2)
    
    # Vectorized quaternion multiplication for omega_z
    w1, x1, y1, z1 = q_dot.unbind(dim=2)
    w2, x2, y2, z2 = q_conj.unbind(dim=2)
    
    omega_z = 2 * (w1*z2 + x1*y2 - y1*x2 + z1*w2)
    mean_omega_z = omega_z.mean(dim=0)  # (n_envs,)
    
    return torch.sign(mean_omega_z)

def load_action_normalization_params(checkpoint_path):
    # Go up two directories and into logs/normalization_params.txt
    exp_dir = os.path.dirname(os.path.dirname(checkpoint_path))
    norm_file = os.path.join(exp_dir, "logs", "normalization_params.txt")
    with open(norm_file, "r") as f:
        lines = f.readlines()
        min_val = float(lines[0].split(":")[1].strip())
        max_val = float(lines[1].split(":")[1].strip())
    return min_val, max_val

def unnormalize_actions(actions, min_val, max_val):
    # actions: torch.Tensor or np.ndarray in [-1, 1]
    return 0.5 * (actions + 1) * (max_val - min_val) + min_val

def count_parameters(model):
    """Count the total number of parameters in a PyTorch model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def get_exp_dir(output_dir, experiment_name, save_enabled=True):
    """
    Create experiment directory from config. If an identical experiment directory
    exists and @auto_remove_exp_dir is False (default), the function will prompt 
    the user on whether to remove and replace it, or keep the existing one and
    add a new subdirectory with the new timestamp for the current run.

    Args:
        auto_remove_exp_dir (bool): if True, automatically remove the existing experiment
            folder if it exists at the same path.
    
    Returns:
        log_dir (str): path to created log directory (sub-folder in experiment directory)
        output_dir (str): path to created models directory (sub-folder in experiment directory)
            to store model checkpoints
        video_dir (str): path to video directory (sub-folder in experiment directory)
            to store rollout videos
    """
    # timestamp for directory names
    t_now = time.time()
    time_str = datetime.datetime.fromtimestamp(t_now).strftime('%Y%m%d%H%M%S')

    # create directory for where to dump model parameters, tensorboard logs, and videos
    base_output_dir = os.path.expanduser(output_dir)
    if not os.path.isabs(base_output_dir):
        # relative paths are specified relative to robomimic module location
        base_output_dir = os.path.join(robomimic.__path__[0], base_output_dir)
    base_output_dir = os.path.join(base_output_dir, experiment_name, time_str)
    if os.path.exists(base_output_dir):
        raise FileExistsError(f"Experiment directory {base_output_dir} already exists!")

    # only make model directory if model saving is enabled
    model_dir = None
    if save_enabled:
        model_dir = os.path.join(base_output_dir, "models")
        os.makedirs(model_dir)

    # tensorboard directory
    log_dir = os.path.join(base_output_dir, "logs")
    os.makedirs(log_dir)

    # video directory
    video_dir = os.path.join(base_output_dir, "videos")
    os.makedirs(video_dir)
    return log_dir, model_dir, video_dir