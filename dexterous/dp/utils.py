# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module with utilities for parsing and loading configurations."""

import importlib
import json
import os
from typing_extensions import override
import yaml
import time
import datetime
import robomimic
import torch
import ast
import argparse

from robomimic.utils import obs_utils as ObsUtils
from robomimic.utils import torch_utils as TorchUtils
from robomimic.utils import file_utils as FileUtils
from robomimic.algo import algo_factory, RolloutPolicy

# Global registry cache
_TASK_REGISTRY_CACHE = None


ADDL_CONFIG_KEYS = ["goal_mode", "goal_horizon", "noise_groups", "mask_observations", "train_mask_prob", "uncond_weight"]
def filter_config_dict(cfg, base_cfg):
    """
    Recursively filter out keys from cfg that are not present in base_cfg.
    """
    if not isinstance(cfg, dict) or not hasattr(base_cfg, 'keys'):
        return cfg
    filtered = {}
    for k, v in cfg.items():
        if k in base_cfg:
            if isinstance(v, dict) and hasattr(base_cfg[k], 'keys'):
                filtered[k] = filter_config_dict(v, base_cfg[k])
            else:
                filtered[k] = v
        elif k in ADDL_CONFIG_KEYS:
            filtered[k] = v
    return filtered


def dict_to_namespace(config_dict):
    for key, value in config_dict.items():
        if isinstance(value, dict):
            config_dict[key] = dict_to_namespace(value)
    return argparse.Namespace(**config_dict)

def policy_from_checkpoint_override_cfg(device=None, ckpt_path=None, ckpt_dict=None, verbose=False, override_config=None):
    """
    From robomimic, but allow config override of policy

    This function restores a trained policy from a checkpoint file or
    loaded model dictionary.

    Args:
        device (torch.device): if provided, put model on this device

        ckpt_path (str): Path to checkpoint file. Only needed if not providing @ckpt_dict.

        ckpt_dict(dict): Loaded model checkpoint dictionary. Only needed if not providing @ckpt_path.

        verbose (bool): if True, include print statements

    Returns:
        model (RolloutPolicy): instance of Algo that has the saved weights from
            the checkpoint file, and also acts as a policy that can easily
            interact with an environment in a training loop

        ckpt_dict (dict): loaded checkpoint dictionary (convenient to avoid
            re-loading checkpoint from disk multiple times)
    """
    ckpt_dict = FileUtils.maybe_dict_from_checkpoint(ckpt_path=ckpt_path, ckpt_dict=ckpt_dict)

    # algo name and config from model dict
    algo_name, _ = FileUtils.algo_name_from_checkpoint(ckpt_dict=ckpt_dict)
    config, _ = FileUtils.config_from_checkpoint(algo_name=algo_name, ckpt_dict=ckpt_dict, verbose=verbose)

    if override_config is not None:
        filtered_config = filter_config_dict(override_config, config)

        with config.unlocked():
            config.update(filtered_config)

    # read config to set up metadata for observation modalities (e.g. detecting rgb observations)
    ObsUtils.initialize_obs_utils_with_config(config)

    # shape meta from model dict to get info needed to create model
    shape_meta = ckpt_dict["shape_metadata"]

    # maybe restore observation normalization stats
    obs_normalization_stats = ckpt_dict.get("obs_normalization_stats", None)
    if obs_normalization_stats is not None:
        assert config.train.hdf5_normalize_obs
        for m in obs_normalization_stats:
            for k in obs_normalization_stats[m]:
                obs_normalization_stats[m][k] = np.array(obs_normalization_stats[m][k])

    if device is None:
        # get torch device
        device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)

    # create model and load weights
    model = algo_factory(
        algo_name,
        config,
        obs_key_shapes=shape_meta["all_shapes"],
        ac_dim=shape_meta["ac_dim"],
        device=device,
    )

    model.deserialize(ckpt_dict["model"])
    model.set_eval()
    model = RolloutPolicy(model, obs_normalization_stats=obs_normalization_stats)
    if verbose:
        print("============= Loaded Policy =============")
        print(model)
    return model, ckpt_dict

def clear_task_registry_cache():
    """Clear the task registry cache to force rebuild."""
    global _TASK_REGISTRY_CACHE
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
                                elif isinstance(k, ast.Constant) and isinstance(v, ast.Call):
                                    # Handle function calls like get_robomimic_entry_point()
                                    kwargs[k.value] = _handle_function_call(v, file_path)
                            registration['kwargs'] = kwargs
                
                if 'id' in registration and 'kwargs' in registration:
                    registrations.append(registration)
                    
    except Exception as e:
        # Skip files that can't be parsed
        pass
    
    return registrations


def _handle_function_call(node: ast.Call, file_path: str) -> str:
    """Handle function calls in gym.register kwargs, specifically get_robomimic_entry_point()."""
    if (isinstance(node.func, ast.Name) and 
        node.func.id == "get_robomimic_entry_point"):
        # Extract the arguments
        if len(node.args) >= 2:
            algo_name = None
            default_entry_point = None
            
            # Get algorithm name (first arg)
            if isinstance(node.args[0], ast.Constant):
                algo_name = node.args[0].value
            
            # Get default entry point (second arg)
            if isinstance(node.args[1], ast.JoinedStr):
                default_entry_point = _reconstruct_fstring(node.args[1], file_path)
            elif isinstance(node.args[1], ast.Constant):
                default_entry_point = node.args[1].value
            
            if algo_name and default_entry_point:
                # Check environment variable
                env_var_name = f"ROBOMIMIC_{algo_name.upper()}_CFG_ENTRY_POINT"
                override = os.environ.get(env_var_name)
                if override:
                    print(f"[INFO] Using override for {env_var_name}: {override}")
                    return override
                else:
                    print(f"[INFO] No override found for {env_var_name}, using default: {default_entry_point}")
                    return default_entry_point
    
    # Fallback: return empty string for unhandled cases
    return ""

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
    current_dir = os.path.dirname(os.path.abspath(__file__))  # dexterous/dp/
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

# def load_action_normalization_params(checkpoint_path):
#     """Load action normalization parameters from checkpoint.
    
#     Returns numpy arrays for per-dimension normalization.
#     Falls back to scalar values for backward compatibility.
#     """
#     import numpy as np
#     import ast
    
#     # Go up two directories and into logs/normalization_params.txt
#     exp_dir = os.path.dirname(os.path.dirname(checkpoint_path))
#     norm_file = os.path.join(exp_dir, "logs", "normalization_params.txt")
    
#     with open(norm_file, "r") as f:
#         lines = f.readlines()
#         min_str = lines[0].split(":", 1)[1].strip()
#         max_str = lines[1].split(":", 1)[1].strip()
        
#         try:
#             # Try to parse as list (per-dimension)
#             min_val = np.array(ast.literal_eval(min_str))
#             max_val = np.array(ast.literal_eval(max_str))
#         except (ValueError, SyntaxError):
#             # Fall back to scalar (backward compatibility)
#             min_val = np.array(float(min_str))
#             max_val = np.array(float(max_str))
    
#     return min_val, max_val


# def save_action_normalization_params(log_dir, action_min, action_max):
#     """Save action normalization parameters alongside a checkpoint.
    
#     Args:
#         log_dir: Directory to save parameters
#         action_min: Minimum values per dimension (scalar or array-like)
#         action_max: Maximum values per dimension (scalar or array-like)
#     """
#     import numpy as np
    
#     # Convert to numpy arrays if needed
#     action_min = np.asarray(action_min)
#     action_max = np.asarray(action_max)
    
#     norm_file = os.path.join(log_dir, "normalization_params.txt")
#     with open(norm_file, "w") as f:
#         f.write(f"min: {action_min.tolist()}\n")
#         f.write(f"max: {action_max.tolist()}\n")

def load_action_normalization_params(checkpoint_path):
    # Go up two directories and into logs/normalization_params.txt
    exp_dir = os.path.dirname(os.path.dirname(checkpoint_path))
    norm_file = os.path.join(exp_dir, "logs", "normalization_params.txt")
    with open(norm_file, "r") as f:
        lines = f.readlines()
        min_val = float(lines[0].split(":")[1].strip())
        max_val = float(lines[1].split(":")[1].strip())
    return min_val, max_val


def unnormalize_actions(actions, min_val, max_val, device='cuda'):
    # actions: torch.Tensor or np.ndarray in [-1, 1]
    actions = actions.to(device)
    min_val = torch.as_tensor(min_val).to(device)
    max_val = torch.as_tensor(max_val).to(device)
    return 0.5 * (actions + 1) * (max_val - min_val) + min_val

def normalize_actions(actions, min_val, max_val):
    # actions: torch.Tensor or np.ndarray in original range
    # normalize to [-1, 1]
    return 2.0 * (actions - min_val) / (max_val - min_val) - 1.0

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