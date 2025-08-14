# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module with utilities for parsing and loading configurations."""

import collections
import gymnasium as gym
import importlib
import inspect
import json
import os
import re
import yaml
import time
import datetime
import robomimic
import torch


def load_cfg_from_registry(task_name: str, entry_point_key: str) -> dict | object:
    """Load default configuration given its entry point from the gym registry.

    This function loads the configuration object from the gym registry for the given task name.
    It supports YAML, JSON, and Python configuration files.

    It expects the configuration to be registered in the gym registry as:

    .. code-block:: python

        gym.register(
            id="My-Awesome-Task-v0",
            ...
            kwargs={"env_entry_point_cfg": "path.to.config:ConfigClass"},
        )

    The parsed configuration object for above example can be obtained as:

    .. code-block:: python

        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

        cfg = load_cfg_from_registry("My-Awesome-Task-v0", "env_entry_point_cfg")

    Args:
        task_name: The name of the environment.
        entry_point_key: The entry point key to resolve the configuration file.

    Returns:
        The parsed configuration object. If the entry point is a YAML or JSON file, it is parsed into a dictionary.
        If the entry point is a Python class, it is instantiated and returned.

    Raises:
        ValueError: If the entry point key is not available in the gym registry for the task.
    """
    # obtain the configuration entry point
    cfg_entry_point = gym.spec(task_name.split(":")[-1]).kwargs.get(entry_point_key)
    # check if entry point exists
    if cfg_entry_point is None:
        # get existing agents and algorithms
        agents = collections.defaultdict(list)
        for k in gym.spec(task_name.split(":")[-1]).kwargs:
            if k.endswith("_cfg_entry_point") and k != "env_cfg_entry_point":
                spec = (
                    k.replace("_cfg_entry_point", "")
                    .replace("rl_games", "rl-games")
                    .replace("rsl_rl", "rsl-rl")
                    .split("_")
                )
                agent = spec[0].replace("-", "_")
                algorithms = [item.upper() for item in (spec[1:] if len(spec) > 1 else ["PPO"])]
                agents[agent].extend(algorithms)
        
        msg = "\nExisting RL library (and algorithms) config entry points: "
        for agent, algorithms in agents.items():
            msg += f"\n  |-- {agent}: {', '.join(algorithms)}"
        # raise error
        raise ValueError(
            f"Could not find configuration for the environment: '{task_name}'."
            f"\nPlease check that the gym registry has the entry point: '{entry_point_key}'."
            f"{msg if agents else ''}"
        )
    # parse the default config file
    if isinstance(cfg_entry_point, str) and (cfg_entry_point.endswith(".yaml") or cfg_entry_point.endswith(".json")):
        if os.path.exists(cfg_entry_point):
            # absolute path for the config file
            config_file = cfg_entry_point
        else:
            # resolve path to the module location
            mod_name, file_name = cfg_entry_point.split(":")
            mod_path = os.path.dirname(importlib.import_module(mod_name).__file__)
            # obtain the configuration file path
            config_file = os.path.join(mod_path, file_name)
        # load the configuration
        print(f"[INFO]: Parsing configuration from: {config_file}")
        with open(config_file, encoding="utf-8") as f:
            if cfg_entry_point.endswith(".yaml"):
                cfg = yaml.full_load(f)
            else:  # .json file
                cfg = json.load(f)
    else:
        if callable(cfg_entry_point):
            # resolve path to the module location
            mod_path = inspect.getfile(cfg_entry_point)
            # load the configuration
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