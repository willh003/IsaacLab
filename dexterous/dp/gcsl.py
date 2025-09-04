"""
Goal-Conditioned Supervised Learning (GCSL) implementation.

This module implements GCSL, which combines imitation learning with goal relabeling
to learn policies that can reach multiple goals. The algorithm:
1. Collects trajectories using the current policy
2. Relabels goals in the buffer to increase data efficiency
3. Trains the policy using supervised learning on state-goal-action tuples
4. Iterates between collection and training phases
"""
# lib imports
import time
from pathlib import Path
from collections import deque, defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import h5py
import json
from tqdm import tqdm
import argparse
import os
import sys
import gymnasium as gym
import gc

def parse_args_early():
    """Parse arguments early to set environment variables before imports."""
    parser = argparse.ArgumentParser()
    
    # Critical arguments needed before import
    parser.add_argument("--config", type=str, default=None, 
                       help="Override robomimic config entry point (e.g., 'path.to.your.config:your_cfg.json')")
    parser.add_argument("--algo", type=str, default="diffusion_policy", help="Algorithm name for config override.")
    
    # Parse only the arguments we need, ignore the rest for now
    args, unknown = parser.parse_known_args()
    return args

# Parse early arguments and set environment variables BEFORE any imports
early_args = parse_args_early()
if early_args.config is not None:
    if os.path.exists(early_args.config):
        cfg_path = early_args.config

    env_var_name = f"ROBOMIMIC_{early_args.algo.upper()}_CFG_ENTRY_POINT"
    os.environ[env_var_name] = early_args.config
    print(f"Pre-import override: {env_var_name} = {early_args.config}")

# Robomimic imports
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.train_utils as TrainUtils
from robomimic.algo import algo_factory, RolloutPolicy
from robomimic.config import config_factory

# Local imports
from dp_model import DiffusionPolicyConfig, DiffusionPolicyUNet
from utils import load_cfg_from_registry_no_gym, get_exp_dir, unnormalize_actions, load_action_normalization_params#, save_action_normalization_params
from gcsl_dataset import gcsl_dataset_factory

# Args
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description="Goal-Conditioned Supervised Learning (GCSL)")

# Task and algorithm
parser.add_argument("--task", type=str, required=True, help="Task name")
parser.add_argument("--algo", type=str, default="diffusion_policy", help="Algorithm name")
parser.add_argument("--initial_policy", type=str, default=None, help="Path to initial policy checkpoint to load instead of using dataset")
parser.add_argument("--config", type=str, default=None, help="Override robomimic config entry point")

# GCSL hyperparameters
parser.add_argument("--num_iterations", type=int, default=100, help="Number of GCSL iterations")
parser.add_argument("--trajectories_per_iter", type=int, default=500, help="Trajectories to collect per iteration")
parser.add_argument("--train_epochs_per_iter", type=int, default=100, help="Training epochs per iteration")
parser.add_argument("--buffer_size", type=int, default=10000, help="Replay buffer capacity")
parser.add_argument("--max_episode_length", type=int, default=50, help="Maximum episode length")

# Environment settings
parser.add_argument("--num_envs", type=int, default=128, help="Number of parallel environments")

# Logging and saving
home_dir = Path.home()
default_log_dir = home_dir / "IsaacLab/dexterous/logs/gcsl"
parser.add_argument("--log_dir", type=str, default=default_log_dir, help="Log directory")
parser.add_argument("--save_checkpoints", action="store_true", default=True, help="Save model checkpoints")
parser.add_argument("--load_initial_data", action="store_true", help="Load initial data from dataset")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--wandb", type=str, default="online", help="Wandb mode")


# IsaacLab imports
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_tasks.utils import parse_env_cfg
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from allegro.utils import get_state_from_env as get_state_from_env_allegro
from allegro.utils import get_goal_from_env as get_goal_from_env_allegro
from leap.utils import get_state_from_env as get_state_from_env_leap

import wandb


class DataBuffer:
    def __init__(self, capacity=1000, train_split=0.8):
        """
        self.train_trajectories is a deque of dicts, each dict is a trajectory with the following keys:
        - obs: list of dicts, each dict is an observation
        - actions: list of dicts, each dict is an action
        - length: int, the length of the trajectory
        """
        self.train_trajectories: deque[dict] = deque(maxlen=int(capacity * train_split))
        self.val_trajectories: deque[dict] = deque(maxlen=int(capacity * (1-train_split)))
    
    def get_trajectory(self, idx, is_val=False):
        pass

class GCSLBuffer(DataBuffer):
    """Replay buffer for GCSL that stores complete trajectories for sequence-based learning."""
    
    def __init__(self, capacity=1000, action_norm_params=None, train_split=0.8):  # Store trajectory count, not individual transitions
        super().__init__(capacity, train_split)
        self.action_min, self.action_max = action_norm_params
        self.train_split = train_split
        
    def add_trajectory(self, obs_traj, action_traj):
        """Add a trajectory to the buffer.
        
        Args:
            obs_traj: List of observations [T, obs_dim]
            action_traj: List of actions [T-1, action_dim] (should be unnormalized)
        """
        # Update action normalization parameters with new data
        # TODO: for now not updating these so it doesn't have a moving target
        #self._update_action_normalization_params(action_traj)
        
        # Normalize actions before storing
        normalized_action_traj = self._normalize_actions_for_storage(action_traj)

        n_trajs = len(self.train_trajectories) + len(self.val_trajectories)
        traj = {
                'obs': obs_traj,
                'actions': normalized_action_traj,
                'length': len(obs_traj)
            }

        val_interval = int(1 / (1 - self.train_split))  # For 0.8 train_split: every 5th trajectory
        if n_trajs % val_interval == (val_interval - 1):  # 0-indexed: 4, 9, 14, ... for interval=5
            self.val_trajectories.append(traj)
        else:
            self.train_trajectories.append(traj)
    
    def get_trajectory(self, idx, is_val=False):
        """Get a trajectory by index."""
        if is_val:
            return self.val_trajectories[idx]
        else:
            return self.train_trajectories[idx]


    def len(self, is_val=False):
        """Get the total number of trajectories."""
        if is_val:
            return len(self.val_trajectories)
        else:
            return len(self.train_trajectories)

    def get_trajectories(self, is_val=False):
        """Get all trajectories."""
        if is_val:
            return list(self.val_trajectories)
        else:
            return list(self.train_trajectories)

    def _update_action_normalization_params(self, action_traj):
        """Update action normalization parameters based on new trajectory data."""
        if len(action_traj) == 0:
            return
            
        # Stack all actions in trajectory
        actions_tensor = torch.stack(action_traj, dim=0)  # [T-1, action_dim]
        
        # Compute min/max across timesteps
        traj_min = torch.min(actions_tensor, dim=0)[0]  # [action_dim]
        traj_max = torch.max(actions_tensor, dim=0)[0]  # [action_dim]
        
        self.action_min = torch.min(self.action_min, traj_min)
        self.action_max = torch.max(self.action_max, traj_max)
    
    def _normalize_actions_for_storage(self, action_traj):
        """Normalize actions to [-1, 1] range for storage."""
        if len(action_traj) == 0:
            return action_traj
        
        normalized_actions = []
        for action in action_traj:
            # Use utils.normalize_actions function
            if self.action_min is not None and self.action_max is not None:
                normalized_action = 2.0 * (action - self.action_min) / (self.action_max - self.action_min) - 1.0
                normalized_actions.append(normalized_action)
            else:
                normalized_actions.append(action)
        
        return normalized_actions
    
    def get_action_normalization_params(self):
        """Get current action normalization parameters."""
        return self.action_min, self.action_max


def collect_trajectories_and_evaluate(env, policy, buffer, num_trajectories, min_length=1, max_steps=200, is_dp=False):
    """Collect trajectories using the current policy and compute evaluation metrics."""
    # Record GPU memory at start of collection
    start_memory = get_gpu_memory_info()
    if start_memory:
        print(f"Collection start GPU memory - Allocated: {start_memory['allocated_gb']:.2f}GB, "
              f"Reserved: {start_memory['reserved_gb']:.2f}GB, "
              f"Free: {start_memory['free_gb']:.2f}GB")
    
    num_envs = env.unwrapped.num_envs
        
    # Collection state
    trajectories = []
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    failed_count = 0
    action_norm_params = buffer.get_action_normalization_params()
    
    with tqdm(total=num_trajectories, desc="Collecting trajectories") as pbar:
        while len(trajectories) < num_trajectories:
            obs, _ = env.reset()
            policy.start_episode()
            
            # Run episode steps
            episode_data = _run_episode(env, policy, obs, num_envs, max_steps, is_dp, action_norm_params)

            # Process completed episodes
            for i, (obs_traj, action_traj, reward, length, success, failed) in enumerate(episode_data):
                if not failed and len(obs_traj) > min_length:
                    # Save individual trajectory for this environment to buffer
                    buffer.add_trajectory(obs_traj, action_traj)
                    
                    # Track metrics
                    trajectories.append(length)
                    episode_rewards.append(reward)
                    episode_lengths.append(length)
                    
                    if success:  # Success
                        success_count += 1

                    if failed:
                        failed_count += 1
                    
                    pbar.update(1)

    
    # Record GPU memory at end of collection
    end_memory = get_gpu_memory_info()
    if end_memory and start_memory:
        memory_change = end_memory['allocated_gb'] - start_memory['allocated_gb']
        print(f"Collection end GPU memory - Allocated: {end_memory['allocated_gb']:.2f}GB, "
              f"Reserved: {end_memory['reserved_gb']:.2f}GB, "
              f"Free: {end_memory['free_gb']:.2f}GB")
        print(f"Total memory change during collection: {memory_change:+.2f}GB")
    
    # Compute evaluation metrics
    eval_results = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'success_rate': success_count / len(trajectories),
        'failed_rate': failed_count / len(trajectories),
        'all_rewards': episode_rewards,
        'start_memory': start_memory,
        'end_memory': end_memory
    }

    print(f"Train buffer num trajectories: {buffer.len(is_val=False)}, Val buffer num trajectories: {buffer.len(is_val=True)}")
    return eval_results


def _run_episode(env, policy, obs, num_envs, max_steps, is_dp, action_norm_params):
    """Run one episode and return episode data for each environment."""
    # Initialize episode tracking
    obs_trajs = [[] for _ in range(num_envs)]
    action_trajs = [[] for _ in range(num_envs)]
    episode_rewards = [0.0 for _ in range(num_envs)]
    episode_lengths = [0 for _ in range(num_envs)]
    success = [False for _ in range(num_envs)]
    failed = [False for _ in range(num_envs)]
    
    for step in tqdm(range(max_steps)):
        # Get action from policy for all environments (needed for environment stepping)
        obs_dict, goal_dict = _prepare_policy_input(obs, policy, env)
        action = _get_policy_action(policy, obs_dict, goal_dict, is_dp, action_norm_params)
        
        # Store observations and actions for each environment separately
        for i in range(num_envs):
            if not (success[i] or failed[i]):
                # Get policy input for all environments first (needed for observation processing)
                obs_dict, goal_dict = _prepare_policy_input(obs, policy, env)
                
                # Extract individual environment data from the processed observations
                env_obs_dict = _extract_env_data(obs_dict, i, num_envs)
                env_action = _extract_env_data(action, i, num_envs)
                obs_trajs[i].append(env_obs_dict)
                action_trajs[i].append(env_action)
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Update episode state for each environment
        for i in range(num_envs):
            if not (success[i] or failed[i]):
                episode_rewards[i] += reward[i].item()
                episode_lengths[i] += 1
                
                # Check completion conditions
                success[i] = _check_env_reset(env, i)
                failed[i] = terminated[i].item() or truncated[i].item()
        
    
    # Return episode data: (obs_traj, action_traj, reward, length, success, failed)
    ep_data = [(obs_trajs[i], action_trajs[i], episode_rewards[i], episode_lengths[i], success[i], failed[i]) 
            for i in range(num_envs)]
    print(f"Success: {sum(success)}, Failed: {sum(failed)}")
    
    return ep_data

def _get_obs_keys(policy):
    if hasattr(policy.policy, 'nets') and 'policy' in policy.policy.nets:
        if "obs_encoder" in policy.policy.nets['policy']:
            # Diffusion policy
            obs_keys = list(policy.policy.nets['policy']['obs_encoder'].nets['obs'].obs_nets.keys())
            if 'goal' in policy.policy.nets['policy']['obs_encoder'].nets.keys():
                goal_keys = list(policy.policy.nets['policy']['obs_encoder'].nets['goal'].obs_nets.keys())
            else:
                goal_keys = []
        else:
            # BC policy
            obs_keys = list(policy.policy.nets['policy'].nets["encoder"].nets['obs'].obs_nets.keys())
            goal_keys = list(policy.policy.nets['policy'].nets["encoder"].nets['goal'].obs_nets.keys())

    return obs_keys, goal_keys

def _prepare_policy_input(obs, policy, env):
    """Prepare observation and goal dictionaries for policy input."""
    if "leap" in env.spec.id.lower():
        
        obs_dict = get_state_from_env_leap(env.unwrapped, obs)
        goal_dict = None
    elif "allegro" in env.spec.id.lower():
        obs_keys, goal_keys = _get_obs_keys(policy)
        obs_dict = get_state_from_env_allegro(obs['policy'], obs_keys, device=args_cli.device)
        goal_dict = get_goal_from_env_allegro(obs['policy'], goal_keys, device=args_cli.device)
    else:
        raise NotImplementedError(f"Environment not supported")
    return obs_dict, goal_dict


def _get_policy_action(policy, obs_dict, goal_dict, is_dp, action_norm_params):
    """Get action from policy and handle normalization."""
    with torch.no_grad():
        action = policy.policy.get_action(obs_dict=obs_dict, goal_dict=goal_dict)
        
        # Unnormalize if needed
        if is_dp and action_norm_params is not None:
            min_val, max_val = action_norm_params
            action = unnormalize_actions(action, min_val, max_val, device=args_cli.device)
        
        return action


def _extract_env_data(data, env_idx, num_envs):
    """Extract data for a specific environment.
    
    Args:
        data: Either a tensor with shape [n_envs, ...] or a single value
        env_idx: Index of the environment to extract
        num_envs: Total number of environments
    
    Returns:
        Data for the specific environment with environment dimension removed
    """
    if isinstance(data, torch.Tensor):
        if len(data.shape) > 0 and data.shape[0] == num_envs:
            # Data has environment dimension, extract specific environment
            return data[env_idx]
        else:
            # Data doesn't have environment dimension, return as is
            return data
    elif isinstance(data, dict):
        # Handle dictionary of observations
        return {k: _extract_env_data(v, env_idx, num_envs) for k, v in data.items()}
    else:
        # Non-tensor data (e.g., scalars, lists)
        return data


def _check_env_reset(env, env_idx):
    """Check if a specific environment has reset (completed successfully)."""
    if "allegro" in env.spec.id.lower():
        command_term = env.unwrapped.command_manager.get_term("object_pose")
        if hasattr(command_term, 'episode_ended'):
            episode_ended = command_term.episode_ended.cpu().numpy()
            if episode_ended[env_idx]:
                # Clear the episode_ended indicator
                command_term.clear_episode_ended_indicator(torch.tensor([env_idx], device=env.unwrapped.device))
                return True
    return False


def collate_batch_for_training(batch):
    """Custom collate function for GCSL training batches.
    
    Actions in the batch are normalized (from buffer storage).
    Policy sees normalized actions for training.
    """
    import collections
    
    batch_out = collections.defaultdict(list)
    for item in batch:
        for k, v in item.items():
            batch_out[k].append(v)
    
    # Stack non-obs keys (convert numpy to tensor if needed)
    for k in batch_out:
        if k != 'obs' and k != 'goal_obs':
            batch_out[k] = [torch.from_numpy(x) if isinstance(x, np.ndarray) else x for x in batch_out[k]]
            batch_out[k] = torch.stack(batch_out[k], dim=0)
    
    # Helper function to collate observation dictionaries
    def collate_obs_dict(obs_list):
        """Collate a list of observation dictionaries into a batched dictionary.
        
        Expected input: list of dicts, where each dict has keys with tensors of shape [T, ...]
        Output: dict with keys having tensors of shape [B, T, ...]
        """
        if not obs_list:
            return {}
        
        obs_keys = obs_list[0].keys()
        obs_dict = {k: [] for k in obs_keys}
        
        # Collect all values for each key
        for obs in obs_list:
            for k in obs_keys:
                obs_dict[k].append(obs[k])
        
        # Stack obs keys (convert numpy to tensor if needed)
        for k in obs_dict:
            obs_dict[k] = [torch.from_numpy(x) if isinstance(x, np.ndarray) else x for x in obs_dict[k]]
            # Stack along batch dimension: [B, T, ...]
            obs_dict[k] = torch.stack(obs_dict[k], dim=0)
        
        return obs_dict
    
    # Collate obs and goal_obs using the helper function
    batch_out['obs'] = collate_obs_dict(batch_out['obs'])
    batch_out['goal_obs'] = collate_obs_dict(batch_out['goal_obs'])
    
    return dict(batch_out)

def train_policy_iteration(policy, train_loader, val_loader, num_epochs=10):
    """Train the policy for one iteration using supervised learning.
    
    Policy trains on normalized actions (as stored in buffer).
    """
    print(f"Train with {len(train_loader) * train_loader.batch_size} sequence samples and batch size {train_loader.batch_size}")
    print(f"Val with {len(val_loader) * val_loader.batch_size} sequence samples and batch size {val_loader.batch_size}")

    for epoch in range(num_epochs): 

        step_log = TrainUtils.run_epoch(model=policy, data_loader=train_loader, epoch=epoch, num_steps=len(train_loader))
        # policy.on_epoch_end(epoch) # TODO: this handles lr scheduling, which we actually don't want rn
        
        wandb_dict = {f"train/{k}": v for k, v in step_log.items() if "time" not in k.lower()}
        wandb_dict.update({f"time/{k}": v for k, v in step_log.items() if "time" in k.lower()})
        wandb_dict["train/num_samples"] = len(train_loader) * train_loader.batch_size
        wandb_dict["train/epoch"] = epoch
        wandb.log(wandb_dict)

        
        with torch.no_grad():
            step_log = TrainUtils.run_epoch(
                model=policy, data_loader=val_loader, epoch=epoch, validate=True, num_steps=len(val_loader)
            )
        # Log to wandb (no timing stats for val)
        wandb.log({f"validation/{k}": v for k, v in step_log.items() if "time" not in k.lower()})

def get_gpu_memory_info():
    """Get GPU memory information for monitoring OOM issues."""
    try:
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
            memory_reserved = torch.cuda.memory_reserved() / (1024**3)    # GB
            memory_free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / (1024**3)  # GB
            return {
                'allocated_gb': memory_allocated,
                'reserved_gb': memory_reserved,
                'free_gb': memory_free,
                'total_gb': torch.cuda.get_device_properties(0).total_memory / (1024**3)
            }
        else:
            return None
    except Exception as e:
        print(f"Warning: Could not get GPU memory info: {e}")
        return None

def gcsl_main():
    """Main GCSL training loop."""
    
    # Load configuration
    task_name = args_cli.task.split(":")[-1]
    cfg_entry_point_key = f"robomimic_{args_cli.algo}_cfg_entry_point"
    
    print(f"Loading configuration for task: {task_name}")
    ext_cfg = load_cfg_from_registry_no_gym(args_cli.task, cfg_entry_point_key)
    config = config_factory(ext_cfg["algo_name"])
    
    # Update config with external config
    with config.values_unlocked():
        config.update(ext_cfg)
    
    # Set up experiment directory
    config.train.output_dir = os.path.abspath(os.path.join("./logs", args_cli.log_dir, args_cli.task))
    log_dir, ckpt_dir, video_dir = get_exp_dir(config.train.output_dir, config.experiment.name, config.experiment.save.enabled)
    
    # Save the config as a json file (same as train_il.py)
    with open(os.path.join(log_dir, "..", "config.json"), "w") as outfile:
        json.dump(config, outfile, indent=4)
    
    # Monkey-patch wandb.save to avoid uploading large files
    original_save = wandb.save
    def selective_save(path, base_path=None, policy="live"):
        if path.endswith('.pth') or path.endswith('.pt'):
            return None  # Skip saving checkpoint files
        return original_save(path, base_path, policy)
    wandb.save = selective_save
    
    # Initialize wandb logging
    wandb_cfg = dict(config)
    wandb_cfg.update({
        "gcsl_num_iterations": args_cli.num_iterations,
        "gcsl_trajectories_per_iter": args_cli.trajectories_per_iter,
        "gcsl_train_epochs_per_iter": args_cli.train_epochs_per_iter,
        "gcsl_initial_policy": args_cli.initial_policy,
        "gcsl_buffer_size": args_cli.buffer_size,
        "gcsl_max_episode_length": args_cli.max_episode_length,
        "num_envs": args_cli.num_envs,
    })
    
    wandb_tags = ["gcsl"]
    if "diffusion" in args_cli.algo.lower():
        wandb_tags.append("dp")
    
    wandb.init(
        project="dexterous",
        entity="willhu003",
        name=os.path.basename(os.path.dirname(log_dir)), # log dir is under the experiment dir
        config=wandb_cfg,
        mode=args_cli.wandb,
        tags=wandb_tags
    )
    """

    -> if config.experiment.validate:
    (Pdb) x = next(iter(train_loader))
    (Pdb) y = trainset[0]
    (Pdb) y.keys()
    dict_keys(['actions', 'rewards', 'dones', 'obs'])
    (Pdb) y['obs'].keys()
    dict_keys(['goal_pose', 'last_action', 'object_pos', 'object_quat', 'robot0_joint_pos'])
    (Pdb) x.keys()
    dict_keys(['actions', 'rewards', 'dones', 'obs', 'goal_obs'])
    (Pdb) x['obs'].keys()
    dict_keys(['goal_pose', 'last_action', 'object_pos', 'object_quat', 'robot0_joint_pos'])
    (Pdb) x['goal_obs'].keys()
    dict_keys([])

    """
    
    # Set up environment
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg,  render_mode="rgb_array" if args_cli.video else None)
    
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "..", "videos", "train"),
            "episode_trigger": lambda ep: ep % (args_cli.trajectories_per_iter // args_cli.num_envs) == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # Load policy from checkpoint (returns RolloutPolicy wrapper)
    print(f"Loading initial policy from checkpoint: {args_cli.initial_policy}")    
    is_dp = "diffusion" in args_cli.algo.lower()    
    policy_wrapper, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=args_cli.initial_policy, device=args_cli.device, verbose=True)
    obs_keys, goal_keys = _get_obs_keys(policy_wrapper)
    shape_meta = ckpt_dict["shape_metadata"]
    env_meta = ckpt_dict["env_metadata"]
    raw_policy = policy_wrapper.policy

    
    print(f"Initialized policy: {config.algo_name}")
    print(f"Model parameters: {sum(p.numel() for p in raw_policy.nets.parameters()):,}")
    print(f"Policy nets training mode: {raw_policy.nets.training}")
        
    action_norm_params = load_action_normalization_params(args_cli.initial_policy)
    buffer = GCSLBuffer(capacity=args_cli.buffer_size, action_norm_params=action_norm_params)

    # TODO: change this to the policy rollout model (this is for debugging)
    # model_to_train = algo_factory(
    #     algo_name=config.algo_name,
    #     config=config,
    #     obs_key_shapes=shape_meta["all_shapes"],
    #     ac_dim=shape_meta["ac_dim"],
    #     device=args_cli.device,
    # )
    model_to_train = raw_policy

    # Save the min and max values to log directory for compatibility
    min_val, max_val = action_norm_params
    min_val, max_val = np.array(min_val), np.array(max_val)
    with open(os.path.join(log_dir, "normalization_params.txt"), "w") as f:
        f.write(f"min: {min_val.tolist()}\n")
        f.write(f"max: {max_val.tolist()}\n")
    print(f"Saved action normalization parameters to {log_dir}/normalization_params.txt")
    
    # Main GCSL loop
    for iteration in range(args_cli.num_iterations):
        print(f"\n=== GCSL Iteration {iteration + 1}/{args_cli.num_iterations} ===")
        print("Collecting trajectories and evaluating policy...")
        
        # 1. Collect trajectories and evaluate policy simultaneously
        raw_policy.set_eval()
        raw_policy.nets.eval()
        eval_results = collect_trajectories_and_evaluate(env, policy_wrapper, buffer, args_cli.trajectories_per_iter, min_length=config.train.seq_length, max_steps=args_cli.max_episode_length, is_dp=is_dp)    
        
        # Print and log evaluation results
        print(f"Evaluation - Mean Reward: {eval_results['mean_reward']:.3f} ± {eval_results['std_reward']:.3f}")
        print(f"             Success Rate: {eval_results['success_rate']:.3f}")
        print(f"             Failed Rate: {eval_results['failed_rate']:.3f}")
        print(f"             Mean Length: {eval_results['mean_length']:.1f}")
         
        # Log to wandb
        log_dict = {
            "eval/mean_reward": eval_results['mean_reward'],
            "eval/std_reward": eval_results['std_reward'], 
            "eval/success_rate": eval_results['success_rate'],
            "eval/failed_rate": eval_results['failed_rate'],
            "eval/mean_length": eval_results['mean_length'],
            "gcsl/buffer_train_num_trajectories": buffer.len(is_val=False),
            "gcsl/buffer_val_num_trajectories": buffer.len(is_val=True),
            "gcsl/iteration": iteration + 1,
            "gcsl/min_val": min_val.mean().item(),
            "gcsl/max_val": max_val.mean().item(),
        }
        wandb.log(log_dict)
        
        # 2. Train policy on buffer data
        assert buffer.len(is_val=False) >= config.train.batch_size, "Not enough data to train"
        # Ensure policy is in training mode before training
        raw_policy.set_train()
        raw_policy.nets.train()
        
        trainset = gcsl_dataset_factory(buffer, config, obs_keys, is_val=False)
        valset = gcsl_dataset_factory(buffer, config, obs_keys, is_val=True)
        # Use num_workers=0 to avoid CUDA context issues with multiprocessing when data contains CUDA tensors
        train_loader = DataLoader(trainset, batch_size=config.train.batch_size, shuffle=True, collate_fn=collate_batch_for_training, num_workers=0, drop_last=True)
        val_loader = DataLoader(valset, batch_size=config.train.batch_size, shuffle=False, collate_fn=collate_batch_for_training, num_workers=0, drop_last=True)

        # TODO: change this to the policy rollout model (this is for debugging)
        train_policy_iteration(model_to_train, train_loader, val_loader,
                                num_epochs=args_cli.train_epochs_per_iter
                                        )            

        # 3. Save checkpoint (only keep the most recent one)
        if args_cli.save_checkpoints and ckpt_dir:
            # Define current and previous checkpoint paths
            current_checkpoint_path = os.path.join(ckpt_dir, f"ckpt_epoch.pth")
            
            # Save using robomimic format
            TrainUtils.save_model(
                model=model_to_train, # TODO: change this to the policy rollout model (this is for debugging)
                config=config,
                env_meta=env_meta,
                shape_meta=shape_meta,
                ckpt_path=current_checkpoint_path,
                obs_normalization_stats=None,  # GCSL doesn't use obs normalization
            )
            
            # Save action normalization parameters alongside the checkpoint
            action_min, action_max = buffer.get_action_normalization_params()

            exp_dir = os.path.dirname(os.path.dirname(current_checkpoint_path))
            log_dir = os.path.join(exp_dir, "logs")            
            save_action_normalization_params(log_dir, action_min, action_max)

            print(f"Saved checkpoint: {current_checkpoint_path}")
    
    env.close()
    wandb.finish()


if __name__ == "__main__":
    # Parse arguments early for config override

    
    # Run GCSL
    gcsl_main()