"""Script to play a checkpoint of a robomimic imitation learning policy"""

"""Launch Isaac Sim Simulator first."""

# CRITICAL: Early parsing for config override before any imports
import argparse
import os
import sys

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

# Now proceed with AppLauncher setup
from isaaclab.app import AppLauncher
# add argparse arguments
parser = argparse.ArgumentParser(description="Play an IL policy")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to the checkpoint file.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=8, help="Number of environments to simulate.")
parser.add_argument("--n_steps", type=int, default=None, help="Number of steps to run.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--goal_z", type=float, default=None, help="Goal z rotation.")
parser.add_argument("--noise_group_timesteps", type=float,nargs='+', default=None, help="Noise group timesteps.")

#parser.add_argument("--checkpoint", type=str, default=None, help="Path to the rl policy checkpoint.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--config", type=str, default=None, 
                   help="Override robomimic config entry point (e.g., 'path.to.your.config:your_cfg.json')")
parser.add_argument("--algo", type=str, default="diffusion_policy", help="Algorithm name for config override.")
parser.add_argument("--eval", action="store_true",default=False,help="whether to enable evaluation config (overriding other settings like n_env)")

parser.add_argument("--mask_observations", action="store_true", default=False, help="Mask observations.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rl_games import RlGamesVecEnvWrapper
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import robomimic.utils.file_utils as FileUtils
import gymnasium as gym
import os
import time
from collections import deque
import matplotlib.pyplot as plt
from isaaclab.utils.math import quat_from_euler_xyz
import torch
import torch.nn.functional as F
import yaml
import numpy as np
from tqdm import tqdm

from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
# Robomimic imports
# IMPORTANT: do not remove these, because they are required to register the diffusion policy
from dp_model import DiffusionPolicyConfig, DiffusionPolicyUNet
from robomimic.config import config_factory
from robomimic.algo import RolloutPolicy
import sys
import os

# TODO: hacky way to import get_state_from_env_leap
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from dp.utils import count_parameters, load_action_normalization_params, unnormalize_actions, clear_task_registry_cache, load_cfg_from_registry_no_gym, filter_config_dict, policy_from_checkpoint_override_cfg
from leap.utils import get_state_from_env as get_state_from_env_leap
from allegro.utils import get_state_from_env as get_state_from_env_allegro
from allegro.utils import get_goal_from_env as get_goal_from_env_allegro
from evaluation import EpisodeEvaluator, NUM_EVAL_ENVS, NUM_EVAL_STEPS

# Clear the task registry cache after setting environment variable
# This ensures the override is applied when the registry is rebuilt
if early_args.config is not None:
    clear_task_registry_cache()

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def main():
    """Play with RSL-RL agent."""
    
    ############################### CONFIG SETUP ############################### 

    if args_cli.eval: 
        print("WARNING: setting num_envs and n_steps to eval defaults (overriding cli)")
        args_cli.num_envs = NUM_EVAL_ENVS
        args_cli.n_steps = NUM_EVAL_STEPS
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )

    ############################### POLICY LOADING ############################### 
    # Load policy
    if args_cli.checkpoint:
        checkpoint_path = retrieve_file_path(args_cli.checkpoint)
    else:
        raise ValueError("Please provide a checkpoint path through CLI arguments.")

    print(f"[INFO]: Loading model checkpoint from: {checkpoint_path}")
    # policy_from_checkpoint returns (policy, ckpt_dict)
    
    policy_config = None
    if args_cli.config is not None:
        cfg_entry_point_key = f"robomimic_{args_cli.algo}_cfg_entry_point"
        task_name = args_cli.task.split(":")[-1]
        ext_cfg = load_cfg_from_registry_no_gym(task_name, cfg_entry_point_key)
        config = config_factory(ext_cfg["algo_name"])

        filtered_ext_cfg = filter_config_dict(ext_cfg, config)
        # with config.values_unlocked():
        with config.unlocked():
            config.update(filtered_ext_cfg)

        policy_config = config

    policy: RolloutPolicy = policy_from_checkpoint_override_cfg(ckpt_path=checkpoint_path, device=args_cli.device, verbose=True, override_config=policy_config)[0]

    # If it's a diffusion policy  count parameters in the noise prediction network
    if hasattr(policy.policy, 'nets') and 'policy' in policy.policy.nets:
        net = policy.policy.nets
        net_total,net_trainable = count_parameters(net)
        print(f"[INFO]: policy parameters - Total: {net_total:,}, Trainable: {net_trainable:,}")

    if "obs_encoder" in policy.policy.nets['policy']: 
        # robomimic dp implementation
        # Robomimic dps put the goal into the obs_encoder, so no need for separate goal
        is_dp = True
        obs_keys = list(policy.policy.nets['policy']['obs_encoder'].nets['obs'].obs_nets.keys())

        if 'goal' in policy.policy.nets['policy']['obs_encoder'].nets.keys():
            goal_keys = list(policy.policy.nets['policy']['obs_encoder'].nets['goal'].obs_nets.keys())
        else:
            goal_keys = None
    else: 
        # robomimic bc implementation
        is_dp = False
        obs_keys = list(policy.policy.nets['policy'].nets["encoder"].nets['obs'].obs_nets.keys())
        goal_keys = list(policy.policy.nets['policy'].nets["encoder"].nets['goal'].obs_nets.keys())
    
    # Initialize action normalization params
    if is_dp:
        action_norm_params = load_action_normalization_params(checkpoint_path)

    ############################### ENVIRONMENT SETUP ############################### 
    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # wrap for video recording
    log_dir = os.path.dirname(checkpoint_path)
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "..", "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)


    ############################### TRAINING LOOP ############################### 
    policy.start_episode() # reset the policy buffers (i.e. prev observations, goals, etc.)
    obs,_ = env.reset()

    n_steps = 0
    finished = False

    goal_zs = np.arange(-2*np.pi, 2*np.pi, args_cli.goal_z)

    z_idx = 0
    buf_len = 50
    quat_buffer = deque(maxlen=50)
    cc_scores = []
    
    # Initialize per-episode evaluation tracking
    num_envs = env_cfg.scene.num_envs
    evaluator = EpisodeEvaluator(num_envs) if "allegro" in args_cli.task.lower() else None
    # Initialize progress bar
    if args_cli.n_steps is not None:
        pbar = tqdm(total=args_cli.n_steps, desc="Rollout Progress")
    else:
        pbar = tqdm(desc="Rollout Progress")
    
    while simulation_app.is_running() and not finished:

        if "leap" in args_cli.task.lower():
            obs_dict = get_state_from_env_leap(env.unwrapped, obs)
            
            if args_cli.goal_z is not None:
                num_envs = env_cfg.scene.num_envs
                goal_rot = quat_from_euler_xyz(torch.zeros(num_envs), torch.zeros(num_envs), torch.ones(num_envs) * goal_zs[z_idx]).to(args_cli.device)
                goal_dict = {"object_rot": goal_rot}
                
            else:
                goal_dict = None
        elif "allegro" in args_cli.task.lower():
            obs_dict = get_state_from_env_allegro(obs['policy'], obs_keys, device=args_cli.device)
            
            goal_dict = get_goal_from_env_allegro(obs['policy'], goal_keys, device=args_cli.device)
        
        else:
            raise NotImplementedError(f"Task {args_cli.task} not implemented")

        # quat_buffer.append(obs_dict["object_rot"])
        # if len(quat_buffer) == quat_buffer.maxlen:
        #     quaternions = torch.stack(list(quat_buffer), dim=0)

        #     z_rot_direction = detect_z_rotation_direction_batch(quaternions)
        #     cc_percent = (z_rot_direction == 1).float().mean()
        #     print(f"Counterclockwise percent: {cc_percent:.2f} for goal {goal_zs[z_idx]} with quat {goal_dict['object_rot'][0]}")
        #     z_idx = z_idx + 1
        #     quat_buffer.clear()
        #     cc_scores.append(cc_percent.item())
        #     # Reset environment with inference mode disabled
        #     obs = env.reset()
        #     policy.start_episode()
        #     if z_idx >= len(goal_zs):
        #         finished = True
        #     continue  # Skip the rest of the loop iteration to avoid using stale obs_dict

        # Apply actions
        with torch.inference_mode():

            if hasattr(args_cli, 'noise_group_timesteps'):
                noise_group_timesteps = args_cli.noise_group_timesteps
            else:
                noise_group_timesteps = None
            if hasattr(args_cli, 'mask_observations'):
                mask_observations = args_cli.mask_observations
            else:
                mask_observations = False
            
            assert not (noise_group_timesteps is not None and mask_observations), "error: cannot use both noise_group_timesteps and mask_observations"
            
            actions = policy.policy.get_action(obs_dict=obs_dict, goal_dict=goal_dict, noise_group_timesteps=noise_group_timesteps, mask_observations=mask_observations)
            if is_dp:
                min_val, max_val = action_norm_params
                actions = unnormalize_actions(actions, min_val, max_val, device=args_cli.device)

                

        obs, rew, terminated, truncated, extras = env.step(actions)

        if terminated.any() or truncated.any():
            policy.start_episode()

        # Update evaluation tracking and check for episode completion
        if evaluator and "allegro" in args_cli.task.lower():
            evaluator.update_step_evaluation(obs_dict, goal_dict, rew)
            evaluator.check_episode_completion(env, obs_dict, goal_dict, terminated)

        n_steps += 1
        pbar.update(1)

        if args_cli.n_steps is not None and n_steps >= args_cli.n_steps:
            finished = True
        # After the loop, print the mean
    
    # Close progress bar
    pbar.close()
    
    # Finalize a
    # ny in-progress episodes and print evaluation results
    if evaluator:

        evaluator.finalize_all_episodes(obs_dict, goal_dict)
        evaluator.print_evaluation_results()

    env.close()


if __name__ == "__main__":
    # run the main function
    

    main()
    # close sim app
    simulation_app.close()