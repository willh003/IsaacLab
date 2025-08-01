"""Script to play a checkpoint of a robomimic imitation learning policy"""

"""Launch Isaac Sim Simulator first."""

import argparse
from cv2.dnn import Net
import robomimic.utils.file_utils as FileUtils
import gymnasium as gym
import os
import time
import torch
import yaml
import numpy as np
from tqdm import tqdm
from isaaclab.app import AppLauncher

# Robomimic imports
# IMPORTANT: do not remove these, because they are required to register the diffusion policy
from dp import DiffusionPolicyConfig, DiffusionPolicyUNet
from utils import count_parameters, load_action_normalization_params, unnormalize_actions
from robomimic.algo import RolloutPolicy
import sys
import os

# TODO: hacky way to import get_state_from_env_leap
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from leap.utils import get_state_from_env as get_state_from_env_leap

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
#parser.add_argument("--checkpoint", type=str, default=None, help="Path to the rl policy checkpoint.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")


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
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry


def get_clip_params(agent_cfg):
    clip_actions = agent_cfg["params"]["env"]["clip_actions"]
    clip_obs = agent_cfg["params"]["env"]["clip_observations"]
    return clip_actions, clip_obs



def main():
    """Play with RSL-RL agent."""
    
    ############################### CONFIG SETUP ############################### 
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg = load_cfg_from_registry(args_cli.task, "rl_games_cfg_entry_point")

    ############################### POLICY LOADING ############################### 
    # Load policy
    if args_cli.checkpoint:
        checkpoint_path = retrieve_file_path(args_cli.checkpoint)
    else:
        raise ValueError("Please provide a checkpoint path through CLI arguments.")

    print(f"[INFO]: Loading model checkpoint from: {checkpoint_path}")
    # policy_from_checkpoint returns (policy, ckpt_dict)
    policy: RolloutPolicy = FileUtils.policy_from_checkpoint(ckpt_path=checkpoint_path, device=args_cli.device, verbose=True)[0]

    # If it's a diffusion policy  count parameters in the noise prediction network
    if hasattr(policy.policy, 'nets') and 'policy' in policy.policy.nets:
        net = policy.policy.nets
        net_total,net_trainable = count_parameters(net)
        print(f"[INFO]: policy parameters - Total: {net_total:,}, Trainable: {net_trainable:,}")

    if "obs_encoder" in policy.policy.nets['policy']: 
        # robomimic dp implementation
        is_dp = True
        obs_keys = list(policy.policy.nets['policy']['obs_encoder'].nets['obs'].obs_nets.keys())
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
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    clip_actions, clip_obs = get_clip_params(agent_cfg)
    rl_device = args_cli.device
    # rlgames provides a useful vec env wrapper, but there is no other dependency on it
    env = RlGamesVecEnvWrapper(env, clip_actions=clip_actions, clip_obs=clip_obs, rl_device=rl_device)


    ############################### TRAINING LOOP ############################### 
    policy.start_episode() # reset the policy buffers (i.e. prev observations, goals, etc.)
    obs = env.reset()

    n_steps = 0
    finished = False
    rewards = np.array([])

    while simulation_app.is_running() and not finished:

        if "leap" in args_cli.task.lower():
            obs_dict = get_state_from_env_leap(env.unwrapped, obs)
            goal_dict = None
        else:
            raise NotImplementedError(f"Task {args_cli.task} not implemented")

        # Apply actions
        with torch.inference_mode():
            actions = policy.policy.get_action(obs_dict=obs_dict, goal_dict=goal_dict)
            if is_dp:
                min_val, max_val = action_norm_params
                actions = unnormalize_actions(actions, min_val, max_val)

            obs, rew, dones, extras = env.step(actions)

        rewards = np.concatenate([rewards, rew.cpu().numpy()])

        n_steps += 1

        if args_cli.n_steps is not None and n_steps >= args_cli.n_steps:
            finished = True
        # After the loop, print the mean
    
    print(f"Mean reward: {np.mean(rewards):.2f} over {n_steps} steps")

    
    env.close()


if __name__ == "__main__":
    # run the main function
    

    main()
    # close sim app
    simulation_app.close() 