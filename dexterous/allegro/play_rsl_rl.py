# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""
import argparse
from utils import get_state_from_env, get_goal_from_env, get_termination_env_ids

# Import evaluation module
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from dp.evaluation import EpisodeEvaluator, NUM_EVAL_ENVS, NUM_EVAL_STEPS

from robomimic.models.obs_nets import D

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=8, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--n_steps", type=int, default=100000, help="Number of steps to run.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--eval", action="store_true",default=False,help="whether to enable evaluation config (overriding other settings like n_env)")

parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch
import numpy as np
from tqdm import tqdm

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg


# PLACEHOLDER: Extension template (do not remove this comment)


def main():
    """Play with RSL-RL agent."""
    task_name = args_cli.task.split(":")[-1]

    if args_cli.eval: 
        print("WARNING: setting num_envs and n_steps to eval defaults (overriding cli)")
        args_cli.num_envs = NUM_EVAL_ENVS
        args_cli.n_steps = NUM_EVAL_STEPS

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )

    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(task_name, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:

        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    print(f"[INFO] Loading model checkpoint from: {resume_path}")

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
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

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = ppo_runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = ppo_runner.alg.actor_critic

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(
        policy_nn, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )

    #dt = env.unwrapped.step_dt
    dt = 1/16

    ep_len = 0

    rewards = np.array([])
    n_steps = 0

    # Initialize evaluation tracking
    evaluator = EpisodeEvaluator(args_cli.num_envs)
    
    # Define observation keys for allegro (assuming standard keys)
    obs_keys = ["robot0_joint_pos", "robot0_joint_vel", "object_pos", "object_quat", 
                "object_lin_vel", "object_ang_vel", "goal_pose", "goal_quat_diff", 
                "last_action", "fingertip_contacts"]
    goal_keys = ["object_quat"]  # We need object_quat for goal

    # reset environment
    obs, _ = env.get_observations()
    finished = False

    prev_consecutive_success = np.zeros(args_cli.num_envs)
    total_consecutive_success = np.zeros(args_cli.num_envs)
    # simulate environment
    while simulation_app.is_running() and not finished:
        with tqdm(range(args_cli.n_steps)) as pbar:
            for i in pbar:  
                start_time = time.time()
                # run everything in inference mode
                with torch.inference_mode():
                    # agent stepping
                    actions = policy(obs)
                    # env stepping
                    obs, rew, dones, extras = env.step(actions)

                # Extract observations for evaluation
                obs_dict = get_state_from_env(obs, obs_keys, device=args_cli.device)
                goal_dict = get_goal_from_env(obs, goal_keys, device=args_cli.device)
                
                # Update evaluation tracking
                if args_cli.eval:
                    evaluator.update_step_evaluation(obs_dict, goal_dict, rew)
                    evaluator.check_episode_completion(env)
                
                rewards = np.concatenate([rewards, rew.cpu().numpy()])

                n_steps += len(actions)  # increment all envs' episode lengths

                # Get the command term for the object pose

                # Count number of resets (success or timeout)
                # new_consecutive_success = command_term.metrics["consecutive_success"].cpu().numpy()
                # delta_consecutive_success = new_consecutive_success - prev_consecutive_success
                # delta_consecutive_success[delta_consecutive_success < 0] = 0 # if the goal was reset, we don't want to count it as a success
                # total_consecutive_success += delta_consecutive_success
                # prev_consecutive_success = new_consecutive_success
                
                # time delay for real-time evaluation
                sleep_time = dt - (time.time() - start_time)
                if args_cli.real_time and sleep_time > 0:
                    time.sleep(sleep_time)

                pbar.set_description(f"Mean reward: {rew.mean().item():.2f}, max reward: {rew.max().item():.2f}")

        # Finalize any in-progress episodes and print evaluation results
        if args_cli.eval:   
            evaluator.finalize_all_episodes()
            evaluator.print_evaluation_results()

        # After the loop, print the mean
        print(f"Mean reward: {np.mean(rewards):.2f} over {n_steps} steps")
        dones = rewards > 1
        print(f"Number of episodes finished total (including timeout): {total_consecutive_success.sum()} over {n_steps} steps ")
        finished = True

     
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
