# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RL-Games."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher
import torch


# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RL-Games.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--sigma", type=str, default=None, help="The policy's initial standard deviation.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--config_name", type=str, default="rl_games_sac_rlpd_cfg_entry_point", help="Name of the config entry point to use.")
parser.add_argument("--dataset", type=str, default=None, help="Path to HDF5 dataset for replay buffer initialization.")
parser.add_argument("--wandb", type=str, default="online", help="Name of the wandb run.")


# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import math
import os
import random
from datetime import datetime

from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import AlgoObserver, IsaacAlgoObserver
from rl_games.algos_torch.model_builder import register_model, register_network


class WandbAlgoObserver(AlgoObserver):
    """Log statistics from the environment along with the algorithm running stats to wandb."""

    def __init__(self):
        pass

    def after_init(self, algo):
        self.algo = algo
        from rl_games.algos_torch import torch_ext
        self.mean_scores = torch_ext.AverageMeter(1, self.algo.games_to_track).to(self.algo.ppo_device)
        self.ep_infos = []
        self.direct_info = {}
        self.writer = self.algo.writer

    def process_infos(self, infos, done_indices):
        if not isinstance(infos, dict):
            classname = self.__class__.__name__
            raise ValueError(f"{classname} expected 'infos' as dict. Received: {type(infos)}")
        # store episode information
        if "episode" in infos:
            self.ep_infos.append(infos["episode"])
        # log other variables directly
        if len(infos) > 0 and isinstance(infos, dict):  # allow direct logging from env
            self.direct_info = {}
            for k, v in infos.items():
                # only log scalars
                if isinstance(v, float) or isinstance(v, int) or (isinstance(v, torch.Tensor) and len(v.shape) == 0):
                    self.direct_info[k] = v

    def after_clear_stats(self):
        # clear stored buffers
        self.mean_scores.clear()

    def after_print_stats(self, frame, epoch_num, total_time):
        import wandb
        # log scalars from the episode
        if self.ep_infos:
            for key in self.ep_infos[0]:
                info_tensor = torch.tensor([], device=self.algo.device)
                for ep_info in self.ep_infos:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    info_tensor = torch.cat((info_tensor, ep_info[key].to(self.algo.device)))
                value = torch.mean(info_tensor)
                wandb.log({f"Episode/{key}": value}, step=epoch_num)
                self.writer.add_scalar("Episode/" + key, value, epoch_num)
            self.ep_infos.clear()
        # log scalars from env information
        for k, v in self.direct_info.items():
            wandb.log({f"{k}/frame": v}, step=frame)
            wandb.log({f"{k}/iter": v}, step=epoch_num)
            wandb.log({f"{k}/time": v}, step=total_time)
            self.writer.add_scalar(f"{k}/frame", v, frame)
            self.writer.add_scalar(f"{k}/iter", v, epoch_num)
            self.writer.add_scalar(f"{k}/time", v, total_time)
        # log mean reward/score from the env
        if self.mean_scores.current_size > 0:
            mean_scores = self.mean_scores.get_mean()
            wandb.log({"scores/mean": mean_scores}, step=frame)
            wandb.log({"scores/iter": mean_scores}, step=epoch_num)
            wandb.log({"scores/time": mean_scores}, step=total_time)
            self.writer.add_scalar("scores/mean", mean_scores, frame)
            self.writer.add_scalar("scores/iter", mean_scores, epoch_num)
            self.writer.add_scalar("scores/time", mean_scores, total_time)


from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import allegro.sac_utd as sac_utd
from allegro.rl_games_utils import load_dataset_transitions, MixedDatasetSampler, Runner, SACCriticLayerNormBuilder
import h5py
import numpy as np

# [wph] Monkey Patch wandb.save to not save checkpoints
import wandb
original_save = wandb.save
def selective_save(glob_str, base_path=None, policy="live"):
    if any(pattern in glob_str for pattern in [".ckpy", ".pt", ".pth", ".pkl", ".pt.gz", ".pth.gz", ".pkl.gz"]):
        return
    return original_save(glob_str, base_path=base_path, policy=policy)
wandb.save = selective_save

# Use the config_name from command line arguments
@hydra_task_config(args_cli.task, args_cli.config_name)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with RL-Games agent."""
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    agent_cfg["params"]["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"]
    agent_cfg["params"]["config"]["max_epochs"] = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg["params"]["config"]["max_epochs"]
    )
    
    # Add wandb configuration
    agent_cfg["params"]["config"]["wandb_project"] = "dexterous"
    agent_cfg["params"]["config"]["wandb_entity"] = "willhu003"
    agent_cfg["params"]["config"]["wandb_tags"] = ["rl_games", "sac"]
    if args_cli.checkpoint is not None:
        resume_path = retrieve_file_path(args_cli.checkpoint)
        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["load_path"] = resume_path
        print(f"[INFO]: Loading model checkpoint from: {agent_cfg['params']['load_path']}")
    train_sigma = float(args_cli.sigma) if args_cli.sigma is not None else None

    # multi-gpu training config
    if args_cli.distributed:
        agent_cfg["params"]["seed"] += app_launcher.global_rank
        agent_cfg["params"]["config"]["device"] = f"cuda:{app_launcher.local_rank}"
        agent_cfg["params"]["config"]["device_name"] = f"cuda:{app_launcher.local_rank}"
        agent_cfg["params"]["config"]["multi_gpu"] = True
        # update env config device
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"

    # set the environment seed (after multi-gpu config for updated rank from agent seed)
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg["params"]["seed"]

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rl_games", agent_cfg["params"]["config"]["name"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs
    log_dir = agent_cfg["params"]["config"].get("full_experiment_name", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    # set directory into agent config
    # logging directory path: <train_dir>/<full_experiment_name>
    agent_cfg["params"]["config"]["train_dir"] = log_root_path
    agent_cfg["params"]["config"]["full_experiment_name"] = log_dir

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_root_path, log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_root_path, log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_root_path, log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_root_path, log_dir, "params", "agent.pkl"), agent_cfg)

    # read configurations about the agent-training
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rl-games
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)

    # register the environment to rl-games registry
    # note: in agents configuration: environment name must be "rlgpu"
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    # set number of actors into agent config
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    
    # Initialize wandb for logging
    wandb_config = {
        # Task and algorithm info
        "task": args_cli.task,
        "algorithm": "sac",
        "model_name": agent_cfg["params"]["model"]["name"],
        "network_name": agent_cfg["params"]["network"]["name"],
        "network_separate": agent_cfg["params"]["network"]["separate"],
        
        # Environment settings
        "num_envs": env.unwrapped.num_envs,
        "clip_observations": agent_cfg["params"]["env"]["clip_observations"],
        "clip_actions": agent_cfg["params"]["env"]["clip_actions"],
        "normalize_input": agent_cfg["params"]["config"]["normalize_input"],
        "reward_shaper_scale": agent_cfg["params"]["config"]["reward_shaper"]["scale_value"],
    }
    
    wandb.init(
        project=agent_cfg["params"]["config"]["wandb_project"],
        entity=agent_cfg["params"]["config"]["wandb_entity"],
        tags=agent_cfg["params"]["config"]["wandb_tags"],
        name=f"{agent_cfg['params']['config']['name']}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
        config=wandb_config,
        sync_tensorboard=False,  # Disable to avoid "Episode/" prefixes from tensorboard
        mode=args_cli.wandb,
    )
    
    # create runner from rl-games
    runner = Runner(WandbAlgoObserver())


    #runner.algo_factory.register_builder('sac_her', lambda **kwargs: sac_her.SACAgent(**kwargs))
    runner.algo_factory.register_builder('sac_utd', lambda **kwargs: sac_utd.SACUTDAgent(**kwargs))
    #register_network('sac_critic_ln', SACCriticLayerNormBuilder)
    
    runner.load(agent_cfg)
    runner.create_agent()
    # reset the agent and env
    runner.reset()
    # Initialize mixed dataset sampling if dataset is provided
    if args_cli.dataset is not None:
        print(f"[INFO] Loading dataset for mixed sampling: {args_cli.dataset}")
        # Load transitions from dataset
        dataset_transitions, action_min, action_max = load_dataset_transitions(
            dataset_path=args_cli.dataset,
            env_obs_shape=env.observation_space.shape,
            env_action_shape=env.action_space.shape,
            device=torch.device(agent_cfg["params"]["config"]["device"])
        )

        # Get the SAC agent from the runner
        agent = runner.agent
        agent.initialize_action_norm(action_min, action_max)
        if hasattr(agent, 'replay_buffer'):
            # Create mixed dataset sampler
            mixed_sampler = MixedDatasetSampler(dataset_transitions, agent.replay_buffer)
            
            # Replace the agent's replay buffer sample method with mixed sampling
            agent.replay_buffer.sample = mixed_sampler.sample
            agent._mixed_sampler = mixed_sampler  # Keep reference to prevent garbage collection

            replay_buffer_capacity = agent.replay_buffer.capacity
            

            print(f"[INFO] Successfully initialized mixed sampling with {len(dataset_transitions)} dataset transitions")
            print(f"Replay buffer size: {agent.replay_buffer.idx}, capacity: {replay_buffer_capacity}")
            
            wandb.log({"dataset/num_transitions": len(dataset_transitions)})
        else:
            print("[WARNING] Agent does not have a replay_buffer attribute. Dataset initialization skipped.")
    else:
        print("[INFO] No dataset provided. Using standard replay buffer only.")
    # train the agent

    

    if args_cli.checkpoint is not None:
        runner.run({"train": True, "play": False, "sigma": train_sigma, "checkpoint": resume_path})
    else:
        runner.run({"train": True, "play": False, "sigma": train_sigma})

    # close the simulator
    env.close()
    
    # finish wandb logging
    wandb.finish()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
