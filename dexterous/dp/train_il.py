# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# MIT License
#
# Copyright (c) 2021 Stanford Vision and Learning Lab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
The main entry point for training policies from pre-collected data.

This script loads dataset(s), creates a model based on the algorithm specified,
and trains the model. It supports training on various environments with multiple
algorithms from robomimic.

Args:
    algo: Name of the algorithm to run.
    task: Name of the environment.
    name: If provided, override the experiment name defined in the config.
    dataset: If provided, override the dataset path defined in the config.
    log_dir: Directory to save logs.
    normalize_training_actions: Whether to normalize actions in the training data.

This file has been modified from the original robomimic version to integrate with IsaacLab.
"""

"""Rest everything follows."""

# Standard library imports
import argparse
from re import S

# Third-party imports
import gymnasium as gym
import h5py
import json
import numpy as np
import os
import shutil
import sys
import time
import torch
import traceback
from collections import OrderedDict
from torch.utils.data import DataLoader
from tqdm import tqdm   
import psutil
import collections
from pathlib import Path

# Robomimic imports
# IMPORTANT: do not remove these, because they are required to register the diffusion policy
from dp_model import DiffusionPolicyConfig, DiffusionPolicyUNet
from utils import get_exp_dir, detect_z_rotation_direction_batch

import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.train_utils as TrainUtils
from robomimic.algo import algo_factory
from robomimic.config import Config, config_factory
from robomimic.utils.log_utils import DataLogger, PrintLogger

from utils import load_cfg_from_registry_no_gym


import wandb

def normalize_hdf5_actions(config: Config, log_dir: str) -> str:
    """Normalizes actions in hdf5 dataset to [-1, 1] range.

    Args:
        config: The configuration object containing dataset path.
        log_dir: Directory to save normalization parameters.

    Returns:
        Path to the normalized dataset.
    """
    base, ext = os.path.splitext(config.train.data)
    normalized_path = base + "_normalized" + ext

    # Copy the original dataset
    print(f"Creating normalized dataset at {normalized_path}")
    shutil.copyfile(config.train.data, normalized_path)

    # Open the new dataset and normalize the actions
    with h5py.File(normalized_path, "r+") as f:
        dataset_paths = [f"/data/demo_{str(i)}/actions" for i in range(len(f["data"].keys()))]

        # Compute the min and max of the dataset
        dataset = np.array(f[dataset_paths[0]]).flatten()
        for i, path in enumerate(dataset_paths):
            if i != 0:
                data = np.array(f[path]).flatten()
                dataset = np.append(dataset, data)

        max = np.max(dataset)
        min = np.min(dataset)

        # Normalize the actions
        for i, path in enumerate(dataset_paths):
            data = np.array(f[path])
            normalized_data = 2 * ((data - min) / (max - min)) - 1  # Scale to [-1, 1] range
            del f[path]
            f[path] = normalized_data

        # Save the min and max values to log directory
        with open(os.path.join(log_dir, "normalization_params.txt"), "w") as f:
            f.write(f"min: {min}\n")
            f.write(f"max: {max}\n")

    return normalized_path


def train(config: Config, device: str, log_dir: str, ckpt_dir: str, video_dir: str, wandb_mode: str = "online"):
    """Train a model using the algorithm specified in config.

    Args:
        config: Configuration object.
        device: PyTorch device to use for training.
        log_dir: Directory to save logs.
        ckpt_dir: Directory to save checkpoints.
        video_dir: Directory to save videos.
        wandb_mode: Wandb mode.
    """
    # first set seeds
    np.random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)

    print("\n============= New Training Run with Config =============")
    print(config)
    print("")

    print(f">>> Saving logs into directory: {log_dir}")
    print(f">>> Saving checkpoints into directory: {ckpt_dir}")
    print(f">>> Saving videos into directory: {video_dir}")

    # read config to set up metadata for observation modalities (e.g. detecting rgb observations)
    ObsUtils.initialize_obs_utils_with_config(config)

    # make sure the dataset exists
    dataset_path = os.path.expanduser(config.train.data)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset at provided path {dataset_path} not found!")

    # load basic metadata from training file
    print("\n============= Loaded Environment Metadata =============")
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=config.train.data)
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=config.train.data, all_obs_keys=config.all_obs_keys, verbose=True
    )

    if config.experiment.env is not None:
        env_meta["env_name"] = config.experiment.env
        print("=" * 30 + "\n" + "Replacing Env to {}\n".format(env_meta["env_name"]) + "=" * 30)


    # setup for a new training run
    data_logger = DataLogger(log_dir, config=config, log_tb=config.experiment.logging.log_tb)
    model = algo_factory(
        algo_name=config.algo_name,
        config=config,
        obs_key_shapes=shape_meta["all_shapes"],
        ac_dim=shape_meta["ac_dim"],
        device=device,
    )
    # save the config as a json file

    with open(os.path.join(log_dir, "..", "config.json"), "w") as outfile:
        json.dump(config, outfile, indent=4)

    print("\n============= Model Summary =============")
    print(model)  # print model summary
    print("")

    # load training data
    trainset, validset = TrainUtils.load_data_for_training(config, obs_keys=shape_meta["all_obs_keys"])

    train_sampler = trainset.get_dataset_sampler()
    print("\n============= Training Dataset =============")
    print(trainset)
    print("")

    # maybe retrieve statistics for normalizing observations
    obs_normalization_stats = None
    if config.train.hdf5_normalize_obs:
        obs_normalization_stats = trainset.get_obs_normalization_stats()

    def custom_collate_fn(batch):
        # Assume batch is a list of dicts with keys: actions, rewards, dones, obs
        # Each obs is a dict with keys like 'goal_pose', 'goal_quat_diff', ...
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
            """Collate a list of observation dictionaries into a batched dictionary."""
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
                obs_dict[k] = torch.stack(obs_dict[k], dim=0)
            
            return obs_dict
        
        # Collate obs and goal_obs using the helper function
        batch_out['obs'] = collate_obs_dict(batch_out['obs'])
        batch_out['goal_obs'] = collate_obs_dict(batch_out['goal_obs'])
        
        return batch_out

    # initialize data loaders
    train_loader = DataLoader(
        dataset=trainset,
        sampler=train_sampler,
        batch_size=config.train.batch_size,
        shuffle=(train_sampler is None),
        num_workers=config.train.num_data_workers,
        drop_last=True,
        collate_fn=custom_collate_fn,
    )

    if config.experiment.validate:
        # cap num workers for validation dataset at 1
        num_workers = min(config.train.num_data_workers, 1)
        valid_sampler = validset.get_dataset_sampler()
        valid_loader = DataLoader(
            dataset=validset,
            sampler=valid_sampler,
            batch_size=config.train.batch_size,
            shuffle=(valid_sampler is None),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=custom_collate_fn,
        )
    else:
        valid_loader = None
    
    #test_train_loader()

    # Monkey-patch wandb.save to avoid uploading large files
    original_save = wandb.save
    def selective_save(glob_str, base_path=None, policy="live"):
        if any(pattern in glob_str for pattern in [".ckpy", ".pt", ".pth", ".pkl", ".pt.gz", ".pth.gz", ".pkl.gz"]):
            return
        return original_save(glob_str, base_path=base_path, policy=policy)
    wandb.save = selective_save
    
    wandb_cfg = dict(config)
    wandb_cfg["train"]["train_length"] = len(trainset)
    if config.experiment.validate:
        wandb_cfg["train"]["valid_length"] = len(validset)
    
    wandb.init(
        project="dexterous",
        entity="willhu003",
        name=os.path.basename(os.path.dirname(log_dir)), # log dir is under the experiment dir
        config=wandb_cfg,
        dir=log_dir,
        mode=wandb_mode
    )

    # main training loop
    best_valid_loss = None
    last_ckpt_time = time.time()

    # number of learning steps per epoch (defaults to a full dataset pass)
    train_num_steps = config.experiment.epoch_every_n_steps
    valid_num_steps = config.experiment.validation_epoch_every_n_steps

    for epoch in range(1, config.train.num_epochs + 1):  # epoch numbers start at 1
        step_log = TrainUtils.run_epoch(model=model, data_loader=train_loader, epoch=epoch, num_steps=train_num_steps)
        model.on_epoch_end(epoch)

        # setup checkpoint path

        # check for recurring checkpoint saving conditions
        should_save_ckpt = False
        if config.experiment.save.enabled:
            time_check = (config.experiment.save.every_n_seconds is not None) and (
                time.time() - last_ckpt_time > config.experiment.save.every_n_seconds
            )
            epoch_check = (
                (config.experiment.save.every_n_epochs is not None)
                and (epoch > 0)
                and (epoch % config.experiment.save.every_n_epochs == 0)
            )
            epoch_list_check = epoch in config.experiment.save.epochs
            should_save_ckpt = time_check or epoch_check or epoch_list_check
            ckpt_reason = "time" if time_check else "epoch" if epoch_check else "epoch_list" if epoch_list_check else None
        
        if should_save_ckpt:
            last_ckpt_time = time.time()


        print(f"Train Epoch {epoch}")
        print(json.dumps(step_log, sort_keys=True, indent=4))
        for k, v in step_log.items():
            if k.startswith("Time_"):
                data_logger.record(f"Timing_Stats/Train_{k[5:]}", v, epoch)
            else:
                data_logger.record(f"Train/{k}", v, epoch)
        

        wandb_dict = {f"train/{k}": v for k, v in step_log.items() if "time" not in k.lower()}
        wandb_dict.update({f"time/{k}": v for k, v in step_log.items() if "time" in k.lower()})
        wandb.log(wandb_dict, step=epoch)

        # Evaluate the model on validation set
        if config.experiment.validate:
            with torch.no_grad():
                step_log = TrainUtils.run_epoch(
                    model=model, data_loader=valid_loader, epoch=epoch, validate=True, num_steps=valid_num_steps
                )
            for k, v in step_log.items():
                if k.startswith("Time_"):
                    data_logger.record(f"Timing_Stats/Valid_{k[5:]}", v, epoch)
                else:
                    data_logger.record(f"Valid/{k}", v, epoch)
            # Log to wandb
            wandb.log({f"validation/{k}": v for k, v in step_log.items() if "time" not in k.lower()}, step=epoch)

            print(f"Validation Epoch {epoch}")
            print(json.dumps(step_log, sort_keys=True, indent=4))

            # save checkpoint if achieve new best validation loss
            valid_check = "Loss" in step_log
            if valid_check and (best_valid_loss is None or (step_log["Loss"] <= best_valid_loss)):
                best_valid_loss = step_log["Loss"]
                if config.experiment.save.enabled and config.experiment.save.on_best_validation:
                    should_save_ckpt = True
                    ckpt_reason = "valid"

        # Save model checkpoints based on conditions (success rate, validation loss, etc)
        if should_save_ckpt:
            TrainUtils.save_model(
                model=model,
                config=config,
                env_meta=env_meta,
                shape_meta=shape_meta,
                ckpt_path=os.path.join(ckpt_dir, f"ckpt_{ckpt_reason}.pth"),
                obs_normalization_stats=obs_normalization_stats,
            )

        # Finally, log memory usage in MB
        process = psutil.Process(os.getpid())
        mem_usage = int(process.memory_info().rss / 1000000)
        data_logger.record("System/RAM Usage (MB)", mem_usage, epoch)
        print(f"\nEpoch {epoch} Memory Usage: {mem_usage} MB\n")
        
        wandb.log({"System/RAM Usage (MB)": mem_usage}, step=epoch)

    # terminate logging
    data_logger.close()
    
    wandb.finish()


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
    return filtered


def main(args: argparse.Namespace):
    """Train a model on a task using a specified algorithm.

    Args:
        args: Command line arguments.
    """
    # load config
    if args.task is not None:
        # obtain the configuration entry point
        cfg_entry_point_key = f"robomimic_{args.algo}_cfg_entry_point"
        task_name = args.task.split(":")[-1]

        print(f"Loading configuration for task: {task_name}")
        ext_cfg = load_cfg_from_registry_no_gym(args.task, cfg_entry_point_key)
        config = config_factory(ext_cfg["algo_name"])

        filtered_ext_cfg = filter_config_dict(ext_cfg, config)
        with config.values_unlocked():
            config.update(filtered_ext_cfg)
    else:
        raise ValueError("Please provide a task name through CLI arguments.")

    if args.dataset is not None:
        config.train.data = args.dataset
    else:
        raise ValueError("Please provide a dataset path through CLI arguments.")

    if args.obs_cond is not None:
        print(f"Overriding observation conditioning with: {args.obs_cond}")
        config.observation.modalities.obs.low_dim = args.obs_cond
    if args.goal_cond is not None:
        print(f"Overriding goal conditioning with: {args.goal_cond}")
        config.observation.modalities.goal.low_dim = args.goal_cond

    if args.name is not None:
        config.experiment.name = args.name
    
    # change location of experiment directory
    config.train.output_dir = os.path.abspath(os.path.join("./logs", args.log_dir, args.task))

    log_dir, ckpt_dir, video_dir = get_exp_dir(config.train.output_dir, config.experiment.name, config.experiment.save.enabled)
    
    if args.normalize_training_actions:
        config.train.data = normalize_hdf5_actions(config, log_dir)

    # get torch device
    device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)

    config.lock()

    # catch error during training and print it
    res_str = "finished run successfully!"
    try:
        train(config, device, log_dir, ckpt_dir or "", video_dir, wandb_mode=args.wandb)
    except Exception as e:
        res_str = f"run failed with error:\n{e}\n\n{traceback.format_exc()}"
    print(res_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Experiment Name (for tensorboard, saving models, etc.)
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="(optional) if provided, override the experiment name defined in the config",
    )

    # Dataset path, to override the one in the config
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="(optional) if provided, override the dataset path defined in the config",
    )

    home_dir = Path.home()
    log_dir = home_dir / "IsaacLab/dexterous/logs/dexterous"

    parser.add_argument("--task", type=str, default=None, help="Name of the task.")
    parser.add_argument("--algo", type=str, default=None, help="Name of the algorithm.")
    parser.add_argument("--log_dir", type=str, default=log_dir, help="Path to log directory")
    parser.add_argument("--normalize_training_actions", action="store_true", default=True, help="Normalize actions")
    parser.add_argument("--wandb", type=str, default="online", help="Wandb mode")
    parser.add_argument("--obs_cond", type=lambda x: x.split(',') if x is not None else None, default=None, help="Observation conditioning")
    parser.add_argument("--goal_cond", type=lambda x: x.split(',') if x is not None else None, default=None, help="Goal conditioning")

    args = parser.parse_args()

    # run training
    main(args)

