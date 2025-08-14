# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

##
# Full kinematic state observations.
##

gym.register(
    id="Isaac-Repose-Cube-Allegro-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.allegro_env_cfg:AllegroCubeEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AllegroCubePPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "robomimic_bc_cfg_entry_point": f"{agents.__name__}:bc_cfg.json",
        "robomimic_bc_rnn_low_dim_cfg_entry_point": f"{agents.__name__}:bc_rnn_low_dim.json",
        "robomimic_diffusion_policy_cfg_entry_point": f"{agents.__name__}:diffusion_policy_cfg.json",
    },
)

gym.register(
    id="Isaac-Repose-Cube-Allegro-Contact-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.allegro_env_cfg:AllegroCubeContactObsEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AllegroCubePPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "robomimic_bc_cfg_entry_point": f"{agents.__name__}:bc_cfg.json",
        "robomimic_bc_rnn_low_dim_cfg_entry_point": f"{agents.__name__}:bc_rnn_low_dim.json",
        "robomimic_diffusion_policy_cfg_entry_point": f"{agents.__name__}:diffusion_policy_cfg.json",
    },
)

gym.register(
    id="Isaac-Repose-Cube-Allegro-Contact-Reset-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={ 
        "env_cfg_entry_point": f"{__name__}.allegro_env_cfg:AllegroCubeResetEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AllegroCubePPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "robomimic_bc_cfg_entry_point": f"{agents.__name__}:bc_cfg.json",
        "robomimic_bc_rnn_low_dim_cfg_entry_point": f"{agents.__name__}:bc_rnn_low_dim.json",
        "robomimic_diffusion_policy_cfg_entry_point": f"{agents.__name__}:diffusion_policy_cfg.json",
    },
)

gym.register(
    id="Isaac-Repose-Cube-Allegro-Contact-Multi-Reset-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={ 
        "env_cfg_entry_point": f"{__name__}.allegro_env_cfg:AllegroCubeMultiResetEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AllegroCubePPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "robomimic_bc_cfg_entry_point": f"{agents.__name__}:bc_cfg.json",
        "robomimic_bc_rnn_low_dim_cfg_entry_point": f"{agents.__name__}:bc_rnn_low_dim.json",
        "robomimic_diffusion_policy_cfg_entry_point": f"{agents.__name__}:gcdp_cfg.json",
    },
)

gym.register(
    id="Isaac-Repose-Cube-Allegro-Contact-Multi-Reset-Sparse-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={ 
        "env_cfg_entry_point": f"{__name__}.allegro_env_cfg:AllegroCubeMultiResetEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AllegroCubePPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "robomimic_bc_cfg_entry_point": f"{agents.__name__}:bc_cfg.json",
        "robomimic_bc_rnn_low_dim_cfg_entry_point": f"{agents.__name__}:bc_rnn_low_dim.json",
        "robomimic_diffusion_policy_cfg_entry_point": f"{agents.__name__}:gcdp_sparse_cfg.json",
    },
)


gym.register(
    id="Isaac-Repose-Cube-Leap-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.leap_env_cfg:LeapCubeEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LeapCubePPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "robomimic_bc_cfg_entry_point": f"{agents.__name__}:bc_cfg.json",
        "robomimic_bc_rnn_low_dim_cfg_entry_point": f"{agents.__name__}:bc_rnn_low_dim.json",
        "robomimic_diffusion_policy_cfg_entry_point": f"{agents.__name__}:diffusion_policy_cfg.json",
    },
)

gym.register(
    id="Isaac-Repose-Cube-Allegro-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.allegro_env_cfg:AllegroCubeEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AllegroCubePPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "robomimic_bc_cfg_entry_point": f"{agents.__name__}:bc_cfg.json",
        "robomimic_bc_rnn_low_dim_cfg_entry_point": f"{agents.__name__}:bc_rnn_low_dim.json",
        "robomimic_diffusion_policy_cfg_entry_point": f"{agents.__name__}:diffusion_policy_cfg.json",
    },
)

##
# Kinematic state observations without velocity information.
##

gym.register(
    id="Isaac-Repose-Cube-Allegro-NoVelObs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.allegro_env_cfg:AllegroCubeNoVelObsEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AllegroCubeNoVelObsPPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "robomimic_bc_cfg_entry_point": f"{agents.__name__}:bc_cfg.json",
        "robomimic_bc_rnn_low_dim_cfg_entry_point": f"{agents.__name__}:bc_rnn_low_dim.json",
        "robomimic_diffusion_policy_cfg_entry_point": f"{agents.__name__}:diffusion_policy_cfg.json",
    },
)

gym.register(
    id="Isaac-Repose-Cube-Allegro-NoVelObs-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.allegro_env_cfg:AllegroCubeNoVelObsEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AllegroCubeNoVelObsPPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "robomimic_bc_cfg_entry_point": f"{agents.__name__}:bc_cfg.json",
        "robomimic_bc_rnn_low_dim_cfg_entry_point": f"{agents.__name__}:bc_rnn_low_dim.json",
        "robomimic_diffusion_policy_cfg_entry_point": f"{agents.__name__}:diffusion_policy_cfg.json",
    },
)
