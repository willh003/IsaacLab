"""
Leap Inhand Manipulation environment.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

reorient_task_entry = "isaaclab_tasks.direct.leap_hand_reorient"

gym.register(
    id="Isaac-Reorient-Cube-Leap",
    entry_point=f"{reorient_task_entry}.reorientation_env:ReorientationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.leap_hand_env_cfg:LeapHandCubeEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LeapHandPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "robomimic_bc_cfg_entry_point": f"{agents.__name__}:bc_cfg.json",
        "robomimic_bc_rnn_low_dim_cfg_entry_point": f"{agents.__name__}:bc_rnn_low_dim.json",
        "robomimic_diffusion_policy_cfg_entry_point": f"{agents.__name__}:diffusion_policy_cfg.json",
    },
)

gym.register(
    id="Isaac-Reorient-Cube-Clockwise-Leap",
    entry_point=f"{reorient_task_entry}.reorientation_env:ReorientationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.leap_hand_env_cfg:LeapHandCubeClockwiseEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LeapHandPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "robomimic_bc_cfg_entry_point": f"{agents.__name__}:bc_cfg.json",
        "robomimic_bc_rnn_low_dim_cfg_entry_point": f"{agents.__name__}:bc_rnn_low_dim.json",
        "robomimic_diffusion_policy_cfg_entry_point": f"{agents.__name__}:diffusion_policy_cfg.json",
    },
)

gym.register(
    id="Isaac-Reorient-TomatoSoupCan-Leap",
    entry_point=f"{reorient_task_entry}.reorientation_env:ReorientationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.leap_hand_env_cfg:LeapHandTomatoSoupCanEzResetEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LeapHandPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "robomimic_bc_cfg_entry_point": f"{agents.__name__}:bc_cfg.json",
        "robomimic_bc_rnn_low_dim_cfg_entry_point": f"{agents.__name__}:bc_rnn_low_dim.json",
        "robomimic_diffusion_policy_cfg_entry_point": f"{agents.__name__}:diffusion_policy_cfg.json",
    },
)

gym.register(
    id="Isaac-Reorient-MustardBottle-Leap",
    entry_point=f"{reorient_task_entry}.reorientation_env:ReorientationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.leap_hand_env_cfg:LeapHandMustardBottleEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LeapHandPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "robomimic_bc_cfg_entry_point": f"{agents.__name__}:bc_cfg.json",
        "robomimic_bc_rnn_low_dim_cfg_entry_point": f"{agents.__name__}:bc_rnn_low_dim.json",
        "robomimic_diffusion_policy_cfg_entry_point": f"{agents.__name__}:diffusion_policy_cfg.json",
    },
)

gym.register(
    id="Isaac-Reorient-Cube-Leap-EzReset",
    entry_point=f"{reorient_task_entry}.reorientation_ezreset_env:ReorientationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.leap_hand_env_cfg:LeapHandCubeEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LeapHandPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "robomimic_bc_cfg_entry_point": f"{agents.__name__}:bc_cfg.json",
        "robomimic_bc_rnn_low_dim_cfg_entry_point": f"{agents.__name__}:bc_rnn_low_dim.json",
        "robomimic_diffusion_policy_cfg_entry_point": f"{agents.__name__}:diffusion_policy_cfg.json",
    },
)


gym.register(
    id="Isaac-Reorient-Banana-Leap-EzReset",
    entry_point=f"{reorient_task_entry}.reorientation_ezreset_env:ReorientationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.leap_hand_env_cfg:LeapHandBananaEzResetEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LeapHandPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "robomimic_bc_cfg_entry_point": f"{agents.__name__}:bc_cfg.json",
        "robomimic_bc_rnn_low_dim_cfg_entry_point": f"{agents.__name__}:bc_rnn_low_dim.json",
        "robomimic_diffusion_policy_cfg_entry_point": f"{agents.__name__}:diffusion_policy_cfg.json",
    },
)

gym.register(
    id="Isaac-Reorient-TomatoSoupCan-Leap-EzReset",
    entry_point=f"{reorient_task_entry}.reorientation_ezreset_env:ReorientationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.leap_hand_env_cfg:LeapHandTomatoSoupCanEzResetEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LeapHandPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "robomimic_bc_cfg_entry_point": f"{agents.__name__}:bc_cfg.json",
        "robomimic_bc_rnn_low_dim_cfg_entry_point": f"{agents.__name__}:bc_rnn_low_dim.json",
        "robomimic_diffusion_policy_cfg_entry_point": f"{agents.__name__}:diffusion_policy_cfg.json",
    },
)

