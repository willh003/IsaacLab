
import torch


# Define the indices for each observation key (start, end)
OBS_INDICES = {
    "robot0_joint_pos": (0, 16),
    "robot0_joint_vel": (16, 32),
    "object_pos": (32, 35),
    "object_quat": (35, 39),
    "object_lin_vel": (39, 42),
    "object_ang_vel": (42, 45),
    "last_action": (56, 72),
}

GOAL_INDICES = {
    "goal_pose": (45, 52),
    "goal_quat_diff": (52, 56),
}


def get_state_from_env(obs, obs_keys, goal_keys = None, device = 'cuda'):
    """
    Get the full observation and goal dictionaries from the environment observation
    """
    obs_dict = {}

    obs = obs['policy']
    for key in obs_keys:
        current = torch.tensor(obs[:, OBS_INDICES[key][0]:OBS_INDICES[key][1]], dtype=torch.float32, device=device)
        obs_dict[key] = current
    goal_dict = None
    if goal_keys is not None:
        goal_dict = {}
        for key in goal_keys:
            current = torch.tensor(obs[:, GOAL_INDICES[key][0]:GOAL_INDICES[key][1]], dtype=torch.float32, device=device)
            goal_dict[key] = current

    return obs_dict, goal_dict