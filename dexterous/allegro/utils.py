
import torch
import numpy as np

# Define the indices for each observation key (start, end)
OBS_INDICES = {
    "robot0_joint_pos": (0, 16),
    "robot0_joint_vel": (16, 32),
    "object_pos": (32, 35),
    "object_quat": (35, 39),
    "object_lin_vel": (39, 42),
    "object_ang_vel": (42, 45),
    "goal_pose": (45, 52),
    "goal_quat_diff": (52, 56),
    "last_action": (56, 72),
    "fingertip_contacts": (72, 76),
}

def get_termination_env_ids(env):
    """Get the termination reason among "success", "failure", or "time_out"."""
                    # Check for environment resets due to success count reset mechanism (successful episodes)
    term_dones = env.unwrapped.termination_manager._term_dones

    # xor of the three should be True
    assert (term_dones["success"] + term_dones["failure"] <= 1).all(), "Only one of success, failure, or time_out should be True"

    # get the envs that are done
    done_envs = {
        "success": np.nonzero(term_dones["success"]),
        "failure": np.nonzero(term_dones["failure"]),
        "time_out": np.nonzero(term_dones["time_out"])
    }

    return done_envs

def get_state_from_env(obs, obs_keys, device = 'cuda'):
    """
    Get the full observation and goal dictionaries from the environment observation
    """
    obs_dict = {}

    for key in obs_keys:
        current = torch.tensor(obs[:, OBS_INDICES[key][0]:OBS_INDICES[key][1]], dtype=torch.float32, device=device)
        obs_dict[key] = current

    return obs_dict

def get_goal_from_env(obs, goal_keys, device='cuda'):
    """
    extracts a goal from the env given goal_keys, which may correspond to state observations
    supported goal_keys: object_pos, object_quat, goal_pose, goal_quat_diff
    """
    if goal_keys is None:
        return {}
        
    goal_dict = {} 

    goal_pose = torch.tensor(obs[:, OBS_INDICES["goal_pose"][0]:OBS_INDICES["goal_pose"][1]], dtype=torch.float32, device=device)
    goal_quat_diff = torch.tensor(obs[:, OBS_INDICES["goal_quat_diff"][0]:OBS_INDICES["goal_quat_diff"][1]], dtype=torch.float32, device=device)

    for key in goal_keys:
        assert key in ["goal_pose", "goal_quat_diff", "object_pos", "object_quat"], f"Error: unsupported key {key}"

        if key == "object_pos":
            goal_dict[key] = goal_pose[:, :3]
        elif key == "object_quat":            
            goal_dict[key] = goal_pose[:, 3:]
        else:
            current = torch.tensor(obs[:, OBS_INDICES[key][0]:OBS_INDICES[key][1]], dtype=torch.float32, device=device)
            goal_dict[key] = current
    
    return goal_dict