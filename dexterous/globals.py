

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