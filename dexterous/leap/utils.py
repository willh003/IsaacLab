def get_state_from_env(env, obs):
    """
    Get object position and rotation directly from environment.
    
    Args:
        env: The ReorientationEnv instance
        obs: The observation from the environment
    
    Returns: dict of
        object_pos: Object position relative to environment origin (num_envs, 3)
        object_rot: Object rotation as quaternion (num_envs, 4) [w, x, y, z]
        joint_pos: Joint positions (num_envs, 16)
        action_targets: Action targets (num_envs, 16)
    """
    # Make sure intermediate values are computed    
    # Get object position (relative to environment origin)
    object_pos = env.object_pos  # Shape: (num_envs, 3)
    
    # Get object rotation (quaternion format: w, x, y, z)
    object_rot = env.object_rot  # Shape: (num_envs, 4)
    
    # Get current joints and action targets
    cur_state = obs[:, -32:]
    joint_pos = cur_state[:, :16]
    action_targets = cur_state[:, 16:]

    return {"joint_pos": joint_pos, "action_targets": action_targets, "object_pos": object_pos, "object_rot": object_rot}