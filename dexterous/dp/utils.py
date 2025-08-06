import os
import time
import datetime
import shutil

import robomimic
import torch

def detect_z_rotation_direction_batch(quaternions):
    """
    Determine rotation direction about z-axis for multiple environments.
    
    Args:
        quaternions: torch.Tensor of shape (n_steps, n_envs, 4)
    
    Returns:
        torch.Tensor of shape (n_envs) with 1 for CCW, -1 for CW, 0 for no rotation
    """
    # Stack deque into tensor: (n_observations, n_envs, 4)
    
    # Normalize quaternions
    q = quaternions / torch.norm(quaternions, dim=2, keepdim=True)
    
    # Compute derivatives
    q_dot = q[1:] - q[:-1]  # (n_obs-1, n_envs, 4)
    
    # Quaternion conjugate
    q_conj = torch.cat([q[:-1, :, :1], -q[:-1, :, 1:]], dim=2)
    
    # Vectorized quaternion multiplication for omega_z
    w1, x1, y1, z1 = q_dot.unbind(dim=2)
    w2, x2, y2, z2 = q_conj.unbind(dim=2)
    
    omega_z = 2 * (w1*z2 + x1*y2 - y1*x2 + z1*w2)
    mean_omega_z = omega_z.mean(dim=0)  # (n_envs,)
    
    return torch.sign(mean_omega_z)

def load_action_normalization_params(checkpoint_path):
    # Go up two directories and into logs/normalization_params.txt
    exp_dir = os.path.dirname(os.path.dirname(checkpoint_path))
    norm_file = os.path.join(exp_dir, "logs", "normalization_params.txt")
    with open(norm_file, "r") as f:
        lines = f.readlines()
        min_val = float(lines[0].split(":")[1].strip())
        max_val = float(lines[1].split(":")[1].strip())
    return min_val, max_val

def unnormalize_actions(actions, min_val, max_val):
    # actions: torch.Tensor or np.ndarray in [-1, 1]
    return 0.5 * (actions + 1) * (max_val - min_val) + min_val

def count_parameters(model):
    """Count the total number of parameters in a PyTorch model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def get_exp_dir(output_dir, experiment_name, save_enabled=True):
    """
    Create experiment directory from config. If an identical experiment directory
    exists and @auto_remove_exp_dir is False (default), the function will prompt 
    the user on whether to remove and replace it, or keep the existing one and
    add a new subdirectory with the new timestamp for the current run.

    Args:
        auto_remove_exp_dir (bool): if True, automatically remove the existing experiment
            folder if it exists at the same path.
    
    Returns:
        log_dir (str): path to created log directory (sub-folder in experiment directory)
        output_dir (str): path to created models directory (sub-folder in experiment directory)
            to store model checkpoints
        video_dir (str): path to video directory (sub-folder in experiment directory)
            to store rollout videos
    """
    # timestamp for directory names
    t_now = time.time()
    time_str = datetime.datetime.fromtimestamp(t_now).strftime('%Y%m%d%H%M%S')

    # create directory for where to dump model parameters, tensorboard logs, and videos
    base_output_dir = os.path.expanduser(output_dir)
    if not os.path.isabs(base_output_dir):
        # relative paths are specified relative to robomimic module location
        base_output_dir = os.path.join(robomimic.__path__[0], base_output_dir)
    base_output_dir = os.path.join(base_output_dir, experiment_name, time_str)
    if os.path.exists(base_output_dir):
        raise FileExistsError(f"Experiment directory {base_output_dir} already exists!")

    # only make model directory if model saving is enabled
    model_dir = None
    if save_enabled:
        model_dir = os.path.join(base_output_dir, "models")
        os.makedirs(model_dir)

    # tensorboard directory
    log_dir = os.path.join(base_output_dir, "logs")
    os.makedirs(log_dir)

    # video directory
    video_dir = os.path.join(base_output_dir, "videos")
    os.makedirs(video_dir)
    return log_dir, model_dir, video_dir