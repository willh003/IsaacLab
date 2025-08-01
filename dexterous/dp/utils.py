import os
import time
import datetime
import shutil

import robomimic

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