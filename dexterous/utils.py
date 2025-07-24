import os
import time
import datetime
import shutil

import robomimic


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