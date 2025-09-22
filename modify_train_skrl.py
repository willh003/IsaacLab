#!/usr/bin/env python3

import re

# Read the original file
with open('/home/will/IsaacLab/dexterous/allegro/train_skrl.py', 'r') as f:
    content = f.read()

# Add wandb import and monkey patch after the other imports
import_section = """
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)
"""

new_import_section = """
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

# [wph] Monkey Patch wandb.save to not save checkpoints
import wandb
original_save = wandb.save
def selective_save(glob_str, base_path=None, policy="live"):
    if any(pattern in glob_str for pattern in [".ckpy", ".pt", ".pth", ".pkl", ".pt.gz", ".pth.gz", ".pkl.gz"]):
        return
    return original_save(glob_str, base_path=base_path, policy=policy)
wandb.save = selective_save

# PLACEHOLDER: Extension template (do not remove this comment)
"""

# Replace the import section
content = content.replace(import_section, new_import_section)

# Add wandb configuration after environment seed setup
seed_section = """    # set the agent and environment seed from command line
    # note: certain randomization occur in the environment initialization so we set the seed here
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]"""

new_seed_section = """    # set the agent and environment seed from command line
    # note: certain randomization occur in the environment initialization so we set the seed here
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]

    # configure wandb logging
    if "agent" in agent_cfg and "experiment" in agent_cfg["agent"]:
        agent_cfg["agent"]["experiment"]["write_to"] = "wandb"
        agent_cfg["agent"]["experiment"]["wandb_kwargs"] = {
            "project": "dexterous",
            "entity": "willhu003"
        }"""

# Replace the seed section
content = content.replace(seed_section, new_seed_section)

# Write the modified content back to the file
with open('/home/will/IsaacLab/dexterous/allegro/train_skrl.py', 'w') as f:
    f.write(content)

print("Successfully modified train_skrl.py with wandb logging")
