from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.manipulation.inhand.inhand_env_cfg as inhand_env_cfg

##
# Pre-defined configs
##
from isaaclab_assets import LEAP_HAND_CFG  # isort: skip



@configclass
class LeapCubeEnvCfg(inhand_env_cfg.InHandObjectEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to allegro hand
        self.scene.robot = LEAP_HAND_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

