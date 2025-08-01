
# --------------------------------------------------------
# LEAP Hand: Low-Cost, Efficient, and Anthropomorphic Hand for Robot Learning
# https://arxiv.org/abs/2309.06440
# Copyright (c) 2025 Kenneth Shaw, Sri Anumakonda
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Based on:
# https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/direct/inhand_manipulation/inhand_manipulation_env.py
# --------------------------------------------------------
from isaaclab_assets.robots.leap import LEAP_HAND_CFG
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelWithAdditiveBiasCfg
import os
import math
from pathlib import Path
from omegaconf import OmegaConf
from isaaclab.sim.spawners.from_files import UsdFileCfg

from .custom_spawner import PhysicsUsdFileCfg
current_dir = Path(__file__).parent
ASSETS_DIR ="/home/will/IsaacLab/source/isaaclab_assets"
OBJECT_DIR = os.path.join(ASSETS_DIR, "isaaclab_assets", "physics_objects")


@configclass
class EventCfg:
    """Configuration for randomization."""


    # -- robot
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        min_step_count_between_reset=720,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 250,
        },
    )
    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,  
        min_step_count_between_reset=720,  
        mode="reset", 
        params={
            "asset_cfg": SceneEntityCfg("robot"),  
            "stiffness_distribution_params":(3.0, 3.0), 
            "damping_distribution_params": (0.1, 0.1),  
            "distribution": "uniform",
        },
    )
    # -- object
    object_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 250,
        },
    )
    object_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (1.0, 1.0),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    object_scale_size=EventTerm(
            func=mdp.randomize_rigid_body_scale,
            mode="prestartup",
            params={
                "asset_cfg": SceneEntityCfg("object"),
                "scale_range": (1.0,1.25), # adr with object scale is not possible so have to define across all envs
            },
        )

    # @property
    # def object_scale_size(self):
    #     return EventTerm(
    #         func=mdp.randomize_rigid_body_scale,
    #         mode="prestartup",
    #         params={
    #             "asset_cfg": SceneEntityCfg("object"),
    #             "scale_range": (2.0,2.0), # adr with object scale is not possible so have to define across all envs
    #         },
    #     )

@configclass
class LeapHandEnvCfg(DirectRLEnvCfg):
    """Base configuration class for LEAP Hand environment."""
    
    # Object configurations
    _object_configs = {
        "cube": { 
            "usd_path": f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            "scale": (1.0, 1.25),
            "density": 400.0,
        },
        "tomato_soup_can": {
            "usd_path": f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/005_tomato_soup_can.usd",
            "scale": (1.0, 1.25),
            "density": 400.0,
        },
        "banana": {
            "usd_path": f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned/011_banana.usd",
            "scale": (1.0, 1.25),
            "density": 400.0,
        },
        "mug": {
            "usd_path": f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned/025_mug.usd",
            "scale": (1.0, 1.25),
            "density": 400.0,
        },
        "large_marker": {
            "usd_path": f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned/040_large_marker.usd",
            "scale": (1.0, 1.25),
            "density": 400.0,
        },
        "large_clamp": {
            "usd_path": f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned/051_large_clamp.usd",
            "scale": (1.0, 1.25),
            "density": 400.0,
        },
        "mustard_bottle": {
            "usd_path": f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/006_mustard_bottle.usd",
            "scale": (1.0, 1.25),
            "density": 400.0,
        },
    }
    
    # Abstract method to be implemented by subclasses
    @property
    def selected_object(self) -> str:
        """Return the selected object name. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement selected_object property")
    
    # Get the selected object configuration
    def _get_object_config(self):
        if self.selected_object not in self._object_configs:
            raise ValueError(f"Unknown object '{self.selected_object}'. Available objects: {list(self._object_configs.keys())}")
        return self._object_configs[self.selected_object]
    
    # env
    decimation = 4 
    min_episode_length_s = 10.0
    action_space = 16
    hist_len = 3
    store_cur_actions = True
    observation_space = 96
    state_space = 0
    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
            gpu_max_rigid_contact_count=2**25,
            gpu_max_rigid_patch_count=2**25
        ),
    )
    # robot
    robot_cfg: ArticulationCfg = LEAP_HAND_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    actuated_joint_names = [
        'a_0', 'a_1', 'a_2', 'a_3', 'a_4', 'a_5', 'a_6', 'a_7',
        'a_8', 'a_9', 'a_10', 'a_11', 'a_12', 'a_13', 'a_14', 'a_15'
    ]
    fingertip_body_names = [
        'fingertip', 
        'thumb_fingertip', 
        'fingertip_2', 
        'fingertip_3'
    ]
    # in-hand object
    def get_object_cfg(self) -> RigidObjectCfg:
        obj_config = self._get_object_config()
        return  RigidObjectCfg(
            prim_path="/World/envs/env_.*/object",
            spawn=UsdFileCfg(
                usd_path=obj_config["usd_path"],
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=False,
                    disable_gravity=False,
                    enable_gyroscopic_forces=True,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=0,
                    sleep_threshold=0.005,
                    stabilization_threshold=0.0025,
                    max_depenetration_velocity=1000.0,
                ),
                # visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                mass_props=sim_utils.MassPropertiesCfg(density=obj_config["density"]),
                
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.00, -0.1, 0.56), rot=(1.0, 0.0, 0.0, 0.0)),
        )
    # goal object
    def get_goal_object_cfg(self) -> VisualizationMarkersCfg:
        obj_config = self._get_object_config()
        return VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={
            "goal": UsdFileCfg(
                usd_path=obj_config["usd_path"],
                scale=(obj_config["scale"][0],obj_config["scale"][0],obj_config["scale"][0]),
                # visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                )
            },
        )
    
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=8192, env_spacing=0.75, replicate_physics=False)
    # reward scales
    z_rotation_steps = 16
    dist_reward_scale = -10.0
    rot_reward_scale = 1.0
    rot_eps = 0.1
    action_penalty_scale = -0.0002
    torque_penalty_scale = -0.0
    pose_diff_penalty_scale = -0.3
    reach_goal_bonus = 250
    fall_penalty = -10
    fall_dist = 0.07
    success_tolerance = 0.2 
    av_factor = 0.1
    action_type="relative" # absolute
    act_moving_average = 1./24
    #adr config
    enable_adr = True
    starting_adr_increments = 0 # 0 for no DR up to num_adr_increments for max DR 
    min_rot_adr_coeff = 0.15  # min 1 full rotation every 6.67 seconds needed to increase ADR
    min_steps_for_dr_change = 240 * 4 # number of steps
    obs_per_timestep = 32    
    obs_timesteps = 3 # same as hist_len
    wrench_trigger_every = 90 # resample every this many policy steps 
    torsional_radius = 0.0 # m
    wrench_prob_per_rollout = 0.5
    # domain randomization config
    events: EventCfg = EventCfg()
    adr_cfg_dict = {
        "num_increments": 25,
        "robot_physics_material": {
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (0.0, 0.5)
        },
        "robot_joint_stiffness_and_damping": {
            "stiffness_distribution_params": (2.5, 3.1),
            "damping_distribution_params": (0.05, 0.15)
        },
        "object_physics_material": {
            "static_friction_range": (0.3, 1.5),
            "dynamic_friction_range": (0.3, 1.5),
            "restitution_range": (0.0, 0.5)
        },
        "object_scale_mass": {
            "mass_distribution_params": (0.9, 1.3)
        }
    }
    
    adr_custom_cfg_dict = {
            "object_wrench": {
                "max_linear_accel": (0.5, 5.)
            },
            "object_spawn": {
                "x_width_spawn": (0.0, 0.01),
                "y_width_spawn": (0.0, 0.01),
                "x_rotation": (0.0, 0.1),       
                "y_rotation": (0.0, 0.1),  
                "z_rotation": (0.0, 0.1),     
            },
            "object_state_noise": {  # not used as the policy is dependant on action states and not object state
                "object_pos_noise": (0.0, 0.00), # m
                "object_pos_bias": (0.0, 0.0),
                "object_rot_noise": (0.0, 0.0), # rad
                "object_rot_bias": (0.0, 0.0), 
            },
            "robot_spawn": {
                "joint_pos_noise": (0.0, 0.05),
                "joint_vel_noise": (0.0, 0.01)
            },
            "robot_state_noise": {
                "robot_noise": (0.0, 0.05),
                "robot_bias": (0.0, 0.03)
            },
            "robot_action_noise": {
                "hand_noise": (0.1, 0.2)
            },
            "action_latency": {
                "hand_latency":(0.0, 3.0),
            },
            "obs_latency": { 
                "latency":(0.0, 0.0),
            },
        }
    
    # These values are now computed dynamically by the property
    act_max_latency = int(adr_custom_cfg_dict["action_latency"]["hand_latency"][1])
    act_latency_rand = 1
    obs_max_latency = int(adr_custom_cfg_dict["obs_latency"]["latency"][1])
    obs_latency_rand = 1
    

@configclass
class LeapHandCubeEnvCfg(LeapHandEnvCfg):
    """Configuration for LEAP Hand environment with cube object."""
    
    episode_length_s = 100.0

    @property
    def selected_object(self) -> str:
        return "cube"

@configclass
class LeapHandCubeClockwiseEnvCfg(LeapHandEnvCfg):
    """Configuration for LEAP Hand environment with cube object."""
    
    episode_length_s = 100.0

    counterclockwise = False

    @property
    def selected_object(self) -> str:
        return "cube"


@configclass
class LeapHandBananaEnvCfg(LeapHandEnvCfg):
    """Configuration for LEAP Hand environment with banana object."""
    
    episode_length_s = 100.0

    @property
    def selected_object(self) -> str:
        return "banana"


@configclass
class LeapHandTomatoSoupCanEnvCfg(LeapHandEnvCfg):
    """Configuration for LEAP Hand environment with tomato soup can object."""
    
    episode_length_s = 100.0

    @property
    def selected_object(self) -> str:
        return "tomato_soup_can"

@configclass
class LeapHandMustardBottleEnvCfg(LeapHandEnvCfg):
    """Configuration for LEAP Hand environment with mustard bottle object."""
    
    episode_length_s = 100.0

    @property
    def selected_object(self) -> str:
        return "mustard_bottle"


@configclass
class LeapHandCubeEzResetEnvCfg(LeapHandEnvCfg):
    """Configuration for LEAP Hand environment with cube object."""
    episode_length_s = 3.0
    
    @property
    def selected_object(self) -> str:
        return "cube"


@configclass
class LeapHandBananaEzResetEnvCfg(LeapHandEnvCfg):
    """Configuration for LEAP Hand environment with banana object."""
    episode_length_s = 3.0
    
    @property
    def selected_object(self) -> str:
        return "banana"


@configclass
class LeapHandTomatoSoupCanEzResetEnvCfg(LeapHandEnvCfg):
    """Configuration for LEAP Hand environment with tomato soup can object."""
    episode_length_s = 100.0

    @property
    def selected_object(self) -> str:
        return "tomato_soup_can"