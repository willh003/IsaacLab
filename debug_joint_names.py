#!/usr/bin/env python3
"""
Quick script to debug what joint names are available in the articulation.
"""

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import torch
from isaaclab.assets import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass

# Import the Franka-LEAP configuration
from isaaclab_assets.robots.franka_leap import FRANKA_PANDA_LEAP_CFG

@configclass
class FrankaLeapTestSceneCfg(InteractiveSceneCfg):
    """Configuration for testing the Franka-LEAP robot."""
    
    # Add the combined Franka-LEAP robot
    robot: ArticulationCfg = FRANKA_PANDA_LEAP_CFG.replace(prim_path="/World/Robot")

def debug_joint_names():
    """Debug what joint names are available."""
    
    print("🔍 DEBUGGING JOINT NAMES")
    print("=" * 50)
    
    # Create simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 240.0, device="cuda:0")
    sim = sim_utils.SimulationContext(sim_cfg)
    
    # Create scene
    scene_cfg = FrankaLeapTestSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    
    # Initialize simulation
    sim.reset()
    
    # Get robot
    robot = scene["robot"]
    
    print("📋 Available Joint Names:")
    for i, name in enumerate(robot.joint_names):
        print(f"  {i:2d}: {name}")
    
    print(f"\n📊 Total joints: {len(robot.joint_names)}")
    
    # Check pattern matching
    franka_joints = [name for name in robot.joint_names if name.startswith("panda_")]
    leap_joints = [name for name in robot.joint_names if name.startswith("a_")]
    
    print(f"\n🦾 Pattern Analysis:")
    print(f"  Franka joints (panda_*): {len(franka_joints)}")
    for joint in franka_joints:
        print(f"    - {joint}")
        
    print(f"  LEAP joints (a_*): {len(leap_joints)}")
    for joint in leap_joints:
        print(f"    - {joint}")
    
    if not leap_joints:
        print("\n❌ NO LEAP JOINTS FOUND!")
        print("This suggests the joint names in the USD are not being correctly recognized.")
        print("The joints are created in the USD but not showing up in the articulation.")
    
    return robot.joint_names

if __name__ == "__main__":
    debug_joint_names()