#!/usr/bin/env python3
"""
Simple debug functions to inspect prims in the scene.
Use these functions when you hit the breakpoint in the reorientation environment.
"""

import isaaclab.sim as sim_utils
from pxr import UsdPhysics, UsdGeom

def show_all_prims():
    """Show all prims currently in the scene."""
    print("=== ALL PRIMS IN SCENE ===")
    all_prims = sim_utils.get_all_matching_child_prims("/World")
    for i, prim in enumerate(all_prims):
        print(f"{i+1:3d}. {prim.GetPath().pathString} ({prim.GetTypeName()})")
    print(f"Total: {len(all_prims)} prims")

def show_robot_prims():
    """Show robot prims in the scene."""
    print("=== ROBOT PRIMS ===")
    robot_prims = sim_utils.find_matching_prims("/World/envs/env_.*/Robot")
    for i, prim in enumerate(robot_prims):
        print(f"{i+1}. {prim.GetPath().pathString}")
    print(f"Found {len(robot_prims)} robot prims")

def show_object_prims():
    """Show object prims in the scene."""
    print("=== OBJECT PRIMS ===")
    object_prims = sim_utils.find_matching_prims("/World/envs/env_.*/object")
    for i, prim in enumerate(object_prims):
        print(f"{i+1}. {prim.GetPath().pathString}")
    print(f"Found {len(object_prims)} object prims")

def show_env_structure():
    """Show the environment structure."""
    print("=== ENVIRONMENT STRUCTURE ===")
    env_prims = sim_utils.find_matching_prims("/World/envs/env_.*")
    for i, env_prim in enumerate(env_prims):
        env_path = env_prim.GetPath().pathString
        print(f"\nEnvironment {i}: {env_path}")
        children = sim_utils.get_all_matching_child_prims(env_path)
        for child in children:
            print(f"  - {child.GetPath().pathString} ({child.GetTypeName()})")

def show_prim_details(prim_path):
    """Show detailed information about a specific prim."""
    print(f"=== DETAILS FOR {prim_path} ===")
    prims = sim_utils.find_matching_prims(prim_path)
    if not prims:
        print(f"No prims found matching {prim_path}")
        return
    
    for i, prim in enumerate(prims):
        print(f"\nPrim {i+1}:")
        print(f"  Path: {prim.GetPath().pathString}")
        print(f"  Type: {prim.GetTypeName()}")
        print(f"  Valid: {prim.IsValid()}")
        print(f"  Active: {prim.IsActive()}")
        
        # Check APIs
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            print("  Has RigidBodyAPI")
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            print("  Has ArticulationRootAPI")
        if prim.HasAPI(UsdGeom.Imageable):
            print("  Has ImageableAPI")

# Quick commands you can run at the breakpoint:
# show_all_prims()
# show_robot_prims() 
# show_object_prims()
# show_env_structure()
# show_prim_details("/World/envs/env_0/Robot") 