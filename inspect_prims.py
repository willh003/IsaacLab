#!/usr/bin/env python3
"""
Script to inspect prims in the scene at the breakpoint location in the reorientation environment.
This can be used when you hit the breakpoint in the _setup_scene method.
"""

import isaaclab.sim as sim_utils
import isaacsim.core.utils.stage as stage_utils
from pxr import UsdPhysics, UsdGeom

def inspect_scene_prims():
    """Inspect all prims currently in the scene."""
    print("=== INSPECTING SCENE PRIMS ===")
    
    # Get the current stage
    stage = stage_utils.get_current_stage()
    
    # Get all prims in the scene
    all_prims = sim_utils.get_all_matching_child_prims("/World")
    
    print(f"Found {len(all_prims)} prims under /World:")
    print()
    
    # Print each prim with its path and type
    for i, prim in enumerate(all_prims):
        prim_path = prim.GetPath().pathString
        prim_type = prim.GetTypeName()
        print(f"{i+1:3d}. {prim_path} ({prim_type})")
    
    print()
    print("=== DETAILED PRIM INFORMATION ===")
    
    # Get more detailed information about specific prims
    for i, prim in enumerate(all_prims[:10]):  # Limit to first 10 for readability
        prim_path = prim.GetPath().pathString
        prim_type = prim.GetTypeName()
        
        print(f"\n{i+1}. {prim_path}")
        print(f"   Type: {prim_type}")
        print(f"   Valid: {prim.IsValid()}")
        print(f"   Active: {prim.IsActive()}")
        
        # Check for specific APIs
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            print("   Has RigidBodyAPI")
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            print("   Has ArticulationRootAPI")
        if prim.HasAPI(UsdGeom.Imageable):
            print("   Has ImageableAPI")
    
    # Look for specific prims that should be in the reorientation environment
    print("\n=== LOOKING FOR SPECIFIC PRIMS ===")
    
    # Check for robot prims
    robot_prims = sim_utils.find_matching_prims("/World/envs/env_.*/Robot")
    print(f"Robot prims found: {len(robot_prims)}")
    for prim in robot_prims:
        print(f"  - {prim.GetPath().pathString}")
    
    # Check for object prims
    object_prims = sim_utils.find_matching_prims("/World/envs/env_.*/object")
    print(f"Object prims found: {len(object_prims)}")
    for prim in object_prims:
        print(f"  - {prim.GetPath().pathString}")
    
    # Check for ground plane
    ground_prims = sim_utils.find_matching_prims("/World/ground")
    print(f"Ground prims found: {len(ground_prims)}")
    for prim in ground_prims:
        print(f"  - {prim.GetPath().pathString}")
    
    # Check for lights
    light_prims = sim_utils.find_matching_prims("/World/Light")
    print(f"Light prims found: {len(light_prims)}")
    for prim in light_prims:
        print(f"  - {prim.GetPath().pathString}")

def inspect_environment_prims():
    """Inspect prims specifically in the environment structure."""
    print("\n=== INSPECTING ENVIRONMENT STRUCTURE ===")
    
    # Get all environment prims
    env_prims = sim_utils.find_matching_prims("/World/envs/env_.*")
    print(f"Environment prims found: {len(env_prims)}")
    
    for i, env_prim in enumerate(env_prims):
        env_path = env_prim.GetPath().pathString
        print(f"\nEnvironment {i}: {env_path}")
        
        # Get children of this environment
        children = sim_utils.get_all_matching_child_prims(env_path)
        print(f"  Children ({len(children)}):")
        for child in children:
            child_path = child.GetPath().pathString
            child_type = child.GetTypeName()
            print(f"    - {child_path} ({child_type})")

if __name__ == "__main__":
    inspect_scene_prims()
    inspect_environment_prims() 