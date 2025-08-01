#!/usr/bin/env python3
"""
Script to add physics APIs to the tomato soup can USD file.
This will make it compatible with Isaac Lab's RigidObject class.
"""

import omni.kit.commands
from pxr import Usd, UsdPhysics, PhysxSchema
import isaacsim.core.utils.stage as stage_utils

def add_physics_apis_to_usd(usd_path: str):
    """Add physics APIs to a USD file to make it compatible with Isaac Lab."""
    
    # Open the USD file
    stage = Usd.Stage.Open(usd_path)
    
    # Get the default prim (root of the asset)
    default_prim = stage.GetDefaultPrim()
    if not default_prim.IsValid():
        print(f"Error: No default prim found in {usd_path}")
        return False
    
    print(f"Adding physics APIs to: {usd_path}")
    print(f"Default prim: {default_prim.GetPath()}")
    
    # Apply RigidBodyAPI to the root prim
    if not default_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        UsdPhysics.RigidBodyAPI.Apply(default_prim)
        print("Applied UsdPhysics.RigidBodyAPI")
    else:
        print("UsdPhysics.RigidBodyAPI already exists")
    
    # Apply PhysxRigidBodyAPI to the root prim
    if not default_prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
        PhysxSchema.PhysxRigidBodyAPI.Apply(default_prim)
        print("Applied PhysxSchema.PhysxRigidBodyAPI")
    else:
        print("PhysxSchema.PhysxRigidBodyAPI already exists")
    
    # Apply MassAPI to the root prim
    if not default_prim.HasAPI(UsdPhysics.MassAPI):
        UsdPhysics.MassAPI.Apply(default_prim)
        print("Applied UsdPhysics.MassAPI")
    else:
        print("UsdPhysics.MassAPI already exists")
    
    # Apply CollisionAPI to the root prim
    if not default_prim.HasAPI(UsdPhysics.CollisionAPI):
        UsdPhysics.CollisionAPI.Apply(default_prim)
        print("Applied UsdPhysics.CollisionAPI")
    else:
        print("UsdPhysics.CollisionAPI already exists")
    
    # Save the modified USD file
    stage.Save()
    print(f"Saved modified USD file: {usd_path}")
    
    return True

if __name__ == "__main__":
    # Path to the tomato soup can USD file
    tomato_can_path = "/home/will/IsaacLab/source/isaaclab_assets/isaaclab_assets/objects/005_tomato_soup_can.usd"
    
    # Add physics APIs
    success = add_physics_apis_to_usd(tomato_can_path)
    
    if success:
        print("Successfully added physics APIs to the tomato soup can USD file!")
        print("You can now use it with Isaac Lab's RigidObject class.")
    else:
        print("Failed to add physics APIs to the USD file.") 