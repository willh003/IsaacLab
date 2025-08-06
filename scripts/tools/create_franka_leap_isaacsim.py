#!/usr/bin/env python3
"""
Script to create Franka + LEAP hand robot within Isaac Sim.

Run this script from within Isaac Sim's Python console or script editor.
"""

from isaaclab.app import AppLauncher

# create new namespace with headless=True for the launcher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Gf, Sdf, UsdShade
import os
from pathlib import Path

def create_franka_leap_robot():
    """Create a combined Franka + LEAP hand robot USD file with merged articulation."""
    
    # Import the assets utility to get the correct paths
    from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
    franka_usd_path = f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd"
    leap_hand_usd_path = "/home/will/IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/leap_hand_v1_right/leap_hand_right.usd"
    
    # Output path for the combined robot
    output_dir = Path.home() / "IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/franka_leap_combined"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_usd_path = output_dir / "franka_leap_robot.usd"
    
    print(f"Creating merged Franka-LEAP robot USD file at: {output_usd_path}")
    print(f"Using Franka USD: {franka_usd_path}")
    print(f"Using LEAP hand USD: {leap_hand_usd_path}")
    
    # Check if LEAP hand file exists
    if not os.path.exists(leap_hand_usd_path):
        print(f"Error: LEAP hand USD file not found at {leap_hand_usd_path}")
        return False
    
    # Create a new USD stage
    stage = Usd.Stage.CreateNew(str(output_usd_path))
    

    
    # Create the root robot prim
    root_prim = stage.DefinePrim("/Robot", "Xform")
    stage.SetDefaultPrim(root_prim)
    

    # Apply proper articulation APIs
    UsdPhysics.ArticulationRootAPI.Apply(root_prim)
    PhysxSchema.PhysxArticulationAPI.Apply(root_prim)
    
    # Set articulation properties
    root_prim.CreateAttribute("physxArticulation:articulationEnabled", Sdf.ValueTypeNames.Bool).Set(True)
    root_prim.CreateAttribute("physxArticulation:enabledSelfCollisions", Sdf.ValueTypeNames.Bool).Set(True)
    root_prim.CreateAttribute("physxArticulation:sleepThreshold", Sdf.ValueTypeNames.Float).Set(0.005)
    root_prim.CreateAttribute("physxArticulation:solverPositionIterationCount", Sdf.ValueTypeNames.Int).Set(8)
    root_prim.CreateAttribute("physxArticulation:solverVelocityIterationCount", Sdf.ValueTypeNames.Int).Set(0)
    root_prim.CreateAttribute("physxArticulation:stabilizationThreshold", Sdf.ValueTypeNames.Float).Set(0.0005)
    
    print("Applied USD ArticulationRootAPI and PhysX ArticulationAPI")
    
    panda_prim = stage.DefinePrim("/Robot/franka", "Xform")

    panda_prim.GetReferences().AddReference(franka_usd_path, "/panda")

    
    # Remove the hand parts from the Franka robot
    # print("Removing Franka hand parts...")
    # hand_parts_to_remove = ["panda_hand", "panda_leftfinger", "panda_rightfinger"]
    # for part_name in hand_parts_to_remove:
    #     hand_prim = franka_panda.GetChild(part_name)
    #     if hand_prim:
    #         stage.RemovePrim(hand_prim.GetPath())
    #         print(f"  Removed {part_name}")
    
    # Add the LEAP hand as a reference
    print("Adding LEAP hand reference...")
    leap_hand_prim = stage.DefinePrim("/Robot/leap_hand", "Xform")
    leap_hand_prim.GetReferences().AddReference(leap_hand_usd_path, "/leap_right")
    
    # Create a fixed joint to connect the Franka arm to the LEAP hand
    print("Creating connection joint...")
    joint_prim = stage.DefinePrim("/Robot/joints/franka_leap_joint", "PhysicsFixedJoint")
    
    # Set the joint to connect panda_link7 to leap_hand
    joint_prim.CreateRelationship("physics:body0").SetTargets(["/Robot/panda/panda_link7"])
    joint_prim.CreateRelationship("physics:body1").SetTargets(["/Robot/leap_hand"])
    
    # Set local transforms for the connection
    joint_prim.CreateAttribute("physics:localPos0", Sdf.ValueTypeNames.Vector3f).Set(Gf.Vec3f(0, 0, 0.107))
    joint_prim.CreateAttribute("physics:localRot0", Sdf.ValueTypeNames.Quatf).Set(Gf.Quatf(1, 0, 0, 0))
    joint_prim.CreateAttribute("physics:localPos1", Sdf.ValueTypeNames.Vector3f).Set(Gf.Vec3f(0, 0, 0))
    joint_prim.CreateAttribute("physics:localRot1", Sdf.ValueTypeNames.Quatf).Set(Gf.Quatf(1, 0, 0, 0))
    
    print("Created connection joint")
    
    # Update joint relationships to fix all path references
    print("Updating joint relationships...")
    
    # Find all joints and update their body references
    # joints_to_remove = []
    # for prim in stage.Traverse():
    #     if prim.GetTypeName() in ["PhysicsRevoluteJoint", "PhysicsPrismaticJoint", "PhysicsFixedJoint"]:
    #         # Check if this is the problematic hand joint that references non-existent panda_hand
    #         body1_rel = prim.GetRelationship("physics:body1")
    #         if body1_rel:
    #             targets = body1_rel.GetTargets()
    #             for target in targets:
    #                 if str(target).endswith("/panda_hand"):
    #                     print(f"  Removing invalid joint: {prim.GetPath()}")
    #                     joints_to_remove.append(prim.GetPath())
    #                     break
            
    #         # Skip processing joints that will be removed
    #         if prim.GetPath() in joints_to_remove:
    #             continue
                
    #         # Update body relationships to point to the new locations
    #         body0_rel = prim.GetRelationship("physics:body0")
            
    #         def update_target_path(target_str):
    #             """Update a target path to the correct location."""
    #             # Fix Franka paths - they should now be under Robot/panda
    #             if target_str.startswith("/panda/"):
    #                 return target_str.replace("/panda/", "/Robot/panda/")
    #             # Keep other paths as-is
    #             else:
    #                 return target_str
            
    #         if body0_rel:
    #             targets = body0_rel.GetTargets()
    #             new_targets = []
    #             for target in targets:
    #                 target_str = str(target)
    #                 new_target = update_target_path(target_str)
    #                 new_targets.append(new_target)
                
    #             if new_targets != [str(t) for t in targets]:
    #                 body0_rel.SetTargets(new_targets)
    #                 print(f"  Updated body0: {targets} -> {new_targets}")
            
    #         if body1_rel:
    #             targets = body1_rel.GetTargets()
    #             new_targets = []
    #             for target in targets:
    #                 target_str = str(target)
    #                 new_target = update_target_path(target_str)
    #                 new_targets.append(new_target)
                
    #             if new_targets != [str(t) for t in targets]:
    #                 body1_rel.SetTargets(new_targets)
    #                 print(f"  Updated body1: {targets} -> {new_targets}")
    
    # # Remove invalid joints
    # for joint_path in joints_to_remove:
    #     stage.RemovePrim(joint_path)
    #     print(f"  Removed invalid joint: {joint_path}")
    
    # Debug: Print what we actually have
    print("\nDebug: Checking final structure...")
    geometry_count = 0
    material_count = 0
    binding_count = 0
    
    for prim in stage.Traverse():
        print(f"  {prim.GetPath()}")
        if prim.GetTypeName() in ["Mesh", "GeomMesh"]:
            geometry_count += 1
            print(f"  Geometry: {prim.GetPath()}")
        if prim.GetTypeName() in ["Material", "UsdShadeMaterial"]:
            material_count += 1
            print(f"  Material: {prim.GetPath()}")
        if prim.HasAPI(UsdShade.MaterialBindingAPI):
            binding_api = UsdShade.MaterialBindingAPI(prim)
            direct_binding = binding_api.GetDirectBinding()
            if direct_binding:
                material_path = direct_binding.GetMaterialPath()
                if material_path:
                    binding_count += 1
                    print(f"  Binding: {prim.GetPath()} -> {material_path}")
    
    print(f"\nSummary: {geometry_count} geometry prims, {material_count} materials, {binding_count} bindings")
    
    # Save the stage
    stage.Save()
    print(f"Merged robot USD file created successfully at: {output_usd_path}")
    
    return True

def main():
    """Main function."""
    print("Creating Franka + LEAP hand combined robot...")
    
    success = create_franka_leap_robot()
    
    if success:
        print("\nNext steps:")
        print("1. Check the created USD file in Isaac Sim to ensure proper assembly")
        print("2. Adjust joint positions and orientations if needed")
        print("3. Update the robot configuration in franka_leap.py if necessary")
        print("4. Use FRANKA_PANDA_LEAP_CFG in your environment configurations")
    else:
        print("Failed to create combined robot USD file")

if __name__ == "__main__":
    main() 