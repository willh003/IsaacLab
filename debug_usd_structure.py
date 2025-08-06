#!/usr/bin/env python3
"""
Debug script to examine USD structure and understand visibility issues.
"""

from isaaclab.app import AppLauncher

# create new namespace with headless=True for the launcher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Gf, Sdf, UsdShade
import os
from pathlib import Path

def debug_usd_structure():
    """Debug the USD structure to understand visibility issues."""
    
    # Import the assets utility to get the correct paths
    from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
    franka_usd_path = f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd"
    leap_hand_usd_path = "/home/will/IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/leap_hand_v1_right/leap_hand_right.usd"
    
    # Check the generated file
    output_usd_path = Path.home() / "IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/franka_leap_combined/franka_leap_robot.usd"
    
    print("🔍 DEBUGGING USD STRUCTURE")
    print("=" * 50)
    
    # Check original Franka
    print(f"\n📁 ORIGINAL FRANKA: {franka_usd_path}")
    if os.path.exists(franka_usd_path):
        franka_stage = Usd.Stage.Open(franka_usd_path)
        if franka_stage:
            print("  Root prims:")
            for prim in franka_stage.GetPseudoRoot().GetChildren():
                print(f"    {prim.GetPath()}")
            
            # Check for geometry
            print("  Geometry prims:")
            for prim in franka_stage.Traverse():
                if prim.GetTypeName() in ["Mesh", "GeomMesh"]:
                    print(f"    ✓ {prim.GetPath()}")
            
            # Check for materials
            print("  Material prims:")
            for prim in franka_stage.Traverse():
                if prim.GetTypeName() in ["Material", "UsdShadeMaterial"]:
                    print(f"    ✓ {prim.GetPath()}")
            
            # Check for material bindings
            print("  Material bindings:")
            for prim in franka_stage.Traverse():
                if prim.HasAPI(UsdShade.MaterialBindingAPI):
                    binding_api = UsdShade.MaterialBindingAPI(prim)
                    direct_binding = binding_api.GetDirectBinding()
                    if direct_binding:
                        material_path = direct_binding.GetMaterialPath()
                        if material_path:
                            print(f"    {prim.GetPath()} -> {material_path}")
    
    # Check original LEAP hand
    print(f"\n📁 ORIGINAL LEAP HAND: {leap_hand_usd_path}")
    if os.path.exists(leap_hand_usd_path):
        leap_stage = Usd.Stage.Open(leap_hand_usd_path)
        if leap_stage:
            print("  Root prims:")
            for prim in leap_stage.GetPseudoRoot().GetChildren():
                print(f"    {prim.GetPath()}")
            
            # Check for geometry
            print("  Geometry prims:")
            for prim in leap_stage.Traverse():
                if prim.GetTypeName() in ["Mesh", "GeomMesh"]:
                    print(f"    ✓ {prim.GetPath()}")
            
            # Check for materials
            print("  Material prims:")
            for prim in leap_stage.Traverse():
                if prim.GetTypeName() in ["Material", "UsdShadeMaterial"]:
                    print(f"    ✓ {prim.GetPath()}")
    
    # Check generated file
    print(f"\n📁 GENERATED FILE: {output_usd_path}")
    if os.path.exists(output_usd_path):
        generated_stage = Usd.Stage.Open(str(output_usd_path))
        if generated_stage:
            print("  Root prims:")
            for prim in generated_stage.GetPseudoRoot().GetChildren():
                print(f"    {prim.GetPath()}")
            
            # Check for geometry
            print("  Geometry prims:")
            geometry_found = False
            for prim in generated_stage.Traverse():
                if prim.GetTypeName() in ["Mesh", "GeomMesh"]:
                    print(f"    ✓ {prim.GetPath()}")
                    geometry_found = True
            
            if not geometry_found:
                print("    ❌ NO GEOMETRY FOUND!")
            
            # Check for materials
            print("  Material prims:")
            materials_found = False
            for prim in generated_stage.Traverse():
                if prim.GetTypeName() in ["Material", "UsdShadeMaterial"]:
                    print(f"    ✓ {prim.GetPath()}")
                    materials_found = True
            
            if not materials_found:
                print("    ❌ NO MATERIALS FOUND!")
            
            # Check for material bindings
            print("  Material bindings:")
            bindings_found = False
            for prim in generated_stage.Traverse():
                if prim.HasAPI(UsdShade.MaterialBindingAPI):
                    binding_api = UsdShade.MaterialBindingAPI(prim)
                    direct_binding = binding_api.GetDirectBinding()
                    if direct_binding:
                        material_path = direct_binding.GetMaterialPath()
                        if material_path:
                            print(f"    {prim.GetPath()} -> {material_path}")
                            bindings_found = True
            
            if not bindings_found:
                print("    ❌ NO MATERIAL BINDINGS FOUND!")
            
            # Check for visibility attributes
            print("  Visibility attributes:")
            for prim in generated_stage.Traverse():
                if prim.GetTypeName() in ["Mesh", "GeomMesh"]:
                    visibility_attr = prim.GetAttribute("visibility")
                    if visibility_attr and visibility_attr.HasValue():
                        visibility = visibility_attr.Get()
                        print(f"    {prim.GetPath()}: visibility = {visibility}")
                    else:
                        print(f"    {prim.GetPath()}: no visibility attribute")
    else:
        print("    ❌ Generated file not found!")

if __name__ == "__main__":
    debug_usd_structure()