#!/usr/bin/env python3
"""
Script to examine the LEAP hand USD structure and understand its references.
"""

from isaaclab.app import AppLauncher

# create new namespace with headless=True for the launcher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Gf, Sdf, UsdShade
import os
from pathlib import Path

def examine_leap_structure():
    """Examine the LEAP hand USD structure to understand its references."""
    
    leap_hand_usd_path = "/home/will/IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/leap_hand_v1_right/leap_hand_right.usd"
    config_dir = "/home/will/IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/leap_hand_v1_right/configuration"
    
    print("🔍 EXAMINING LEAP HAND STRUCTURE")
    print("=" * 50)
    
    # Check the main LEAP hand file
    print(f"\n📁 MAIN LEAP HAND FILE: {leap_hand_usd_path}")
    if os.path.exists(leap_hand_usd_path):
        leap_stage = Usd.Stage.Open(leap_hand_usd_path)
        if leap_stage:
            print("  Root prims:")
            for prim in leap_stage.GetPseudoRoot().GetChildren():
                print(f"    {prim.GetPath()}")
            
            # Check for sublayers
            print("  Sublayers:")
            for layer in leap_stage.GetLayerStack():
                print(f"    {layer.identifier}")
            
            # Check for geometry
            print("  Geometry prims:")
            geometry_found = False
            for prim in leap_stage.Traverse():
                if prim.GetTypeName() in ["Mesh", "GeomMesh"]:
                    print(f"    ✓ {prim.GetPath()}")
                    geometry_found = True
            
            if not geometry_found:
                print("    ❌ NO GEOMETRY FOUND!")
            
            # Check for materials
            print("  Material prims:")
            for prim in leap_stage.Traverse():
                if prim.GetTypeName() in ["Material", "UsdShadeMaterial"]:
                    print(f"    ✓ {prim.GetPath()}")
    
    # Check configuration files
    print(f"\n📁 CONFIGURATION FILES: {config_dir}")
    if os.path.exists(config_dir):
        for file in os.listdir(config_dir):
            if file.endswith('.usd'):
                config_file_path = os.path.join(config_dir, file)
                print(f"\n  File: {file}")
                
                config_stage = Usd.Stage.Open(config_file_path)
                if config_stage:
                    print("    Root prims:")
                    for prim in config_stage.GetPseudoRoot().GetChildren():
                        print(f"      {prim.GetPath()}")
                    
                    # Check for geometry
                    print("    Geometry prims:")
                    geometry_found = False
                    for prim in config_stage.Traverse():
                        if prim.GetTypeName() in ["Mesh", "GeomMesh"]:
                            print(f"      ✓ {prim.GetPath()}")
                            geometry_found = True
                    
                    if not geometry_found:
                        print("      ❌ NO GEOMETRY FOUND!")
                    
                    # Check for materials
                    print("    Material prims:")
                    for prim in config_stage.Traverse():
                        if prim.GetTypeName() in ["Material", "UsdShadeMaterial"]:
                            print(f"      ✓ {prim.GetPath()}")

if __name__ == "__main__":
    examine_leap_structure() 