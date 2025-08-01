#!/usr/bin/env python3
"""
Custom spawner that applies physics APIs to USD files automatically.
This allows using USD files that don't have physics APIs pre-applied.
"""

from pxr import Usd, UsdPhysics, PhysxSchema
import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.from_files import UsdFileCfg
from isaaclab.utils import configclass
import isaaclab.sim.utils as sim_utils_internal

@configclass
class PhysicsUsdFileCfg(UsdFileCfg):
    """USD file spawner that automatically applies physics APIs."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Override the func to use our custom spawner
        self.func = spawn_physics_usd_file

def spawn_physics_usd_file(prim_path: str, cfg: PhysicsUsdFileCfg, **kwargs):
    """Spawn a USD file and automatically apply physics APIs."""
    
    # First, spawn the USD file normally using the correct function
    from isaaclab.sim.spawners.from_files.from_files import spawn_from_usd
    prim = spawn_from_usd(prim_path, cfg, **kwargs)
    
    # Apply physics APIs to the spawned prim and all its clones
    # Get all matching prims (including clones)
    all_prims = sim_utils_internal.find_matching_prims(prim_path)
    
    for prim_instance in all_prims:
        # Apply physics APIs to each instance
        if not prim_instance.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI.Apply(prim_instance)
        
        if not prim_instance.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
            PhysxSchema.PhysxRigidBodyAPI.Apply(prim_instance)
        
        if not prim_instance.HasAPI(UsdPhysics.MassAPI):
            UsdPhysics.MassAPI.Apply(prim_instance)
        
        if not prim_instance.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI.Apply(prim_instance)
    
    return prim 