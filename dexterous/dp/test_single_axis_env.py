#!/usr/bin/env python3

"""
Simple test script to validate the SingleAxis environment configuration.
This script doesn't need to actually run the environment, just validates imports and config structure.
"""

import sys
sys.path.append('/home/will/IsaacLab/source/isaaclab_tasks')

def test_single_axis_imports():
    """Test that all imports work correctly."""
    try:
        # Test command class import
        from isaaclab_tasks.manager_based.manipulation.inhand.mdp.commands.single_axis_command import SingleAxisCommand
        print("✓ SingleAxisCommand imported successfully")

        # Test configuration import
        from isaaclab_tasks.manager_based.manipulation.inhand.mdp.commands.commands_cfg import SingleAxisCommandCfg
        print("✓ SingleAxisCommandCfg imported successfully")

        # Test environment config import
        from isaaclab_tasks.manager_based.manipulation.inhand.config.allegro_hand.allegro_env_cfg import AllegroCubeSingleAxisEnvCfg
        print("✓ AllegroCubeSingleAxisEnvCfg imported successfully")

        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_configuration_structure():
    """Test that the configuration structure is valid."""
    try:
        from isaaclab_tasks.manager_based.manipulation.inhand.mdp.commands.commands_cfg import SingleAxisCommandCfg
        from isaaclab_tasks.manager_based.manipulation.inhand.mdp.commands.single_axis_command import SingleAxisCommand

        # Create a config instance
        config = SingleAxisCommandCfg(
            asset_name="test_object",
            init_pos_offset=(0.0, 0.0, 0.0),
            update_goal_on_success=True,
            orientation_success_threshold=0.1,
            make_quat_unique=False,
            angle_range=1.57  # π/2 radians
        )

        # Verify the config has the expected attributes
        assert hasattr(config, 'class_type'), "Missing class_type attribute"
        assert config.class_type == SingleAxisCommand, "Incorrect class_type"
        assert hasattr(config, 'angle_range'), "Missing angle_range attribute"
        assert config.angle_range == 1.57, "Incorrect angle_range value"

        print("✓ Configuration structure is valid")
        return True

    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def test_goal_generation_logic():
    """Test the goal generation logic without needing Isaac Sim."""
    try:
        import torch
        import sys
        sys.path.append('/home/will/IsaacLab/source/isaaclab')
        import isaaclab.utils.math as math_utils

        # Simulate the goal generation logic
        num_envs = 10
        device = 'cpu'
        angle_range = 3.14159265359  # π radians

        # Randomly choose which axis to rotate around for each environment
        selected_axes = torch.randint(0, 3, (num_envs,), device=device)

        # Sample random angles within the specified range
        angles = (2 * torch.rand(num_envs, device=device) - 1) * angle_range

        # Define unit vectors for X, Y, Z axes
        x_unit = torch.tensor([1.0, 0.0, 0.0], device=device).unsqueeze(0).expand(num_envs, -1)
        y_unit = torch.tensor([0.0, 1.0, 0.0], device=device).unsqueeze(0).expand(num_envs, -1)
        z_unit = torch.tensor([0.0, 0.0, 1.0], device=device).unsqueeze(0).expand(num_envs, -1)

        # Initialize result quaternions
        quats = torch.zeros((num_envs, 4), device=device)

        # Process each axis separately
        for axis_idx in range(3):
            mask = selected_axes == axis_idx
            if not mask.any():
                continue

            axis_envs = mask.nonzero(as_tuple=False).squeeze(-1)
            if axis_envs.dim() == 0:
                axis_envs = axis_envs.unsqueeze(0)

            if axis_idx == 0:  # X-axis (roll)
                axis_vector = x_unit[axis_envs]
            elif axis_idx == 1:  # Y-axis (pitch)
                axis_vector = y_unit[axis_envs]
            else:  # Z-axis (yaw)
                axis_vector = z_unit[axis_envs]

            # Generate quaternions for this axis
            quat = math_utils.quat_from_angle_axis(angles[axis_envs], axis_vector)
            quats[axis_envs] = quat

        # Verify we got valid quaternions
        assert quats.shape == (num_envs, 4), f"Expected shape ({num_envs}, 4), got {quats.shape}"

        # Check that quaternions are normalized (approximately)
        norms = torch.norm(quats, dim=1)
        valid_quats = (norms > 0.95) & (norms < 1.05)  # Allow small numerical errors
        assert valid_quats.sum() == num_envs, f"Only {valid_quats.sum()}/{num_envs} quaternions are properly normalized"

        print("✓ Goal generation logic works correctly")
        print(f"  - Generated {num_envs} quaternions")
        print(f"  - Axis distribution: X={torch.sum(selected_axes==0)}, Y={torch.sum(selected_axes==1)}, Z={torch.sum(selected_axes==2)}")
        print(f"  - Angle range: [{angles.min():.2f}, {angles.max():.2f}] radians")
        return True

    except Exception as e:
        print(f"✗ Goal generation test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing SingleAxis environment implementation...")
    print("=" * 50)

    tests = [
        test_single_axis_imports,
        test_configuration_structure,
        test_goal_generation_logic
    ]

    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            print()

    print("=" * 50)
    print(f"Test Results: {passed}/{len(tests)} tests passed")

    if passed == len(tests):
        print("🎉 All tests passed! The SingleAxis environment is ready to use.")
        print("\nYou can now use the environment with:")
        print("  gym.make('Isaac-Repose-Cube-Allegro-SingleAxis-v0')")
    else:
        print("❌ Some tests failed. Please check the implementation.")

    return passed == len(tests)

if __name__ == "__main__":
    main()