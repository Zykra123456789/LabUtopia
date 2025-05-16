import os
import numpy as np
from typing import List, Optional

import omni.isaac.motion_generation as mg
from omni.isaac.core.utils.extensions import get_extension_path_from_name
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.franka.controllers.rmpflow_controller import RMPFlowController


class FrankaTrajectoryController(RMPFlowController):
    """Trajectory controller for Franka robotic arm, supporting continuous trajectory generation and execution."""

    def __init__(
        self, 
        name: str, 
        robot_articulation: Articulation, 
        physics_dt: float = 1.0/60.0
    ) -> None:
        """Initialize the Franka trajectory controller.

        Args:
            name (str): Name of the controller.
            robot_articulation (Articulation): The articulated robot object.
            physics_dt (float, optional): Physics simulation time step. Defaults to 1.0/60.0.
        """
        super().__init__(name=name, robot_articulation=robot_articulation, physics_dt=physics_dt)
        
        # Retrieve motion generation configuration path
        mg_extension_path = get_extension_path_from_name("omni.isaac.motion_generation")
        rmp_config_dir = os.path.join(mg_extension_path, "motion_policy_configs")
        
        # Initialize trajectory generator
        self._c_space_trajectory_generator = mg.LulaCSpaceTrajectoryGenerator(
            robot_description_path=rmp_config_dir + "/franka/rmpflow/robot_descriptor.yaml",
            urdf_path=rmp_config_dir + "/franka/lula_franka_gen.urdf"
        )
        
        # Initialize kinematics solver
        self._kinematics_solver = mg.LulaKinematicsSolver(
            robot_description_path=rmp_config_dir + "/franka/rmpflow/robot_descriptor.yaml",
            urdf_path=rmp_config_dir + "/franka/lula_franka_gen.urdf"
        )
        
        self._action_sequence = []
        self._action_sequence_index = 0
        self._end_effector_name = "panda_hand"
        self._physics_dt = physics_dt

    def generate_trajectory(
        self, 
        waypoints: np.ndarray,
        timestamps: Optional[np.ndarray] = None
    ) -> None:
        """Generate a joint space trajectory.

        Args:
            waypoints (np.ndarray): Array of shape (N, 8) containing N waypoints with joint angles and gripper positions.
            timestamps (Optional[np.ndarray], optional): Array of timestamps for waypoints. Defaults to None.
        """
        # Separate joint angles and gripper positions
        joint_waypoints = waypoints[:, :7]  # First 7 joints
        self.gripper_positions = waypoints[:, 7]  # Last column is gripper position
        
        # Retrieve joint angle limits
        joint_limits = self._c_space_trajectory_generator.get_c_space_position_limits()
        joint_min, joint_max = joint_limits[0], joint_limits[1]
        
        # Constrain joint angles within valid range
        joint_waypoints = np.clip(joint_waypoints, joint_min, joint_max)
        
        # Create action sequence
        self._action_sequence = []
        for i in range(len(joint_waypoints)):
            action = ArticulationAction(
                joint_positions=np.concatenate([joint_waypoints[i], [self.gripper_positions[i], self.gripper_positions[i]]]),
                joint_velocities=None,
                joint_efforts=None
            )
            self._action_sequence.append(action)
            
        # Initialize gripper indices
        total_actions = len(self._action_sequence)
        self.gripper_indices = np.linspace(0, len(self.gripper_positions)-1, total_actions, dtype=int)
        self._action_sequence_index = 0

    def get_next_action(self) -> Optional[ArticulationAction]:
        """Retrieve the next action in the sequence.

        Returns:
            Optional[ArticulationAction]: The next action to execute, or None if the sequence is complete.
        """
        if not self._action_sequence or self._action_sequence_index >= len(self._action_sequence):
            return None
            
        action = self._action_sequence[self._action_sequence_index]
        
        self._action_sequence_index += 1
        return action

    def is_trajectory_complete(self) -> bool:
        """Check if the trajectory execution is complete.

        Returns:
            bool: True if the trajectory is complete, False otherwise.
        """
        return len(self._action_sequence) == 0 or self._action_sequence_index >= len(self._action_sequence)

    def reset(self) -> None:
        """Reset the controller state."""
        super().reset()
        self._action_sequence = []
        self._action_sequence_index = 0
        self.gripper_positions = []
        self.gripper_indices = []