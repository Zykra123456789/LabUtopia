from controllers.robot_controllers.grapper_manager import Gripper
from omni.isaac.core.controllers import BaseController
from omni.isaac.core.utils.stage import get_stage_units
from omni.isaac.core.utils.types import ArticulationAction
import numpy as np
import typing
from omni.isaac.core.utils.rotations import euler_angles_to_quat

class CloseController(BaseController):
    """
    Drawer/Cabinet door closing controller.
    
    Controls the state machine for closing drawers or cabinet doors.

    States include:
    - State 0: Move to the initial position
    - State 1: Wait for completion

    Args:
        name (str): Controller identifier
        events_dt (list of float, optional): Duration of each state. Defaults to [0.4, 0.1]
        speed (float, optional): Action speed multiplier. Defaults to 1.0
    """

    def __init__(
        self,
        name: str,
        cspace_controller: BaseController,
        gripper: Gripper,
        events_dt: typing.Optional[typing.List[float]] = None,
        speed: float = 1.0,
    ) -> None:
        BaseController.__init__(self, name=name)
        self._event = 0
        self._t = 0
        self._cspace_controller = cspace_controller
        self._gripper = gripper
        if events_dt is None:
            self._events_dt = [0.01 / speed, 0.01 / speed]
        else:
            self._events_dt = events_dt
        self._pause = False

    def forward(
        self,
        handle_position: np.ndarray,
        current_joint_positions: np.ndarray,
        furniture_type: str = "drawer",
        end_effector_orientation: typing.Optional[np.ndarray] = None
    ) -> ArticulationAction:
        """
        Perform one step of the closing action control.

        Args:
            handle_position (np.ndarray): Handle position
            current_joint_positions (np.ndarray): Current joint positions
            furniture_type (str): Furniture type ("drawer" for drawers, "door" for cabinet doors)

        Returns:
            ArticulationAction: Control action
        """
        if self._pause or self.is_done():
            target_joint_positions = [None] * current_joint_positions.shape[0]
            return ArticulationAction(joint_positions=target_joint_positions)

        if end_effector_orientation is None:
            end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi, 0]))
            
        target_joint_positions = self._execute_phase(
            handle_position, 
            end_effector_orientation, 
            current_joint_positions,
        )
        
        self._t += self._events_dt[self._event]
        if self._t >= 1.0:
            self._event += 1
            self._t = 0

        return target_joint_positions
    
    def _execute_phase(self, handle_position, end_effector_orientation, current_joint_positions):
        """Execute the current phase of the closing action"""
        if self._event == 0:
            handle_position[0] -= 0.04
            target_joint_positions = self._cspace_controller.forward(
                    target_end_effector_position=handle_position, 
                    target_end_effector_orientation=end_effector_orientation
                )
        elif self._event == 1:
            target_joint_positions = [None] * current_joint_positions.shape[0]
            gripper_distance = 0.00
            target_joint_positions[7] = gripper_distance
            target_joint_positions[8] = gripper_distance
            target_joint_positions = ArticulationAction(joint_positions=target_joint_positions)
            self.target_position = handle_position
            self.target_position[0] += 0.2
        elif self._event == 2 or self._event == 3:
            # Check if we're close enough to target position
            if np.linalg.norm(handle_position - self.target_position) <= 0.01:
                self._event = 4  # Skip to next state
                self._t = 0
                target_joint_positions = [None] * current_joint_positions.shape[0]
                gripper_distance = 0.04
                target_joint_positions[7] = gripper_distance
                target_joint_positions[8] = gripper_distance
                target_joint_positions = ArticulationAction(joint_positions=target_joint_positions)
            else:
                handle_position[0] += 0.05
                target_joint_positions = self._cspace_controller.forward(
                    target_end_effector_position=handle_position, 
                    target_end_effector_orientation=end_effector_orientation
                )
        elif self._event == 4:
            target_joint_positions = [None] * current_joint_positions.shape[0]
            gripper_distance = 0.04
            target_joint_positions[7] = gripper_distance
            target_joint_positions[8] = gripper_distance
            target_joint_positions = ArticulationAction(joint_positions=target_joint_positions)
        else:
            handle_position[0] -= 0.1
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=handle_position, 
                target_end_effector_orientation=end_effector_orientation
            )
            
        return target_joint_positions
    
    def reset(self) -> None:
        """Reset the controller state"""
        BaseController.reset(self)
        self._event = 0
        self._t = 0
        self._pause = False

    def is_done(self) -> bool:
        """Check if the controller has completed all states"""
        return self._event >= len(self._events_dt)

    def pause(self) -> None:
        """Pause the controller"""
        self._pause = True

    def resume(self) -> None:
        """Resume the controller"""
        self._pause = False
