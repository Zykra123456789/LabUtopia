from omni.isaac.core.controllers import BaseController
from omni.isaac.core.utils.stage import get_stage_units
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.rotations import euler_angles_to_quat
import numpy as np
import typing
from scipy.spatial.transform import Rotation as R

class StirController(BaseController):
    """
    A state machine controller for performing a stirring action with a robotic arm.

    This controller follows a sequence of states to perform a stirring action:
    - State 0: Lift the glass rod.
    - State 1: Move the glass rod above the beaker.
    - State 2: Lower the glass rod into the beaker.
    - State 3: Perform stirring motion.

    Args:
        name (str): Identifier for the controller.
        cspace_controller (BaseController): A cartesian space controller returning an ArticulationAction type.
        gripper (Gripper, optional): A gripper controller for open/close actions.
        events_dt (list of float, optional): Time duration for each phase. Defaults to [lift, move, lower, stir] durations.
        speed (float, optional): Speed multiplier for phase durations. Defaults to 1.0.

    Raises:
        Exception: If 'events_dt' is not a list or numpy array.
        Exception: If 'events_dt' length is greater than 4.
    """

    def __init__(
        self,
        name: str,
        cspace_controller: BaseController,
        events_dt: typing.Optional[typing.List[float]] = None,
        speed: float = 1.0,
        stir_radius: float = 0.009,
        stir_speed: float = 3.0,
    ) -> None:
        super().__init__(name=name)
        self._forward_start = False
        self._event = 0
        self._t = 0.0
        self._events_dt = events_dt
        self._speed = speed

        if self._events_dt is None:
            # Default durations for [lift, move, lower, stir] in seconds
            self._events_dt = [1.0 / speed, 4 / speed, 2.0 / speed, 5.0 / speed, 2.0 / speed]
        else:
            if not isinstance(self._events_dt, (list, np.ndarray)):
                raise Exception("events_dt needs to be a list or numpy array")
            if isinstance(self._events_dt, np.ndarray):
                self._events_dt = self._events_dt.tolist()
            if len(self._events_dt) > 4:
                raise Exception("events_dt needs to have a length of 4 or less")

        self._cspace_controller = cspace_controller
        self._pause = False
        self._start = True
        self._stir_radius = stir_radius / get_stage_units()
        self._stir_speed = stir_speed / get_stage_units()
        self._current_stir_angle = 0.0
        self._total_stir_time = 0.0
        return

    def is_paused(self) -> bool:
        """
        Check if the state machine is paused.

        Returns:
            bool: True if paused, False otherwise.
        """
        return self._pause

    def forward(
        self,
        center_position: np.ndarray,
        current_joint_positions: np.ndarray,
        end_effector_orientation: typing.Optional[np.ndarray] = None,
    ) -> ArticulationAction:
        """
        Execute one step of the controller based on the current state.

        Args:
            center_position (np.ndarray): Reference position for movement.
            current_joint_positions (np.ndarray): Current joint positions of the robot.
            end_effector_orientation (np.ndarray, optional): Orientation of the end effector. Defaults to None.

        Returns:
            ArticulationAction: Action to be executed by the ArticulationController.
        """
        if end_effector_orientation is None:
            end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi, 0]))

        if not self._forward_start:
            self._forward_start = True
            self._iter_added = False
            self._pause = False
            self._start = True
            self.reset()
        
        if self._pause or self._event >= len(self._events_dt):
            # If paused or all events completed, hold position
            target_joint_positions = [None] * current_joint_positions.shape[0]
            return ArticulationAction(joint_positions=target_joint_positions)

        if self._event == 0 and self._start:
            # Initialize Lift State
            self._start = False

        if self._event == 0:
            # Lift the glass rod
            center_position[2] += 0.3 / get_stage_units()
            target_joints = self._cspace_controller.forward(
                target_end_effector_position=center_position,
                target_end_effector_orientation=end_effector_orientation
            )

        elif self._event == 1:
            center_position[2] += 0.3 / get_stage_units()
            # Move the glass rod above the beaker
            target_joints = self._cspace_controller.forward(
                target_end_effector_position=center_position,
                target_end_effector_orientation=end_effector_orientation
            )

        elif self._event == 2:
            center_position[2] += 0.12 / get_stage_units()
            # Lower the glass rod into the beaker
            target_joints = self._cspace_controller.forward(
                target_end_effector_position=center_position,
                target_end_effector_orientation=end_effector_orientation
            )

        elif self._event == 3:
            # Stirring State
            angle_increment = self._stir_speed * 0.01  # Assuming forward is called every 0.01s
            self._current_stir_angle += angle_increment
            self._total_stir_time += 0.01

            # Calculate circular motion around the center_position
            x_offset = self._stir_radius * np.cos(self._current_stir_angle)
            y_offset = self._stir_radius * np.sin(self._current_stir_angle)
            position_target = center_position + np.array([x_offset, y_offset, 0.1 / get_stage_units()])  # Assuming stirring at fixed height

            target_joints = self._cspace_controller.forward(
                target_end_effector_position=position_target,
                target_end_effector_orientation=end_effector_orientation
            )
        elif self._event == 4:
            center_position[2] += 0.2 / get_stage_units()
            # Lower the glass rod into the beaker
            target_joints = self._cspace_controller.forward(
                target_end_effector_position=center_position,
                target_end_effector_orientation=end_effector_orientation
            )
        else:
            # Undefined state, hold position
            target_joints = [None] * current_joint_positions.shape[0]

        # print(self._event, self._events_dt[self._event])
        # Increment time and check for state transition
        self._t += 0.01  # Assuming forward is called every 0.01s
        if self._t >= self._events_dt[self._event]:
            self._event += 1
            self._t = 0.0

        return target_joints

    def reset(self, events_dt: typing.Optional[typing.List[float]] = None) -> None:
        """
        Reset the state machine to start from the first phase.

        Args:
            events_dt (list of float, optional): Time duration for each phase. Defaults to the initial durations.

        Raises:
            Exception: If 'events_dt' is not a list or numpy array.
            Exception: If 'events_dt' length is greater than 4.
        """
        super().reset()
        self._cspace_controller.reset()
        self._event = 0
        self._t = 0.0
        self._pause = False
        self._start = True
        self._current_stir_angle = 0.0
        self._total_stir_time = 0.0

        if events_dt is not None:
            self._events_dt = events_dt
            if not isinstance(self._events_dt, (list, np.ndarray)):
                raise Exception("events_dt needs to be a list or numpy array")
            if isinstance(self._events_dt, np.ndarray):
                self._events_dt = self._events_dt.tolist()
            if len(self._events_dt) > 4:
                raise Exception("events_dt needs to have a length of 4 or less")

        return

    def is_done(self) -> bool:
        """
        Check if the state machine has completed all phases.

        Returns:
            bool: True if all phases are completed, False otherwise.
        """
        return self._event >= len(self._events_dt)

    def pause(self) -> None:
        """
        Pause the state machine's time and phase.
        """
        self._pause = True
        return

    def resume(self) -> None:
        """
        Resume the state machine's time and phase.
        """
        self._pause = False
        return