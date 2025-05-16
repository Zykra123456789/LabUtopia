from omni.isaac.core.controllers import BaseController
from omni.isaac.core.controllers.articulation_controller import ArticulationController
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.manipulators.grippers.gripper import Gripper
from omni.isaac.core.utils.rotations import euler_angles_to_quat

from pxr import Gf
import numpy as np
import typing
from scipy.spatial.transform import Rotation as R
from utils.object_utils import ObjectUtils
class PourController(BaseController):
    """
    A state machine for performing a pouring action with a gripper.

    This controller follows a sequence of states to perform a pouring action.

    The states are:
    - State 0: Start pouring (e.g., tilt the gripper forward).
    - State 1: Pause for a moment.
    - State 2: Stop pouring (e.g., tilt the gripper back to the original position).

    Args:
        name (str): Identifier for the controller.
        cspace_controller (BaseController): A cartesian space controller returning an ArticulationAction type.
        gripper (Gripper): A gripper controller for open/close actions.
        Sim_Container1 (Sim_Container, optional): The first simulation container involved in the pouring action.
        Sim_Container2 (Sim_Container, optional): The second simulation container involved in the pouring action.
        pour_volume (int, optional): The volume of liquid to pour.
        events_dt (list of float, optional): Time duration for each phase. Defaults to [0.004, 0.002, 0.004] divided by speed if not specified.
        speed (float, optional): Speed multiplier for phase durations. Defaults to 1.0.

    Raises:
        Exception: If 'events_dt' is not a list or numpy array.
        Exception: If 'events_dt' length is greater than 3.
    """

    def __init__(
        self,
        name: str,
        cspace_controller: BaseController,
        events_dt: typing.Optional[typing.List[float]] = None,
        speed: float = 1
    ) -> None:
        BaseController.__init__(self, name=name)
        self._event = 0
        self._t = 0
        self._events_dt = events_dt
        if self._events_dt is None:
            self._events_dt = [dt / speed for dt in [0.002, 0.01, 0.009, 0.005, 0.009, 0.5]]
        else:
            if not isinstance(self._events_dt, np.ndarray) and not isinstance(self._events_dt, list):
                raise Exception("events dt need to be list or numpy array")
            elif isinstance(self._events_dt, np.ndarray):
                self._events_dt = self._events_dt.tolist()
            assert len(self._events_dt) == 6, "events dt need have length of 6 or less"
        self._cspace_controller = cspace_controller

        self._pour_default_speed = -120.0 / 180.0 * np.pi
        self.object_utils = ObjectUtils.get_instance()
        self._position_threshold = 0.01

        self._height_range_1 = (0.3, 0.4)
        self._height_range_2 = (0.2, 0.3)
        self._random_height_1 = np.random.uniform(*self._height_range_1)
        self._random_height_2 = np.random.uniform(*self._height_range_2)
        return

    def forward(
        self,
        franka_art_controller: ArticulationController,
        source_size: np.ndarray,
        target_position: np.ndarray,
        current_joint_velocities: np.ndarray,
        gripper_position: np.ndarray,
        source_name: str = None,
        pour_speed: float = None,
        
    ) -> ArticulationAction:
        """
        Execute one step of the controller.

        Args:
            franka_art_controller (ArticulationController): The articulation controller for the Franka robot.
            source_size (np.ndarray): Size of the source object being poured.
            current_joint_velocities (np.ndarray): Current joint velocities of the robot.
            pour_speed (float, optional): Speed for the pouring action. Defaults to None.

        Returns:
            ArticulationAction: Action to be executed by the ArticulationController.
        """

        if pour_speed is None:
            self._pour_speed = self._pour_default_speed
        else:
            self._pour_speed = pour_speed
            
        if  self._event >= len(self._events_dt):
            franka_art_controller.switch_dof_control_mode(dof_index=6, mode="velocity")
            target_joint_velocities = [None] * current_joint_velocities.shape[0]
            return ArticulationAction(joint_velocities=target_joint_velocities)
        
        if self._event == 0:
            target_position[2] += self._random_height_1
            target_joints = self._cspace_controller.forward(
                target_end_effector_position=target_position, 
                target_end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 10])).as_quat()
            )
            
            self._random_height_1 = np.random.uniform(*self._height_range_1)
            
            xy_distance = np.linalg.norm(gripper_position[:2] - target_position[:2])
            if xy_distance < 0.08:
                self._event += 1
                self._t = 0
                return target_joints
                
        elif self._event == 1:
            target_position[2] += self._random_height_2
            target_position[1] -= self.get_pickz_offset(source_name, source_size)
            target_joints = self._cspace_controller.forward(
                target_end_effector_position=target_position, 
                target_end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 0])).as_quat()
            )
            self._random_height_2 = np.random.uniform(*self._height_range_2)
            xy_distance = np.linalg.norm(gripper_position[:2] - target_position[:2])
            if xy_distance < self._position_threshold:
                self._event += 1
                self._t = 0
                return target_joints

        elif self._event == 2:
            franka_art_controller.switch_dof_control_mode(dof_index=6, mode="velocity")
            target_joint_velocities = [None] * current_joint_velocities.shape[0]
            target_joint_velocities[6] = self._pour_speed
            target_joints = ArticulationAction(joint_velocities=target_joint_velocities)
        elif self._event == 3:
            franka_art_controller.switch_dof_control_mode(dof_index=6, mode="velocity")
            target_joint_velocities = [None] * current_joint_velocities.shape[0]
            target_joint_velocities[6] = 0
            target_joints = ArticulationAction(joint_velocities=target_joint_velocities)
        elif self._event == 4:
            franka_art_controller.switch_dof_control_mode(dof_index=6, mode="velocity")
            target_joint_velocities = [None] * current_joint_velocities.shape[0]
            target_joint_velocities[6] = -self._pour_speed
            target_joints = ArticulationAction(joint_velocities=target_joint_velocities)
        elif self._event == 5:
            franka_art_controller.switch_dof_control_mode(dof_index=6, mode="velocity")
            target_joint_velocities = [None] * current_joint_velocities.shape[0]
            target_joint_velocities[6] = 0
            target_joints = ArticulationAction(joint_velocities=target_joint_velocities)

        self._t += self._events_dt[self._event]
        if self._t >= 1.0:
            self._event += 1
            self._t = 0

        return target_joints

    def reset(self, events_dt: typing.Optional[typing.List[float]] = None) -> None:
        """
        Reset the state machine to start from the first phase.

        Args:
            events_dt (list of float, optional): Time duration for each phase. Defaults to None.

        Raises:
            Exception: If 'events_dt' is not a list or numpy array.
            Exception: If 'events_dt' length is greater than 3.
        """
        BaseController.reset(self)
        self._cspace_controller.reset()
        self._event = 0
        self._t = 0
        self._start = True
        if events_dt is not None:
            self._events_dt = events_dt
            if not isinstance(self._events_dt, np.ndarray) and not isinstance(self._events_dt, list):
                raise Exception("events dt need to be list or numpy array")
            elif isinstance(self._events_dt, np.ndarray):
                self._events_dt = self._events_dt.tolist()
            if len(self._events_dt) > 3:
                raise Exception("events dt need have length of 3 or less")

        self._random_height_1 = np.random.uniform(*self._height_range_1)
        self._random_height_2 = np.random.uniform(*self._height_range_2)
        return

    def is_done(self) -> bool:
        """
        Check if the state machine has reached the last phase.

        Returns:
            bool: True if the last phase is reached, False otherwise.
        """
        return self._event >= len(self._events_dt)
    
    def get_pickz_offset(self, item_name, source_size):
        if item_name is None:
            return source_size[2] / 2
            
        offset = {
            "conical_bottle03": 0.06,
            "conical_bottle04": 0.08,
            "beaker2": 0.02,
            "graduated_cylinder_01": 0.0,
            "graduated_cylinder_02": 0.0,
            "graduated_cylinder_03": 0.0,
            "graduated_cylinder_04": 0.0,
            "volume_flask": 0.05,
        }
        
        for key in offset:
            if key in item_name.lower():
                return source_size[2] / 2 - offset[key]
        
        return source_size[2] / 2