from omni.isaac.core.controllers import BaseController
from omni.isaac.core.controllers.articulation_controller import ArticulationController
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.manipulators.grippers.gripper import Gripper
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.isaac.core.utils.stage import get_stage_units, get_current_stage

from pxr import Gf
import numpy as np
import typing
from scipy.spatial.transform import Rotation as R

class ShakeController(BaseController):
    

    def __init__(
        self,
        name: str,
        cspace_controller: BaseController,
        gripper: Gripper,
        events_dt: typing.Optional[typing.List[float]] = None,
        shake_distance: float = 0.1   
    ) -> None:
        BaseController.__init__(self, name=name)  
        self._forward_start = False  
        self._event = 0  
        self._t = 0  
        self._events_dt = events_dt  
        if self._events_dt is None:  
            
            self._events_dt = [0.02, 0.018, 0.018, 0.018, 0.018, 0.018, 0.018, 0.018, 0.018, 0.018]
        else:  
            if not isinstance(self._events_dt, (np.ndarray, list)):  
                raise Exception(" NumPy ")
            elif isinstance(self._events_dt, np.ndarray):  
                self._events_dt = self._events_dt.tolist()  
            if len(self._events_dt) != 10:  
                raise Exception(" 10")
        self._cspace_controller = cspace_controller  
        self._gripper = gripper  
        self._pause = False  
        self._start = True  
        self._shake_distance = shake_distance / get_stage_units() 
        self._initial_position = np.array([0.25, 0, 1.0])  
        return

    def forward(
        self,
        franka_art_controller: ArticulationController,
        current_joint_positions: np.ndarray,
        current_joint_velocities: np.ndarray,
        end_effector_orientation: typing.Optional[np.ndarray] = None,
    ) -> ArticulationAction:
        
        if end_effector_orientation is None:
            end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi, 0]))

        if self._forward_start is False:  
            self._iter_added = False  
            self._forward_start = True  
            target_joint_positions = [None] * current_joint_positions.shape[0]
            return ArticulationAction(joint_positions=target_joint_positions)
        
        if self._pause or self._event >= len(self._events_dt):  
            target_joint_positions = [None] * current_joint_positions.shape[0]
            return ArticulationAction(joint_positions=target_joint_positions)
            
        if self._event == 0 and self._start:  
            self._start = False  
            self._initial_orientation = self._gripper.get_world_pose()[1]  
        
        
        if self._event == 0:
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=self._initial_position,
                target_end_effector_orientation=end_effector_orientation
            )

        
        elif self._event == 1:
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=self._initial_position,
                target_end_effector_orientation=end_effector_orientation
            )

        
        elif self._event == 2:
            target_position = self._initial_position + np.array([0, -self._shake_distance, 0])
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=target_position,
                target_end_effector_orientation=end_effector_orientation
            )

        
        elif self._event == 3:
            target_position = self._initial_position + np.array([0, self._shake_distance, 0])
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=target_position,
                target_end_effector_orientation=end_effector_orientation
            )

        
        elif self._event == 4:
            target_position = self._initial_position + np.array([0, -self._shake_distance, 0])
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=target_position,
                target_end_effector_orientation=end_effector_orientation
            )

        
        elif self._event == 5:
            target_position = self._initial_position + np.array([0, self._shake_distance, 0])
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=target_position,
                target_end_effector_orientation=end_effector_orientation
            )

        
        elif self._event == 6:
            target_position = self._initial_position + np.array([0, -self._shake_distance, 0])
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=target_position,
                target_end_effector_orientation=end_effector_orientation
            )

        
        elif self._event == 7:
            target_position = self._initial_position + np.array([0, self._shake_distance, 0])
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=target_position,
                target_end_effector_orientation=end_effector_orientation
            )

        
        elif self._event == 8:
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=self._initial_position,
                target_end_effector_orientation=end_effector_orientation
            )

        
        elif self._event == 9:
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=self._initial_position,
                target_end_effector_orientation=end_effector_orientation
            )

        
        self._t += self._events_dt[self._event]
        if self._t >= 1.0:
            self._event += 1
            self._t = 0

        return target_joint_positions  

    def reset(self, events_dt: typing.Optional[typing.List[float]] = None) -> None:
        
        BaseController.reset(self)  
        self._cspace_controller.reset()  
        self._event = 0  
        self._t = 0  
        self._pause = False  
        self._start = True  
        if events_dt is not None:  
            self._events_dt = events_dt
            if not isinstance(self._events_dt, (np.ndarray, list)):
                raise Exception(" NumPy ")
            elif isinstance(self._events_dt, np.ndarray):
                self._events_dt = self._events_dt.tolist()
            if len(self._events_dt) != 10:  
                raise Exception(" 10")
        return

    def is_done(self) -> bool:
        
        return self._event >= len(self._events_dt)  

    def is_shaking(self) -> bool:
        
        return 2 <= self._event <= 7

    def pause(self) -> None:
        
        self._pause = True
        return

    def resume(self) -> None:
        
        self._pause = False
        return