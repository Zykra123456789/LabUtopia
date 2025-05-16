from omni.isaac.core.controllers import BaseController
from omni.isaac.core.utils.stage import get_stage_units, get_current_stage
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.rotations import euler_angles_to_quat
import numpy as np
import typing
from omni.isaac.manipulators.grippers.gripper import Gripper

class PlaceController(BaseController):
    

    def __init__(
        self,
        name: str,
        cspace_controller: BaseController,
        gripper: Gripper,
        end_effector_initial_height: typing.Optional[float] = None,
        events_dt: typing.Optional[typing.List[float]] = None,
        speed: float = 1.0
    ) -> None:
        BaseController.__init__(self, name=name)
        self._event = 0
        self._t = 0
        self._h1 = end_effector_initial_height
        if self._h1 is None:
            self._h1 = 0.32 / get_stage_units()
        self._h0 = None
        self._events_dt = events_dt
        if events_dt is None:
            self._events_dt = [dt / speed for dt in [0.005, 0.01, 0.08, 0.05, 0.01, 0.1]]
        else:
            if not isinstance(self._events_dt, np.ndarray) and not isinstance(self._events_dt, list):
                raise Exception("events dt  numpy ")
            elif isinstance(self._events_dt, np.ndarray):
                self._events_dt = self._events_dt.tolist()
            if len(self._events_dt) > 10:
                raise Exception("events dt  7")
        self._cspace_controller = cspace_controller
        self._gripper = gripper
        self._pause = False
        self._start = True
        self.target_position = None  
        return

    def is_paused(self) -> bool:
        
        return self._pause

    def get_current_event(self) -> int:
        
        return self._event

    def forward(
        self,
        place_position: np.ndarray,
        current_joint_positions: np.ndarray,
        gripper_control,
        end_effector_offset: typing.Optional[np.ndarray] = None,
        end_effector_orientation: typing.Optional[np.ndarray] = None,
    ) -> ArticulationAction:
        
        if end_effector_offset is None:
            end_effector_offset = np.array([0, 0, 0])
        if self._start:
            self._start = False
            target_joint_positions = [None] * current_joint_positions.shape[0]
            return ArticulationAction(joint_positions=target_joint_positions)
        if self._pause or self.is_done():
            self.pause()
            target_joint_positions = [None] * current_joint_positions.shape[0]
            return ArticulationAction(joint_positions=target_joint_positions)
        
        if end_effector_orientation is None:
            end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi, 0]))
        
        if self._event == 0:
            
            target_position = place_position
            target_position[2] += 0.3 / get_stage_units()
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=target_position,
                target_end_effector_orientation=end_effector_orientation
            )
        elif self._event == 1:
            
            target_position = place_position
            target_position[2] += 0.10 / get_stage_units()
            target_position = place_position
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=target_position,
                target_end_effector_orientation=end_effector_orientation
            )
        elif self._event == 2:
            
            target_joint_positions = ArticulationAction(joint_positions=[None] * current_joint_positions.shape[0])
        elif self._event == 3:
            
            target_joint_positions = self._gripper.forward(action="open")
            self.target_position = place_position
            self.target_position[2] += 0.05 / get_stage_units()
            self.target_position[0] -= 0.1 / get_stage_units()
            gripper_control.release_object()
        elif self._event == 4:
            
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=self.target_position,
                target_end_effector_orientation=end_effector_orientation
            )
        else:
            
            target_joint_positions = ArticulationAction(joint_positions=[None] * current_joint_positions.shape[0])
        
        self._t += self._events_dt[self._event]
        if self._t >= 1.0:
            self._event += 1
            self._t = 0

        return target_joint_positions

    def _get_alpha(self):
        
        if self._event < 5:
            return 0
        elif self._event == 5:
            return self._mix_sin(self._t)
        elif self._event in [6, 7, 8]:
            return 1.0
        elif self._event == 9:
            return 1
        else:
            raise ValueError("")

    def _get_target_hs(self, target_height):
        
        if self._event == 0:
            h = self._h1
        elif self._event == 1:
            a = self._mix_sin(max(0, self._t))
            h = self._combine_convex(self._h1, self._h0, a)
        elif self._event == 3:
            h = self._h0
        elif self._event == 4:
            a = self._mix_sin(max(0, self._t))
            h = self._combine_convex(self._h0, self._h1, a)
        elif self._event == 5:
            h = self._h1
        elif self._event == 6:
            h = self._combine_convex(self._h1, target_height, self._mix_sin(self._t))
        elif self._event == 7:
            h = target_height
        elif self._event == 8:
            h = self._combine_convex(target_height, self._h1, self._mix_sin(self._t))
        elif self._event == 9:
            h = self._h1
        else:
            raise ValueError("")
        return h

    def _mix_sin(self, t):
        
        return 0.5 * (1 - np.cos(t * np.pi))

    def _combine_convex(self, a, b, alpha):
        
        return (1 - alpha) * a + alpha * b

    def reset(
        self,
        end_effector_initial_height: typing.Optional[float] = None,
        events_dt: typing.Optional[typing.List[float]] = None,
    ) -> None:
        
        BaseController.reset(self)
        self._cspace_controller.reset()
        self._event = 0
        self._t = 0
        if end_effector_initial_height is not None:
            self._h1 = end_effector_initial_height
        self._pause = False
        if events_dt is not None:
            self._events_dt = events_dt
            if not isinstance(self._events_dt, np.ndarray) and not isinstance(self._events_dt, list):
                raise Exception("event velocities  numpy ")
            elif isinstance(self._events_dt, np.ndarray):
                self._events_dt = self._events_dt.tolist()
            if len(self._events_dt) > 10:
                raise Exception("events dt  10")
        return

    def is_done(self) -> bool:
        
        return self._event >= len(self._events_dt)

    def pause(self) -> None:
        
        self._pause = True
        return

    def resume(self) -> None:
        
        self._pause = False
        return