from omni.isaac.core.controllers import BaseController
from omni.isaac.core.utils.stage import get_stage_units
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.rotations import euler_angles_to_quat
import numpy as np
import typing
from omni.isaac.manipulators.grippers.gripper import Gripper

class PressController(BaseController):
    
    
    def __init__(
        self,
        name: str,
        cspace_controller: BaseController,
        gripper: Gripper,
        end_effector_initial_height: typing.Optional[float] = None,
        initial_offset: typing.Optional[float] = None,
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

        self._initial_offset = initial_offset if initial_offset is not None else 0.2 / get_stage_units()
        
        
        if events_dt is None:
            self._events_dt = [0.005, 0.01, 0.01]  
        else:
            self._events_dt = events_dt
            if not isinstance(self._events_dt, (np.ndarray, list)):
                raise Exception("events_dt  NumPy ")
            elif isinstance(self._events_dt, np.ndarray):
                self._events_dt = events_dt.tolist()
            if len(self._events_dt) > 3:
                raise Exception("events_dt  3")
        
        self._cspace_controller = cspace_controller  
        self._gripper = gripper  
        self._pause = False  
        self._start = True  

    def is_paused(self) -> bool:
        
        
        if self._start:
            
            self._start = False
            target_joint_positions = [None] * current_joint_positions.shape[0]
            target_joint_positions[7] = 0.04 / get_stage_units()  
            target_joint_positions[8] = 0.04 / get_stage_units()  
            return ArticulationAction(joint_positions=target_joint_positions)
        
        if self._pause or self.is_done():
            
            target_joint_positions = [None] * current_joint_positions.shape[0]
            return ArticulationAction(joint_positions=target_joint_positions)
        
        if end_effector_orientation is None:
            end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi, 0]))
            
        
        
        target_joint_positions = self._execute_phase(
            target_position, 
            end_effector_orientation, 
            current_joint_positions,
            gripper_control
        )
        
        
        self._update_state()
        
        return target_joint_positions
    
    def _execute_phase(self, target_position, end_effector_orientation, current_joint_positions, gripper_control):
        
        if self._event == 0:
            
            target_position[0] -= self._initial_offset  
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=target_position,
                target_end_effector_orientation=end_effector_orientation
            )
        elif self._event == 1:
            
            target_joint_positions = [None] * current_joint_positions.shape[0]
            gripper_distance = 0.0015 / get_stage_units()  
            target_joint_positions[7] = gripper_distance
            target_joint_positions[8] = gripper_distance
            target_joint_positions = ArticulationAction(joint_positions=target_joint_positions)
        elif self._event == 2:
            
            target_position[0]-= 0.040/ get_stage_units() 
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=target_position,
                target_end_effector_orientation=end_effector_orientation
            )
        
        return target_joint_positions
    
    def _update_state(self):
        
        self._t += self._events_dt[self._event]
        if self._t >= 1.0:
            self._event += 1
            self._t = 0
    
    def reset(
        self,
        initial_offset: typing.Optional[float] = None,
        events_dt: typing.Optional[typing.List[float]] = None
    ) -> None:
        
        BaseController.reset(self)
        self._cspace_controller.reset()
        self._event = 0
        self._t = 0
        if initial_offset is not None:
            self._initial_offset = initial_offset
        if events_dt is not None:
            self._events_dt = events_dt
            if not isinstance(self._events_dt, (np.ndarray, list)):
                raise Exception("events_dt  NumPy ")
            elif isinstance(self._events_dt, np.ndarray):
                self._events_dt = events_dt.tolist()
            if len(self._events_dt) > 3:
                raise Exception("events_dt  3")
        self._pause = False
        self._start = True
    
    def is_done(self) -> bool:
        
        return self._event >= len(self._events_dt)
    
    def pause(self) -> None:
        
        self._pause = True
    
    def resume(self) -> None:
        
        self._pause = False
