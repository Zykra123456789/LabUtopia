from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.franka.controllers.rmpflow_controller import RMPFlowController
from scipy.spatial.transform import Rotation as R
import numpy as np
import os

from .atomic_actions.pick_controller import PickController
from .atomic_actions.shake_controller import ShakeController
from .base_controller import BaseController
from .robot_controllers.grapper_manager import Gripper
from factories.collector_factory import create_collector

class ShakeTaskController(BaseController):
    def __init__(self, cfg, robot, object_utils):

        super().__init__(cfg, robot, object_utils)
        
        
        self.shake_collector = create_collector(
            cfg.collector.type,
            camera_configs=cfg.cameras,
            save_dir=os.path.join(cfg.multi_run.run_dir, "shake_data"),
            max_episodes=cfg.max_episodes,
            compression=cfg.collector.compression
        )
        
        
        self.pick_controller = PickController(
            name="pick_controller",
            cspace_controller=RMPFlowController(
                name="target_follower_controller", 
                robot_articulation=robot
            ),
            gripper=robot.gripper,
            events_dt=[0.016, 0.005, 0.01, 0.2, 0.05, 0.01]
        )
        
        
        self.shake_controller = ShakeController(
            name="shake_controller",
            cspace_controller=RMPFlowController(
                name="target_follower_controller", 
                robot_articulation=robot
            ),
            gripper=robot.gripper,
        )
        
        self.gripper_control = Gripper()
        self._pick_completed = False  
        self._last_pick_joint_data = None  
        
    def reset(self):
        
        super().reset()
        self.pick_controller.reset()
        self.shake_controller.reset()
        self._pick_completed = False
        
    def step(self, state):
        if not self.pick_controller.is_done():
            
            action = self.pick_controller.forward(
                picking_position=state['object_position'],
                current_joint_positions=state['joint_positions'],
                object_size=state['object_size'],
                object_name="beaker",
                gripper_control=self.gripper_control,
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 10])).as_quat(),
            )
            
            if self.pick_controller.is_done():
                self._pick_completed = True
                
            return action, False, False
            
        
        if not self.shake_controller.is_done():            
            action = self.shake_controller.forward(
                franka_art_controller=self.robot.get_articulation_controller(),
                current_joint_positions=self.robot.get_joint_positions(),
                current_joint_velocities=self.robot.get_joint_velocities(),
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 10])).as_quat(),
            )
            
            if 'camera_data' in state:
                
                self.shake_collector.cache_step(
                    camera_images=state['camera_data'],
                    joint_angles=state['joint_positions'][:-1]
                )
                
            return action, False, False
            
        
        final_object_position = self.object_utils.get_object_xform_position(
            object_path=self.cfg.sub_obj_path
        )

        if final_object_position is not None and final_object_position[2] > 0.85:
            
            self.shake_collector.write_cached_data(state['joint_positions'][:-1])
            self._last_success = True
            self.reset_needed = True
            return None, True, True
        else: 
            
            self.shake_collector.clear_cache()
            self._last_success = False
            self.reset_needed = True
            return None, True, False
        
    def close(self):
        
        self.shake_collector.close()
        
    def get_current_collector(self):
        
        return self._current_collector
        
    def is_done(self):
        
        return self._pick_completed and self.shake_controller.is_done()
        
    def is_success(self):
        
        return self._last_success

    def eposide_num(self):
        
        return self.shake_collector.episode_count