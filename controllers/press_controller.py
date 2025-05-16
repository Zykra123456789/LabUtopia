import os
from factories.collector_factory import create_collector
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.franka.controllers.rmpflow_controller import RMPFlowController
from scipy.spatial.transform import Rotation as R
import numpy as np

from .atomic_actions.press_controller import PressController
from .robot_controllers.grapper_manager import Gripper
from .base_controller import BaseController

class PressTaskController(BaseController):
    
    def __init__(self, cfg, robot, object_utils):
        super().__init__(cfg, robot, object_utils)
        self.press_controller = PressController(
            name="press_controller",
            cspace_controller=RMPFlowController(
                name="target_follower_controller",
                robot_articulation=robot
            ),
            gripper=robot.gripper,
        )
        self.gripper_control = Gripper()
        self.data_collector = create_collector(
            cfg.collector.type,
            camera_configs=cfg.cameras,
            save_dir=os.path.join(cfg.multi_run.run_dir, "press_data"),
            max_episodes=cfg.max_episodes,
            compression=cfg.collector.compression
        )
        
    def reset(self):
        
        super().reset()
        self.press_controller.reset()
        
    def step(self, state):
        if not self.press_controller.is_done():
            
            action = self.press_controller.forward(
                target_position=state['object_position'],
                current_joint_positions=state['joint_positions'],
                gripper_control=self.gripper_control,
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 10])).as_quat(),
            )
            
            if 'camera_data' in state:
                
                self.data_collector.cache_step(
                    camera_images=state['camera_data'],
                    joint_angles=state['joint_positions'][:-1]
                )
            
            return action, False, False
        
        
        final_object_position = self.object_utils.get_object_xform_position(
            object_path=self.cfg.sub_obj_path
        )
        
        if final_object_position is not None and final_object_position[0] > 0.40:
            
            self.data_collector.write_cached_data(state['joint_positions'][:-1])
            self._last_success = True
            self.reset_needed = True
            return None, True, True
            
        
        self.data_collector.clear_cache()
        self._last_success = False
        self.reset_needed = True
        return None, True, False
        
    def is_done(self):
        
        return self.press_controller.is_done()
        
    def is_success(self):
        
        return self._last_success
        
    def close(self):
        
        self.data_collector.close()

    def eposide_num(self):
        
        return self.data_collector.episode_count