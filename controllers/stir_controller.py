from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.franka.controllers.rmpflow_controller import RMPFlowController
from scipy.spatial.transform import Rotation as R
import numpy as np
import os

from .atomic_actions.pick_controller import PickController
from .atomic_actions.stir_controller import StirController  
from .base_controller import BaseController
from .robot_controllers.grapper_manager import Gripper
from factories.collector_factory import create_collector

class StirTaskController(BaseController):
    
    def __init__(self, cfg, robot, object_utils):
        super().__init__(cfg, robot, object_utils)
        
        
        self.stir_collector = create_collector(
            cfg.collector.type,
            camera_configs=cfg.cameras,
            save_dir=os.path.join(cfg.multi_run.run_dir, "stir_data"),  
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
        
        
        self.stir_controller = StirController(
            name="stir_controller",
            cspace_controller=RMPFlowController(
                name="stir_controller", 
                robot_articulation=robot
            ),
            events_dt=[1.0 , 2.5 , 2.0 , 4.0 , 1.0],
        )
        
        self.gripper_control = Gripper()
        self.gripper_control.release_object()
        self._pick_completed = False  
        self._last_pick_joint_data = None  
        
    def reset(self):
        
        super().reset()
        self.pick_controller.reset()
        self.stir_controller.reset()  
        self._pick_completed = False
        self.gripper_control.release_object()
        
    def step(self, state):
        if not self.pick_controller.is_done():
            
            action = self.pick_controller.forward(
                picking_position=state['object_position'],
                current_joint_positions=state['joint_positions'],
                object_size=state['object_size'],
                object_name="glass_rod",
                gripper_control=self.gripper_control,
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 10])).as_quat(),
            )
            
            if self.pick_controller.is_done():
                self._pick_completed = True
            
            self.gripper_control.update_grasped_object_position()

            return action, False, False
            
        
        if not self.stir_controller.is_done():            
            
            target_position = self.object_utils.get_object_xform_position(
                object_path=state['target_beaker']
            )
            if target_position is None:
                target_position = state['target_position']  
            
            action = self.stir_controller.forward(
                center_position=target_position,  
                current_joint_positions=state['joint_positions'],
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 10])).as_quat(),
            )
            
            if 'camera_data' in state:
                
                self.stir_collector.cache_step(
                    camera_images=state['camera_data'],
                    joint_angles=state['joint_positions'][:-1]
                )

            self.gripper_control.update_grasped_object_position()
                
            return action, False, False
            
        
        final_object_position = self.object_utils.get_object_xform_position(
            object_path=self.cfg.sub_obj_path
        )
        target_position = state['target_position']  
        if (final_object_position is not None and 
            final_object_position[2] > 0.85 and
            abs(final_object_position[0] - target_position[0]) < 0.025 and
            abs(final_object_position[1] - target_position[1]) < 0.025 ):
            
            self.stir_collector.write_cached_data(state['joint_positions'][:-1])  
            self._last_success = True
            self.reset_needed = True
            return None, True, True
        else: 
            
            self.stir_collector.clear_cache()  
            self._last_success = False
            self.reset_needed = True
            self.gripper_control.release_object()
            return None, True, False
        
    def close(self):
        
        self.stir_collector.close()  
        
    def get_current_collector(self):
        
        return self.stir_collector  
        
    def is_done(self):
        
        return self._pick_completed and self.stir_controller.is_done()  
        
    def is_success(self):
        
        return self._last_success

    def eposide_num(self):
        
        return self.stir_collector.episode_count  