import numpy as np
import random
from tasks.base_task import BaseTask
from utils.camera_utils import process_camera_image
from omni.isaac.core.utils.prims import set_prim_visibility
from pxr import UsdShade

class PlaceTask(BaseTask):
    def __init__(self, cfg, world, stage, robot):
        """Initialize the Pick and Pour task.

        Args:
            cfg: Configuration object for the task.
            world: The simulation world instance.
            stage: The USD stage for the simulation.
            robot: The robot instance used in the task.
        """
        super().__init__(cfg, world, stage, robot)

    def reset(self):
        """Reset the task state."""
        super().reset()
        self.robot.initialize()

        
        self.source_beaker = self.cfg.task.obj_paths[0]['path']
        self.target_plat = self.cfg.task.obj_paths[1]['path']

        obj_position_range = self.cfg.task.obj_paths[0]['position_range'] 
        object_position = np.array([
                        np.random.uniform(obj_position_range['x'][0], obj_position_range['x'][1]),
                        np.random.uniform(obj_position_range['y'][0], obj_position_range['y'][1]),
                        obj_position_range['z'][0]
                    ])
        self.object_utils.set_object_position(object_path=self.source_beaker, position=object_position)

        target_position_range = self.cfg.task.obj_paths[1]['position_range']
        target_position = np.array([
                        np.random.uniform(target_position_range['x'][0], target_position_range['x'][1]),
                        np.random.uniform(target_position_range['y'][0], target_position_range['y'][1]),
                        target_position_range['z'][0]
                    ])
        self.object_utils.set_object_position(object_path=self.target_plat, position=target_position)
   

    def step(self):
        """Execute one simulation step.

        Returns:
            dict: A dictionary containing simulation state data, or None if not ready.
        """
        self.frame_idx += 1
        if self.frame_idx < 5:
            return None
        elif self.frame_idx > self.cfg.task.max_steps:
            self.reset_needed = True
        
        joint_positions = self.robot.get_joint_positions()
        if joint_positions is None:
            return None
            
        source_position = self.object_utils.get_object_position(object_path=self.source_beaker)
        source_size = self.object_utils.get_object_size(object_path=self.source_beaker)
        target_position = self.object_utils.get_object_position(object_path=self.target_plat)
        
        camera_data = {}
        for camera, cam_cfg in zip(self.cameras, self.cfg.cameras):
            record, _ = process_camera_image(camera, cam_cfg.image_type)
            if record is not None:
                camera_data[cam_cfg.name] = record

        camera_data, display_data = self.get_camera_data()
                
        return {
            'joint_positions': joint_positions,
            'object_position': source_position,
            'object_size': source_size,
            'object_path': self.source_beaker,
            'object_name': self.source_beaker.split("/")[-1],
            'target_position': target_position,
            'target_name': self.target_plat.split("/")[-1],
            'target_path': self.target_plat,
            'camera_data': camera_data,
            'camera_display': display_data,
            'done': self.reset_needed,
            'gripper_position': self.object_utils.get_transform_position(object_path="/World/Franka/panda_hand/tool_center")
        }
