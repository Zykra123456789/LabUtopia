import numpy as np
import random
from .base_task import BaseTask
from utils.camera_utils import process_camera_image
from omni.isaac.core.utils.prims import set_prim_visibility
from pxr import UsdShade

class PickPourTask(BaseTask):
    def __init__(self, cfg, world, stage, robot):
        """Initialize the Pick and Pour task.

        Args:
            cfg: Configuration object for the task.
            world: The simulation world instance.
            stage: The USD stage for the simulation.
            robot: The robot instance used in the task.
        """
        super().__init__(cfg, world, stage, robot)
        self.target_path = cfg.target_path

    def on_task_complete(self, success):
        """Handle task progress updates.

        Args:
            success (bool): Indicates if the task was completed successfully.
        """
        if success:
            self.current_obj_episodes += 1
            if self.current_obj_episodes >= self.episodes_per_obj:
                self.current_obj_idx = (self.current_obj_idx + 1) % len(self.obj_configs)
                self.current_obj_episodes = 0
            if self.available_materials:
                self.current_material_idx = (self.current_material_idx + 1) % len(self.available_materials)

    def reset(self):
        """Reset the task state."""
        super().reset()
        self.robot.initialize()
        
        # Update material
        if self.material_config and self.available_materials:
            target_prim = self.stage.GetPrimAtPath(self.material_config.path)
            if target_prim.IsValid():
                material_path = self.available_materials[self.current_material_idx]
                mtl_prim = self.stage.GetPrimAtPath(material_path)
                if mtl_prim.IsValid():
                    cube_mat_shade = UsdShade.Material(mtl_prim)
                    UsdShade.MaterialBindingAPI(target_prim).Bind(
                        cube_mat_shade, 
                        UsdShade.Tokens.strongerThanDescendants
                    )

        # Set object position and visibility
        for i, obj_config in enumerate(self.obj_configs):
            obj_path = obj_config['path']
            position_range = obj_config['position_range']
            prim = self.stage.GetPrimAtPath(obj_path)
            
            if prim.IsValid():
                if i == self.current_obj_idx:
                    # Place current target object in workspace
                    right_pos = np.array([
                        np.random.uniform(position_range['x'][0], position_range['x'][1]),
                        np.random.uniform(position_range['y'][0], position_range['y'][1]),
                        position_range['z'][0]
                    ])
                    left_pos = right_pos.copy()
                    left_pos[1] -= random.random() * 0.08 + 0.16
                    left_pos[0] -= random.random() * 0.1 - 0.1
                    left_pos[2] += 0.05
                    self.object_utils.set_object_position(object_path=obj_path, position=right_pos)
                    self.object_utils.set_object_position(object_path=self.target_path, position=left_pos)
                    set_prim_visibility(prim, True)
                else:
                    # Move other objects far away and hide
                    angle = 2 * np.pi * i / len(self.obj_configs)
                    far_position = np.array([
                        10 * np.cos(angle),
                        10 * np.sin(angle),
                        0.1
                    ])
                    self.object_utils.set_object_position(object_path=obj_path, position=far_position)
                    set_prim_visibility(prim, False)
                    
        self.current_obj_path = self.obj_configs[self.current_obj_idx]['path']

    def step(self):
        """Execute one simulation step.

        Returns:
            dict: A dictionary containing simulation state data, or None if not ready.
        """
        self.frame_idx += 1
        if self.frame_idx < 5:
            return None
        elif self.frame_idx > self.cfg.task.max_steps:
            self.on_task_complete(True)
            self.reset_needed = True
        
        joint_positions = self.robot.get_joint_positions()
        if joint_positions is None:
            return None
            
        source_position = self.object_utils.get_geometry_center(object_path=self.current_obj_path)
        source_size = self.object_utils.get_object_size(object_path=self.current_obj_path)
        target_position = self.object_utils.get_geometry_center(object_path=self.target_path)
        source_quaternion = self.object_utils.get_transform_quat(object_path=self.current_obj_path+"/mesh")
                
        camera_data, display_data = self.get_camera_data()
        return {
            'joint_positions': joint_positions,
            'object_position': source_position,
            'object_size': source_size,
            'object_path': self.current_obj_path,
            'object_name': self.current_obj_path.split("/")[-1],
            'object_quaternion': source_quaternion,
            'target_position': target_position,
            'target_name': self.target_path.split("/")[-1],
            'target_path': self.target_path,
            'camera_data': camera_data,
            'done': self.reset_needed,
            'camera_display': display_data,
            'gripper_position': self.object_utils.get_transform_position(object_path="/World/Franka/panda_hand/tool_center")
        }
