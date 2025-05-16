import numpy as np
from .base_task import BaseTask
from utils.camera_utils import process_camera_image
from omni.isaac.core.utils.prims import set_prim_visibility
from pxr import UsdShade

class PickTask(BaseTask):
    """
    A task class for robotic picking operations.
    Manages object placement, material switching, and task state transitions.
    """
    def __init__(self, cfg, world, stage, robot):
        super().__init__(cfg, world, stage, robot)
        
    def on_task_complete(self, success):
        """
        Handles task completion logic.
        Updates object and material indices when success conditions are met.

        Args:
            success (bool): Whether the task was completed successfully
        """
        if success:
            self.current_obj_episodes += 1
            if self.current_obj_episodes >= self.episodes_per_obj:
                self.current_obj_idx = (self.current_obj_idx + 1) % len(self.obj_configs)
                self.current_obj_episodes = 0
            if self.available_materials:
                self.current_material_idx = (self.current_material_idx + 1) % len(self.available_materials)
            
    def reset(self):
        """
        Resets the task state.
        Initializes robot position and updates object/material states.
        Places objects in their starting positions and updates visibility.
        """
        super().reset()
        self.robot.initialize()
        
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
                    
        for i, obj_config in enumerate(self.obj_configs):
            obj_path = obj_config['path']
            position_range = obj_config['position_range']
            prim = self.stage.GetPrimAtPath(obj_path)
            
            if prim.IsValid():
                if i == self.current_obj_idx:
                    object_position = np.array([
                        np.random.uniform(position_range['x'][0], position_range['x'][1]),
                        np.random.uniform(position_range['y'][0], position_range['y'][1]),
                        np.random.uniform(position_range['z'][0], position_range['z'][1])
                    ])
                    self.object_utils.set_object_position(object_path=obj_path, position=object_position)
                    set_prim_visibility(prim, True)
                else:
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
        """
        Executes one simulation step and returns current state.

        Returns:
            dict: Current state dictionary containing:
                - joint_positions: Robot joint positions
                - object_position: Target object position
                - object_size: Target object dimensions
                - camera_data: Camera image data
                - done: Whether episode is complete
                - gripper_position: End effector position
                - object_name: Name of current target object
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
        
        object_position = self.object_utils.get_geometry_center(
            object_path=self.obj_configs[self.current_obj_idx]['path']
        )
        object_size = self.object_utils.get_object_size(
            object_path=self.obj_configs[self.current_obj_idx]['path']
        )
        
        camera_data, display_data = self.get_camera_data()
        
        return {
            'joint_positions': joint_positions,
            'object_position': object_position,
            'object_size': object_size,
            'camera_data': camera_data,
            'camera_display': display_data,
            'done': self.reset_needed,
            'gripper_position': self.object_utils.get_transform_position(object_path="/World/Franka/panda_hand/tool_center"),
            'object_name': self.current_obj_path.split("/")[-1]
        }
