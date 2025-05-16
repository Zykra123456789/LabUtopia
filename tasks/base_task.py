from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np
from omni.isaac.sensor import Camera
from utils.object_utils import ObjectUtils
from omni.isaac.core.utils.semantics import add_update_semantics
from utils.camera_utils import process_camera_image

class BaseTask(ABC):
    """
    Base class for all simulation tasks.
    
    Attributes:
        cfg: Task configuration
        world: Simulation world instance
        cameras: Camera settings
        reset_needed (bool): Flag indicating if task needs reset
        frame_idx (int): Current frame index
        object_utils (ObjectUtils): Utility instance for object operations
    """
    def __init__(self, cfg, world, stage, robot):
        """
        Initialize the task.
        
        Args:
            cfg: Task configuration
            world: Simulation world instance
            cameras: Camera settings
        """
        self.cfg = cfg
        self.world = world
        self.stage = stage
        self.robot = robot
        self.reset_needed = False
        self.frame_idx = 0
        self.object_utils = ObjectUtils.get_instance()
        self.setup_cameras()
        self.setup_objects()
        self.setup_materials()
        
        self.current_material_idx = 0
        if len(self.obj_configs) != 0:
            self.episodes_per_obj = int(cfg.max_episodes / len(self.obj_configs))
        else:
            self.episodes_per_obj = 0
        self.current_obj_idx = 0
        self.current_obj_episodes = 0
        
    def reset(self) -> None:
        """
        Reset the task state and simulation world.
        """
        self.world.reset()
        self.reset_needed = False
        self.frame_idx = 0
    
    @abstractmethod
    def step(self) -> Dict[str, Any]:
        """
        Execute one step of the task.
        
        Returns:
            Dict[str, Any]: Task step information
        """
        pass

    def get_task_info(self) -> Dict[str, Any]:
        """
        Get task-related information.
        
        Returns:
            Dict[str, Any]: Dictionary containing task information
        """
        return {
            "frame_idx": self.frame_idx,
            "reset_needed": self.reset_needed
        }
        
    def need_reset(self) -> bool:
        """
        Check if task needs to be reset.
        
        Returns:
            bool: True if reset is needed, False otherwise
        """
        return self.reset_needed

    def on_task_complete(self, success: bool) -> None:
        """
        Handle task completion event.
        
        Args:
            success (bool): Whether the task completed successfully
        """
        self.reset_needed = True
        
    def setup_cameras(self) -> None:
        """
        Set up cameras for the task.
        """
        self.cameras = []
        for cam_cfg in self.cfg.cameras:
            camera = Camera(
                prim_path=cam_cfg.prim_path,
                translation=np.array(cam_cfg.translation),
                name=cam_cfg.name,
                frequency=60,
                resolution=tuple(cam_cfg.resolution)
            )
            camera.set_focal_length(cam_cfg.focal_length)
            camera.set_local_pose(orientation=np.array(cam_cfg.orientation), camera_axes="usd")
            camera.set_clipping_range(near_distance=0.1, far_distance=10.0)
            self.world.scene.add(camera)
            self.cameras.append(camera)
            
        self.world.reset()
        for camera, cam_cfg in zip(self.cameras, self.cfg.cameras):
            camera.initialize()
            
            image_types = cam_cfg.image_type.split('+') if '+' in cam_cfg.image_type else [cam_cfg.image_type]
            
            for image_type in image_types:
                if image_type == "depth":
                    camera.add_distance_to_image_plane_to_frame()
                elif image_type == "pointcloud":
                    camera.add_distance_to_image_plane_to_frame()
                    camera.add_pointcloud_to_frame()
                elif image_type == "segmentation":
                    camera.add_instance_segmentation_to_frame()
                    for class_id, class_to_prim in cam_cfg.class_to_prim.items():
                        e_prim = self.stage.GetPrimAtPath(class_to_prim)
                        add_update_semantics(e_prim, class_id)
                elif image_type == "semantic_pointcloud":
                    camera.add_instance_segmentation_to_frame()
                    camera.add_distance_to_image_plane_to_frame()
                    camera.add_pointcloud_to_frame()
                    for class_id, class_to_prim in cam_cfg.class_to_prim.items():
                        e_prim = self.stage.GetPrimAtPath(class_to_prim)
                        add_update_semantics(e_prim, class_id)
    
    def setup_objects(self) -> None:
        """
        Set up objects in the simulation world.
        """
        self.obj_configs = []
        if hasattr(self.cfg.task, 'obj_paths'):
            for obj in self.cfg.task.obj_paths:
                if isinstance(obj, str):
                    self.obj_configs.append({
                        'path': obj,
                        'position_range': {
                            'x': [0.24, 0.30],
                            'y': [-0.05, 0.05],
                            'z': [0.85, 0.85]
                        }
                    })
                else:
                    self.obj_configs.append(obj)
                
    def setup_materials(self) -> None:
        """
        Set up materials for the objects.
        """
        self.material_config = None
        if hasattr(self.cfg.task, 'material_paths'):
            self.material_config = self.cfg.task.material_paths[0] if self.cfg.task.material_paths else None
            self.available_materials = self.material_config.materials if self.material_config else []
        else:
            self.available_materials = []
    
    def get_camera_data(self):
        camera_data = {}
        display_data = {}
        for camera, cam_cfg in zip(self.cameras, self.cfg.cameras):
            record, display = process_camera_image(camera, cam_cfg.image_type)
            if record is not None:
                if isinstance(record, dict):
                    for k, v in record.items():
                        camera_data[f"{cam_cfg.name}_{k}"] = v
                else:
                    camera_data[f"{cam_cfg.name}_{cam_cfg.image_type}"] = record
            if display is not None:
                display_data[cam_cfg.name] = display
        return camera_data, display_data
    
    def get_num_episode(self,num_episode):
        self.num_episode = num_episode