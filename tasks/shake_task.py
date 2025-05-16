import numpy as np
import random
from .base_task import BaseTask
from utils.camera_utils import process_camera_image
from omni.isaac.sensor import Camera
from omni.isaac.core.utils.semantics import add_update_semantics

class ShakeTask(BaseTask):
    def __init__(self, cfg, world, stage, cameras, robot, object_utils):
        super().__init__(cfg, world, cameras)
        self.robot = robot
        self.object_utils = object_utils
        self.stage = stage
        
        
        self.cameras = []
        for cam_cfg in cfg.cameras:
            camera = Camera(
                prim_path=cam_cfg.prim_path,
                translation=np.array(cam_cfg.translation),
                name=cam_cfg.name,
                frequency=20,
                resolution=tuple(cam_cfg.resolution)
            )
            camera.set_focal_length(cam_cfg.focal_length)
            camera.set_local_pose(orientation=np.array(cam_cfg.orientation), camera_axes="usd")
            world.scene.add(camera)
            self.cameras.append(camera)
            
        self.world.reset()
        self._setup_cameras()
        
    def _setup_cameras(self):
        
        for camera, cam_cfg in zip(self.cameras, self.cfg.cameras):
            camera.initialize()
            if cam_cfg.image_type == "depth":
                camera.add_distance_to_image_plane_to_frame()
            elif cam_cfg.image_type == "segmentation":
                camera.add_instance_segmentation_to_frame()
                for class_id, class_to_prim in cam_cfg.class_to_prim.items():
                    e_prim = self.stage.GetPrimAtPath(class_to_prim)
                    add_update_semantics(e_prim, class_id)
            elif cam_cfg.image_type == "point":
                camera.add_instance_segmentation_to_frame()
                for class_id, class_to_prim in cam_cfg.class_to_prim.items():
                    e_prim = self.stage.GetPrimAtPath(class_to_prim)
                    add_update_semantics(e_prim, class_id)

            
    def reset(self):
        
        super().reset()
        self.robot.initialize()
        
        self.source_beaker = self.cfg.obj_path

        object_position = np.array([np.random.uniform(0.19, 0.34), np.random.uniform(-0.075, 0.075), 0.77])
        self.object_utils.set_object_position(obj_path=self.source_beaker, position=object_position)
        
            
    def step(self):
        
        self.frame_idx += 1
        if self.frame_idx < 5:
            return None
        elif self.frame_idx > 2000:
            self.reset_needed=True
        
        joint_positions = self.robot.get_joint_positions()
        source_position = self.object_utils.get_object_position(object_path=self.source_beaker)
        source_size = self.object_utils.get_object_size(object_path=self.source_beaker)
        
        
        camera_data = {}
        for camera, cam_cfg in zip(self.cameras, self.cfg.cameras):
            record, _ = process_camera_image(camera, cam_cfg.image_type)
            if record is not None:
                camera_data[cam_cfg.name] = record
                
        return {
            'joint_positions': joint_positions,
            'object_position': source_position,
            'object_size': source_size,
            'camera_data': camera_data,
            'source_beaker': self.source_beaker,
            'done': self.reset_needed
        }
