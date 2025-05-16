import numpy as np
from scipy.spatial.transform import Rotation as R
from .base_task import BaseTask
from utils.object_utils import ObjectUtils
from utils.camera_utils import process_camera_image
import cv2
import os

from omni.isaac.sensor import Camera
from omni.isaac.core.utils.semantics import add_update_semantics
# from omni.isaac.core.utils.numpy.rotations import euler_angles_to_quat

class CloseTask(BaseTask):
    def __init__(self, cfg, world, stage, cameras, robot):
        super().__init__(cfg, world, cameras)
        self.robot = robot
        
        
        self.cameras = []
        for cam_cfg in cfg.cameras:
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
            world.scene.add(camera)
            self.cameras.append(camera)
            
        self.world.reset()
        for camera, cam_cfg in zip(self.cameras, cfg.cameras):
            camera.initialize()
            if cam_cfg.image_type == "depth":
                camera.add_distance_to_image_plane_to_frame()
            elif cam_cfg.image_type == "segmentation":
                camera.add_instance_segmentation_to_frame()
                for class_id, class_to_prim in cam_cfg.class_to_prim.items():
                    e_prim = stage.GetPrimAtPath(class_to_prim)
                    add_update_semantics(e_prim, class_id)
            elif cam_cfg.image_type == "point":
                camera.add_instance_segmentation_to_frame()
                for class_id, class_to_prim in cam_cfg.class_to_prim.items():
                    e_prim = stage.GetPrimAtPath(class_to_prim)
                    add_update_semantics(e_prim, class_id)
                    
    def reset(self):
        super().reset()
        self.robot.initialize()
        
        object_position = np.array([
            np.random.uniform(0.75, 0.8),
            np.random.uniform(-0.08, 0.08),
            1.09
        ])
        self.object_utils.set_object_position(obj_path=self.cfg.obj_path, position=object_position)
        object_position = self.object_utils.get_object_xform_position(object_path="/World/sektion_cabinet_instanceable/drawer_bottom")
        object_position[0] -= 0.1
        self.object_utils.set_object_position(obj_path="/World/sektion_cabinet_instanceable/drawer_bottom", position=object_position)
        
        
    def step(self):
        
        self.frame_idx += 1
        if self.frame_idx < 5:
            return None
        elif self.frame_idx > 1000:
            self.reset_needed = True
            
        
        joint_positions = self.robot.get_joint_positions()
        if joint_positions is None:
            return None
        
        object_position = self.object_utils.get_object_position(object_path=self.cfg.sub_obj_path)
        object_size = self.object_utils.get_object_size(object_path=self.cfg.sub_obj_path)
        
        
        camera_data = {}
        for camera, cam_cfg in zip(self.cameras, self.cfg.cameras):
            record, _ = process_camera_image(camera, cam_cfg.image_type)
            if record is not None:
                camera_data[cam_cfg.name] = record
        
        # print(object_position, object_size)
        
        return {
            'joint_positions': joint_positions,
            'object_position': object_position,
            'object_size': object_size,
            'camera_data': camera_data,
            'done': self.reset_needed,
        }
