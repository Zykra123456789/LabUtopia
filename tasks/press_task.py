import numpy as np
from scipy.spatial.transform import Rotation as R
from .base_task import BaseTask
from utils.object_utils import ObjectUtils
from utils.camera_utils import process_camera_image
import cv2
import os

from omni.isaac.sensor import Camera
from omni.isaac.core.utils.semantics import add_update_semantics
import random
from utils.Material_utils import bind_material_to_object

class PressTask(BaseTask):
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

        
        instrument_position = np.array([0.72889,-0.1 ,0.64346])
        instrument_path = self.cfg.instrument_path
        self.object_utils.set_object_position(obj_path=instrument_path, position=instrument_position)

        self.target_button_path = self.cfg.target_button_path
        self.distractor_button1_path = self.cfg.distractor_button1_path
        self.distractor_button2_path = self.cfg.distractor_button2_path
        self.table_path = self.cfg.table_path

        self.button_material_paths = self.cfg.button_material_paths
        self.table_material_paths = self.cfg.table_material_paths

        self.num_episode = 0
        self.per_episode = self.cfg.per_episode
        self.material_types = self.cfg.material_types
        self.button_types = self.cfg.button_types
                    
    def reset(self):
        super().reset()
        self.robot.initialize()

        
        self.position1 = np.array([0.40, 0, 1.1+np.random.uniform(-0.1,0.1)])
        self.position2 = self.position1+np.array([0.0, -0.20, 0])
        self.position3 = self.position1+np.array([0.0, -0.40, 0])

        positions = [self.position1, self.position2, self.position3]
        random.shuffle(positions)  

        
        self.object_utils.set_object_position(obj_path=self.target_button_path, position=positions[0])
        self.object_utils.set_object_position(obj_path=self.distractor_button1_path, position=positions[1])
        self.object_utils.set_object_position(obj_path=self.distractor_button2_path, position=positions[2])

        
        random_material_path = random.choice(self.button_material_paths[:self.button_types])
        bind_material_to_object(stage=self.stage,
                                obj_path=self.cfg.sub_obj_path,
                                material_path=random_material_path)
        
        
        table_material_index = int(self.num_episode/self.per_episode) 
        table_material_index = table_material_index % self.material_types
        bind_material_to_object(stage=self.stage,
                                obj_path=self.table_path,
                                material_path=self.table_material_paths[table_material_index])
        
    def step(self):
        
        self.frame_idx += 1
        if self.frame_idx < 5:
            return None
        elif self.frame_idx > 1000:
            self.reset_needed = True
            
        
        joint_positions = self.robot.get_joint_positions()
        if joint_positions is None:
            return None
        
        object_position = self.object_utils.get_object_xform_position(object_path=self.target_button_path)
        object_size = self.object_utils.get_object_size(object_path=self.target_button_path)
        
        
        camera_data = {}
        for camera, cam_cfg in zip(self.cameras, self.cfg.cameras):
            record, _ = process_camera_image(camera, cam_cfg.image_type)
            if record is not None:
                camera_data[cam_cfg.name] = record
                
        return {
            'joint_positions': joint_positions,
            'object_position': object_position,
            'object_size': object_size,
            'camera_data': camera_data,
            'done': self.reset_needed,
        }
    
    def get_num_episode(self,num_episode):
        self.num_episode = num_episode
        # print("num_episode:",self.num_episode)
