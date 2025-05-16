import numpy as np
import random
from .base_task import BaseTask
from utils.camera_utils import process_camera_image
from omni.isaac.sensor import Camera
from omni.isaac.core.utils.semantics import add_update_semantics
from utils.Material_utils import bind_material_to_object

class CleanBeakerTask(BaseTask):
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

        self.table_material_paths = self.cfg.table_material_paths

        self.table1_surface_1 = self.cfg.table1_surface_1
        self.table1_surface_2 = self.cfg.table1_surface_2
        self.table2_surface_1 = self.cfg.table2_surface_1
        self.table2_surface_2 = self.cfg.table2_surface_2

        self.num_episode = 0
        self.per_episode = self.cfg.per_episode
        self.material_types = self.cfg.material_types
        
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
        
        self.target_beaker = "/World/target_beaker"

        self.beaker_1 = "/World/beaker_hard_1"
        self.beaker_2 = "/World/beaker_hard_2"

        self.plat_1 = "/World/target_plat_1"
        self.plat_2 = "/World/target_plat_2"


        beaker_1_position = np.array([np.random.uniform(0.20, 0.25), np.random.uniform(-0.05, -0.1), 0.77])
        self.object_utils.set_object_position(obj_path=self.beaker_1, position=beaker_1_position)

        plat_1_position = beaker_1_position + [0.03,0,-0.057]
        self.object_utils.set_object_position(obj_path=self.plat_1, position=plat_1_position)

        beaker_2_position = np.array([np.random.uniform(0.20, 0.25), np.random.uniform(0.20, 0.25), 0.77])
        self.object_utils.set_object_position(obj_path=self.beaker_2, position=beaker_2_position)

        plat_2_position = np.array([0.056, np.random.uniform(0.27, 0.32), 0.713])
        self.object_utils.set_object_position(obj_path=self.plat_2, position=plat_2_position)

        
        table_material_index = int(self.num_episode/self.per_episode) 
        table_material_index = table_material_index % self.material_types
        print("table_material_index:",table_material_index)
        bind_material_to_object(stage=self.stage,
                                obj_path=self.table1_surface_1,
                                material_path=self.table_material_paths[table_material_index])
        
        bind_material_to_object(stage=self.stage,
                                obj_path=self.table1_surface_2,
                                material_path=self.table_material_paths[table_material_index])
        
        bind_material_to_object(stage=self.stage,
                                obj_path=self.table2_surface_1,
                                material_path=self.table_material_paths[table_material_index])
        
        bind_material_to_object(stage=self.stage,
                                obj_path=self.table2_surface_2,
                                material_path=self.table_material_paths[table_material_index])

            
    def step(self):
        
        self.frame_idx += 1
        if self.frame_idx < 5:
            return None
        elif self.frame_idx > 20000:
            self.reset_needed=True
        
        joint_positions = self.robot.get_joint_positions()

        beaker_1_position = self.object_utils.get_object_position(object_path=self.beaker_1)
        beaker_2_position = self.object_utils.get_object_position(object_path=self.beaker_2)

        plat_1_position = self.object_utils.get_object_position(object_path=self.plat_1)
        plat_2_position = self.object_utils.get_object_position(object_path=self.plat_2)

        source_size = self.object_utils.get_object_size(object_path=self.target_beaker)
        target_position = self.object_utils.get_object_position(object_path=self.target_beaker)
        
        
        camera_data = {}
        for camera, cam_cfg in zip(self.cameras, self.cfg.cameras):
            record, _ = process_camera_image(camera, cam_cfg.image_type)
            if record is not None:
                camera_data[cam_cfg.name] = record
                
        return {
            'joint_positions': joint_positions,
            'object_size': source_size,
            'target_position': target_position,
            'beaker_1_position': beaker_1_position,
            'beaker_2_position': beaker_2_position,
            'plat_1_position': plat_1_position,
            'plat_2_position': plat_2_position,
            'camera_data': camera_data,
            'target_beaker': self.target_beaker,
            'beaker_2': self.beaker_2,
            'beaker_1': self.beaker_1,
            'done': self.reset_needed
        }
    
    def get_num_episode(self,num_episode):
        self.num_episode = num_episode
        # print("num_episode:",self.num_episode)
