import os
import numpy as np
import torch
import hydra
from collections import deque
from omni.isaac.core.utils.types import ArticulationAction
from omegaconf import OmegaConf
from policy.model.common.normalizer import LinearNormalizer
from policy.dataset.base_dataset import BaseImageDataset
from utils.object_utils import ObjectUtils
from .base_controller import BaseController
from .robot_controllers.trajectory_controller import FrankaTrajectoryController
from .robot_controllers.grapper_manager import Gripper

class InferStirGlassrodController(BaseController):
    def __init__(self, cfg, robot, object_utils):
        super().__init__(cfg, robot, object_utils)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if not OmegaConf.has_resolver("eval"):
            OmegaConf.register_new_resolver("eval", lambda x: eval(x))
        
        self.gripper_control = Gripper()
        
        
        self.checkpoint = torch.load(cfg.stir_model_path, map_location=self.device, weights_only=False)
        policy_cfg = OmegaConf.load(cfg.stir_config_path)
        self.policy = hydra.utils.instantiate(policy_cfg.policy)
        self.policy.load_state_dict(self.checkpoint['state_dicts']['model'])
        self.policy.eval().to(self.device)
        
        
        if hasattr(cfg, 'stir_normalizer_path'):
            normalizer = LinearNormalizer()
            normalizer.load_state_dict(torch.load(cfg.stir_normalizer_path, map_location=self.device))
        else:
            dataset: BaseImageDataset = hydra.utils.instantiate(policy_cfg.task.dataset)
            normalizer = dataset.get_normalizer().to(self.device)
        self.policy.set_normalizer(normalizer)
        
        
        self.n_obs_steps = policy_cfg.n_obs_steps
        self.obs_history_dict = {
            cam_name: deque(maxlen=self.n_obs_steps) 
            for cam_name in self.cfg.cameras_names
        }
        self.obs_history_pose = deque(maxlen=self.n_obs_steps)
        
        
        self.trajectory_controller = FrankaTrajectoryController(
            name="trajectory_controller",
            robot_articulation=robot
        )
        self.frame_count = 0
        self.obj_added = False
        
    def reset(self):
        
        super().reset()
        self.obs_history_dict = {
            cam_name: deque(maxlen=self.n_obs_steps) 
            for cam_name in self.cfg.cameras_names
        }
        self.obs_history_pose = deque(maxlen=self.n_obs_steps)
        self.trajectory_controller.reset()
        self.frame_count = 0
        self.gripper_control.release_object()
        print("glassrod is released！！！")
        self.obj_added = False
        
    def step(self, state):
        
        self.state = state
        
        
        for cam_name, image in state['camera_data'].items():
            self.obs_history_dict[cam_name].append(image)
        self.obs_history_pose.append(state['joint_positions'][:-1])
        
        
        histories_complete = (
            len(self.obs_history_pose) == self.n_obs_steps and
            all(len(hist) == self.n_obs_steps for hist in self.obs_history_dict.values())
        )
        
        if self.trajectory_controller.is_trajectory_complete() and histories_complete:
            
            obs_dict = {}
            for camera_name, history in self.obs_history_dict.items():
                camera_tensor = torch.from_numpy(
                    np.stack(list(history))
                ).float() / 255.0
                obs_dict[camera_name] = camera_tensor.to(self.device)
                
            pose_tensor = torch.from_numpy(
                np.stack(list(self.obs_history_pose))
            ).float()
            obs_dict['agent_pose'] = pose_tensor.to(self.device)
            
            
            if self.n_obs_steps != 1:
                for camera_name in self.obs_history_dict.keys():
                    obs_dict[camera_name] = obs_dict[camera_name].unsqueeze(0).to(self.device) # [1, T, C, H, W]
                obs_dict['agent_pose'] = obs_dict['agent_pose'].unsqueeze(0).to(self.device)  # [1, T, D]
            else:
                for camera_name in self.obs_history_dict.keys():
                    obs_dict[camera_name] = obs_dict[camera_name].to(self.device)  # [1(T), C, H, W]
                obs_dict['agent_pose'] = obs_dict['agent_pose'].to(self.device)  # [1(T), D]

            
            with torch.no_grad():
                prediction = self.policy.predict_action(obs_dict)
                joint_positions = prediction['action'][0].cpu().numpy()
            
            
            self.trajectory_controller.generate_trajectory(joint_positions)
        
        
        action = self.trajectory_controller.get_next_action()
        if action is not None:
            if len(action.joint_positions) == 9:
                if action.joint_positions[7] == None and action.joint_positions[8] == None:
                    action.joint_positions[7] = 0.015
                    action.joint_positions[8] = 0.015

                if action.joint_positions[7] < np.float32(0.02) and action.joint_positions[8] < np.float32(0.02) and not self.obj_added:
                    self.gripper_control.add_object_to_gripper("/World/glass_rod", "/World/Franka/panda_hand/tool_center")
                    print("glassrod is added to franka gripper center！！！")
                    self.obj_added = True
                    
        self.gripper_control.update_grasped_object_position()
        return action, False, self.is_success()
        
    def is_done(self):
        
        return self.trajectory_controller.is_trajectory_complete()
        
    def is_success(self):
        
        Maxframe = 1500
        self.frame_count += 1
        if self.frame_count > Maxframe:
            self.gripper_control.release_object()
            self.reset_needed = True
            return True
        
        # object_pos = self.object_utils.get_object_xform_position(
        #     object_path=self.cfg.sub_obj_path
        # )
        # target_position = self.state['target_position']
        # if (object_pos is not None and object_pos[2] > 0.85 and
        #     abs(object_pos[0] - target_position[0]) < 0.025 and
        #     abs(object_pos[1] - target_position[1]) < 0.025):
        #     return True
        return False

    def eposide_num(self):
        
        return self._eposide_num