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

class InferShakeBeakerController(BaseController):
    def __init__(self, cfg, robot, object_utils):
        super().__init__(cfg, robot, object_utils)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if not OmegaConf.has_resolver("eval"):
            OmegaConf.register_new_resolver("eval", lambda x: eval(x))
        
        
        self.shake_checkpoint = torch.load(cfg.shake_model_path, map_location=self.device,weights_only=False)
        shake_policy_cfg = OmegaConf.load(cfg.shake_config_path)
        self.shake_policy = hydra.utils.instantiate(shake_policy_cfg.policy)
        self.shake_policy.load_state_dict(self.shake_checkpoint['state_dicts']['model'])
        self.shake_policy.eval().to(self.device)
        
        
        if hasattr(cfg, 'shake_normalizer_path'):
            normalizer = LinearNormalizer()
            normalizer.load_state_dict(torch.load(cfg.shake_normalizer_path, map_location=self.device))
        else:
            dataset: BaseImageDataset = hydra.utils.instantiate(shake_policy_cfg.task.dataset)
            normalizer = dataset.get_normalizer().to(self.device)
        self.shake_policy.set_normalizer(normalizer)
        
        
        self.n_obs_steps = shake_policy_cfg.n_obs_steps
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
        self.change_flag = False
        
    def reset(self):
        
        super().reset()
        self.obs_history_dict = {
            cam_name: deque(maxlen=self.n_obs_steps) 
            for cam_name in self.cfg.cameras_names
        }
        self.obs_history_pose = deque(maxlen=self.n_obs_steps)
        self.trajectory_controller.reset()
        self.frame_count = 0
        self.change_flag = False
        
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
                    obs_dict[camera_name] = obs_dict[camera_name].unsqueeze(0).to(self.device)  # [1, T, C, H, W]
                obs_dict['agent_pose'] = obs_dict['agent_pose'].unsqueeze(0).to(self.device)    # [1, T, D]
            else:
                for camera_name in self.obs_history_dict.keys():
                    obs_dict[camera_name] = obs_dict[camera_name].to(self.device)  # [1(T), C, H, W]
                obs_dict['agent_pose'] = obs_dict['agent_pose'].to(self.device)  # [1(T), D]

            
            with torch.no_grad():
                prediction = self.shake_policy.predict_action(obs_dict)
                joint_positions = prediction['action'][0].cpu().numpy()
            
            
            self.trajectory_controller.generate_trajectory(joint_positions)
        
        
        action = self.trajectory_controller.get_next_action()
        if self.n_obs_steps != 1 and action is not None:
            if len(action.joint_positions) == 9:
                if action.joint_positions[7] is not None or action.joint_positions[8] is not None:
                    if action.joint_positions[7] - np.float32(0.015) < np.float32(0.015):
                        self.change_flag = True

                    if self.change_flag:
                        action.joint_positions[7] = np.float32(0.015)
                        action.joint_positions[8] = np.float32(0.015)

        return action, self.is_done(), self.is_success()
        
    def is_done(self):
        
        return self.trajectory_controller.is_trajectory_complete()
        
    def is_success(self):
        
        Maxframe = 650 
        self.frame_count += 1
        if self.frame_count > Maxframe:
            self.reset_needed = True
            return True
        return False

    def episode_num(self):
        
        return self._episode_num