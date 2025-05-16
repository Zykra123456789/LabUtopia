import torch
import hydra
import numpy as np
from omegaconf import OmegaConf
from collections import deque
from policy.model.common.normalizer import LinearNormalizer
from .base_controller import BaseController
from policy.dataset.base_dataset import BaseImageDataset
from .robot_controllers.trajectory_controller import FrankaTrajectoryController

class InferPressController(BaseController):
    def __init__(self, cfg, robot, object_utils):
        super().__init__(cfg, robot, object_utils)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        
        if not OmegaConf.has_resolver("eval"):
            OmegaConf.register_new_resolver("eval", lambda x: eval(x))

        self.press_checkpoint = torch.load(cfg.press_model_path, map_location=self.device,weights_only=False)
        self.press_config = OmegaConf.load(cfg.press_config_path)
        
        
        self.press_policy = hydra.utils.instantiate(self.press_config.policy)
        self.press_policy.load_state_dict(self.press_checkpoint['state_dicts']['model'])
        self.press_policy.eval().to(self.device)
        self.press_policy.to(self.device)

        
        if hasattr(cfg, 'normalizer_path'):
            normalizer = LinearNormalizer()
            normalizer.load_state_dict(torch.load(cfg.normalizer_path, map_location=self.device))
        else:
            dataset: BaseImageDataset = hydra.utils.instantiate(self.press_config.task.dataset)
            normalizer = dataset.get_normalizer()
        normalizer.to(self.device)
        self.press_policy.set_normalizer(normalizer)

        
        self.n_obs_steps = self.press_config.n_obs_steps
        self.obs_history_dict = {
            cam_name: deque(maxlen=self.n_obs_steps) 
            for cam_name in self.cfg.cameras_names
        }
        self.obs_history_pose = deque(maxlen=self.n_obs_steps)
        
        self.trajectory_controller = FrankaTrajectoryController(
            name="trajectory_controller",
            robot_articulation=robot
        )

        self.success_episode = 0
        self.print_success = False

    def reset(self):
        
        if self.print_success:
            print(f"Task Success!Success rate:{self.success_episode}/{self._eposide_num}({self.get_success_rate()}%)")

        super().reset()
        self.obs_history_dict = {
            cam_name: deque(maxlen=self.n_obs_steps) 
            for cam_name in self.cfg.cameras_names
        }
        self.obs_history_pose = deque(maxlen=self.n_obs_steps)
        self.trajectory_controller.reset()

        
    def step(self, state):
        
        
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
                prediction = self.press_policy.predict_action(obs_dict)
                joint_positions = prediction['action'][0].cpu().numpy()
                
            
            self.trajectory_controller.generate_trajectory(joint_positions)
            
        
        action = self.trajectory_controller.get_next_action()
        
        return action, False, self.is_success()
        
    def is_success(self):
        
        object_pos = self.object_utils.get_object_xform_position(
            object_path=self.cfg.sub_obj_path
        )
        if object_pos is not None and object_pos[0] > 0.41:
            self.success_episode += 1
            self.reset_needed = True
            self.print_success = True
            return True
        return False
    
    def is_done(self):
        
        return False
        
    def eposide_num(self):
        
        
        return self._eposide_num
    
    def get_success_rate(self):
        episode_num = self.eposide_num()
        success_rate = self.success_episode/episode_num * 100
        return success_rate