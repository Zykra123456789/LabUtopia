import os
import numpy as np
import torch
import hydra
from collections import deque
from omni.isaac.core.utils.types import ArticulationAction
from omegaconf import OmegaConf
from policy.model.common.normalizer import LinearNormalizer
from policy.dataset.base_dataset import BaseImageDataset
from scipy.spatial.transform import Rotation as R
from utils.object_utils import ObjectUtils
from .base_controller import BaseController
from .robot_controllers.trajectory_controller import FrankaTrajectoryController
from .atomic_actions.pick_controller import PickController
from omni.isaac.franka.controllers.rmpflow_controller import RMPFlowController
from .robot_controllers.grapper_manager import Gripper

class InferStirController(BaseController):
    def __init__(self, cfg, robot, object_utils):
        super().__init__(cfg, robot, object_utils)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if not OmegaConf.has_resolver("eval"):
            OmegaConf.register_new_resolver("eval", lambda x: eval(x))
        
        
        self.pick_controller = PickController(
            name="pick_controller",
            cspace_controller=RMPFlowController(
                name="target_follower_controller",
                robot_articulation=robot
            ),
            gripper=robot.gripper,
            events_dt=[0.016, 0.005, 0.01, 0.2, 0.05, 0.01]
        )
        self.gripper_control = Gripper()
        
        
        self.stir_checkpoint = torch.load(cfg.stir_model_path, map_location=self.device, weights_only=False)
        stir_policy_cfg = OmegaConf.load(cfg.stir_config_path)
        self.stir_policy = hydra.utils.instantiate(stir_policy_cfg.policy)
        self.stir_policy.load_state_dict(self.stir_checkpoint['state_dicts']['model'])
        self.stir_policy.eval().to(self.device)
        
        
        if hasattr(cfg, 'stir_normalizer_path'):
            normalizer = LinearNormalizer()
            normalizer.load_state_dict(torch.load(cfg.stir_normalizer_path, map_location=self.device))
        else:
            dataset: BaseImageDataset = hydra.utils.instantiate(stir_policy_cfg.task.dataset)
            normalizer = dataset.get_normalizer().to(self.device)
        self.stir_policy.set_normalizer(normalizer)
        
        
        self.n_obs_steps = stir_policy_cfg.n_obs_steps
        self.obs_history_dict = {
            cam_name: deque(maxlen=self.n_obs_steps) 
            for cam_name in self.cfg.cameras_names
        }
        self.obs_history_pose = deque(maxlen=self.n_obs_steps)
        
        
        self.trajectory_controller = FrankaTrajectoryController(
            name="trajectory_controller",
            robot_articulation=robot
        )
        self.use_stir_model = False
        self.change_flag = False
        self.frame_count = 0
        
    def reset(self):
        
        super().reset()
        self.pick_controller.reset()
        self.obs_history_dict = {
            cam_name: deque(maxlen=self.n_obs_steps) 
            for cam_name in self.cfg.cameras_names
        }
        self.obs_history_pose = deque(maxlen=self.n_obs_steps)
        self.use_stir_model = False
        self.trajectory_controller.reset()
        self.frame_count = 0
        
    def step(self, state):
        
        self.state = state
        
        if not self.pick_controller.is_done():
            action = self.pick_controller.forward(
                picking_position=state['object_position'],
                current_joint_positions=state['joint_positions'],
                object_size=state['object_size'],
                object_name="glass_rod",
                gripper_control=self.gripper_control,
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 10])).as_quat(),
            )
            
            final_object_position = self.object_utils.get_object_xform_position(
                object_path=self.cfg.sub_obj_path
            )
            if final_object_position is not None and final_object_position[2] > 0.82:
                self.use_stir_model = True

            self.gripper_control.update_grasped_object_position()

            return action, False, False
        
        
        if self.use_stir_model:
            
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
                    prediction = self.stir_policy.predict_action(obs_dict)
                    joint_positions = prediction['action'][0].cpu().numpy()
                
                
                self.trajectory_controller.generate_trajectory(joint_positions)
            
            
            action = self.trajectory_controller.get_next_action()
            self.gripper_control.update_grasped_object_position()

            return action, False, self.is_success()
        
        
        return ArticulationAction(), False, False
        
    def is_done(self):
        
        if not self.use_stir_model:
            return self.pick_controller.is_done()
        return self.trajectory_controller.is_trajectory_complete()
        
    def is_success(self):
        
        Maxframe = 700 # DP 380
        self.frame_count +=1
        if self.frame_count > Maxframe:
            self.gripper_control.release_object()
            self.reset_needed = True
            return True
        if not self.use_stir_model:
            return False
        object_pos = self.object_utils.get_object_xform_position(
            object_path=self.cfg.sub_obj_path
        )
        target_position = self.state['target_position']
        if (object_pos is not None and object_pos[2] > 0.85 and
            abs(object_pos[0] - target_position[0]) < 0.025 and
            abs(object_pos[1] - target_position[1]) < 0.025):
                return True
        return False

    def eposide_num(self):
        
        return self._eposide_num