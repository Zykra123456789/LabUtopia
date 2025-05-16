import os
import numpy as np
import torch
import hydra
from collections import deque
from omegaconf import OmegaConf
from policy.model.common.normalizer import LinearNormalizer
from .base_controller import BaseController
from policy.dataset.base_dataset import BaseImageDataset
from utils.object_utils import ObjectUtils
from .robot_controllers.trajectory_controller import FrankaTrajectoryController

class InferCleanBeaker7PolicyController(BaseController):
    def __init__(self, cfg, robot, object_utils):
        super().__init__(cfg, robot, object_utils)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Validate configuration paths
        if len(cfg.config_paths) != 7 or len(cfg.model_paths) != 7:
            raise ValueError("Expected exactly 7 policy config paths and model paths.")
        if hasattr(cfg, 'normalizer_paths') and len(cfg.normalizer_paths) != 7:
            raise ValueError("Expected exactly 7 normalizer paths if provided.")
        
        if not OmegaConf.has_resolver("eval"):
            OmegaConf.register_new_resolver("eval", lambda x: eval(x))
        
        # Store configuration paths
        self.policy_cfg_paths = cfg.config_paths
        self.policy_paths = cfg.model_paths
        self.normalizer_paths = cfg.normalizer_paths if hasattr(cfg, 'normalizer_paths') else None
        self.current_policy = None  # Current active policy model
        
        # Observation history buffers
        self.n_obs_steps = OmegaConf.load(cfg.config_paths[0]).n_obs_steps
        self.obs_history_dict = {
            cam_name: deque(maxlen=self.n_obs_steps)
            for cam_name in self.cfg.cameras_names
        }
        self.obs_history_pose = deque(maxlen=self.n_obs_steps)
        
        # Trajectory controller
        self.trajectory_controller = FrankaTrajectoryController(
            name="trajectory_controller",
            robot_articulation=robot
        )
        self.current_policy_idx = 0
        self.frame_count = 0
        self.success_episode = [0] * 7
        self._eposide_num = 0
        self.model_1_loaded = False
        self.model_2_loaded = False
        self.model_3_loaded = False
        self.model_4_loaded = False
        self.model_5_loaded = False
        self.model_6_loaded = False
        self.model_7_loaded = False

    def load_policy_model(self, policy_idx):
        """Load policy model for the given index, clearing previous model and CUDA memory."""
        if not (0 <= policy_idx < 7):
            raise ValueError(f"policy_idx must be between 0 and 6, got {policy_idx}")
        
        # Clear current policy
        if self.current_policy is not None:
            del self.current_policy
            self.current_policy = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Load new policy
        try:
            checkpoint = torch.load(self.policy_paths[policy_idx], map_location=self.device, weights_only=False)
            policy_cfg = OmegaConf.load(self.policy_cfg_paths[policy_idx])
            policy = hydra.utils.instantiate(policy_cfg.policy)
            policy.load_state_dict(checkpoint['state_dicts']['model'])
            policy.eval().to(self.device)
            self.current_policy = policy
            
            # Setup normalizer
            if self.normalizer_paths:
                normalizer = LinearNormalizer()
                normalizer.load_state_dict(torch.load(self.normalizer_paths[policy_idx], map_location=self.device))
            else:
                dataset: BaseImageDataset = hydra.utils.instantiate(policy_cfg.task.dataset)
                normalizer = dataset.get_normalizer().to(self.device)
            self.current_policy.set_normalizer(normalizer)
        
        except Exception as e:
            raise RuntimeError(f"Failed to load policy {policy_idx}: {str(e)}")

    def reset(self):
        """Reset controller state and clear current policy model."""
        if self._eposide_num > 0:
            success_rate = self.get_success_rate()
            self.print_success_rate(success_rate)
        
        super().reset()
        self.obs_history_dict = {
            cam_name: deque(maxlen=self.n_obs_steps)
            for cam_name in self.cfg.cameras_names
        }
        self.obs_history_pose = deque(maxlen=self.n_obs_steps)
        self.current_policy_idx = 0
        self.trajectory_controller.reset()
        self.frame_count = 0
        self._eposide_num += 1

        self.model_1_loaded = False
        self.model_2_loaded = False
        self.model_3_loaded = False
        self.model_4_loaded = False
        self.model_5_loaded = False
        self.model_6_loaded = False
        self.model_7_loaded = False
        
        # Clear current policy
        if self.current_policy is not None:
            del self.current_policy
            self.current_policy = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def step(self, state):
        """Execute one control step."""
        self.current_beaker_pos = None
        self.target_pos = None
        
        # Check policy transition conditions
        if self.current_policy_idx == 0:
            if not self.model_1_loaded:
                self.load_policy_model(0)
                self.model_1_loaded = True
            # 1. Pick beaker2
            self.frame_count += 1
            self.current_beaker_pos = self.object_utils.get_object_xform_position(object_path=self.cfg.beaker_sub_2)
            if self.current_beaker_pos is not None and self.current_beaker_pos[2] > 0.89:
                self.trajectory_controller.reset()
                self.success_episode[self.current_policy_idx] += 1
                self.current_policy_idx = 1
                self.frame_count = 0
            elif self.frame_count > 2000:
                self.frame_count = 0
                self.reset_needed = True

        elif self.current_policy_idx == 1:
            if not self.model_2_loaded:
                self.load_policy_model(1)
                self.model_2_loaded = True
            # 2. Pour beaker2 to beaker1
            self.frame_count += 1
            self.current_beaker_pos = self.object_utils.get_object_xform_position(object_path=self.cfg.beaker_sub_2)
            self.target_pos = self.object_utils.get_object_xform_position(object_path=self.cfg.beaker_sub_1)
            if (self.current_beaker_pos is not None and self.target_pos is not None and
                self.current_beaker_pos[2] <= 1.12 and
                abs(self.current_beaker_pos[0] - self.target_pos[0]) < 0.05 and
                abs(self.current_beaker_pos[1] - self.target_pos[1]) < 0.05 and
                self.frame_count > 1100):
                self.success_episode[self.current_policy_idx] += 1
                self.current_policy_idx = 2
                self.frame_count = 0
            elif self.frame_count > 2000:
                self.frame_count = 0
                self.reset_needed = True
            
        elif self.current_policy_idx == 2:
            if self.model_3_loaded:
                self.load_policy_model(2)
                self.model_3_loaded = True
            # 3. Place beaker2 to plat2
            self.frame_count += 1
            self.current_beaker_pos = self.object_utils.get_object_xform_position(object_path=self.cfg.beaker_sub_2)
            self.target_pos = self.object_utils.get_object_position(object_path=self.cfg.plat_2)
            if (self.current_beaker_pos is not None and self.target_pos is not None and
                self.current_beaker_pos[2] <= 0.76 and
                abs(self.current_beaker_pos[0] - self.target_pos[0]) < 0.15 and
                abs(self.current_beaker_pos[1] - self.target_pos[1]) < 0.15 and
                self.frame_count > 520):
                self.success_episode[self.current_policy_idx] += 1
                self.current_policy_idx = 3
                self.frame_count = 0
            elif self.frame_count > 2000:
                self.frame_count = 0
                self.reset_needed = True

        elif self.current_policy_idx == 3:
            if self.model_4_loaded:
                self.load_policy_model(3)
                self.model_4_loaded = True
            # 4. Pick beaker1
            self.frame_count += 1
            self.current_beaker_pos = self.object_utils.get_object_xform_position(object_path=self.cfg.beaker_sub_1)
            if self.current_beaker_pos is not None and self.current_beaker_pos[2] > 0.82 and self.frame_count > 350:
                self.success_episode[self.current_policy_idx] += 1
                self.current_policy_idx = 4
                self.frame_count = 0
            elif self.frame_count > 2000:
                self.frame_count = 0
                self.reset_needed = True

        elif self.current_policy_idx == 4:
            if self.model_5_loaded:
                self.load_policy_model(4)
                self.model_5_loaded = True
            # 5. Shake beaker1
            self.frame_count += 1
            self.current_beaker_pos = self.object_utils.get_object_xform_position(object_path=self.cfg.beaker_sub_1)
            if self.current_beaker_pos is not None and self.current_beaker_pos[2] >= 0.87 and self.frame_count > 700:
                self.success_episode[self.current_policy_idx] += 1
                self.current_policy_idx = 5
                self.frame_count = 0
            elif self.frame_count > 2000:
                self.frame_count = 0
                self.reset_needed = True

        elif self.current_policy_idx == 5:
            if not self.model_6_loaded:
                self.load_policy_model(5)
                self.model_6_loaded = True
            # 6. Pour beaker1 to target beaker
            self.frame_count += 1
            self.current_beaker_pos = self.object_utils.get_object_xform_position(object_path=self.cfg.beaker_sub_1)
            self.target_pos = self.object_utils.get_object_xform_position(object_path=self.cfg.target_sub_beaker)
            if (self.current_beaker_pos is not None and self.target_pos is not None and
                self.current_beaker_pos[2] <= 1.1 and
                abs(self.current_beaker_pos[0] - self.target_pos[0]) < 0.05 and
                abs(self.current_beaker_pos[1] - self.target_pos[1]) < 0.05 and
                self.frame_count > 1800):
                self.success_episode[self.current_policy_idx] += 1
                self.current_policy_idx = 6
                self.frame_count = 0
            elif self.frame_count > 2000:
                self.frame_count = 0
                self.reset_needed = True

        elif self.current_policy_idx == 6:
            if self.model_7_loaded:
                self.load_policy_model(6)
                self.model_7_loaded = True
            # 7. Final action
            self.frame_count += 1
            if self.frame_count > 600:
                self.current_policy_idx = 7
                self.frame_count = 0
            elif self.frame_count > 2000:
                self.frame_count = 0
                self.reset_needed = True
        
        # Update observation history
        for cam_name, image in state['camera_data'].items():
            self.obs_history_dict[cam_name].append(image)
        self.obs_history_pose.append(state['joint_positions'][:-1])
        
        # Check if observation history is complete
        histories_complete = (
            len(self.obs_history_pose) == self.n_obs_steps and
            all(len(hist) == self.n_obs_steps for hist in self.obs_history_dict.values())
        )
        
        if self.trajectory_controller.is_trajectory_complete() and histories_complete and self.current_policy_idx < 7:
            # Prepare model input
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
            
            # Handle DP and ACT input formats
            if self.n_obs_steps != 1:
                for camera_name in self.obs_history_dict.keys():
                    obs_dict[camera_name] = obs_dict[camera_name].unsqueeze(0).to(self.device)
                obs_dict['agent_pose'] = obs_dict['agent_pose'].unsqueeze(0).to(self.device)
            else:
                for camera_name in self.obs_history_dict.keys():
                    obs_dict[camera_name] = obs_dict[camera_name].to(self.device)
                obs_dict['agent_pose'] = obs_dict['agent_pose'].to(self.device)
            
            # Perform inference with current policy
            if self.current_policy is not None:
                with torch.no_grad():
                    prediction = self.current_policy.predict_action(obs_dict)
                    joint_positions = prediction['action'][0].cpu().numpy()
                
                # Generate new trajectory
                self.trajectory_controller.generate_trajectory(joint_positions)
            else:
                raise RuntimeError(f"No policy loaded for policy_idx {self.current_policy_idx}")
        
        # Get next action
        action = self.trajectory_controller.get_next_action()
        
        if self.n_obs_steps != 1 and action is not None:
            if len(action.joint_positions) == 9:
                if action.joint_positions[7] is not None or action.joint_positions[8] is not None:
                    if action.joint_positions[7] - np.float32(0.015) < np.float32(0.015):
                        action.joint_positions[7] = np.float32(0.015)
                        action.joint_positions[8] = np.float32(0.015)
        
        return action, False, self.is_success()

    def is_done(self):
        """Check if the task is complete."""
        if self.current_policy_idx < 6:
            return False
        return self.trajectory_controller.is_trajectory_complete()

    def is_success(self):
        """Check if the task was successfully completed."""
        if self.current_policy_idx < 6:
            return False
        
        beaker1_pos = self.object_utils.get_object_xform_position(object_path=self.cfg.beaker_sub_1)
        beaker2_pos = self.object_utils.get_object_xform_position(object_path=self.cfg.beaker_sub_2)
        plat_1_position = self.object_utils.get_object_position(object_path=self.cfg.plat_1)
        plat_2_position = self.object_utils.get_object_position(object_path=self.cfg.plat_2)
        
        success = (
            beaker1_pos is not None and
            abs(beaker1_pos[0] - plat_1_position[0]) < 0.2 and
            abs(beaker1_pos[1] - plat_1_position[1]) < 0.2 and
            beaker1_pos[2] <= 0.78 and
            beaker2_pos is not None and
            abs(beaker2_pos[0] - plat_2_position[0]) < 0.2 and
            abs(beaker2_pos[1] - plat_2_position[1]) < 0.2 and
            beaker2_pos[2] <= 0.78
        )
        
        if success and self.current_policy_idx == 7:
            self.success_episode[self.current_policy_idx-1] += 1
            self.reset_needed = True
            return True
        return False

    def eposide_num(self):
        """Return the current episode number."""
        return self._eposide_num

    def get_success_rate(self):
        """Calculate success rate for each policy."""
        episode_num = self._eposide_num
        success_rate = []
        for per_success_episode in self.success_episode:
            per_success_rate = per_success_episode / episode_num * 100 if episode_num > 0 else 0
            success_rate.append(per_success_rate)
        return success_rate

    def print_success_rate(self, success_rate):
        """Print success rate for each policy."""
        for idx, per_success_rate in enumerate(success_rate, start=1):
            print(f"Sequence {idx}: success rate: {per_success_rate:.2f}% ({self.success_episode[idx-1]}/{self._eposide_num})")