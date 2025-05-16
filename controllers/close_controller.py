from controllers.atomic_actions.close_controller import CloseController
from controllers.robot_controllers.grapper_manager import Gripper
from controllers.robot_controllers.trajectory_controller import FrankaTrajectoryController
from factories.collector_factory import create_collector
from .base_controller import BaseController
import numpy as np
from omni.isaac.franka.controllers.rmpflow_controller import RMPFlowController
from scipy.spatial.transform import Rotation as R
import torch
import hydra
from collections import deque
from omegaconf import OmegaConf
from policy.model.common.normalizer import LinearNormalizer

class CloseTaskController(BaseController):
    def __init__(self, cfg, robot):
        super().__init__(cfg, robot)
        self.mode = cfg.mode  
        
        if self.mode == "collect":
            self._init_collect_mode(cfg, robot)
        else:
            self._init_infer_mode(cfg, robot)
            
        self.initial_handle_position = None
            
    def _init_collect_mode(self, cfg, robot):
        
        rmp_controller = RMPFlowController(
            name="target_follower_controller",
            robot_articulation=robot
        )
        
        self.close_controller = CloseController(
            name="close_controller",
            cspace_controller=rmp_controller,
            gripper=robot.gripper,
            events_dt=[0.008, 0.02, 0.008, 0.01, 0.05, 0.01]
        )
        
        self.gripper_control = Gripper()
        
        self.data_collector = create_collector(
            cfg.collector.type,
            camera_configs=cfg.cameras,
            save_dir=cfg.multi_run.run_dir,
            max_episodes=cfg.max_episodes,
            compression=cfg.collector.compression
        )

    def _init_infer_mode(self, cfg, robot):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        
        self.checkpoint = torch.load(cfg.policy_model_path, map_location=self.device)
        self.config = OmegaConf.load(cfg.policy_config_path)
        
        self.policy = hydra.utils.instantiate(self.config.policy)
        self.policy.load_state_dict(self.checkpoint['state_dicts']['model'])
        self.policy.eval()
        self.policy.to(self.device)

        
        if hasattr(cfg, 'normalizer_path'):
            normalizer = LinearNormalizer()
            normalizer.load_state_dict(torch.load(cfg.normalizer_path, map_location=self.device))
        else:
            dataset = hydra.utils.instantiate(self.config.task.dataset)
            normalizer = dataset.get_normalizer()
        normalizer.to(self.device)
        self.policy.set_normalizer(normalizer)

        self.n_obs_steps = self.config.n_obs_steps
        self.obs_history_dict = {
            cam_name: deque(maxlen=self.n_obs_steps) 
            for cam_name in self.cfg.cameras_names
        }
        self.obs_history_pose = deque(maxlen=self.n_obs_steps)
        
        self.trajectory_controller = FrankaTrajectoryController(
            name="trajectory_controller",
            robot_articulation=robot
        )

    def reset(self):
        
        super().reset()
        self.initial_handle_position = None
        if self.mode == "collect":
            self.close_controller.reset()
        else:
            self.obs_history_dict = {
                cam_name: deque(maxlen=self.n_obs_steps) 
                for cam_name in self.cfg.cameras_names
            }
            self.obs_history_pose = deque(maxlen=self.n_obs_steps)
            self.trajectory_controller.reset()

    def step(self, state):
        if self.initial_handle_position is None:
            self.initial_handle_position = state['object_position']
        if self.mode == "collect":
            return self._step_collect(state)
        else:
            return self._step_infer(state)

    def _step_collect(self, state):
        if not self.close_controller.is_done():
            action = self.close_controller.forward(
                handle_position=state['object_position'],
                current_joint_positions=state['joint_positions'],
                end_effector_orientation=np.array([0.50434, 0.49562, 0.50434, 0.49562])
            )
            
            if 'camera_data' in state:
                self.data_collector.cache_step(
                    camera_images=state['camera_data'],
                    joint_angles=state['joint_positions'][:-1]
                )
            
            return action, False, False

        success = self._check_success()
        if success:
            print("Task success!")
            self.data_collector.write_cached_data(state['joint_positions'][:-1])
            self._last_success = True
        else:
            print("Task failed!")
            self.data_collector.clear_cache()
            self._last_success = False
            
        self.reset_needed = True
        return None, True, success

    def _step_infer(self, state):
        
        for cam_name, image in state['camera_data'].items():
            self.obs_history_dict[cam_name].append(image)
        self.obs_history_pose.append(state['joint_positions'][:-1])
        
        histories_complete = (
            len(self.obs_history_pose) == self.n_obs_steps and
            all(len(hist) == self.n_obs_steps for hist in self.obs_history_dict.values())
        )
        
        if self.trajectory_controller.is_trajectory_complete() and histories_complete:
            obs_dict = {
                cam_name: torch.from_numpy(np.stack(list(hist))).float().to(self.device) / 255.0
                for cam_name, hist in self.obs_history_dict.items()
            }
            obs_dict['agent_pose'] = torch.from_numpy(
                np.stack(list(self.obs_history_pose))
            ).float().to(self.device)
            
            for cam_name in obs_dict.keys():
                if obs_dict[cam_name].shape[0] != 1:
                    obs_dict[cam_name] = obs_dict[cam_name].unsqueeze(0)
            if obs_dict['agent_pose'].shape[0] != 1:
                obs_dict['agent_pose'] = obs_dict['agent_pose'].unsqueeze(0)
                
            with torch.no_grad():
                prediction = self.policy.predict_action(obs_dict)
                joint_positions = prediction['action'][0].cpu().numpy()
            
            self.trajectory_controller.generate_trajectory(joint_positions)
            
        action = self.trajectory_controller.get_next_action()
        
        success = self._check_success()
        if success:
            print("Task success!")
            self._last_success = True
            self.reset_needed = True
            return None, True, True
            
        return action, False, False
        
    def _check_success(self):
        
        current_pos = self.object_utils.get_object_xform_position(
            object_path=self.cfg.sub_obj_path
        )
        
        end_effector_pos = self.object_utils.get_object_xform_position(
            object_path="/World/Franka/panda_hand/tool_center"
        )
                
        if current_pos is None or self.initial_handle_position is None or end_effector_pos is None:
            return False
            
        distance = abs(np.linalg.norm(np.array(current_pos) - np.array(self.initial_handle_position)))
        
        
        end_effector_distance = abs(np.linalg.norm(np.array(end_effector_pos) - np.array(current_pos)))
            
        return distance > 0.2 and end_effector_distance > 0.05
        
    def is_done(self):
        
        if self.mode == "collect":
            return self.close_controller.is_done()
        return self.reset_needed
        
    def _check_success(self):
        
        return self._last_success

    def episode_num(self):
        
        if self.mode == "collect":
            return self.data_collector.episode_count
        return self._episode_num
