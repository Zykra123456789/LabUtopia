import torch
import hydra
import numpy as np
from omegaconf import OmegaConf
from collections import deque
from enum import Enum
from policy.model.common.normalizer import LinearNormalizer
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.franka.controllers.rmpflow_controller import RMPFlowController

from .base_controller import BaseController
from .atomic_actions.open_controller import OpenController
from .atomic_actions.close_controller import CloseController
from .robot_controllers.grapper_manager import Gripper
from .robot_controllers.trajectory_controller import FrankaTrajectoryController
from factories.collector_factory import create_collector

class Phase(Enum):
    OPENING = "opening"
    CLOSING = "closing"
    FINISHED = "finished"

class OpenCloseController(BaseController):
    def __init__(self, cfg, robot):
        super().__init__(cfg, robot)
        self.mode = cfg.mode  # "collect" or "infer"
        self.initial_handle_position = None  
        self.success_count = 0  
        self.total_episodes = 0  
        
        if self.mode == "collect":
            self._init_collect_mode(cfg, robot)
        else:
            self._init_infer_mode(cfg, robot)
            
    def _init_collect_mode(self, cfg, robot):
        
        rmp_controller = RMPFlowController(
            name="target_follower_controller",
            robot_articulation=robot
        )
        
        self.open_controller = OpenController(
            name="open_controller",
            cspace_controller=rmp_controller,
            gripper=robot.gripper,
            events_dt=[0.005, 0.015, 0.08, 0.01, 0.05, 0.05, 0.08]
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
        self.current_phase = Phase.OPENING
        self.active_controller = self.open_controller

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
        self.current_phase = Phase.OPENING

    def reset(self):
        super().reset()
        self.current_phase = Phase.OPENING
        
        self.initial_handle_position = None
        if self.mode == "collect":
            self.active_controller = self.open_controller
            self.active_controller.reset()
        else:
            self.obs_history_dict = {
                cam_name: deque(maxlen=self.n_obs_steps) 
                for cam_name in self.cfg.cameras_names
            }
            self.obs_history_pose = deque(maxlen=self.n_obs_steps)
            self.trajectory_controller.reset()

    def _check_phase_success(self):
        
        current_pos = self.object_utils.get_object_position(
            object_path=self.cfg.sub_obj_path
        )
        end_effector_pos = self.object_utils.get_object_xform_position(
            object_path="/World/Franka/panda_hand/tool_center"
        )
        
        if current_pos is None or self.initial_handle_position is None or end_effector_pos is None:
            return False
            
        if self.current_phase == Phase.OPENING:
            
            distance = abs(np.linalg.norm(np.array(current_pos) - np.array(self.initial_handle_position)))
            return distance > 0.1
        elif self.current_phase == Phase.CLOSING:
            
            
            handle_distance = abs(np.linalg.norm(np.array(current_pos) - np.array(self.initial_handle_position)))
            
            end_effector_distance = abs(np.linalg.norm(np.array(end_effector_pos) - np.array(current_pos)))
            
            return handle_distance < 0.02 and abs(end_effector_distance - 0.05) < 0.01
        return False

    def step(self, state):
        if self.initial_handle_position is None:
            self.initial_handle_position = state['object_position']
        if self.mode == "collect":
            return self._step_collect(state)
        else:
            return self._step_infer(state)

    def _step_collect(self, state):
        if self.current_phase == Phase.FINISHED:
            return None, True, self._last_success

        if not self.active_controller.is_done():
            action = self.active_controller.forward(
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

        success = self._check_phase_success()
        if success:
            if self.current_phase == Phase.OPENING:
                print("Open task success! Switching to close...")
                self.current_phase = Phase.CLOSING
                self.active_controller = self.close_controller
                self.active_controller.reset()
                return None, False, False
            elif self.current_phase == Phase.CLOSING:
                print("Close task success!")
                self.data_collector.write_cached_data(state['joint_positions'][:-1])
                self._last_success = True
                self.reset_needed = True
                self.current_phase = Phase.FINISHED
                return None, True, True
        
        print(f"{self.current_phase.value} task failed!")
        self.data_collector.clear_cache()
        self._last_success = False
        self.reset_needed = True
        self.current_phase = Phase.FINISHED
        return None, True, False

    def _step_infer(self, state):
        if self.current_phase == Phase.FINISHED:
            self.reset_needed = True
            return None, True, self._last_success

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
            
            self.trajectory_controller.generate_trajectory(joint_positions[0:120,:])
            
        action = self.trajectory_controller.get_next_action()
        
        success = self._check_phase_success()
        if success and self.current_phase == Phase.OPENING:
            print("Open task success! Switching to close...")
            self.current_phase = Phase.CLOSING
            self.trajectory_controller.reset()
        elif success and self.current_phase == Phase.CLOSING:
            print("Close task success!")
            self._last_success = True
            self.current_phase = Phase.FINISHED
            return None, True, True
            
        return action, False, False

    def is_done(self):
        return self.current_phase == Phase.FINISHED

    def is_success(self):
        return self._last_success

    def close(self):
        if self.mode == "collect":
            self.data_collector.close()

    def episode_num(self):
        if self.mode == "collect":
            return self.data_collector.episode_count
        return self._episode_num
