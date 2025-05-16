from omni.isaac.franka.controllers.rmpflow_controller import RMPFlowController
import numpy as np
import torch
import hydra
from collections import deque
from omegaconf import OmegaConf
from policy.model.common.normalizer import LinearNormalizer

from controllers.atomic_actions.open_controller import OpenController
from .base_controller import BaseController
from .robot_controllers.trajectory_controller import FrankaTrajectoryController
from factories.collector_factory import create_collector
from omni.isaac.core.utils.numpy.rotations import euler_angles_to_quats

class OpenTaskController(BaseController):
    """Controller for managing the task of opening a door in collect or infer mode.

    Args:
        cfg: Configuration object containing mode and other parameters.
        robot: Robot articulation instance.
    """

    def __init__(self, cfg, robot):
        super().__init__(cfg, robot)
        self.mode = cfg.mode  # "collect" or "infer"
        self.REQUIRED_SUCCESS_STEPS = 120
        self.success_counter = 0
        
        if self.mode == "collect":
            self._init_collect_mode(cfg, robot)
        elif self.mode == "infer":
            self._init_infer_mode(cfg, robot)
        else:
            raise ValueError(f"Invalid mode: {self.mode}. Expected 'collect' or 'infer'.")
            
        self.initial_handle_position = None
        self.state = None
            
    def _init_collect_mode(self, cfg, robot):
        """Initializes the controller for data collection mode.

        Args:
            cfg: Configuration object for collect mode.
            robot: Robot articulation instance.
        """
        rmp_controller = RMPFlowController(
            name="target_follower_controller",
            robot_articulation=robot
        )
        
        self.open_controller = OpenController(
            name="open_controller",
            cspace_controller=rmp_controller,
            gripper=robot.gripper,
            events_dt=[0.0025, 0.005, 0.08, 0.002, 0.05, 0.05, 0.01, 0.008],
            furniture_type="door",
            door_open_direction="clockwise"
        )
        
        self.data_collector = create_collector(
            cfg.collector.type,
            camera_configs=cfg.cameras,
            save_dir=cfg.multi_run.run_dir,
            max_episodes=cfg.max_episodes,
            compression=cfg.collector.compression
        )

    def _init_infer_mode(self, cfg, robot):
        """Initializes the controller for inference mode using a trained policy.

        Args:
            cfg: Configuration object for infer mode.
            robot: Robot articulation instance.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.checkpoint = torch.load(cfg.infer.policy_model_path, map_location=self.device)
        self.config = OmegaConf.load(cfg.infer.policy_config_path)
        
        self.policy = hydra.utils.instantiate(self.config.policy)
        self.policy.load_state_dict(self.checkpoint['state_dicts']['model'])
        self.policy.eval()
        self.policy.to(self.device)

        if hasattr(cfg.infer, 'normalizer_path'):
            normalizer = LinearNormalizer()
            normalizer.load_state_dict(torch.load(cfg.infer.normalizer_path, map_location=self.device))
        else:
            dataset = hydra.utils.instantiate(self.config.task.dataset)
            normalizer = dataset.get_normalizer()
        normalizer.to(self.device)
        self.policy.set_normalizer(normalizer)
        self.obs_names = cfg.infer.obs_names  
        self.camera_to_obs = {k: v for k, v in self.obs_names.items()}  
        self.n_obs_steps = self.config.n_obs_steps
        self.obs_history_dict = {
            obs_key: deque(maxlen=self.n_obs_steps) 
            for obs_key in self.obs_names.values()
        }
        self.obs_history_pose = deque(maxlen=self.n_obs_steps)
        
        self.trajectory_controller = FrankaTrajectoryController(
            name="trajectory_controller",
            robot_articulation=robot
        )

    def reset(self):
        """Resets the controller to its initial state."""
        super().reset()
        self.initial_handle_position = None
        self.success_counter = 0
        if self.mode == "collect":
            self.open_controller.reset()
        else:
            self.obs_history_dict = {
                obs_key: deque(maxlen=self.n_obs_steps) 
                for obs_key in self.obs_names.values()
            }
            self.obs_history_pose = deque(maxlen=self.n_obs_steps)
            self.trajectory_controller.reset()

    def step(self, state):
        """Executes one step of the task based on the current state.

        Args:
            state: Current state of the environment.

        Returns:
            Tuple containing the action, done flag, and success flag.
        """
        if self.initial_handle_position is None:
            self.initial_handle_position = state['object_position']
        if self.mode == "collect":
            return self._step_collect(state)
        else:
            return self._step_infer(state)

    def _step_collect(self, state):
        """Executes a step in collect mode using the open controller.

        Args:
            state: Current state of the environment.

        Returns:
            Tuple containing the action, done flag, and success flag.
        """
        if not self.open_controller.is_done():
            action = self.open_controller.forward(
                handle_position=state['object_position'],
                current_joint_positions=state['joint_positions'],
                revolute_joint_position=state['revolute_joint_position'],
                gripper_position=state['gripper_position'],
                end_effector_orientation=euler_angles_to_quats([0, 110, 0], degrees=True, extrinsic=False),
            )
            
            if 'camera_data' in state:
                self.data_collector.cache_step(
                    camera_images=state['camera_data'],
                    joint_angles=state['joint_positions'][:-1]
                )
            
            if self._check_success(state):
                self.success_counter += 1
            else:
                self.success_counter = 0
                
            return action, False, False

        success = self.success_counter >= self.REQUIRED_SUCCESS_STEPS
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
        """Executes a step in infer mode using the trained policy.

        Args:
            state: Current state of the environment.

        Returns:
            Tuple containing the action, done flag, and success flag.
        """
        for cam_name, image in state['camera_data'].items():
            if cam_name in self.camera_to_obs:
                obs_key = self.camera_to_obs[cam_name]
                self.obs_history_dict[obs_key].append(image)
                
        self.obs_history_pose.append(state['joint_positions'][:-1])
        
        histories_complete = (
            len(self.obs_history_pose) == self.n_obs_steps and
            all(len(hist) == self.n_obs_steps for hist in self.obs_history_dict.values())
        )
        
        if self.trajectory_controller.is_trajectory_complete() and histories_complete:
            obs_dict = {
                obs_key: torch.from_numpy(np.stack(list(hist))).float().to(self.device) / 255.0
                for obs_key, hist in self.obs_history_dict.items()
            }
            obs_dict['agent_pose'] = torch.from_numpy(
                np.stack(list(self.obs_history_pose))
            ).float().to(self.device)
            
            for key in obs_dict.keys():
                if obs_dict[key].shape[0] != 1:
                    obs_dict[key] = obs_dict[key].unsqueeze(0)
            if obs_dict['agent_pose'].shape[0] != 1:
                obs_dict['agent_pose'] = obs_dict['agent_pose'].unsqueeze(0)
                
            with torch.no_grad():
                prediction = self.policy.predict_action(obs_dict)
                joint_positions = prediction['action'][0].cpu().numpy()
            
            self.trajectory_controller.generate_trajectory(joint_positions)
            
        action = self.trajectory_controller.get_next_action()
        
        if self._check_success(state):
            self.success_counter += 1
        else:
            self.success_counter = 0
            
        success = self.success_counter >= self.REQUIRED_SUCCESS_STEPS
        if success:
            print("Task success!")
            self._last_success = True
            self.reset_needed = True
            return None, True, True
            
        return action, False, False
        
    def _check_success(self, state):
        """Checks if the task has been successfully completed.

        Args:
            state: Current state of the environment.

        Returns:
            bool: True if the task is successful, False otherwise.
        """
        current_pos = state['object_position']
        gripper_position = state['gripper_position']
        
        if current_pos is None or self.initial_handle_position is None or gripper_position is None:
            return False
        
        end_effector_distance = abs(np.linalg.norm(np.array(gripper_position) - np.array(current_pos)))
        distance = abs(np.linalg.norm(np.array(current_pos) - self.initial_handle_position))
        
        return distance > 0.13 and end_effector_distance > 0.04
    
    def is_done(self):
        """Checks if the task is complete.

        Returns:
            bool: True if the task is done, False otherwise.
        """
        if self.mode == "collect":
            return self.open_controller.is_done()
        return self.reset_needed

    def is_success(self):
        """Checks if the last task attempt was successful.

        Returns:
            bool: True if the last task was successful, False otherwise.
        """
        return self._last_success

    def close(self):
        """Closes the data collector in collect mode."""
        if self.mode == "collect":
            self.data_collector.close()

    def episode_num(self):
        """Returns the current episode number.

        Returns:
            int: Number of episodes completed.
        """
        if self.mode == "collect":
            return self.data_collector.episode_count
        return self._episode_num