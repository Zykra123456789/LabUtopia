import torch
import hydra
import numpy as np
from omegaconf import OmegaConf
from collections import deque
from policy.model.common.normalizer import LinearNormalizer
from omni.isaac.franka.controllers.rmpflow_controller import RMPFlowController
from scipy.spatial.transform import Rotation as R

from .base_controller import BaseController
from .atomic_actions.pick_controller import PickController
from .robot_controllers.grapper_manager import Gripper
from .robot_controllers.trajectory_controller import FrankaTrajectoryController
from factories.collector_factory import create_collector

class PickTaskController(BaseController):
    """
    Controller for pick-and-place tasks with two operation modes:
    - Collection mode: Gathers training data through demonstrations
    - Inference mode: Executes learned policies for autonomous picking

    Attributes:
        mode (str): Operation mode ("collect" or "infer")
        REQUIRED_SUCCESS_STEPS (int): Number of consecutive steps needed for success
        success_counter (int): Counter for tracking successful steps
    """
    def __init__(self, cfg, robot):
        super().__init__(cfg, robot)
        self.mode = cfg.mode  # "collect" or "infer"
        self.initial_position = None
        self.REQUIRED_SUCCESS_STEPS = 120
        self.success_counter = 0
        
        if self.mode == "collect":
            self._init_collect_mode(cfg, robot)
        else:
            self._init_infer_mode(cfg, robot)
            
    def _init_collect_mode(self, cfg, robot):
        """
        Initializes components for data collection mode.
        Sets up pick controller, gripper control, and data collector.

        Args:
            cfg: Configuration object containing collection settings
            robot: Robot instance to control
        """
        self.pick_controller = PickController(
            name="pick_controller",
            cspace_controller=RMPFlowController(
                name="target_follower_controller",
                robot_articulation=robot
            ),
            events_dt=[0.004, 0.002, 0.01, 0.2, 0.05, 0.004, 0.008]
        )
        self.gripper_control = Gripper()
        self.data_collector = create_collector(
            cfg.collector.type,
            camera_configs=cfg.cameras,
            save_dir=cfg.multi_run.run_dir,
            max_episodes=cfg.max_episodes,
            compression=cfg.collector.compression
        )

    def reset(self):
        super().reset()
        if self.mode == "collect":
            self.pick_controller.reset()
        else:
            self.obs_history_dict = {
                cam_name: deque(maxlen=self.n_obs_steps) 
                for cam_name in self.cfg.cameras_names
            }
            self.obs_history_pose = deque(maxlen=self.n_obs_steps)
            self.trajectory_controller.reset()
        self.initial_position = None
        self.success_counter = 0
    
    def step(self, state):
        if self.initial_position is None:
            self.initial_position = state['object_position']
        self.state = state
        if self.mode == "collect":
            return self._step_collect(state)
        else:
            return self._step_infer(state)
            
    def _check_success(self):
        return self.state['object_position'][2] > self.initial_position[2] + 0.1

    def _init_infer_mode(self, cfg, robot):
        """
        Initializes components for inference mode.
        Loads policy model, normalizer, and sets up trajectory controller.

        Args:
            cfg: Configuration object containing model paths and settings
            robot: Robot instance to control
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
        
    def _step_collect(self, state):
        """
        Executes one step in collection mode.
        Records demonstrations and manages episode transitions.

        Args:
            state (dict): Current environment state

        Returns:
            tuple: (action, done, success) indicating control output and episode status
        """
        if self._check_success():
            self.success_counter += 1
        else:
            self.success_counter = 0
        
        if not self.pick_controller.is_done():
            action = self.pick_controller.forward(
                picking_position=state['object_position'],
                current_joint_positions=state['joint_positions'],
                object_size=state['object_size'],
                object_name=state['object_name'],
                gripper_control=self.gripper_control,
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 10])).as_quat(),
                gripper_position=state['gripper_position'],
            )
            
            if 'camera_data' in state:
                self.data_collector.cache_step(
                    camera_images=state['camera_data'],
                    joint_angles=state['joint_positions'][:-1]
                )
            
            return action, False, False
        
        self._last_success = self.success_counter >= self.REQUIRED_SUCCESS_STEPS
        if self._last_success:
            self.data_collector.write_cached_data(state['joint_positions'][:-1])
            self.reset_needed = True
            return None, True, True

        self.data_collector.clear_cache()
        self._last_success = False
        self.reset_needed = True
        return None, True, False
        
    def _step_infer(self, state):
        """
        Executes one step in inference mode.
        Processes observations and generates control actions using learned policy.

        Args:
            state (dict): Current environment state

        Returns:
            tuple: (action, done, success) indicating control output and episode status
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
            
            self.trajectory_controller.generate_trajectory(joint_positions[0:120,:])
            
        action = self.trajectory_controller.get_next_action()
        
        if self._check_success():
            self.success_counter += 1
        else:
            self.success_counter = 0
            
        self._last_success = self.success_counter >= self.REQUIRED_SUCCESS_STEPS
        if self._last_success:
            self.reset_needed = True
            return action, True, True
        return action, False, False

    def episode_num(self):
        if self.mode == "collect":
            return self.data_collector.episode_count
        return self._episode_num
