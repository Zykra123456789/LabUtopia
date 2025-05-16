from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.franka.controllers.rmpflow_controller import RMPFlowController
from scipy.spatial.transform import Rotation as R
import numpy as np
import os
import torch
import hydra
from collections import deque
from omegaconf import OmegaConf
from enum import Enum
from policy.model.common.normalizer import LinearNormalizer
from utils.task_utils import TaskUtils

from .atomic_actions.pick_controller import PickController
from .atomic_actions.pour_controller import PourController
from .base_controller import BaseController
from .robot_controllers.grapper_manager import Gripper
from .robot_controllers.trajectory_controller import FrankaTrajectoryController
from factories.collector_factory import create_collector


class Phase(Enum):
    PICKING = "picking"
    POURING = "pouring"
    FINISHED = "finished"

class PourTaskController(BaseController):
    def __init__(self, cfg, robot):
        """Initialize the pick and pour task controller.
        
        Args:
            cfg: Configuration object containing controller settings
            robot: Robot instance to control
        """
        super().__init__(cfg, robot)
        self.mode = cfg.mode
        self.success_count = 0
        self.total_episodes = 0
        self.initial_position = None
        self.initial_size = None
        self.task_utils = TaskUtils.get_instance()
        self.initial_quaternion = None
        self.pour_timer = 0
        self.pour_complete = False
        self.return_complete = False
        self.return_timer = 0
        self.frame_count = 0
        
        if self.mode == "collect":
            self._init_collect_mode(cfg, robot)
        elif self.mode == "infer":
            self._init_infer_mode(cfg, robot)
            
        self.current_phase = Phase.PICKING
            
    def _init_collect_mode(self, cfg, robot):
        """Initialize controller for data collection mode."""
        rmp_controller = RMPFlowController(
            name="target_follower_controller",
            robot_articulation=robot
        )
        
        self.pick_controller = PickController(
            name="pick_controller",
            cspace_controller=rmp_controller,
            events_dt=[0.002, 0.002, 0.005, 0.2, 0.05, 0.01, 0.1]
        )
        
        self.pour_controller = PourController(
            name="pour_controller",
            cspace_controller=rmp_controller,
            events_dt=[0.006, 0.005, 0.009, 0.005, 0.009, 0.5]
        )
        
        self.gripper_control = Gripper()
        self.active_controller = self.pick_controller
        
        self.data_collector = create_collector(
            cfg.collector.type,
            camera_configs=cfg.cameras,
            save_dir=cfg.multi_run.run_dir,
            max_episodes=cfg.max_episodes,
            compression=cfg.collector.compression
        )

    def _init_infer_mode(self, cfg, robot):
        """Initialize controller for inference mode."""
        
        self.pick_controller = PickController(
            name="pick_controller",
            cspace_controller=RMPFlowController(
                name="target_follower_controller",
                robot_articulation=robot
            ),
            gripper=robot.gripper,
            events_dt=[0.002, 0.002, 0.005, 0.2, 0.05, 0.01, 0.1]
        )
        self.gripper_control = Gripper()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.checkpoint = torch.load(cfg.infer.policy_model_path, map_location=self.device, weights_only=False)
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

        # Initialize observation history
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
        """Reset controller state and phase."""
        super().reset()
        self.current_phase = Phase.PICKING
        self.initial_position = None
        self.initial_size = None
        self.initial_quaternion = None
        self.pour_timer = 0
        self.pour_complete = False
        self.return_complete = False
        self.return_timer = 0
        self.frame_count = 0
        
        if self.mode == "collect":
            self.active_controller = self.pick_controller
            self.pick_controller.reset()
            self.pour_controller.reset()
        else:
            self.pick_controller.reset()
            self.obs_history_dict = {
                obs_key: deque(maxlen=self.n_obs_steps) 
                for obs_key in self.obs_names.values()
            }
            self.obs_history_pose = deque(maxlen=self.n_obs_steps)
            self.trajectory_controller.reset()

    def _check_phase_success(self):
        """Check if current phase is successful."""
        self.frame_count +=1
        object_pos = self.state['object_position']
        # print(object_pos, self.initial_position)
        if self.current_phase == Phase.PICKING:
            return object_pos[2] > 1.0
            
        elif self.current_phase == Phase.POURING:
            return object_pos[2] > 0.85 and self.frame_count > 1200

    def step(self, state):
        """Execute one step of control.
        
        Args:
            state: Current state dictionary containing sensor data and robot state
            
        Returns:
            Tuple containing action, done flag, and success flag
        """
        self.state = state
        if self.initial_position is None:
            self.initial_position = self.state['object_position']
        if self.initial_size is None:
            self.initial_size = self.state['object_size']
        if self.mode == "collect":
            return self._step_collect(state)
        else:
            return self._step_infer(state)

    def _step_collect(self, state):
        """Execute collection mode step."""
        success = self._check_phase_success()
        
        if success:
            if self.current_phase == Phase.PICKING:
                print("Pick task success! Switching to pour...")
                self.current_phase = Phase.POURING
                self.active_controller = self.pour_controller
                return None, False, False
            elif self.current_phase == Phase.POURING:
                print("Pour task success!")
                self.data_collector.write_cached_data(state['joint_positions'][:-1])
                self._last_success = True
                self.current_phase = Phase.FINISHED
                self.frame_count = 0  
                return None, True, True
        
        
        if self.current_phase == Phase.FINISHED:
            if self.frame_count > 2000:
                self.reset_needed = True
                return None, True, self._last_success
            return None, False, False  

        
        if not self.active_controller.is_done():
            action = None
            if self.current_phase == Phase.PICKING:
                action = self.pick_controller.forward(
                    picking_position=state['object_position'],
                    current_joint_positions=state['joint_positions'],
                    object_size=state['object_size'],
                    object_name=state['object_name'],
                    gripper_control=self.gripper_control,
                    gripper_position=state['gripper_position'],
                    end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 10])).as_quat(),
                    flag=True,
                )
            else:
                action = self.pour_controller.forward(
                    franka_art_controller=self.robot.get_articulation_controller(),
                    source_size=self.initial_size,
                    target_position=state['target_position'],
                    current_joint_velocities=self.robot.get_joint_velocities(),
                    pour_speed=-1,
                    source_name=state['object_name'],
                    gripper_position=state['gripper_position'],
                )
                
                if 'camera_data' in state:
                    self.data_collector.cache_step(
                        camera_images=state['camera_data'],
                        joint_angles=state['joint_positions'][:-1]
                    )
            
            return action, False, False
        
        
        if self.frame_count > 2000:
            print(f"{self.current_phase.value} task failed!")
            self.data_collector.clear_cache()
            self._last_success = False
            self.current_phase = Phase.FINISHED
            self.frame_count = 0  
            return None, True, False
        
        return None, False, False

    def _step_infer(self, state):
        """Execute inference mode step."""
        self.state = state
        if self.current_phase == Phase.FINISHED:
            self.reset_needed = True
            return None, True, self._last_success
        
        
        if not self.pick_controller.is_done():
            action = None
            action = self.pick_controller.forward(
                    picking_position=state['object_position'],
                    current_joint_positions=state['joint_positions'],
                    object_size=state['object_size'],
                    object_name=state['object_name'],
                    gripper_control=self.gripper_control,
                    gripper_position=state['gripper_position'],
                    end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 15])).as_quat(),
                )
            
        else:
            # Update observation history
            for cam_name, image in state['camera_data'].items():
                if cam_name in self.camera_to_obs:
                    obs_key = self.camera_to_obs[cam_name]
                    self.obs_history_dict[cam_name].append(image)
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
            if self.n_obs_steps != 1 and action is not None:
                if len(action.joint_positions) == 9:
                    if action.joint_positions[7] != None or action.joint_positions[8] != None:
                        if action.joint_positions[7] - np.float32(0.015) < np.float32(0.015):
                            self.change_flag = True

                        if self.change_flag:
                            action.joint_positions[7] = np.float32(0.015)
                            action.joint_positions[8] = np.float32(0.015)
        
        success = self._check_phase_success()
        if success and self.current_phase == Phase.PICKING:
            print("Pick task success! Switching to pour...")
            self.current_phase = Phase.POURING
            self.trajectory_controller.reset()
        elif success and self.current_phase == Phase.POURING:
            print("Pour task success!")
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