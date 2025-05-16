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

from .atomic_actions.pick_controller import PickController
from .atomic_actions.place_controller import PlaceController
from .base_controller import BaseController
from .robot_controllers.grapper_manager import Gripper
from .robot_controllers.trajectory_controller import FrankaTrajectoryController
from factories.collector_factory import create_collector

class Phase(Enum):
    PICKING = "picking"
    PLACING = "placing"
    FINISHED = "finished"

class PlaceTaskController(BaseController):
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
        
        self.place_controller = PlaceController(
            name="place_controller",
            cspace_controller=rmp_controller,
            gripper=robot.gripper,
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

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.checkpoint = torch.load(cfg.policy_model_path, map_location=self.device,weights_only=False)
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

        # Initialize observation history
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
        """Reset controller state and phase."""
        super().reset()
        self.current_phase = Phase.PICKING
        self.initial_position = None
        self.initial_size = None
        self.frame_count = 0
        
        if self.mode == "collect":
            self.active_controller = self.pick_controller
            self.pick_controller.reset()
            self.place_controller.reset()
        else:
            self.pick_controller.reset()
            self.obs_history_dict = {
                cam_name: deque(maxlen=self.n_obs_steps) 
                for cam_name in self.cfg.cameras_names
            }
            self.obs_history_pose = deque(maxlen=self.n_obs_steps)
            self.trajectory_controller.reset()

    def _check_phase_success(self):
        """Check if current phase is successful based on object position."""
        object_pos = self.state['object_position']
        target_position = self.state['target_position']
        
        if self.current_phase == Phase.PICKING:
            return object_pos[2] > 0.82
        elif self.current_phase == Phase.PLACING:
            success = (object_pos is not None and object_pos[2] <= 0.78 and
                        abs(object_pos[0] - target_position[0]) < 0.05 and
                        abs(object_pos[1] - target_position[1]) < 0.05)
            if self.mode != "collect":
                success = success and self.frame_count > 1200
            return success


    def step(self, state):
        """Execute one step of control.
        
        Args:
            state: Current state dictionary containing sensor data and robot state
            
        Returns:
            Tuple containing action, done flag, and success flag
        """
        self.state = state
        self.frame_count+=1
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
        if self.current_phase == Phase.FINISHED:
            self.reset_needed = True
            return None, True, self._last_success

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
                action = self.place_controller.forward(
                    place_position = state['target_position'],
                    current_joint_positions=self.robot.get_joint_velocities(),
                    gripper_control=self.gripper_control,
                    end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 10])).as_quat(),
                )
                
                if 'camera_data' in state:
                    self.data_collector.cache_step(
                        camera_images=state['camera_data'],
                        joint_angles=state['joint_positions'][:-1]
                    )
            
            return action, False, False

        if success:
            if self.current_phase == Phase.PICKING:
                print("Pick task success! Switching to pour...")
                self.current_phase = Phase.PLACING
                self.active_controller = self.place_controller
                return None, False, False
            elif self.current_phase == Phase.PLACING:
                print("Pour task success!")
                self.data_collector.write_cached_data(state['joint_positions'][:-1])
                self._last_success = True
                self.current_phase = Phase.FINISHED
                return None, True, True
            else:
                print(f"{self.current_phase.value} task failed!")
                self.data_collector.clear_cache()
                self._last_success = False
                self.current_phase = Phase.FINISHED
                return None, True, False
        
        return None, False, False

    def _step_infer(self, state):
        """Execute inference mode step."""
        self.state = state
        if self.current_phase == Phase.FINISHED:
            self.reset_needed = True
            return None, True, self._last_success
        
        
        if self.current_phase == Phase.PICKING:
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
               
        return action, False, self.is_success()

    def is_done(self):
        return self.current_phase == Phase.FINISHED

    def is_success(self):
        
        object_pos = self.state["object_position"]
        target_position = self.state['target_position']
        if (object_pos is not None and object_pos[2] > 0.85 and
            abs(object_pos[0] - target_position[0]) < 0.025 and
            abs(object_pos[1] - target_position[1]) < 0.025 and
            self.frame_count > 1200):
            self.current_phase == Phase.FINISHED
            return True
        return False

    def close(self):
        if self.mode == "collect":
            self.data_collector.close()

    def episode_num(self):
        if self.mode == "collect":
            return self.data_collector.episode_count
        return self._episode_num