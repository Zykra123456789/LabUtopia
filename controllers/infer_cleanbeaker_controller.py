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

class InferCleanBeakerController(BaseController):
    def __init__(self, cfg, robot, object_utils):
        super().__init__(cfg, robot, object_utils)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if not OmegaConf.has_resolver("eval"):
            OmegaConf.register_new_resolver("eval", lambda x: eval(x))
        
        
        self.checkpoint = torch.load(cfg.model_path, map_location=self.device, weights_only=False)
        policy_cfg = OmegaConf.load(cfg.config_path)

        self.policy = hydra.utils.instantiate(policy_cfg.policy)
        self.policy.load_state_dict(self.checkpoint['state_dicts']['model'])
        self.policy.eval().to(self.device)
        self.policy.to(self.device)
        
        
        if hasattr(cfg, 'normalizer_path'):
            normalizer = LinearNormalizer()
            normalizer.load_state_dict(torch.load(cfg.normalizer_path, map_location=self.device))
        else:
            dataset: BaseImageDataset = hydra.utils.instantiate(policy_cfg.task.dataset)
            normalizer = dataset.get_normalizer().to(self.device)
        self.policy.set_normalizer(normalizer)
        
        
        self.n_obs_steps = policy_cfg.n_obs_steps
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
        self.frame_count = 0
        
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
                    obs_dict[camera_name] = obs_dict[camera_name].unsqueeze(0).to(self.device)  # [1, T, C, H, W]
                obs_dict['agent_pose'] = obs_dict['agent_pose'].unsqueeze(0).to(self.device)  # [1, T, D]
            else:
                for camera_name in self.obs_history_dict.keys():
                    obs_dict[camera_name] = obs_dict[camera_name].to(self.device)  # [1(T), C, H, W]
                obs_dict['agent_pose'] = obs_dict['agent_pose'].to(self.device)  # [1(T), D]
            
            
            with torch.no_grad():
                prediction = self.policy.predict_action(obs_dict)
                joint_positions = prediction['action'][0].cpu().numpy()
                
            
            self.trajectory_controller.generate_trajectory(joint_positions)
            
        
        action = self.trajectory_controller.get_next_action()

        if action is not None:
            if len(action.joint_positions) == 9:
                if action.joint_positions[7] == None and action.joint_positions[8] == None:
                    action.joint_positions[7] = 0.020
                    action.joint_positions[8] = 0.020

                if action.joint_positions[7] < np.float32(0.03) and action.joint_positions[8] < np.float32(0.03):
                    action.joint_positions[7] = 0.020
                    action.joint_positions[8] = 0.020
        
        return action, False, self.is_success()
        
    def is_done(self):
        
        return self.trajectory_controller.is_trajectory_complete()
        
    def is_success(self):
        
        Maxframe = 5000  # DP 380
        self.frame_count += 1
        # print(f"frame count:{self.frame_count}")
        if self.frame_count > Maxframe:
            self.reset_needed = True
            return True
        
        beaker1_pos = self.object_utils.get_object_xform_position(object_path=self.cfg.beaker_1+"/mesh")
        beaker2_pos = self.object_utils.get_object_xform_position(object_path=self.cfg.beaker_2+"/mesh")

        plat1_pos = self.object_utils.get_object_xform_position(object_path=self.cfg.plat_1)
        plat2_pos = self.object_utils.get_object_xform_position(object_path=self.cfg.plat_2)

        if self.frame_count > 5000:
            
            
            print("===  ===")
            print(f"beaker1_pos: {beaker1_pos}")
            print(f"plat1_pos: {plat1_pos}")
            print(f"beaker2_pos: {beaker2_pos}")
            print(f"plat2_pos: {plat2_pos}")

            
            print("\n===  ===")

            
            cond1 = beaker1_pos is not None
            print(f" 1: beaker1_pos is not None -> {cond1}")

            
            diff_x1 = abs(beaker1_pos[0] - plat1_pos[0])
            cond2 = diff_x1 < 0.04
            print(f" 2: abs(beaker1_pos[0] - plat1_pos[0]) = {diff_x1:.6f} < 0.04 -> {cond2}")

            
            diff_y1 = abs(beaker1_pos[1] - plat1_pos[1])
            cond3 = diff_y1 < 0.04
            print(f" 3: abs(beaker1_pos[1] - plat1_pos[1]) = {diff_y1:.6f} < 0.04 -> {cond3}")

            
            z1 = beaker1_pos[2]
            cond4 = z1 <= 0.78
            print(f" 4: beaker1_pos[2] = {z1:.6f} <= 0.78 -> {cond4}")

            
            cond5 = beaker2_pos is not None
            print(f" 5: beaker2_pos is not None -> {cond5}")

            
            diff_x2 = abs(beaker2_pos[0] - plat2_pos[0])
            cond6 = diff_x2 < 0.04
            print(f" 6: abs(beaker2_pos[0] - plat2_pos[0]) = {diff_x2:.6f} < 0.04 -> {cond6}")

            
            diff_y2 = abs(beaker2_pos[1] - plat2_pos[1])
            cond7 = diff_y2 < 0.04
            print(f" 7: abs(beaker2_pos[1] - plat2_pos[1]) = {diff_y2:.6f} < 0.04 -> {cond7}")

            
            z2 = beaker2_pos[2]
            cond8 = z2 <= 0.78
            print(f" 8: beaker2_pos[2] = {z2:.6f} <= 0.78 -> {cond8}")

            
            
            success = cond1 and cond2 and cond3 and cond4 and cond5 and cond6 and cond7 and cond8
            print("\n===  ===")
            print(f"success = {success}")
            if success:
                print("：（1-8） True")
            else:
                print("： False：")
                if not cond1: print("-  1: beaker1_pos is not None")
                if not cond3: print("-  3: abs(beaker1_pos[1] - plat1_pos[1]) < 0.04")
                if not cond4: print("-  4: beaker1_pos[2] <= 0.78")
                if not cond5: print("-  5: beaker2_pos is not None")
                if not cond6: print("-  6: abs(beaker2_pos[0] - plat2_pos[0]) < 0.04")
                if not cond7: print("-  7: abs(beaker2_pos[1] - plat2_pos[1]) < 0.04")
                if not cond8: print("-  8: beaker2_pos[2] <= 0.78")

                # success = (
                #     beaker1_pos is not None and
                #     abs(beaker1_pos[0] - plat1_pos[0]) < 0.2 and
                #     abs(beaker1_pos[1] - plat1_pos[1]) < 0.2 and
                #     beaker1_pos[2] <= 0.78 and
                #     beaker2_pos is not None and
                #     abs(beaker2_pos[0] - plat2_pos[0]) < 0.2 and
                #     abs(beaker2_pos[1] - plat2_pos[1]) < 0.2 and
                #     beaker2_pos[2] <= 0.78
                # )
                # success = False

        else:
            success = False

        if success:
            self.success_episode += 1
            self.reset_needed = True
            self.print_success = True
            return True
        return False

    def eposide_num(self):
        
        return self._eposide_num
    
    def get_success_rate(self):
        episode_num = self.eposide_num()
        success_rate = self.success_episode/episode_num * 100
        return success_rate