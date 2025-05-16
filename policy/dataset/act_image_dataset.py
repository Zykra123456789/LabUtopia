import copy
import os
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional
from policy.dataset.base_dataset import BaseImageDataset
from policy.model.common.normalizer import LinearNormalizer
from policy.model.common.normalizer import SingleFieldLinearNormalizer
from policy.common.normalize_util import get_image_range_normalizer
from torch.nn.utils.rnn import pad_sequence

class ACTImageDataset(BaseImageDataset):
    def __init__(self, 
                 shape_meta,
                 dataset_path: str,
                 seed: int = 42,
                 horizon: int = None,
                 n_obs_steps: int = None,
                 val_ratio: float = 0.00,
                 max_train_episodes: Optional[int] = None):
        
        self.dataset_path = dataset_path
        self.shape_meta = shape_meta
        self.seed = seed
        self.val_ratio = val_ratio
        self.max_train_episodes = max_train_episodes
        
        h5_path = os.path.join(dataset_path, "episode_data.hdf5")
        self.h5_file = h5py.File(h5_path, 'r')
        self.episode_ids = list(self.h5_file.keys())
        self.camera_names = ['camera_1', 'camera_2']
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        
        self.episode_map = []
        
        h5_path = os.path.join(dataset_path, "episode_data.hdf5")
        self.h5_file = h5py.File(h5_path, 'r')
        
        for episode_name in self.h5_file.keys():
            n_frames = self.h5_file[episode_name]['actions'].shape[0]
            self.episode_map.append((
                episode_name,
                n_frames
            ))
        
        self.sequences = []
        for episode_name, n_frames in self.episode_map:
            total_steps = n_frames
            # if self.horizon is not None and self.n_obs_steps is not None:
            #     total_steps = n_frames - (self.horizon + self.n_obs_steps) + 1
            for start_idx in range(total_steps):
                self.sequences.append((episode_name, start_idx))
                
    def __len__(self):
        return len(self.sequences)

    def get_all_actions(self) -> torch.Tensor:
        all_actions = []
        for episode_name in self.episode_ids:
            episode = self.h5_file[episode_name]
            actions = torch.from_numpy(episode['actions'][:].astype(np.float32))
            all_actions.append(actions)
        return torch.cat(all_actions, dim=0)
    
    def get_normalizer(self, mode='limits', **kwargs):
        normalizer = LinearNormalizer()
        
        
        normalizer['action'] = SingleFieldLinearNormalizer.create_fit(
            self.get_all_actions().numpy())
        
        
        all_poses = []
        for episode_name in self.episode_ids:
            episode = self.h5_file[episode_name]
            poses = episode['agent_pose'][:].astype(np.float32)
            all_poses.append(poses)
        all_poses = np.concatenate(all_poses, axis=0)
        normalizer['agent_pose'] = SingleFieldLinearNormalizer.create_fit(all_poses)
        
        
        normalizer['camera_1'] = get_image_range_normalizer()
        normalizer['camera_2'] = get_image_range_normalizer()
        
        return normalizer
    
    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.train = False
        return val_set
    
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        episode_name, start_idx = self.sequences[index]
        episode = self.h5_file[episode_name]
        
        original_action_shape = episode['actions'].shape
        
        obs_start_idx = start_idx
        obs_end_idx = start_idx + self.n_obs_steps
        action_start_idx = obs_end_idx
        action_end_idx = action_start_idx + self.horizon
        
        episode_len = original_action_shape[0]
        

        qpos = episode['agent_pose'][obs_start_idx]
        image_dict = dict()
        image_dict['camera_1'] = episode['camera_1'][obs_start_idx]
        image_dict['camera_2'] = episode['camera_2'][obs_start_idx]

        action = episode['agent_pose'][action_start_idx:]
        action_len = episode_len - action_start_idx
        
        
        padded_action = np.zeros(original_action_shape, dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(episode_len)
        is_pad[action_len:] = 1

        
        cam1_obs = torch.from_numpy(image_dict['camera_1']).float() / 255.0
        cam2_obs = torch.from_numpy(image_dict['camera_2']).float() / 255.0
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        
        return {
            'obs': {
                'camera_1': cam1_obs,        # updated variable name
                'camera_2': cam2_obs,        # updated variable name
                'agent_pose': qpos_data,     # corrected variable name
            },
            'action': action_data,        # corrected variable name
            'is_pad': is_pad              # corrected variable name
        }

    @staticmethod
    def collate_fn(batch):
        
        cam1_images = [item['obs']['camera_1'] for item in batch]
        cam2_images = [item['obs']['camera_2'] for item in batch]
        robot_eef_pose = [item['obs']['agent_pose'] for item in batch]
        actions = [item['action'] for item in batch]
        is_pad = [item['is_pad'] for item in batch]

        
        padded_cam1_images = pad_sequence([img for img in cam1_images], batch_first=True)
        padded_cam2_images = pad_sequence([img for img in cam2_images], batch_first=True)
        padded_robot_eef_pose = pad_sequence([pose for pose in robot_eef_pose], batch_first=True)
        padded_actions = pad_sequence([action for action in actions], batch_first=True)
        padded_is_pad = pad_sequence([pad for pad in is_pad], batch_first=True)

        return {
            'obs': {
                'camera_1': padded_cam1_images,    # [B,T,3,H,W]
                'camera_2': padded_cam2_images,    # [B,T,3,H,W]
                'agent_pose': padded_robot_eef_pose # [B,T,7]
            },
            'action': padded_actions,            # [B,T,7]
            'is_pad': padded_is_pad
        }


from torch.utils.data import DataLoader
def main():
    
    dataset_path = '/home/ubuntu/Documents/chemistry-lab-simulator/outputs/data/2025.03.25/12.35.02_rgb_camera/dataset'  
    shape_meta = {'camera_1': (3, 256, 256), 'camera_2': (3, 256, 256), 'agent_pose': (8,)}  

    
    dataset = ACTImageDataset(
        shape_meta=shape_meta,
        dataset_path=dataset_path,
        seed=42,
        val_ratio=0.1,
        n_obs_steps=1,
        horizon=60
    )

    val_dataset = dataset.get_validation_dataset()
    
    print(f": {len(dataset)}")
    print(f": {len(val_dataset)}")
    
    # print(dataset[0])
    
    batch_size = 1  # Assuming a batch size, adjust as needed
    train_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=ACTImageDataset.collate_fn, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=ACTImageDataset.collate_fn, shuffle=False)

    
    for batch in train_loader:
        print(":")
        print("Camera 1:", batch['obs']['camera_1'].shape)  # [B, T, 3, H, W]
        print("Camera 2:", batch['obs']['camera_2'].shape)  # [B, T, 3, H, W]
        print("Agent Pose:", batch['obs']['agent_pose'].shape)  # [B, T, 2]
        print("Actions:", batch['action'].shape)  # [B, T, 2]
        camera_1_tensor = batch['obs']['camera_1']

        
        max_value = torch.max(camera_1_tensor)
        min_value = torch.min(camera_1_tensor)
        mean_value = torch.mean(camera_1_tensor)
        
        print(f": {max_value.item()}")
        print(f": {min_value.item()}")
        print(f": {mean_value.item()}")
        print("Agent Pose:", batch['obs']['agent_pose'])
        print(batch['action'])
        break
    

if __name__ == "__main__":
    main()
