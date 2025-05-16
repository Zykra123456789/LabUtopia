import os
import numpy as np
import h5py
import torch
import copy
from typing import Dict
from diffusion_policy_3d.dataset.base_dataset import BaseDataset
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer

class H5Dataset(BaseDataset):
    def __init__(self,
                dataset_path: str,
                horizon: int,
                n_obs_steps: int,
                pad_before: int = 0,
                pad_after: int = 0,
                seed: int = 42,
                val_ratio: float = 0.0,
                max_train_episodes: int = None):
        
        self.dataset_path = dataset_path
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.n_obs_steps = n_obs_steps
        self.train = True

        
        h5_path = os.path.join(dataset_path, "episode_data.hdf5")
        self.h5_file = h5py.File(h5_path, 'r')
        
        
        self.episode_map = []
        for episode_name in self.h5_file.keys():
            n_frames = self.h5_file[episode_name]['actions'].shape[0]
            self.episode_map.append((episode_name, n_frames))
            
        
        self.sequences = []
        for episode_name, n_frames in self.episode_map:
            total_steps = n_frames
            if self.horizon is not None and self.n_obs_steps is not None:
                total_steps = n_frames - (self.horizon + self.n_obs_steps) + 1
            for start_idx in range(total_steps):
                self.sequences.append((episode_name, start_idx))

        
        n_val = int(len(self.sequences) * val_ratio)
        if self.train:
            self.sequences = self.sequences[:-n_val]
        else:
            self.sequences = self.sequences[-n_val:]

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.train = False
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        normalizer = LinearNormalizer()
        
        
        all_actions = []
        all_poses = []
        for episode_name, _ in self.episode_map:
            episode = self.h5_file[episode_name]
            actions = episode['actions'][:].astype(np.float32)
            poses = episode['agent_pose'][:].astype(np.float32)
            all_actions.append(actions)
            all_poses.append(poses)
        
        all_actions = np.concatenate(all_actions, axis=0)
        all_poses = np.concatenate(all_poses, axis=0)
        
        normalizer['action'] = SingleFieldLinearNormalizer.create_fit(all_actions)
        normalizer['agent_pose'] = SingleFieldLinearNormalizer.create_fit(all_poses)
        
        
        normalizer['point_cloud'] = SingleFieldLinearNormalizer.create_identity()
        
        return normalizer

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        episode_name, start_idx = self.sequences[idx]
        episode = self.h5_file[episode_name]
        
        
        obs_start_idx = start_idx
        obs_end_idx = start_idx + self.n_obs_steps
        action_start_idx = obs_end_idx
        action_end_idx = action_start_idx + self.horizon
        
        
        cam1_obs = episode['camera_1_semantic_pointcloud'][obs_start_idx:obs_end_idx]  # [T,3,H,W]
        agent_pose = episode['agent_pose'][obs_start_idx:obs_end_idx]  # [T,D]
        actions = episode['actions'][action_start_idx:action_end_idx]  # [T,D]

        
        point_cloud = torch.from_numpy(cam1_obs).float() / 255.0
        agent_pose = torch.from_numpy(agent_pose).float()
        actions = torch.from_numpy(actions).float()

        return {
            'obs': {
                'point_cloud': point_cloud,
                'agent_pose': agent_pose,
            },
            'action': actions
        }

    def __del__(self):
        if hasattr(self, 'h5_file') and self.h5_file is not None:
            self.h5_file.close()

    @staticmethod
    def collate_fn(batch):
        
        point_cloud = torch.stack([item['obs']['point_cloud'] for item in batch])
        agent_pose = torch.stack([item['obs']['agent_pose'] for item in batch])
        actions = torch.stack([item['action'] for item in batch])
        
        return {
            'obs': {
                'point_cloud': point_cloud,    # [B,T,6]
                'agent_pose': agent_pose,        # [B,T,D]
            },
            'action': actions,                 # [B,T,D]
        }

import torch
from torch.utils.data import DataLoader

def main():
    dataset_path = '/home/ubuntu/Documents/LabSim/outputs/collect/2025.05.07/15.10.49_Level3_PourLiquid/dataset'
    horizon = 16
    n_obs_steps = 2
    batch_size = 1

    dataset = H5Dataset(
        # shape_meta=shape_meta,
        dataset_path=dataset_path,
        horizon=horizon,
        n_obs_steps=n_obs_steps,
        seed=42,
        val_ratio=0.1
    )

    val_dataset = dataset.get_validation_dataset()
    train_loader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            collate_fn=H5Dataset.collate_fn,
                            shuffle=False)

    
    for batch in train_loader:
        print("Point Cloud:", batch['obs']['point_cloud'].shape)
        print("Agent Pos:", batch['obs']['agent_pose'].shape)
        print("Actions:", batch['action'].shape)
        break

if __name__ == "__main__":
    main()
