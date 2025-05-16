import numpy as np
from omni.isaac.core.utils.types import ArticulationAction

def joint_positions_to_action(joint_positions: np.ndarray) -> ArticulationAction:
    
    
    if not isinstance(joint_positions, np.ndarray):
        joint_positions = np.array(joint_positions)
        
    
    action_size = joint_positions.shape[0] + 1
    target_joint_positions = [None] * action_size
    
    
    for i in range(action_size-1):
        target_joint_positions[i] = joint_positions[i]
    target_joint_positions[-1] = joint_positions[-1]
    
    return ArticulationAction(joint_positions=target_joint_positions)
