from abc import ABC, abstractmethod 
from typing import Dict, Any, Tuple
import torch
import hydra
from omegaconf import OmegaConf
from controllers.robot_controllers.trajectory_controller import FrankaTrajectoryController
from utils.object_utils import ObjectUtils
from policy.model.common.normalizer import LinearNormalizer
from collections import deque

class BaseController(ABC):
    """Base class for all controllers in the chemistry lab simulator.
    
    Provides common functionality for robot control, state management,
    and episode tracking.
    """
    
    def __init__(self, cfg, robot):
        """Initialize the base controller.
        
        Args:
            cfg: Configuration object containing controller settings
            robot: Robot instance to control
            object_utils: Utility class for object manipulation
        """
        self.cfg = cfg
        self.robot = robot
        self.object_utils = ObjectUtils.get_instance()
        self.reset_needed = False
        self._last_success = False
        self._episode_num = 0
        self.success_count = 0 
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        OmegaConf.register_new_resolver("eval", lambda x: eval(x))
    
    @abstractmethod
    def step(self, state: Dict[str, Any]) -> Tuple[Any, bool, bool]:
        """Execute one step of control.
        
        Args:
            state: Current state dictionary containing sensor data and robot state
            
        Returns:
            Tuple containing:
            - action: Control action to execute
            - done: Whether the episode is complete
            - is_success: Whether the task was completed successfully
        """
        pass
    
    def episode_num(self) -> int:
        """Get the current episode number.
        
        Returns:
            int: Current episode number
        """
        return self._episode_num
    
    def reset(self) -> None:
        """Reset the controller state between episodes."""
        if self._last_success:
            self.success_count += 1
        self._episode_num += 1
        print(f"Episode Stats: Success Rate = {self.success_count}/{self._episode_num} ({(self.success_count/self._episode_num)*100:.2f}%)")
        
        self.reset_needed = False
        self._last_success = False

        
    def close(self) -> None:
        """Clean up resources used by the controller."""
        if hasattr(self, 'data_collector'):
            self.data_collector.close()
        
    def need_reset(self) -> bool:
        """Check if the controller needs to be reset.
        
        Returns:
            bool: True if reset is needed, False otherwise
        """
        return self.reset_needed