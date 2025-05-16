from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.franka.controllers.rmpflow_controller import RMPFlowController
from scipy.spatial.transform import Rotation as R
import numpy as np
import os

from .atomic_actions.pick_controller import PickController
from .atomic_actions.place_controller import PlaceController
from .atomic_actions.pour_controller import PourController
from .atomic_actions.shake_controller import ShakeController
from .base_controller import BaseController
from .robot_controllers.grapper_manager import Gripper
from factories.collector_factory import create_collector

class CleanBeakerTaskController(BaseController):

    def __init__(self, cfg, robot, object_utils):
        
        super().__init__(cfg, robot, object_utils)

        
        self.collector = create_collector(
            cfg.collector.type,
            camera_configs=cfg.cameras,
            save_dir=os.path.join(cfg.multi_run.run_dir, "cleanbeaker_data"),  
            max_episodes=cfg.max_episodes,
            compression=cfg.collector.compression
        )

        
        self.gripper_control = Gripper()

        # 1. Pick beaker2
        self.pick_beaker2 = PickController(
            name="pick_beaker2",
            cspace_controller=RMPFlowController(
                name="pick_beaker2_controller",
                robot_articulation=robot
            ),
            gripper=robot.gripper,
            events_dt=[0.016, 0.005, 0.01, 0.2, 0.05, 0.01]
        )

        # 2. Pour beaker2 to beaker1
        self.pour_beaker2 = PourController(
            name="pour_beaker2",
            cspace_controller=RMPFlowController(
                name="pour_beaker2_controller",
                robot_articulation=robot
            ),
            gripper=robot.gripper,
            events_dt=[0.006, 0.005, 0.009, 0.005, 0.009, 0.5]
        )

        # 3. Place beaker2 to plat2
        self.place_beaker2 = PlaceController(
            name="place_beaker2",
            cspace_controller=RMPFlowController(
                name="place_beaker2_controller",
                robot_articulation=robot
            ),
            gripper=robot.gripper,
            events_dt=[0.005, 0.01, 0.08, 0.05, 0.01, 0.1]
        )

        # 4. Pick beaker1
        self.pick_beaker1 = PickController(
            name="pick_beaker1",
            cspace_controller=RMPFlowController(
                name="pick_beaker1_controller",
                robot_articulation=robot
            ),
            gripper=robot.gripper,
            events_dt=[0.016, 0.005, 0.01, 0.2, 0.05, 0.01]
        )

        # 5. Shake beaker1
        self.shake_beaker1 = ShakeController(
            name="shake_beaker1",
            cspace_controller=RMPFlowController(
                name="shake_beaker1_controller",
                robot_articulation=robot
            ),
            gripper=robot.gripper
        )

        # 6. Pour beaker1 to target_beaker
        self.pour_beaker1 = PourController(
            name="pour_beaker1",
            cspace_controller=RMPFlowController(
                name="pour_beaker1_controller",
                robot_articulation=robot
            ),
            gripper=robot.gripper,
            events_dt=[0.006, 0.005, 0.009, 0.005, 0.009, 0.5]
        )

        # 7. Place beaker1 to plat1
        self.place_beaker1 = PlaceController(
            name="place_beaker1",
            cspace_controller=RMPFlowController(
                name="place_beaker1_controller",
                robot_articulation=robot
            ),
            gripper=robot.gripper
        )

        
        self._current_step = 1  
        self._last_success = False
        self.reset_needed = False

    def reset(self):
        
        super().reset()
        self.pick_beaker2.reset()
        self.pour_beaker2.reset()
        self.place_beaker2.reset()
        self.pick_beaker1.reset()
        self.shake_beaker1.reset()
        self.pour_beaker1.reset()
        self.place_beaker1.reset()
        self._current_step = 1
        self._last_success = False
        self.reset_needed = False

    def step(self, state):
        action = None
        done = False
        success = False

        
        if 'camera_data' in state:
            self.collector.cache_step(
                camera_images=state['camera_data'],
                joint_angles=state['joint_positions'][:-1]
            )

        
        if self._current_step == 1:
            # 1. Pick beaker2
            action = self.pick_beaker2.forward(
                picking_position=state['beaker_2_position'],
                current_joint_positions=state['joint_positions'],
                object_size=state['object_size'],
                object_name="beaker_2",
                gripper_control=self.gripper_control,
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 10])).as_quat()
            )
            if self.pick_beaker2.is_done():
                self._current_step = 2

        elif self._current_step == 2:
            # 2. Pour beaker2 to beaker1
            action = self.pour_beaker2.forward(
                franka_art_controller=self.robot.get_articulation_controller(),
                target_name=state['beaker_1'],
                current_joint_velocities=self.robot.get_joint_velocities(),
                pour_speed=-1,
            )
            if self.pour_beaker2.is_done():
                self._current_step = 3

        elif self._current_step == 3:
            # 3. Place beaker2 to plat2
            action = self.place_beaker2.forward(
                place_position=state['plat_2_position'],
                current_joint_positions=state['joint_positions'],
                gripper_control=self.gripper_control,
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 10])).as_quat()
            )
            if self.place_beaker2.is_done():
                self._current_step = 4

        elif self._current_step == 4:
            # 4. Pick beaker1
            action = self.pick_beaker1.forward(
                picking_position=state['beaker_1_position'],
                current_joint_positions=state['joint_positions'],
                object_size=state['object_size'],
                object_name="beaker1",
                gripper_control=self.gripper_control,
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 10])).as_quat()
            )
            if self.pick_beaker1.is_done():
                self._current_step = 5

        elif self._current_step == 5:
            # 5. Shake beaker1
            action = self.shake_beaker1.forward(
                franka_art_controller=self.robot.get_articulation_controller(),
                current_joint_positions=self.robot.get_joint_positions(),
                current_joint_velocities=self.robot.get_joint_velocities(),
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 10])).as_quat(),
            )
            if self.shake_beaker1.is_done():
                self._current_step = 6

        elif self._current_step == 6:
            # 6. Pour beaker1 to target_beaker
            action = self.pour_beaker1.forward(
                franka_art_controller=self.robot.get_articulation_controller(),
                target_name=state['target_beaker'],
                current_joint_velocities=self.robot.get_joint_velocities(),
                pour_speed=-1,
            )
            if self.pour_beaker1.is_done():
                self._current_step = 7

        elif self._current_step == 7:
            # 7. Place beaker1 to plat1
            action = self.place_beaker1.forward(
                place_position=state['plat_1_position'],
                current_joint_positions=state['joint_positions'],
                gripper_control=self.gripper_control,
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 10])).as_quat()
            )
            if self.place_beaker1.is_done():
                
                beaker1_pos = self.object_utils.get_object_xform_position(object_path=self.cfg.beaker_1)
                beaker2_pos = self.object_utils.get_object_xform_position(object_path=self.cfg.beaker_2)

                success = (
                    beaker1_pos is not None and
                    abs(beaker1_pos[0] - state['plat_1_position'][0]) < 0.2 and
                    abs(beaker1_pos[1] - state['plat_1_position'][1]) < 0.2 and
                    beaker1_pos[2] <= 0.78 and
                    beaker2_pos is not None and
                    abs(beaker2_pos[0] - state['plat_2_position'][0]) < 0.2 and
                    abs(beaker2_pos[1] - state['plat_2_position'][1]) < 0.2 and
                    beaker2_pos[2] <= 0.78
                )
                if success:
                    self.collector.write_cached_data(state['joint_positions'][:-1])
                    self._last_success = True
                else:
                    self.collector.clear_cache()
                    self._last_success = False
                done = True
                self.reset_needed = True
                action = None

        return action, done, success

    def close(self):
        
        self.collector.close()

    def get_current_collector(self):
        
        return self.collector

    def is_done(self):
        
        return self._current_step == 7 and self.place_beaker1.is_done()

    def is_success(self):
        
        return self._last_success

    def eposide_num(self):
        
        return self.collector.episode_count