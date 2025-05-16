import os
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import hydra
from omegaconf import DictConfig
import cv2
import numpy as np

from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
import omni.usd

from factories.robot_factory import create_robot
from utils.object_utils import ObjectUtils
from factories.task_factory import create_task
from factories.controller_factory import create_controller

@hydra.main(version_base=None, config_path="config", config_name="level3_Heat_Liquid")
def main(cfg: DictConfig):
    world = World(stage_units_in_meters=1.0, physics_prim_path="/physicsScene", backend="numpy")
    
    # world = World(stage_units_in_meters=1, device="cpu")
    # physx_interface = omni.physx.get_physx_interface()
    # physx_interface.overwrite_gpu_setting(1)

    robot = create_robot(
        cfg.robot.type,
        position=np.array(cfg.robot.position)
    )
    
    stage = omni.usd.get_context().get_stage()
    add_reference_to_stage(usd_path=os.path.abspath(cfg.usd_path), prim_path="/World")
    
    ObjectUtils.get_instance(stage)
    
    task = create_task(
        cfg.task_type,
        cfg=cfg,
        world=world,
        stage=stage,
        robot=robot,
    )
    
    task_controller = create_controller(
        cfg.controller_type,
        cfg=cfg,
        robot=robot,
    )
    
    video_writer = None
    task.reset()
    
    while simulation_app.is_running():
        world.step(render=True)
        
        if world.is_stopped():
            task_controller.reset_needed = True
            
        if world.is_playing():
            if task_controller.need_reset() or task.need_reset():
                if video_writer is not None:
                    video_writer.release()
                    video_writer = None
                           
                task_controller.reset()
                if task_controller.episode_num() >= cfg.max_episodes:
                    task_controller.close()
                    simulation_app.close()
                    cv2.destroyAllWindows()
                    break
                task.reset()
                
                continue
                
            state = task.step()
            if state is None:
                continue
            
            action, done, is_success = task_controller.step(state)
            if action is not None:
                robot.get_articulation_controller().apply_action(action)
            if done:
                task.on_task_complete(is_success)
                continue
            
            if cfg.save_video or cfg.show_video:
                camera_images = []
                for _, image_data in state['camera_display'].items():
                    display_img = cv2.cvtColor(image_data.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
                    camera_images.append(display_img)
                
                if camera_images:
                    combined_img = np.hstack(camera_images)
                    total_width = 0
                    for idx, img in enumerate(camera_images):
                        label = f"Camera {idx+1} ({cfg.cameras[idx].image_type})"
                        cv2.putText(combined_img, label, (total_width + 2, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)
                        total_width += img.shape[1]
                    if cfg.show_video:
                        cv2.imshow('Camera Views', combined_img)
                        cv2.waitKey(1)
                    if cfg.save_video:
                        output_dir = os.path.join(cfg.multi_run.run_dir, "video")
                        os.makedirs(output_dir, exist_ok=True)
                        output_path = os.path.join(output_dir, f"episode_{task_controller._episode_num}.mp4")
                        if video_writer is None:
                            height, width = combined_img.shape[:2]
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                            video_writer = cv2.VideoWriter(output_path, fourcc, 60.0, (width, height))
                        video_writer.write(combined_img)


if __name__ == "__main__":
    main()
