from typing import Dict, Type
from tasks.base_task import BaseTask
from tasks.close_task import CloseTask
from tasks.open_task import OpenTask
from tasks.pick_task import PickTask
from tasks.pour_task import PourTask
from tasks.place_task import PlaceTask
from tasks.press_task import PressTask
from tasks.shake_task import ShakeTask
from tasks.stir_task import StirTask
from tasks.pickpour_task import PickPourTask
from tasks.pickplace_task import PickPlaceTask
from tasks.placepress_task import PlacePressTask
from tasks.cleanbeaker_task import CleanBeakerTask

_task_registry: Dict[str, Type[BaseTask]] = {}

def register_task(name: str, task_class: Type[BaseTask]):
    
    _task_registry[name] = task_class

def create_task(task_name: str, *args, **kwargs) -> BaseTask:
    
    if task_name not in _task_registry:
        raise ValueError(f": {task_name}")
    return _task_registry[task_name](*args, **kwargs)


register_task("pick", PickTask)
register_task("pour", PourTask)
register_task("place", PlaceTask)
register_task("press", PressTask)
register_task("shake", ShakeTask)
register_task("stir", StirTask)
register_task("open", OpenTask)
register_task("close", CloseTask)

register_task("pickpour", PickPourTask)
register_task("pickplace", PickPlaceTask)
register_task("placepress", PlacePressTask)

register_task("cleanbeaker", CleanBeakerTask)

