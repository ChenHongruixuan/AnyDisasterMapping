"""Task handler registry."""

from src.tasks.base import TaskHandler
from src.tasks.cd import TaskCD
from src.tasks.scd import TaskSCD

TASK_REGISTRY = {
    "seg": TaskHandler,
    "cd": TaskCD,
    "scd": TaskSCD,
}


def get_task_handler(task_name: str) -> TaskHandler:
    cls = TASK_REGISTRY.get(task_name.lower())
    if cls is None:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(TASK_REGISTRY.keys())}")
    return cls()


def register_task(name: str, handler_cls):
    TASK_REGISTRY[name] = handler_cls
