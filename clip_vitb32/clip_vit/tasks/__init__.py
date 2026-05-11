from .base import Task
from .datasets import (
    SUN397Task,
    RESISC45Task,
    StanfordCarsTask,
    SVHNTask,
    GTSRBTask,
    DTDTask,
)

_TASKS = {
    "sun397": SUN397Task,
    "resisc45": RESISC45Task,
    "stanford_cars": StanfordCarsTask,
    "svhn": SVHNTask,
    "gtsrb": GTSRBTask,
    "dtd": DTDTask,
}


def get_task(name: str) -> Task:
    """Instantiate the Task implementation for a dataset name."""
    key = name.strip().lower()
    if key in _TASKS:
        return _TASKS[key]()
    raise ValueError(f"Unsupported dataset task: {name}")
