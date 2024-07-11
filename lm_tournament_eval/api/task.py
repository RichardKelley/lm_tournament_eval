from datasets import load_dataset
from dataclasses import dataclass, field
from typing import Optional, Union

import logging

from .instance import OutputType, Instance

@dataclass
class TaskConfig:
    task : Optional[str] = None

    dataset_path : Optional[str] = None
    dataset_name : Optional[str] = None
    dataset_kwargs : Optional[str] = None
    training_split : Optional[str] = None
    validation_split : Optional[str] = None
    test_split : Optional[str] = None
    description : str = ""

    metric_list : Optional[list] = None
    output_type : OutputType = "multiple_choice"

    def __getitem__(self, item):
        return getattr(self, item)
    
    def __setitem__(self, item, value):
        setattr(self, item, value)


class Task:
    def __init__(self, config) -> None:
        self.config = config
        if config.task is not None:
            self._task_name = config.task

        self._instances = []

        self._download()
        self._create_instances()

    @property
    def task_name(self) -> str:
        return self._task_name

    def _download(self) -> None:
        logging.log(logging.INFO, f"Downloading dataset {self.task_name}")
        self._dataset = load_dataset(self.task_name)

    def _create_instances(self) -> None:
        logging.log(logging.INFO, "Preparing instances for evaluation...")        
        for req in self._dataset["validation"]:
            for i, ending in enumerate(req["endings"]):
                self._instances.append(Instance(request_type=self.config.output_type,
                                                   doc=req,
                                                   arguments=(req['ctx'],ending),
                                                   idx=i
                                                  ))