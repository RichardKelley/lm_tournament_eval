import datasets
from dataclasses import dataclass, field
from typing import Optional, Union

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


class Task:
    def __init__(self, config) -> None:
        self.config = config

    def download(self):
        dataset = datasets.load_dataset('hellaswag')
        return dataset

    def create_instances(self, dataset):
        self.instance_list = []
        for req in dataset["validation"]:
            for i, ending in enumerate(req["endings"]):
                self.instance_list.append(Instance(request_type=self.config.output_type,
                                                   doc=req,
                                                   arguments=(req['ctx'],ending),
                                                   idx=i
                                                  ))