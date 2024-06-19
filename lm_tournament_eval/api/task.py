from dataclasses import dataclass, field
from typing import Optional, Union

from lm_tournament_eval.api.instance import OutputType, Instance

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
    def __init__(self) -> None:
        pass
        