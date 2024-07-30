# a match is a sampled subset of points from a dataset.

from lm_tournament_eval.api.task import TaskConfig, Task

from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class MatchResult:
    """
    A record of the results of a match between two models.
    """
    # model0_name: str
    # model1_name: str
    model0_old_elo: float
    model1_old_elo: float
    model0_new_elo: float
    model1_new_elo: float
    # task_config : TaskConfig

    # task_indices : List[int] = field(
    #     metadata={"doc" : "Indices from the task that make up the match"},
    #     default_factory=list
    # )

    # match_points : List[int] = field(
    #    metadata={"doc" : "Array of winner for each 'point' - model 0 or model 1."},
    #    default_factory=list
    # )

class Match:
    def __init__(self):
        pass
