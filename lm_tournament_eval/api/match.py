# a match is a sampled subset of points from a dataset.

from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class MatchResult:
    """
    A record of the results of a match between two models.
    """
    model1_name: str
    model2_name: str
    model1_old_elo: float
    model2_old_elo: float
    model1_new_elo: float
    model2_new_elo: float
    task: str
    task_split: Optional[str]
    task_indices: List[int]

class Match:
    def __init__(self):
        pass