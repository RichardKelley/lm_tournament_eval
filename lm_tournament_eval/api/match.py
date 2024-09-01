# a match is a sampled subset of points from a dataset.

from lm_tournament_eval.api.task import TaskConfig, Task

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Union, Dict

@dataclass
class MatchResult:
    """
    A record of the results of a match between two models.
    """
    model0_name: str
    model1_name: str
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

@dataclass
class MultipleChoiceDocument:
    ind : int
    activity_label: str
    ctx: str
    endings : List[str]
    split : str
    gold : int

    def __repr__(self):
        return f"""{{   
            'ind' : {self.ind},
            'activity_label' : {self.activity_label},
            'ctx' : {self.ctx},
            'endings' : {self.endings},
            'split' : {self.split},
            'gold' : {self.gold}
        }}"""
    
@dataclass
class GenerateUntilDocument:
    question: str
    answer: str

    def __repr__(self):
        return f"""{{
          'question' : {self.question},
          'answer' : {self.answer}
        }}"""

@dataclass
class LoglikelihoodRollingDocument:
    text: str

    def __repr__(self):
        return f"""{{
          'str' : {self.str}
        }}"""

class InstanceResult:
    def __init__(self,
                 doc_id : int,
                 doc : Union[MultipleChoiceDocument, GenerateUntilDocument, LoglikelihoodRollingDocument],
                 target : Union[str, int],
                 arguments : List[Tuple],
                 resps : List,
                 filtered_resps : List,
                 doc_hash : str,
                 prompt_hash : str,
                 target_hash : str
                 ) -> None:
        self.doc_id = doc_id
        self.doc = doc
        self.target = target
        self.arguments = arguments
        self.resps = resps
        self.filtered_resps = filtered_resps
        self.doc_hash = doc_hash
        self.prompt_hash = prompt_hash
        self.target_hash = target_hash


class Match:

    def __init__(self, 
                 tournament_name : str,
                 results0 : Dict, 
                 results1 : Dict, 
                 model0_key : Tuple,
                 model1_key : Tuple,
                 schedule : List[int]):
        '''
        Record the metadata of a match.
        '''
        self.tournament_name = tournament_name

        self.results0 = results0
        self.results1 = results1

        self.task = results0["task"]
        self.dataset_path = results0["dataset_path"]
        self.output_type = results0["output_type"]

        if "training_split" in results0:
            self.training_split = results0["training_split"]

        if "validation_split" in results0:
            self.validation_split = results0["validation_split"]
    
        if "test_split" in results0:
            self.test_split = results0["test_split"]

        self.num_fewshot = results0["num_fewshot"]
        self.repeats = results0["repeats"]

        self.model0_key = model0_key
        self.model1_key = model1_key

        self.schedule = schedule
        self.match_size = len(self.schedule)

    def __repr__(self):
        ret = "Match("
        ret += f"task={self.task}, "
        ret += f"output_type={self.output_type}, "
        ret += f"model0={self.model0_key}, "
        ret += f"model1={self.model1_key}, "
        ret += f"schedule={self.schedule}"
        ret += ")"
        return ret