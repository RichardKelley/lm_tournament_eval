# this is a collection of matches, models, and a schedule of "play"

import logging
import time

from dataclasses import dataclass, field
from .task import Task, TaskConfig

from .tasks import TaskManager

@dataclass
class TournamentConfig:
    name : str
    model0_name : str
    model1_name : str
    task_name : str
    rounds : int


class Tournament:
    def __init__(self, config : TournamentConfig):
        pass

    def tournament_evaluate(
        self,
        model0,
        model1,
        tasks,
        gen_kwargs,
        task_manager,
        random_seed,
        numpy_random_seed,
        torch_random_seed
    ):
        logging.info("Running tournament!")
        
        start_date = time.time()

        # TODO set random seeds

        if tasks is None:
            tasks = []
        if len(tasks) == 0:
            raise ValueError(
                "No tasks specified or no tasks found."
            )
        
        # TODO set up gen_kwargs

        # TODO handle model initialization. TODO decide if we want to allow strings.

        if task_manager is None:
            task_manager = TaskManager()

        # TODO setup match parameters

        

        # for number of rounds
        # create match
        # run match
        # record results
        # update Elo

        return {}


    