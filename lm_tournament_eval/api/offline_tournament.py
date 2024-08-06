# this is a collection of matches, models, and a schedule of "play"

from dataclasses import dataclass, field
from .offline_match_scheduler import OfflineMatchSchedulerConfig, OfflineMatchScheduler
from .task import TaskConfig
from .match import MatchResult, Match
import json
import numpy as np
from lm_tournament_eval.api.elo import ELO


@dataclass
class OfflineTournamentConfig:
    name : str
    offline_file_0 : str
    offline_file_1 : str
    task_name : str
    rounds : int
    num_samples : int
    task_config : TaskConfig
    model0_name : str
    model1_name : str


class OfflineTournament:
    def __init__(self, config : OfflineTournamentConfig):
        self.config = config
        self.elo = ELO()
        # read the offline results in 
        self.responses_0 = []
        self.responses_1 = []
        with open(config.offline_file_0, 'r') as file:
            for line in file:
                self.responses_0.append(json.loads(line))
        with open(config.offline_file_1, 'r') as file:
            for line in file:
                self.responses_1.append(json.loads(line))

        self.scheduler_cfg = OfflineMatchSchedulerConfig(rounds = config.rounds,
                                                         num_samples = config.num_samples)
        self.scheduler = OfflineMatchScheduler(self.scheduler_cfg)
        self.match_result_list = [MatchResult(model0_name=self.config.model0_name,
                                              model1_name=self.config.model1_name,           
                                              model0_old_elo=self.elo.score_0,
                                              model1_old_elo=self.elo.score_1,
                                              model0_new_elo=self.elo.score_0,
                                              model1_new_elo=self.elo.score_1)
                                              for i in range(self.config.rounds)]

    def run_tournament(self):
        for n in range(self.config.rounds):
            self.match_result_list[n].model0_old_elo = self.elo.score_0
            self.match_result_list[n].model1_old_elo = self.elo.score_1
            task_indices = self.scheduler.schedule_tournament()
            self.elo.offline_elo_update(self.responses_0, self.responses_1, self.config.task_name, task_indices)
            self.match_result_list[n].model0_new_elo = self.elo.score_0
            self.match_result_list[n].model1_new_elo = self.elo.score_1
        return {}

