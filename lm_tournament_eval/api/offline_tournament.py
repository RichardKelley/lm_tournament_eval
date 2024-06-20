# this is a collection of matches, models, and a schedule of "play"

from dataclasses import dataclass, field
from .offline_match_scheduler import OfflineMatchSchedulerConfig, OfflineMatchScheduler
from .task import TaskConfig
from .match import MatchResult, Match
import json
import numpy as np

@dataclass
class OfflineTournamentConfig:
    name : str
    offline_file_1 : str
    offline_file_2 : str
    task_name : str
    rounds : int
    num_samples : int
    task_config : TaskConfig


class OfflineTournament:
    def __init__(self, config : OfflineTournamentConfig):
        self.config = config
        # set the initial scores
        self.score_1 = 1200
        self.score_2 = 1200
        self.k = 16
        # read the offline results in 
        self.responses_1 = []
        self.responses_2 = []
        with open(config.offline_file_1, 'r') as file:
            for line in file:
                self.responses_1.append(json.loads(line))
        with open(config.offline_file_2, 'r') as file:
            for line in file:
                self.responses_2.append(json.loads(line))

        self.scheduler_cfg = OfflineMatchSchedulerConfig(rounds = config.rounds,
                                                         num_samples = config.num_samples)
        self.scheduler = OfflineMatchScheduler(self.scheduler_cfg)
        self.match_result_list = [MatchResult(model0_name="test",
                                              model1_name="test",           
                                              model0_old_elo=self.score_1,
                                              model1_old_elo=self.score_2,
                                              model0_new_elo=self.score_1,
                                              model1_new_elo=self.score_2,
                                              task_config=self.config.task_config)
                                              for i in range(self.config.rounds)]

    def run_tournament(self):
        # run match
        for n in range(self.config.rounds):
            self.match_result_list[n].model0_old_elo = self.score_1
            self.match_result_list[n].model1_old_elo = self.score_2
            # sample indices
            task_indices = self.scheduler.schedule_tournament()
            # calculate the wins, losses, and draws
            as_1 = []
            as_2 = []
            for i in task_indices:
                # draw
                if self.responses_1[i]['acc'] == self.responses_2[i]['acc']:
                    as_1.append(0)
                    as_2.append(0)
                # model 1 won 
                elif self.responses_1[i]['acc'] > self.responses_2[i]['acc']:
                    as_1.append(1)
                # model 2 won
                elif self.responses_1[i]['acc'] < self.responses_2[i]['acc']:
                    as_2.append(1)
            expected_score_1 = 1/(1+10**((self.score_1-self.score_2)/400))
            expected_score_2 = 1/(1+10**((self.score_2-self.score_1)/400))
            
            # update Elo
            if sum(as_1) > sum(as_2):
                self.score_1 = self.score_1 + self.k*(1 - expected_score_1)
                self.score_2 = self.score_2 + self.k*(0 - expected_score_2)
            elif sum(as_2) > sum(as_1):
                self.score_1 = self.score_1 + self.k*(0 - expected_score_1)
                self.score_2 = self.score_2 + self.k*(1 - expected_score_2)
            elif sum(as_1) == sum(as_2):
                self.score_1 = self.score_1 + self.k*(0.5 - expected_score_1)
                self.score_2 = self.score_2 + self.k*(0.5 - expected_score_2)           
            print(f"score_1, 2 {self.score_1}, {self.score_2}")
            print("----------------------------")
            
            self.match_result_list[n].model0_new_elo = self.score_1
            self.match_result_list[n].model1_new_elo = self.score_2
            
            
        return {}

