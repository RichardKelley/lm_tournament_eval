# this is a collection of matches, models, and a schedule of "play"

from dataclasses import dataclass, field
from .offline_match_scheduler import OfflineMatchSchedulerConfig, OfflineMatchScheduler
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

    def run_tournament(self):
        print("Running tournament!")

        # run match
        for n in range(self.config.rounds):
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
        return {}

