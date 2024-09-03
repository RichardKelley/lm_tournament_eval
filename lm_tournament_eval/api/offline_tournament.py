# this is a collection of matches, models, and a schedule of "play"

from dataclasses import dataclass, field
from .offline_match_scheduler import OfflineMatchSchedulerConfig, OfflineMatchScheduler
from .task import TaskConfig
from .match import MatchResult, Match
import json
import numpy as np
from lm_tournament_eval.api.elo import ELO
from lm_tournament_eval.score_database import ScoreDatabase
import logging


@dataclass
class OfflineTournamentConfig:
    name : str
    offline_results_file_0 : str
    offline_sample_file_0 : str
    offline_results_file_1 : str
    offline_sample_file_1 : str
    task_name : str
    rounds : int
    num_samples : int
    task_config : TaskConfig
    model0_name : str
    model1_name : str


class OfflineTournament:
    def __init__(self, config : OfflineTournamentConfig,
                 scheduler = None,
                 db: ScoreDatabase = None):
        self.config = config
        # read the offline results in 
        self.responses_0 = []
        self.responses_1 = []
        with open(config.offline_results_file_0) as json_file:
            self.results_0 = json.load(json_file)
        with open(config.offline_results_file_1) as json_file:
            self.results_1 = json.load(json_file)
        with open(config.offline_sample_file_0, 'r') as file:
            for line in file:
                self.responses_0.append(json.loads(line))
        with open(config.offline_sample_file_1, 'r') as file:
            for line in file:
                self.responses_1.append(json.loads(line))

        self.scheduler_cfg = OfflineMatchSchedulerConfig(rounds = config.rounds,
                                                         num_samples = config.num_samples)
        self.scheduler = OfflineMatchScheduler(self.scheduler_cfg)


        model0_bpw = '16'
        if self.results_0["config"]["model_args"] is not None:
            if "load_in_4bit" in self.results_0["config"]["model_args"]:
                model0_bpw = '4'
            elif "load_in_8bit" in self.results_0["config"]["model_args"]:
                model0_bpw = '8'
               
        model1_bpw = '16'
        if self.results_1["config"]["model_args"] is not None:
            if "load_in_4bit" in self.results_1["config"]["model_args"]:
                model1_bpw = '4'
            if "load_in_8bit" in self.results_1["config"]["model_args"]:
                model1_bpw = '8'

        self.model0_key = (
            config.model0_name, 
            model0_bpw, 
            self.results_0["config"]["model_args"] if self.results_0["config"]["model_args"] is not None else 'None'
        )
        self.model1_key = (
            config.model1_name, 
            model1_bpw, 
            self.results_1["config"]["model_args"] if self.results_1["config"]["model_args"] is not None else 'None'
        )

        if not db.check_model_exists(*self.model0_key):
            logging.info("Inserting model0 into DB")
            self.db.insert_model(*self.model0_key)

        if not db.check_model_exists(*self.model1_key):
            self.db.insert_model(*self.model1_key)

        self.elo = ELO(self.model0_key, 
                       self.model1_key, 
                       self.db
                    )

    def run_tournament(self):
        for n in range(self.config.rounds):
            self.match_result_list[n].model0_old_elo = self.elo.score_0
            self.match_result_list[n].model1_old_elo = self.elo.score_1
            task_indices = self.scheduler.schedule_tournament()
            self.elo.offline_elo_update(self.responses_0, self.responses_1, self.config.task_name, task_indices)
            self.match_result_list[n].model0_new_elo = self.elo.score_0
            self.match_result_list[n].model1_new_elo = self.elo.score_1
        return {}

