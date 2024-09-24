# this is a collection of matches, models, and a schedule of "play"

import logging
import time
import random
import numpy as np
import torch
import gc

from hflm import LM

import wandb

from dataclasses import dataclass
from lm_tournament_eval.caching.cache import delete_cache

from lm_tournament_eval.tasks import TaskManager, get_task_dict
from lm_tournament_eval.api.task_utils import prepare_tasks, create_requests, create_subtask
from lm_tournament_eval.api.model_utils import load_model, parse_model_name
from lm_tournament_eval.tournament_evaluator import evaluate
from lm_tournament_eval.utils import simple_parse_args_string

from typing import Optional, Union, Dict, List, Tuple
from lm_tournament_eval.loggers import EvaluationTracker
from lm_tournament_eval.evaluator_utils import get_task_groups
from lm_tournament_eval.api.elo import ELO
from lm_tournament_eval.api.bt import BradleyTerryModel
from lm_tournament_eval.api.glicko import GlickoSystem
from lm_tournament_eval.api.trueskill import CustomTrueSkill

from lm_tournament_eval.models.huggingface_model import HFLM
from lm_tournament_eval.api.match import MatchResult
from lm_tournament_eval.api.scheduler import FileScheduler, SamplingScheduler
from lm_tournament_eval.api.task import Task
from lm_tournament_eval.score_database import ScoreDatabase
from lm_tournament_eval.api.match import Match
from lm_tournament_eval.api.group import ConfigurableGroup

from lm_tournament_eval.loggers.utils import (
     add_env_info, 
     add_tokenizer_info, 
     get_git_commit_hash
)

from copy import deepcopy

@dataclass
class TournamentConfig:
    name : str
    model0_name : str
    model0_args : str
    model1_name : str
    model1_args : str
    task_names : str
    rounds : int
    batch_size : Union[str, int]
    gen_kwargs : str
    device : str
    limit : int
    match_size : int
    cmd_filter : str
    random_seed : int
    numpy_random_seed : int
    torch_random_seed : int
    fewshot_random_seed : int
    use_wandb : bool
    elo_dynamics : str
    ranking_system : str

class Tournament:
    def __init__(self, 
                 config : TournamentConfig, 
                 tasks, 
                 task_manager, 
                 verbosity, 
                 scheduler = None, 
                 db : ScoreDatabase = None):
        self.config = config
        self.tasks = tasks
        self.task_manager = task_manager
        self.verbosity = verbosity
        self.scheduler = scheduler
        self.db = db

        # get model0_key
        model0_bpw = '16'
        if config.model0_args is not None:
            print(config.model0_args)
            if "load_in_4bit" in config.model0_args:
                model0_bpw = '4'
            elif "load_in_8bit" in config.model0_args:
                model0_bpw = '8'
               
        model1_bpw = '16'
        if config.model1_args is not None:
            print(config.model1_args)
            if "load_in_4bit" in config.model1_args:
                model1_bpw = '4'
            if "load_in_8bit" in config.model1_args:
                model1_bpw = '8'

        self.model0_key = (
            config.model0_name, 
            model0_bpw, 
            config.model0_args if config.model0_args is not None else 'None'
        )
        self.model1_key = (
            config.model1_name, 
            model1_bpw, 
            config.model1_args if config.model1_args is not None else 'None'
        )

        logging.info(f"model0 key = {self.model0_key}")

        if self.config.ranking_system == "elo":
            self.elo = ELO(self.model0_key, 
                           self.model1_key, 
                           self.db,
                           True if config.elo_dynamics == "unbounded" else False)
        if self.config.ranking_system == "bt":
            self.bt = BradleyTerryModel(self.model0_key, 
                                        self.model1_key, 
                                        self.db)       
        if self.config.ranking_system == "glicko":
            self.glicko = GlickoSystem(self.model0_key, 
                                       self.model1_key, 
                                       self.db)
        if self.config.ranking_system == "trueskill":
            self.ts = CustomTrueSkill(self.model0_key, 
                                       self.model1_key, 
                                       self.db)
    def tournament_evaluate(
        self,
        model: str,
        lm: LM,
        requests: Dict,
        task : Task,
        eval_tasks: List,
        task_dict: Dict,
        padding_requests: Dict,
        model_args: Optional[Union[str, dict]] = None,
        batch_size: Optional[Union[int, str]] = None,
        device: Optional[str] = None,
        use_cache: Optional[str] = None,
        delete_requests_cache: bool = False,
        limit: Optional[Union[int, float]] = None,
        bootstrap_iters: int = 100000,
        gen_kwargs: Optional[str] = None,
    ) -> Dict:

        start_date = time.time()

        if delete_requests_cache:
            logging.info("Deleteing requests cache.")
            delete_cache()

        if gen_kwargs is not None:
            gen_kwargs = simple_parse_args_string(gen_kwargs)
            logging.warning(
                "generation_kwargs specified through cli, these settings will update set parameters in yaml tasks. "
                "Ensure 'do_sample=True' for non-greedy decoding!"
            )
            if gen_kwargs == "":
                gen_kwargs = None

        # We don't currently support CachingLM

        results = evaluate(
            lm=lm,
            requests=requests,
            task=task,
            eval_tasks=eval_tasks,
            task_dict=task_dict,
            padding_requests=padding_requests,
            limit=limit,
        )

        # post-process results
        if lm.rank == 0:
            if isinstance(model, str):
                model_name = model
            elif hasattr(model, "config") and hasattr(model.config, "_name_or_path"):
                model_name = model.config._name_or_path
            else:
                model_name = type(model).__name__

            results["config"] = {
                "model": model_name,
                "model_args": model_args,
            }

            if isinstance(lm, HFLM):
                results["config"].update(lm.get_model_info())

            results["config"].update(
                {
                    "batch_size": batch_size,
                    "batch_sizes": (
                        list(lm.batch_sizes.values()) if hasattr(lm, "batch_sizes") else []
                    ),
                    "device": device,
                    "use_cache": use_cache,
                    "limit": limit,
                    "bootstrap_iters": bootstrap_iters,
                    "gen_kwargs": gen_kwargs,
                    "random_seed": self.config.random_seed,
                    "numpy_seed": self.config.numpy_random_seed,
                    "torch_seed": self.config.torch_random_seed,
                    "fewshot_seed": self.config.fewshot_random_seed,
                }
            )

            results["git_hash"] = get_git_commit_hash()
            results["date"] = start_date
            add_env_info(results)  # additional environment info to results
            add_tokenizer_info(results, lm)  # additional info about tokenizer
        else:
            return None

        return results

    def run_tournament(self):

        seed_message = []

        if self.config.random_seed is not None:
            seed_message.append(f"Setting random seed to {self.config.random_seed}")
            random.seed(self.config.random_seed)

        if self.config.numpy_random_seed is not None:
            seed_message.append(f"Setting numpy seed to {self.config.numpy_random_seed}")
            np.random.seed(self.config.numpy_random_seed)

        if self.config.torch_random_seed is not None:
            seed_message.append(f"Setting torch manual seed to {self.config.torch_random_seed}")
            torch.manual_seed(self.config.torch_random_seed)

        if seed_message:
            logging.info(" | ".join(seed_message))

        for task_name in self.tasks:
            task_idx = self.tasks.index(task_name)
            filter = self.config.cmd_filter[task_idx]

            eval_tasks, task_dict = prepare_tasks([task_name], 
                                                  self.task_manager, 
                                                  self.verbosity,
                                                  cmd_filter=filter,
                                                  gen_kwargs=self.config.gen_kwargs)

            task_dict = get_task_groups(task_dict)

            for task_name, task_output in zip(task_dict.keys(), eval_tasks):
                logging.info(f"Current task: {task_name}")
                self.scheduler.set_task_size(len(task_dict[task_name].eval_docs))

                if not self.db.task_exists(task_name):
                    task = task_dict[task_name]
                    self.db.insert_task(task_name=task_name, 
                                        output_type=task.config.output_type, 
                                        num_instances=len(task.eval_docs))

                original_task = deepcopy(task_dict[task_name])
                rank = 0
                for match_schedule in self.scheduler:
                    logging.info(f"Current match indices: {match_schedule}")

                    subtask = create_subtask(original_task, schedule=match_schedule)
                    
                    model0_type, model0_name = parse_model_name(self.config.model0_name)
                    model0 = load_model(model0_type, 
                                        model0_name,
                                        self.config.model0_args,
                                        batch_size=self.config.batch_size,
                                        device=self.config.device)        

                    rank = model0._rank
                    requests0, padding_reqests0 = create_requests(model0, 
                                                                task=subtask,
                                                                limit=self.config.limit)

                    results0 = self.tournament_evaluate(model=self.config.model0_name,
                                                        lm=model0,
                                                        model_args=self.config.model0_args,
                                                        requests=requests0,
                                                        task=subtask,
                                                        eval_tasks=[task_output],
                                                        task_dict=task_dict,
                                                        padding_requests=padding_reqests0,
                                                        batch_size=self.config.batch_size,
                                                        device=self.config.device,
                                                        limit=self.config.limit
                                                    )
                    del model0._model
                    gc.collect()
                    if model0.device != "mps":
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()

                    model1_type, model1_name = parse_model_name(self.config.model1_name)
                    model1 = load_model(model1_type,
                                        model1_name,
                                        self.config.model1_args,
                                        batch_size=self.config.batch_size,
                                        device=self.config.device)

                    requests1, padding_reqests1 = create_requests(model1,
                                                                task=subtask,
                                                                limit=self.config.limit)

                    results1 = self.tournament_evaluate(model=self.config.model1_name,
                                                        lm=model1,
                                                        model_args=self.config.model1_args,
                                                        requests=requests1,
                                                        task=subtask,
                                                        eval_tasks=[task_output],
                                                        task_dict=task_dict,
                                                        padding_requests=padding_reqests1,
                                                        batch_size=self.config.batch_size,
                                                        device=self.config.device,
                                                        limit=self.config.limit
                                                    )
                    del model1._model
                    gc.collect()
                    if model1.device != "mps":
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()

                    match_dict0 = results0['configs'][task_name]
                    match_dict1 = results1['configs'][task_name]
                    m = Match(self.config.name, match_dict0, match_dict1, self.model0_key, self.model1_key, match_schedule)
                    match_id = self.db.insert_match(m)

                    self.db.update_instance_records(m, 
                                                    results0["samples"][task_name], 
                                                    results1["samples"][task_name])

                    rounds_per_task = []
                    match_results = {}
                    if rank == 0:
                        if self.config.ranking_system == "elo":
                            self.elo.online_elo_update(match_id=match_id, m=m, results0=results0, results1=results1)
                            self.db.set_model_score(*self.model0_key, self.elo.score_0)
                            self.db.set_model_score(*self.model1_key, self.elo.score_1)
                            if self.config.use_wandb:
                                wandb.log({
                                    str(self.model0_key) : self.elo.score_0,
                                    str(self.model1_key) : self.elo.score_1,                            
                                })
                        elif self.config.ranking_system == "bt":
                            self.bt.create_results(match_id=match_id, m=m, results0=results0, results1=results1)
                            self.db.set_model_score(*self.model0_key, self.bt.score_0)
                            self.db.set_model_score(*self.model1_key, self.bt.score_1)
                            if self.config.use_wandb:
                                wandb.log({
                                    str(self.model0_key) : self.bt.score_0,
                                    str(self.model1_key) : self.bt.score_1,                            
                                })
                        elif self.config.ranking_system == "glicko":
                            self.glicko.create_results(match_id=match_id, m=m, results0=results0, results1=results1)
                            self.db.set_model_score(*self.model0_key, self.glicko.score_0, self.glicko.rd_0, self.glicko.vol_0)
                            self.db.set_model_score(*self.model1_key, self.glicko.score_1, self.glicko.rd_1, self.glicko.vol_1)
                            if self.config.use_wandb:
                                wandb.log({
                                    str(self.model0_key) : self.glicko.score_0,
                                    str(self.model1_key) : self.glicko.score_1,                            
                                })
                        elif self.config.ranking_system == "trueskill":
                            self.ts.create_results(match_id=match_id, m=m, results0=results0, results1=results1)
                            self.db.set_model_score(*self.model0_key, self.ts.score_0, self.ts.sigma_0)
                            self.db.set_model_score(*self.model1_key, self.ts.score_1, self.ts.sigma_1)
                            if self.config.use_wandb:
                                wandb.log({
                                    str(self.model0_key) : self.ts.score_0,
                                    str(self.model1_key) : self.ts.score_1,                            
                                })