# this is a collection of matches, models, and a schedule of "play"

import logging
import time
import random
import numpy as np
import torch

from dataclasses import dataclass
from lm_tournament_eval.caching.cache import delete_cache

from lm_tournament_eval.tasks import TaskManager
from lm_tournament_eval.api.task_utils import create_requests
from lm_tournament_eval.api.model_utils import load_model
from lm_tournament_eval.tournament_evaluator import evaluate
from lm_tournament_eval.utils import simple_parse_args_string

from typing import Optional, Union, Dict, List, Tuple
from lm_tournament_eval.loggers import EvaluationTracker
from lm_tournament_eval.api.elo import ELO
from lm_tournament_eval.models.huggingface_model import HFLM
from lm_tournament_eval.api.match import MatchResult

from lm_tournament_eval.loggers.utils import (
     add_env_info, 
     add_tokenizer_info, 
     get_git_commit_hash
)

@dataclass
class TournamentConfig:
    name : str
    model0_name : str
    model0_args : str
    model1_name : str
    model1_args : str
    task_names : str
    rounds : int
    batch_size : int
    device : str
    limit : int
    match_size : int 


class Tournament:
    def __init__(self, config : TournamentConfig, tasks, task_manager, verbosity):
        self.config = config
        self.tasks = tasks
        self.task_manager = task_manager
        self.verbosity = verbosity
        self.elo = ELO()

    def tournament_evaluate(
        self,
        model: str,
        lm: HFLM,
        requests: Dict,
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
        random_seed: int = 0,
        numpy_random_seed: int = 1234,
        torch_random_seed: int = 1234,
        fewshot_random_seed: int = 1234
    ) -> Dict:

        start_date = time.time()

        if delete_requests_cache:
            logging.info("Deleteing requests cache.")
            delete_cache()

        seed_message = []

        if random_seed is not None:
            seed_message.append(f"Setting random seed to {random_seed}")
            random.seed(random_seed)

        if numpy_random_seed is not None:
            seed_message.append(f"Setting numpy seed to {numpy_random_seed}")
            np.random.seed(numpy_random_seed)

        if torch_random_seed is not None:
            seed_message.append(f"Setting torch manual seed to {torch_random_seed}")
            torch.manual_seed(torch_random_seed)

        if seed_message:
            logging.info(" | ".join(seed_message))
        
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
                    "random_seed": random_seed,
                    "numpy_seed": numpy_random_seed,
                    "torch_seed": torch_random_seed,
                    "fewshot_seed": fewshot_random_seed,
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
        model0 = load_model("hf", 
                            self.config.model0_name,
                            self.config.model0_args,
                            batch_size=self.config.batch_size,
                            max_batch_size=self.config.batch_size,
                            device=self.config.device)

        model1 = load_model("hf",
                            self.config.model1_name,
                            self.config.model1_args,
                            batch_size=self.config.batch_size,
                            max_batch_size=self.config.batch_size,
                            device=self.config.device)

        requests0, eval_tasks0, task_dict0, padding_reqests0 = create_requests(model0,
                                                                               self.tasks,
                                                                               self.task_manager,
                                                                               self.verbosity,
                                                                               self.config.limit)
                #TODO: add all the other params here so that build_all_requests is happy 
        requests1, eval_tasks1, task_dict1, padding_reqests1 = create_requests(model1,
                                                                               self.tasks,
                                                                               self.task_manager,
                                                                               self.verbosity,
                                                                               self.config.limit)
                #TODO: add all the other params here so that build_all_requests is happy 

        results0 = self.tournament_evaluate(model=self.config.model0_name,
                                            lm=model0,
                                            model_args=self.config.model0_args,
                                            requests=requests0,
                                            eval_tasks=eval_tasks0,
                                            task_dict=task_dict0,
                                            padding_requests=padding_reqests0,
                                            batch_size=self.config.batch_size,
                                            device=self.config.device,
                                            limit=self.config.limit
                                        )
        results1 = self.tournament_evaluate(model=self.config.model1_name,
                                            lm=model1,
                                            model_args=self.config.model1_args,
                                            requests=requests1,
                                            eval_tasks=eval_tasks1,
                                            task_dict=task_dict1,
                                            padding_requests=padding_reqests1,
                                            batch_size=self.config.batch_size,
                                            device=self.config.device,
                                            limit=self.config.limit
                                        )
        rounds_per_task = []
        match_results = {}
        for i,task_name in enumerate(self.config.task_names):
            rounds_per_task.append(len(results0["samples"][task_name])//self.config.match_size)
            match_results[task_name] = [MatchResult(model0_name=self.config.model0_name,
                                                    model1_name=self.config.model1_name,
                                                    model0_old_elo=self.elo.score_0,
                                                    model0_new_elo=self.elo.score_0,
                                                    model1_old_elo=self.elo.score_1,
                                                    model1_new_elo=self.elo.score_1)
                                                    for i in range(rounds_per_task[i])]
        #calculate ELO updates
        self.elo.online_elo_update(results0, results1, self.config.task_names, self.config.match_size, match_results)