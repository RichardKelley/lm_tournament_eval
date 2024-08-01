# this is a collection of matches, models, and a schedule of "play"

import logging
import time

from dataclasses import dataclass, field
from .task import Task, TaskConfig

from lm_tournament_eval.tasks import TaskManager
from lm_tournament_eval.api.task_utils import create_requests

from typing import Optional, Union, Dict, List, Tuple
from lm_tournament_eval.loggers import EvaluationTracker
from lm_tournament_eval.api.elo import ELO


@dataclass
class TournamentConfig:
    name : str
    model0_name : str
    model1_name : str
    task_names : str
    rounds : int
    batch_size : int
    device : str
    limit : int


class Tournament:
    def __init__(self, config : TournamentConfig, tasks, task_manager, verbosity):
        self.config = config
        self.tasks = tasks
        self.task_manager = task_manager
        self.verbosity = verbosity

    def tournament_evaluate(
        self,
        model_type,
        model0,
        model1,
        requests,
        eval_tasks,
        model0_args: Optional[Union[str, dict]] = None,
        model1_args: Optional[Union[str, dict]] = None,
        num_fewshot: Optional[int] = None,
        batch_size: Optional[Union[int, str]] = None,
        max_batch_size: Optional[int] = None,
        device: Optional[str] = None,
        use_cache: Optional[str] = None,
        cache_requests: bool = False,
        rewrite_requests_cache: bool = False,
        delete_requests_cache: bool = False,
        limit: Optional[Union[int, float]] = None,
        bootstrap_iters: int = 100000,
        check_integrity: bool = False,
        write_out: bool = False,
        log_samples: bool = True,
        evaluation_tracker: Optional[EvaluationTracker] = None,
        system_instruction: Optional[str] = None,
        apply_chat_template: bool = False,
        fewshot_as_multiturn: bool = False,
        gen_kwargs: Optional[str] = None,
        verbosity: str = "INFO",
        predict_only: bool = False,
        random_seed: int = 0,
        numpy_random_seed: int = 1234,
        torch_random_seed: int = 1234,
        fewshot_random_seed: int = 1234
    ) -> Tuple[Dict, Dict]:
        
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

        # LOAD MODEL0 to CPU
        if isinstance(model0, str):
            if model0_args is None:
                logging.warning("model0_args not specified. Using defaults")
                model0_args = ""

            if isinstance(model0_args, dict):
                model0_args.update({"model": model0})
                logging.info(
                    f"Initializing {model0} model, with arguments: {model0_args}."
                )

                lm0 = lm_tournament_eval.api.registry.get_model(model_type).create_from_arg_obj(
                    model0_args,
                    {
                        "batch_size" : batch_size,
                        "max_batch_size" : max_batch_size,
                        "device":  device,
                    },
                )
                
            else:
                model0_args += f"model={model0}"
                logging.info(
                    f"Initializing {model0} model, with arguments: {simple_parse_args_string(model0_args)}"
                )
                lm0 = lm_tournament_eval.api.registry.get_model(model_type).create_from_arg_string(
                    model0_args,
                    {
                        "batch_size": batch_size,
                        "max_batch_size": max_batch_size,
                        "device": device,
                    }
                )
        else:
            if not isinstance(model0, lm_tournament_eval.api.model.LM):
                raise TypeError
            logging.info("Using pre-initialized model")
            lm0 = model0

        # now do model1
        if isinstance(model1, str):
            if model1_args is None:
                logging.warning("model1_args not specified. Using defaults")
                model1_args = ""

            if isinstance(model1_args, dict):
                model1_args.update({"model": model1})
                logging.info(
                    f"Initializing {model1} model, with arguments: {model1_args}."
                )

                lm1 = lm_tournament_eval.api.registry.get_model(model_type).create_from_arg_obj(
                    model1_args,
                    {
                        "batch_size" : batch_size,
                        "max_batch_size" : max_batch_size,
                        "device":  device,
                    },
                )
                
            else:
                model1_args += f"model={model1}"
                logging.info(
                    f"Initializing {model1} model, with arguments: {simple_parse_args_string(model1_args)}"
                )
                lm1 = lm_tournament_eval.api.registry.get_model(model_type).create_from_arg_string(
                    model1_args,
                    {
                        "batch_size": batch_size,
                        "max_batch_size": max_batch_size,
                        "device": device,
                    }
                )
        else:
            if not isinstance(model1, lm_tournament_eval.api.model.LM):
                raise TypeError
            logging.info(f"Using pre-initialized model for model 1.")
            lm1 = model1

        # We don't currently support CachingLM

        results0 = evaluate(
            lm=lm0,
            requests=requests,
            eval_tasks=eval_tasks,
            limit=limit,
        )

        results1 = evaluate(
            lm=lm1,
            requests=requests,
            eval_tasks=eval_tasks,
            limit=limit
        )

        # post-process results0 and results1
        if lm0.rank == 0:
            if isinstance(model0, str):
                model0_name = model0
            elif hasattr(model0, "config") and hasattr(model0.config, "_name_or_path"):
                model0_name = model0.config._name_or_path
            else:
                model0_name = type(model0).__name__

            results0["config"] = {
                "model0": model0_name,
                "model0_args": model0_args,
            }

            if isinstance(lm0, lm_tournament_eval.models.huggingface_model.HFLM):
                results0["config"].update(lm0.get_model_info())

            results0["config"].update(
                {
                    "batch_size": batch_size,
                    "batch_sizes": (
                        list(lm0.batch_sizes.values()) if hasattr(lm0, "batch_sizes") else []
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

            results0["git_hash"] = get_git_commit_hash()
            results0["date"] = start_date
            add_env_info(results0)  # additional environment info to results
            add_tokenizer_info(results0, lm0)  # additional info about tokenizer
        else:
            return None

        if lm1.rank == 0:
            if isinstance(model1, str):
                model1_name = model1
            elif hasattr(model1, "config") and hasattr(model1.config, "_name_or_path"):
                model1_name = model1.config._name_or_path
            else:
                model1_name = type(model1).__name__

            results1["config"] = {
                "model1": model1_name,
                "model1_args": model1_args,
            }

            if isinstance(lm1, lm_tournament_eval.models.huggingface_model.HFLM):
                results1["config"].update(lm1.get_model_info())

            results1["config"].update(
                {
                    "batch_size": batch_size,
                    "batch_sizes": (
                        list(lm1.batch_sizes.values()) if hasattr(lm1, "batch_sizes") else []
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

            results1["git_hash"] = get_git_commit_hash()
            results1["date"] = start_date
            add_env_info(results1)  # additional environment info to results
            add_tokenizer_info(results1, lm1)  # additional info about tokenizer
        else:
            return None
        
        return (results0, results1)

    def run_tournament(self):
        elo = ELO()
        #TODO: create function that initializes the models; ie. rip the code out of tournament_evaluate
        #      pass the one of the models into create requests
        requests, eval_tasks = create_requests(
                self.tasks,
                self.task_manager,
                self.verbosity,
                self.config.limit)
                #TODO: add all the other params here so that build_all_requests is happy 
        # for rounds:
        for round in range(self.config.rounds):
            #TODO: create subset of requests of len(match_size)
            
            #run tournament evaluate on that subset
            results0, results1 = tournament_evaluate(
                model_type="hf",
                model0=self.config.model0_name,
                model1=self.config.model1_name,
                model0_args=self.config.model0_args,
                model1_args=self.config.model1_args,
                requests = requests,
                eval_tasks=eval_tasks,
                batch_size=self.config.batch_size,
                device=self.config.device,
                limit=self.config.limit
            )
            #calculate ELO updates
            elo.online_elo_update(results0, results1, self.config.task_name)
