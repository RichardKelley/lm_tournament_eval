import time
import logging
from typing import Optional, Union, Dict, List
import random

import numpy as np

import torch

from lm_tournament_eval.loggers import EvaluationTracker
from lm_tournament_eval.tasks import TaskManager
from lm_tournament_eval.caching.cache import delete_cache

from lm_tournament_eval.utils import (
    eval_logger,
    handle_non_serializable,
    hash_string,
    positional_deprecated,
    simple_parse_args_string,
)


def tournament_evaluate(
    model0,
    model1,
    model0_args: Optional[Union[str, dict]] = None,
    model1_args: Optional[Union[str, dict]] = None,
    tasks: Optional[List[Union[str, dict, object]]] = None,
    num_fewshot: Optional[int] = int,
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
    task_manager: Optional[TaskManager] = None,
    verbosity: str = "INFO",
    predict_only: bool = False,
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

    if tasks is None:
        tasks = []
    if len(tasks) == 0:
        raise ValueError(
            "No tasks specified, or no tasks found."
        )
    
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
            logging.info(
                f"Initializing {model0} model, with arguments: {model0_args}."
            )

            # TODO START HERE


    # TODO LOAD MODEL1 to CPU

    # TODO...