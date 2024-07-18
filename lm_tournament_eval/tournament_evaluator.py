import time
import logging
from typing import Optional, Union, Dict, List, Tuple
import random
import json
from collections import defaultdict

import numpy as np

import torch

import lm_tournament_eval.api.registry
import lm_tournament_eval.api.metrics
import lm_tournament_eval.api.task

import lm_tournament_eval.models

from lm_tournament_eval.loggers import EvaluationTracker
from lm_tournament_eval.loggers.utils import (
     add_env_info, 
     add_tokenizer_info, 
     get_git_commit_hash
)

from lm_tournament_eval.tasks import (
    TaskManager,
    get_task_dict
)
from lm_tournament_eval.caching.cache import delete_cache

from lm_tournament_eval.utils import (
    eval_logger,
    handle_non_serializable,
    hash_string,
    positional_deprecated,
    simple_parse_args_string,
)

from lm_tournament_eval.evaluator_utils import (
    consolidate_group_results,
    consolidate_results,
    get_sample_size,
    get_subtask_list,
    get_task_list,
    prepare_print_tasks,
    print_writeout,
    run_task_tests,
)

def tournament_evaluate(
    model_type,
    model0,
    model1,
    model0_args: Optional[Union[str, dict]] = None,
    model1_args: Optional[Union[str, dict]] = None,
    tasks: Optional[List[Union[str, dict, object]]] = None,
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
    task_manager: Optional[TaskManager] = None,
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

    if task_manager is None:
        task_manager = TaskManager(verbosity)

    task_dict = get_task_dict(tasks, task_manager)
    
    # helper function to recursively apply config overrides to leaf subtasks, skipping their constituent groups.
    # (setting of num_fewshot ; bypassing metric calculation ; setting fewshot seed)
    def _adjust_config(task_dict):
        adjusted_task_dict = {}
        for task_name, task_obj in task_dict.items():
            if isinstance(task_obj, dict):
                adjusted_task_dict = {
                    **adjusted_task_dict,
                    **{task_name: _adjust_config(task_obj)},
                }

            else:
                if task_obj.get_config("output_type") == "generate_until":
                    if gen_kwargs is not None:
                        task_obj.set_config(
                            key="generation_kwargs", value=gen_kwargs, update=True
                        )

                if predict_only:
                    eval_logger.info(
                        f"Processing {task_name} in output-only mode. Metrics will not be calculated!"
                    )
                    # we have to change the class properties post-hoc. This is pretty hacky.
                    task_obj.override_metric(metric_name="bypass")

                # override tasks' fewshot values to the provided num_fewshot arg value
                # except if tasks have it set to 0 manually in their configs--then we should never overwrite that
                if num_fewshot is not None:
                    if (default_num_fewshot := task_obj.get_config("num_fewshot")) == 0:
                        eval_logger.info(
                            f"num_fewshot has been set to 0 for {task_name} in its config. Manual configuration will be ignored."
                        )
                    else:
                        eval_logger.warning(
                            f"Overwriting default num_fewshot of {task_name} from {default_num_fewshot} to {num_fewshot}"
                        )
                        task_obj.set_config(key="num_fewshot", value=num_fewshot)
                else:
                    # if num_fewshot not provided, and the task does not define a default one, default to 0
                    if (
                        default_num_fewshot := task_obj.get_config("num_fewshot")
                    ) is None:
                        task_obj.set_config(key="num_fewshot", value=0)
                # fewshot_random_seed set for tasks, even with a default num_fewshot (e.g. in the YAML file)
                task_obj.set_fewshot_seed(seed=fewshot_random_seed)
                eval_logger.info(
                    f"Setting fewshot random generator seed to {fewshot_random_seed}"
                )

                adjusted_task_dict[task_name] = task_obj

        return adjusted_task_dict

    task_dict = _adjust_config(task_dict)

    results0 = evaluate(
        lm=lm0,
        task_dict=task_dict,
        limit=limit,
    )

    results1 = evaluate(
        lm=lm1,
        task_dict=task_dict,
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

def evaluate(
    lm: "LM",
    task_dict,
    limit: Optional[int] = None,
    cache_requests: bool = False,
    rewrite_requests_cache: bool = False,
    bootstrap_iters: Optional[int] = 100000,
    write_out: bool = False,
    log_samples: bool = True,
    system_instruction: Optional[str] = None,
    apply_chat_template: bool = False,
    fewshot_as_multiturn: bool = False,
    verbosity: str = "INFO",
):
    """Instantiate and evaluate a model on a list of tasks.

    :param lm: obj
        Language Model
    :param task_dict: dict[str, Task]
        Dictionary of tasks. Tasks will be taken to have name type(task).config.task .
    :param limit: int, optional
        Limit the number of examples per task (only use this for testing)
    :param bootstrap_iters:
        Number of iterations for bootstrap statistics, used when calculating stderr. Set to 0 for skipping all stderr calculations.
    :param write_out: bool
        If True, write out an example document and model input for checking task integrity
    :param log_samples: bool
        If True, write out all model outputs and documents for per-sample measurement and post-hoc analysis
    :param system_instruction: str
        System instruction to be applied to the prompt
    :param apply_chat_template: bool
        If True, apply chat template to the prompt
    :param fewshot_as_multiturn: bool
        Whether to provide the fewshot examples as a multiturn conversation or a single user turn.
    :return
        Dictionary of results
    """

    #breakpoint()

    eval_logger.setLevel(getattr(logging, f"{verbosity}"))

    # tracks all Instances/requests a model must generate output on.
    requests = defaultdict(list)
    # stores the amount to pad out reqs per req. type so that
    # number of fwd passes per distributed rank is equal
    padding_requests = defaultdict(int)

    # get lists of group hierarchy and each type of request
    eval_tasks = get_task_list(task_dict)
    if not log_samples:
        if not all(
            "bypass" not in getattr(task_output.task, "_metric_fn_list", {}).keys()
            for task_output in eval_tasks
        ):
            raise ValueError("log_samples must be True for 'bypass' metric-only tasks")
    for task_output in eval_tasks:
        task: Task = task_output.task
        limit = get_sample_size(task, limit)
        task.build_all_requests(
            limit=limit,
            rank=lm.rank,
            world_size=lm.world_size,
            cache_requests=cache_requests,
            rewrite_requests_cache=rewrite_requests_cache,
            system_instruction=system_instruction,
            apply_chat_template=apply_chat_template,
            fewshot_as_multiturn=fewshot_as_multiturn,
            chat_template=getattr(lm, "apply_chat_template")
            if apply_chat_template
            else None,
            tokenizer_name=getattr(lm, "tokenizer_name", "")
            if apply_chat_template
            else "",
        )
        eval_logger.debug(
            f"Task: {task_output.task_name}; number of requests on this rank: {len(task.instances)}"
        )
        if write_out:
            print_writeout(task)
        # aggregate Instances by LM method requested to get output.
        for instance in task.instances:
            reqtype = instance.request_type
            requests[reqtype].append(instance)

        if lm.world_size > 1:
            instances_rnk = torch.tensor(len(task._instances), device=lm.device)
            gathered_item = (
                lm.accelerator.gather(instances_rnk).cpu().detach().numpy().tolist()
            )
            # "multiple_choice" task types dispatch (several) "loglikelihood" request types
            reqtype = (
                "loglikelihood"
                if task.OUTPUT_TYPE == "multiple_choice"
                else task.OUTPUT_TYPE
            )
            # compute number of pseudo-batches to pad with (FSDP/DDP require even batches among ranks)
            numpad = max(gathered_item) - gathered_item[lm.rank]
            # todo: may not account for padding in cases like SquadV2 which has multiple req types
            padding_requests[reqtype] += numpad

    ### Run LM on inputs, get all outputs ###
    # execute each type of request
    for reqtype, reqs in requests.items():
        eval_logger.info(f"Running {reqtype} requests")
        # create `K` copies of each request `req` based off `K = req.repeats`
        cloned_reqs = []
        for req in reqs:
            cloned_reqs.extend([req] * req.repeats)

        if (lm.world_size > 1) and (padding_requests[reqtype] > 0):
            for _ in range(padding_requests[reqtype]):
                cloned_reqs.extend([req] * req.repeats)

        # run requests through model
        resps = getattr(lm, reqtype)(cloned_reqs)

        # put responses from model into a list of length K for each request.
        for x, req in zip(resps, cloned_reqs):
            req.resps.append(x)

        if lm.world_size > 1:
            lm.accelerator.wait_for_everyone()

    RANK = lm.rank
    WORLD_SIZE = lm.world_size
    ### Postprocess outputs ###
    # TODO: del model here, maybe (idea: allow user to specify device of e.g. reward model separately)
    for task_output in eval_tasks:
        task = task_output.task
        task.apply_filters()

        ### Collect values of metrics on all datapoints ###
        # # unpack results and sort back in order and return control to Task
        # TODO: make it possible to use a different metric per filter
        # Pre-process task.instances to group by doc_id
        instances_by_doc_id = defaultdict(list)
        for instance in task.instances:
            instances_by_doc_id[instance.doc_id].append(instance)
        # Sort instances within each group
        for instances in instances_by_doc_id.values():
            instances.sort(key=lambda x: x.idx)
        # iterate over different filters used
        for filter_key in task.instances[0].filtered_resps.keys():
            doc_iterator = task.doc_iterator(
                rank=RANK, limit=limit, world_size=WORLD_SIZE
            )
            for doc_id, doc in doc_iterator:
                requests = instances_by_doc_id[doc_id]
                metrics = task.process_results(
                    doc, [req.filtered_resps[filter_key] for req in requests]
                )
                if log_samples:
                    target = task.doc_to_target(doc)
                    example = {
                        "doc_id": doc_id,
                        "doc": doc,
                        "target": target,
                        "arguments": [req.args for req in requests],
                        "resps": [req.resps for req in requests],
                        "filtered_resps": [
                            req.filtered_resps[filter_key] for req in requests
                        ],
                        "doc_hash": hash_string(
                            json.dumps(
                                requests[0].doc,
                                indent=2,
                                default=handle_non_serializable,
                                ensure_ascii=False,
                            )
                        ),
                        "prompt_hash": hash_string(requests[0].arguments[0]),
                        "target_hash": hash_string(str(target)),
                    }
                    example.update(metrics)
                    task_output.logged_samples.append(example)
                for metric, value in metrics.items():
                    task_output.sample_metrics[(metric, filter_key)].append(value)

    if WORLD_SIZE > 1:
        # if multigpu, then gather data across all ranks to rank 0
        # first gather logged samples across all ranks
        for task_output in eval_tasks:
            if log_samples:
                # for task_name, task_samples in list(samples.items()):
                full_samples = [None] * WORLD_SIZE if RANK == 0 else None
                torch.distributed.gather_object(
                    obj=task_output.logged_samples,
                    object_gather_list=full_samples,
                    dst=0,
                )

                if RANK == 0:
                    task_output.logged_samples = list(
                        itertools.chain.from_iterable(full_samples)
                    )

            # then collect metrics across all ranks
            for metrics in task_output.sample_metrics:
                metric_list = [None] * WORLD_SIZE if RANK == 0 else None
                torch.distributed.gather_object(
                    obj=task_output.sample_metrics[metrics],
                    object_gather_list=metric_list,
                    dst=0,
                )
                if RANK == 0:
                    task_output.sample_metrics[metrics] = list(
                        itertools.chain.from_iterable(metric_list)
                    )

    if RANK == 0:
        ### Aggregate results over all datapoints ###
        # aggregate results ; run bootstrap CIs
        for task_output in eval_tasks:
            task_output.calculate_aggregate_metric(bootstrap_iters=bootstrap_iters)
        (
            results,
            samples,
            configs,
            versions,
            num_fewshot,
            higher_is_better,
        ) = consolidate_results(eval_tasks)

        ### Calculate group metrics ###
        if bool(results):
            results, versions, show_group_table, *_ = consolidate_group_results(
                results, versions, task_dict
            )

        results_agg, group_agg = prepare_print_tasks(task_dict, results)
        subtask_list = get_subtask_list(task_dict)

        # collect all higher_is_better values for metrics
        # in the group's subtasks.
        # TODO: clean this up ; unify with the below metric_list loop?
        _higher_is_better = {}
        for group, task_list in subtask_list.items():
            if (
                len(task_list) != 0
            ):  # subtask list will list "task_name": [] for solo tasks
                for task in task_list:
                    for m, h in higher_is_better[task].items():
                        if m not in _higher_is_better.keys():
                            _higher_is_better[m] = h

                        if (
                            m in _higher_is_better
                            and _higher_is_better[m] is not None
                            and _higher_is_better[m] != h
                        ):
                            eval_logger.warning(
                                f"Higher_is_better values for metric {m} in group {group} are not consistent. Defaulting to None."
                            )
                            _higher_is_better[m] = None
                higher_is_better[group] = _higher_is_better

        results_dict = {
            "results": dict(results_agg.items()),
            **(
                {"groups": dict(group_agg.items())}
                if (bool(group_agg) & show_group_table)
                else {}
            ),
            "group_subtasks": dict(reversed(subtask_list.items())),
            "configs": dict(sorted(configs.items())),
            "versions": dict(sorted(versions.items())),
            "n-shot": dict(sorted(num_fewshot.items())),
            "higher_is_better": dict(sorted(higher_is_better.items())),
            "n-samples": {
                task_output.task_name: {
                    "original": len(task_output.task.eval_docs),
                    "effective": min(
                        limit if limit else len(task_output.task.eval_docs),
                        len(task_output.task.eval_docs),
                    ),
                }
                for task_output in eval_tasks
            },
        }
        if log_samples:
            results_dict["samples"] = dict(samples)

        return results_dict

    else:
        return None
