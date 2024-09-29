from lm_tournament_eval import utils
from lm_tournament_eval.api.scheduler import (
    Scheduler,
    FileScheduler, 
    SamplingScheduler, 
    DefaultScheduler
)

import argparse
import logging
import sys
import os
import time

from typing import List, Tuple

def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--model0", "-m0", type=str, help="Name of first competing model.")
    parser.add_argument("--model0_args", type=str, help="Arguments for model 0.")
    parser.add_argument("--model1", "-m1", type=str, help="Name of second competing model.")
    parser.add_argument("--model1_args", type=str, help="Arguments for model 1.")
    parser.add_argument("--tasks", "-t", default=None, type=str, metavar="task1,task2")
    parser.add_argument("--filter", default='none', type=str, metavar="filter1,filter2")
    parser.add_argument("--num_rounds", default=1, type=int)
    parser.add_argument("--batch_size", "-b", default="1", type=str)
    parser.add_argument("--gen_kwargs", type=str, default=None, help=("String arguments for model generation on greedy_until tasks, e.g. `temperature=0,top_k=0,top_p=0`."))
    parser.add_argument("--match_size", default=1, type=int)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_path", "-o", type=str, default=".")
    parser.add_argument("--tournament_name", type=str, default="")
    parser.add_argument("--log_samples", "-s", type=bool, default=True)
    parser.add_argument("--system_instruction", type=str, default="")
    parser.add_argument("--apply_chat_template", type=bool, default=False)
    parser.add_argument("--verbosity", type=str, default="INFO", metavar="CRITICAL|ERROR|WARNING|INFO|DEBUG")
    parser.add_argument("--wandb_args", type=str, default="", help="Comma-separated string arguments passed to wandb.init")
    parser.add_argument("--random_seed", type=int, default=1234)
    parser.add_argument("--numpy_random_seed", type=int, default=1234)
    parser.add_argument("--torch_random_seed", type=int, default=1234)
    parser.add_argument("--fewshot_random_seed", type=int, default=1234)
    parser.add_argument("--include_path", type=str, default=None, metavar="DIR", 
                        help="Additional path to include if there are external tasks to include.")
    parser.add_argument("--trust_remote_code",
                        action="store_true",
                        help="Sets trust_remote_code to True to execute code to create HF Datasets from the Hub")
    parser.add_argument("--limit", type=int)

    parser.add_argument("--offline", type=bool, default=False, 
                        help="If True, run offline analysis of two output files from lm-evaluation-harness.")
    parser.add_argument("--offline_file_0", type=str, default="",
                        help="File path for first model results.")
    parser.add_argument("--offline_file_1", type=str, default="",
                        help="File path for second model results.")
        
    parser.add_argument("--file_schedule", type=str, default=None,
                        help="Path to a file containing a CSV of match indices.")
    parser.add_argument("--sampling_schedule", type=bool, default=None,
                        help="Triggers sampling with replacement.")
    
    parser.add_argument("--db_path", type=str, default="elo.db",
                        help="Path to sqlite3 database file.")

    parser.add_argument("--model_list", type=str, default=None,
                        help="Comma-separated list of models for roundrobin evaluation.")
    parser.add_argument("--model_arg_list", type=str, default=None,
                        help="Semicolon-separated list of model args")
    parser.add_argument("--roundrobin_file", type=str, default=None,
                        help="Path to a file containing models to use for roundrobin evaluation.")
    
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="Name of a Weights and Biases project to record elos at.")
    
    parser.add_argument("--elo_dynamics", type=str, default="unbounded", metavar="bounded|unbounded",
                        help="Whether to use bounded or unbounded elo updates. Default unbounded.")
    parser.add_argument("--k", type=int, default=10,
                        help="the scaling factor for elo")
    return parser

def setup_wandb(args) -> bool:
    if args.wandb_project is not None:
        import wandb
        wandb.init(project=args.wandb_project, name="run-" + str(time.time()))
        return True
    else:
        return False

def setup_roundrobin_models(args) -> List[Tuple]:
    if args.model_list is not None and args.roundrobin_file is not None:
        logging.error("At most one of model_list and roundrobin_file can be not None.")
        sys.exit()

    if args.roundrobin_file is None and args.model_list is None:
        logging.error(f"Need --model_list or --roundrobin_file.")
        sys.exit()

    model_name_list = []
    arg_list = []

    if args.roundrobin_file is not None:
        with open(args.roundrobin_file, 'r') as f:
            for line in f:
                if line[0] == "#": 
                    continue # skip comments
                assert(';' in line)
                model_str, args_str = line.split(';')
                model_str = model_str.strip()
                args_str = args_str.strip()
                model_name_list.append(model_str)
                arg_list.append(args_str)

    if args.model_list is not None:
    
        model_name_list = args.model_list.split(',')
        model_name_list = [name.strip() for name in model_name_list]

        if args.model_arg_list is not None:
            arg_list = args.model_arg_list.split(';')
            assert(len(arg_list) == len(model_name_list))
        else:
            arg_list = ['' for _ in model_name_list]

    ret = zip(model_name_list, arg_list)

    return list(ret)

def validate_tasks(args, task_manager):
    if args.include_path is not None:
        logging.info(f"Including path: {args.include_path}")

    if args.tasks is None:
        logging.error("Need to specify a task to evaluate.")    
        sys.exit()
    elif args.tasks == "list":
        print(task_manager.list_all_tasks())
        sys.exit()
    elif args.tasks == "list_groups":
        print(task_manager.list_all_tasks(list_subtasks=False, list_tags=False))
        sys.exit()
    elif args.tasks == "list_tags":
        print(task_manager.list_all_tasks(list_groups=False, list_subtasks=False))
        sys.exit()
    elif args.tasks == "list_subtasks":
        print(task_manager.list_all_tasks(list_groups=False, list_tags=False))
        sys.exit()
    else:
        if os.path.isdir(args.tasks):
            import glob

            task_names = []
            yaml_path = os.path.join(args.tasks, "*.yaml")
            for yaml_file in glob.glob(yaml_path):
                config = utils.load_yaml_config(yaml_file)
                task_names.append(config)
        else:
            task_list = args.tasks.split(",")
            task_names = task_manager.match_tasks(task_list)

            if set(task_list) == set(task_names):
                task_names = task_list

            for task in [task for task in task_list if task not in task_names]:
                if os.path.isfile(task):
                    config = utils.load_yaml_config(task)
                    task_names.append(config)
            task_missing = [
                task for task in task_list if task not in task_names and "*" not in task
            ]  # we don't want errors if a wildcard ("*") task name was used

            if task_missing:
                missing = ", ".join(task_missing)
                logging.error(
                    f"Tasks were not found: {missing}\n"
                    f"{utils.SPACING}Try `lm-tournament-eval --tasks list` for list of available tasks",
                )
                raise ValueError(
                    f"Tasks not found: {missing}. Try `lm-tournament-eval --tasks {{list_groups,list_subtasks,list_tags,list}}` to list out all available names for task groupings; only (sub)tasks; tags; or all of the above, or pass '--verbosity DEBUG' to troubleshoot task registration issues."
                )
            
    logging.info(f"Selected Tasks: {task_names}")

    return task_names

def setup_scheduler(args) -> Scheduler:
    
    # set up scheduler
    if args.file_schedule is not None and args.sampling_schedule is not None:
        logging.error("Cannot set file_schedule and sampling_schedule at same time.")
        sys.exit(1)

    if args.file_schedule is not None:
        logging.info("Using {args.file_schedule} for match schedule.")
        scheduler = FileScheduler(args.file_schedule)
    elif args.sampling_schedule is not None and args.sampling_schedule:

        logging.info("Using {args.match_size} for sample size.")
        scheduler = SamplingScheduler(rounds=args.num_rounds, match_size=args.match_size)
    else:
        scheduler = DefaultScheduler(rounds=args.num_rounds, match_size=args.match_size)

    return scheduler

def setup_trust_remote_code(args):
    # Respect user's value passed in via CLI, otherwise default to True and add to comma-separated model args
    if args.trust_remote_code:
        logging.info(
            "Passed `--trust_remote_code`, setting environment variable `HF_DATASETS_TRUST_REMOTE_CODE=true`"
        )
        # HACK: import datasets and override its HF_DATASETS_TRUST_REMOTE_CODE value internally,
        # because it's already been determined based on the prior env var before launching our
        # script--`datasets` gets imported by lm_eval internally before these lines can update the env.
        import datasets

        datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True

        args.model_args = args.model_args + ",trust_remote_code=True"

    return args

def setup_filter_list(args, task_names):
    if ',' in args.filter:
        filter_list = args.filter.split(',')
        if len(filter_list) != len(task_names):
            raise ValueError(
                f"Filter list length {len(filter_list)} does not match task list length {len(task_names)}. Provide one filter per task."
            )
    else:
        filter_list = [args.filter]

    return filter_list

def setup_batch_size(args):
    if args.batch_size == 'auto':
        return args.batch_size
    else:
        return int(args.batch_size)