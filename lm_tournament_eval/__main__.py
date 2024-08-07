import argparse
import json
import sys
import logging
import datetime
import os
import csv

from lm_tournament_eval import utils
from lm_tournament_eval.api.tournament import TournamentConfig, Tournament
from lm_tournament_eval.api.offline_tournament import OfflineTournamentConfig, OfflineTournament
from lm_tournament_eval.api.task import TaskConfig
from lm_tournament_eval.tasks import TaskManager
from lm_tournament_eval.evaluator_utils import request_caching_arg_to_dict

def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--model0", "-m0", type=str, help="Name of first competing model.")
    parser.add_argument("--model0_args", type=str, help="Arguments for model 0.")
    parser.add_argument("--model1", "-m1", type=str, help="Name of second competing model.")
    parser.add_argument("--model1_args", type=str, help="Arguments for model 1.")
    parser.add_argument("--tasks", "-t", default=None, type=str, metavar="task1,task2")
    parser.add_argument("--num_rounds", default=1, type=int)
    parser.add_argument("--batch_size", "-b", default=1, type=int)
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
    
    parser.add_argument("--elo_csv_in", type=str, default=None,
                        help="Path to CSV file with initial ELO scores.")
    #parser.add_argument("--save_scores", type=bool, default=False,
    #                    help="Write scores back out to input file.")
    parser.add_argument("--elo_csv_out", type=str, default=None,
                        help="Path to CSV file to write updated ELO scores.")

    return parser


def run_tournament():
    # print("Running tournament!")

    # handle arguments.
    parser = setup_parser()
    args = parser.parse_args()

    if args.include_path is not None:
        logging.info(f"Including path: {args.include_path}")
    task_manager = TaskManager(args.verbosity, include_path=args.include_path)

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
                    f"{utils.SPACING}Try `lm-eval --tasks list` for list of available tasks",
                )
                raise ValueError(
                    f"Tasks not found: {missing}. Try `lm-eval --tasks {{list_groups,list_subtasks,list_tags,list}}` to list out all available names for task groupings; only (sub)tasks; tags; or all of the above, or pass '--verbosity DEBUG' to troubleshoot task registration issues."
                )

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

    logging.info(f"Selected Tasks: {task_names}")

    # TODO support caching?
    #request_caching_args = request_caching_arg_to_dict(
    #    cache_requests=args.cache_requests
    #)

    args.tournament_name = "{}-{}-{}".format(datetime.datetime.now(), args.model0, args.model1)
    
    # set up local logger.
    # set up wandb logger.

    initial_elos = {}
    if args.elo_csv_in is not None:
        with open(args.elo_csv_in, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                model, bpw, elo = row
                initial_elos.update({(model, bpw): float(elo)})

    logging.info(f"Using initial elo scores {initial_elos}")

    if args.offline == True:
        # validate tournament parameters.
        task_config = TaskConfig()
        cfg = OfflineTournamentConfig(name=args.tournament_name,
                                      offline_file_0=args.offline_file_0,
                                      offline_file_1=args.offline_file_1,
                                      task_name=args.tasks,
                                      rounds=args.num_rounds,
                                      num_samples=args.match_size,
                                      task_config=task_config,
                                      model0_name = args.model0,
                                      model1_name = args.model1
                                     )
        # create offline tournament
        tournament = OfflineTournament(cfg)

        # run tournament evaluator.
        result = tournament.run_tournament()    
    else:
        # validate tournament parameters.
        cfg = TournamentConfig(name=args.tournament_name,
                              rounds=args.num_rounds,
                              model0_name=args.model0,
                              model0_args=args.model0_args,
                              model1_name=args.model1,
                              model1_args=args.model1_args,
                              task_names=task_names,
                              batch_size=args.batch_size,
                              device=args.device,
                              limit=args.limit,
                              match_size=args.match_size
                             )

        #create tournament
        tournament = Tournament(
            cfg, 
            task_names, 
            task_manager, 
            args.verbosity, 
            initial_elos,
            args.elo_csv_out)

        #logging.info(f"Running tournament {cfg}")
        tournament.run_tournament()

        #newline = '\n'
        # print(f"{results0}{newline*10}{results1}")

    # save tournament results to disk.


if __name__ == "__main__":
    run_tournament()
