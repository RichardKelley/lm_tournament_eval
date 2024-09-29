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
from lm_tournament_eval.api.scheduler import FileScheduler, SamplingScheduler, DefaultScheduler

from lm_tournament_eval.score_database import ScoreDatabase

from lm_tournament_eval.scripts.script_args import (
    setup_parser,
    validate_tasks,
    setup_scheduler,
    setup_trust_remote_code,
    setup_filter_list,
    setup_wandb,
    setup_batch_size
)

def run_tournament():

    parser = setup_parser()
    args = parser.parse_args()
    use_wandb = setup_wandb(args)
    args.batch_size = setup_batch_size(args)

    task_manager = TaskManager(args.verbosity, include_path=args.include_path)
    task_names = validate_tasks(args, task_manager)
    args = setup_trust_remote_code(args)

    rank = int(os.environ.get('LOCAL_RANK',-1))

    if rank == 0 or rank == -1:
        db = ScoreDatabase(args.db_path)
    else:
        db = None

    filter_list = setup_filter_list(args=args, task_names=task_names)
    scheduler = setup_scheduler(args)

    tournament_name = f'{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.{args.model0}.{args.model1}'

    # set up wandb logger.

    if args.offline == True:
        # validate tournament parameters.
        task_config = TaskConfig()
        cfg = OfflineTournamentConfig(name=tournament_name,
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
        cfg = TournamentConfig(name=tournament_name,
                              rounds=args.num_rounds,
                              model0_name=args.model0,
                              model0_args=args.model0_args,
                              model1_name=args.model1,
                              model1_args=args.model1_args,
                              task_names=task_names,
                              batch_size=args.batch_size,
                              gen_kwargs=args.gen_kwargs,
                              device=args.device,
                              limit=args.limit,
                              match_size=args.match_size,
                              cmd_filter=filter_list,
                              random_seed=args.random_seed,
                              numpy_random_seed=args.numpy_random_seed,
                              torch_random_seed=args.torch_random_seed,
                              fewshot_random_seed=args.fewshot_random_seed,
                              use_wandb=use_wandb,
                              elo_dynamics=args.elo_dynamics,
                              ranking_system=args.ranking_system
                             )

        tournament = Tournament(
            cfg, 
            task_names, 
            task_manager, 
            args.verbosity, 
            scheduler,
            db=db)
        if rank == 0 or rank == -1:
            db.record_tournament(tournament)

        tournament.run_tournament()


if __name__ == "__main__":
    run_tournament()
