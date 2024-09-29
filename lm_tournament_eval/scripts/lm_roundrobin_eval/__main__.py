import argparse
import json
import sys
import logging
import datetime
import os
import csv

from tqdm import tqdm
import wandb

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
    setup_roundrobin_models,
    setup_wandb,
    setup_batch_size
)

from lm_tournament_eval.api.model_utils import load_model, parse_model_name
from lm_tournament_eval.utils import simple_parse_args_string

def get_roundrobin_schedule(num_models):
    players = list(range(num_models))
    if num_models % 2 != 0:
        players.append(None)  # Add a dummy player if num_models is odd
    
    n = len(players)
    matches = []
    
    for round in range(n - 1):
        round_matches = []
        for i in range(n // 2):
            if players[i] is not None and players[n - 1 - i] is not None:
                round_matches.append((players[i], players[n - 1 - i]))
        matches.extend(round_matches)
        
        players = [players[0]] + [players[-1]] + players[1:-1]
    
    return matches


def run_roundrobin_eval():
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

    model_list = setup_roundrobin_models(args)

    logging.info(f"Dry run loading models to meta device...")
    for model in model_list:
        logging.info(f"Loading {model}")
        try:
            model_type, model_name = parse_model_name(model[0])
            kwargs = simple_parse_args_string(model[1])
            if "trust_remote_code" in kwargs:
                trust_remote_code = kwargs["trust_remote_code"]
            else:
                trust_remote_code = False
            model = load_model(model_type, 
                                model_name,
                                {"trust_remote_code" : trust_remote_code},
                                device="meta")
            logging.info(f"Successfully loaded {model}")
        except Exception as e:
            logging.error(e)
    
    tournament_idxs = get_roundrobin_schedule(len(model_list))
    tournament_list = [(model_list[i], model_list[j]) for (i,j) in tournament_idxs]

    pbar = tqdm(total=len(tournament_list), disable=False, desc="Running roundrobin tournament.")

    for t in tournament_list:
        model0 = t[0]
        model1 = t[1]

        tournament_name = f'{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.{model0[0]}.{model1[0]}'

        logging.info(f"Running tournament {tournament_name}")

        cfg = TournamentConfig(name=tournament_name,
                              rounds=args.num_rounds,
                              model0_name=model0[0],
                              model0_args=model0[1],
                              model1_name=model1[0],
                              model1_args=model1[1],
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
        pbar.update(1)

if __name__ == '__main__':
    run_roundrobin_eval()