import argparse
import json
import logging

from lm_tournament_eval.api.tournament import TournamentConfig, Tournament
from lm_tournament_eval.api.offline_tournament import OfflineTournamentConfig, OfflineTournament
from lm_tournament_eval.api.task import TaskConfig

def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--model1", "-m1", type=str, help="Name of first competing model.")
    parser.add_argument("--model2", "-m2", type=str, help="Name of second competing model.")
    parser.add_argument("--tasks", "-t", default=None, type=str, metavar="task1,task2")
    parser.add_argument("--num_rounds", default=1, type=int)
    parser.add_argument("--match_size", default=1, type=int)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_path", "-o", type=str, default=".")
    parser.add_argument("--log_samples", "-s", type=bool, default=True)
    parser.add_argument("--system_instruction", type=str, default="")
    parser.add_argument("--apply_chat_template", type=bool, default=False)
    parser.add_argument("--verbosity", type=str, default="INFO", metavar="CRITICAL|ERROR|WARNING|INFO|DEBUG")
    parser.add_argument("--wandb_args", type=str, default="", help="Comma-separated string arguments passed to wandb.init")
    parser.add_argument("--random_seed", type=int, default=1234)
    parser.add_argument("--numpy_random_seed", type=int, default=1234)
    parser.add_argument("--torch_random_seed", type=int, default=1234)

    parser.add_argument("--offline", type=bool, default=False, 
                        help="If True, run offline analysis of two output files from lm-evaluation-harness.")
    parser.add_argument("--offline_file_1", type=str, default="",
                        help="File path for first model results.")
    parser.add_argument("--offline_file_2", type=str, default="",
                        help="File path for second model results.")

    return parser


def run_tournament():
    print("Running tournament!")

    # handle arguments.
    parser = setup_parser()
    args = parser.parse_args()

    # set up local logger.
    # set up wandb logger.

    if args.offline == True:
        # validate tournament parameters.
        task_config = TaskConfig()
        cfg = OfflineTournamentConfig(name="test_offline_tournament",
                                      offline_file_1=args.offline_file_1,
                                      offline_file_2=args.offline_file_2,
                                      task_name="test_task_name",
                                      rounds=args.num_rounds,
                                      num_samples=args.match_size,
                                      task_config=task_config
                                     )
        # create offline tournament
        tournament = OfflineTournament(cfg)

        # run tournament evaluator.
        result = tournament.run_tournament()    
    else:
        # validate tournament parameters.
        cfg = TournamentConfig()

        # create tournament
        tournament = Tournament(cfg)

        # run tournament evaluator.
        result = tournament.run_tournament()

    # save tournament results to disk.

    print("done!")

if __name__ == "__main__":
    run_tournament()
