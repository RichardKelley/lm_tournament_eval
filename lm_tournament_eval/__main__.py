import argparse
import json

def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model1", "-m1", type=str, help="Name of first competing model."
    )
    parser.add_argument(
        "--model2", "-m2", type=str, help="Name of second competing model."
    )

    parser.add_argument(
        "--tasks",
        "-t",
        default=None,
        type=str,
        metavar="task1,task2",
    )

    # --batch_size
    # --device
    # --output_path
    # --log_samples
    # --system_instruction
    # --apply_chat_template
    # --verbosity
    # --wandb_args
    # --random_seed
    # --numpy_random_seed
    # --torch_random_seed


def run_tournament():
    print("Running tournament!")

    # handle arguments.

    # set up local logger.

    # set up wandb logger.

    # validate tournament parameters.

    # validate task list.

    # run tournament evaluator.

    # save tournament results to disk.

if __name__ == "__main__":
    run_tournament()