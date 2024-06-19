# this is a collection of matches, models, and a schedule of "play"

from dataclasses import dataclass, field

@dataclass
class TournamentConfig:
    name : str
    model1_name : str
    model2_name : str
    task_name : str
    rounds : int


class Tournament:
    def __init__(self, config : TournamentConfig):
        pass

    def run_tournament(self):
        print("Running tournament!")

        # for number of rounds
        # create match
        # run match
        # record results
        # update Elo

        return {}


    