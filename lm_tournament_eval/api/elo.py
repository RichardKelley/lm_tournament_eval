import math
from .match import MatchResult, Match
from typing import List, Dict
import logging
import csv
from lm_tournament_eval.score_database import ScoreDatabase
from lm_tournament_eval.api.match import Match

def argmax(iterable):
    return max(enumerate(iterable), key=lambda x: x[1])[0]

MAX_SCORE_DIFF = 800
RATING_FLOOR = 100

class ELO:
    def __init__(self, model0_key, model1_key, db : ScoreDatabase, unbounded : bool = True):
        self.model0_key = model0_key
        self.model1_key = model1_key
        self.db = db
        self.unbounded = unbounded

        if self.db.check_model_exists(*model0_key):
            self.score_0 = self.db.get_model_score(*model0_key)
        else:
            self.db.insert_model(*model0_key)
            self.score_0 = 1200.0
        
        if self.db.check_model_exists(*model1_key):
            self.score_1 = self.db.get_model_score(*model1_key)
        else:
            self.db.insert_model(*model1_key)
            self.score_1 = 1200.0

        self.k = 10
        self.soft_ceiling=3000
        self.decay_factor=0.01

    def _soft_ceiling(self, current_rating, rating_change):
        if current_rating < self.soft_ceiling:
            # Below the soft ceiling, apply the full rating change
            new_rating = current_rating + rating_change
        else:
            # Above the soft ceiling, apply a diminishing rating change
            distance_above_ceiling = current_rating - self.soft_ceiling
            damping_factor = math.exp(-self.decay_factor * distance_above_ceiling)
            adjusted_change = rating_change * damping_factor
            new_rating = current_rating + adjusted_change
        
        return new_rating


    def online_elo_update(self, match_id: int, m : Match, results0 : Dict, results1 : Dict):
        index = 0
        print(f"initial scores : score_0, 1 {self.score_0}, {self.score_1}")
        print("----------------------------")

        answers0 = []
        answers1 = []

        model0_old_elo = self.score_0
        model1_old_elo = self.score_1

        for i in range(0, len(results0["samples"][m.task]), m.match_size):
            match m.output_type:
                case 'generate_until':
                    for result0, result1 in zip(results0["samples"][m.task][i:i+m.match_size], results1["samples"][m.task][i:i+m.match_size]):
                        if 'exact_match' in result0.keys():
                            if result0['exact_match'] == 1.0:
                                answers0.append(1)
                            else:
                                answers0.append(0)
                            if result1['exact_match'] == 1.0:
                                answers1.append(1)
                            else:
                                answers1.append(0)
                        #ifeval case: average prompt_level_strict_acc and inst_level_strict_acc
                        elif 'prompt_level_strict_acc' in result0.keys():
                            answers0.append((int(result0['prompt_level_strict_acc']) + sum(result0['inst_level_strict_acc'])/len(result0['inst_level_strict_acc']))/2)
                            answers1.append((int(result1['prompt_level_strict_acc']) + sum(result1['inst_level_strict_acc'])/len(result0['inst_level_strict_acc']))/2)
                case 'loglikelihood' | "multiple_choice":
                    for result0, result1 in zip(results0["samples"][m.task][i:i+m.match_size], results1["samples"][m.task][i:i+m.match_size]):
                        if "acc_norm" in result0.keys():
                            if result0["acc_norm"] == 1.0:
                                answers0.append(1)
                            else:
                                answers0.append(0)
                        elif "acc" in result0.keys():
                            if result0["acc"] == 1.0:
                                answers0.append(1)
                            else:
                                answers0.append(0)

                        if "acc_norm" in result1.keys():
                            if result1["acc_norm"] == 1.0:
                                answers1.append(1)
                            else:
                                answers1.append(0)
                        elif "acc" in result1.keys():
                            if result1["acc"] == 1.0:
                                answers1.append(1)
                            else:
                                answers1.append(0)
                case 'loglikelihood_rolling':
                    # TODO check this logic on a real task...
                    for result0, result1 in zip(results0["samples"][m.task][i:i+m.match], results1["samples"][m.task][i:i+m.match_size]):
                        nll0 = [response[0] for response in result0["resps"]]
                        nll1 = [response[0] for response in result1["resps"]]
                        prediction0 = argmax(nll0)
                        prediction1 = argmax(nll1)
                        if prediction0 == result0["target"]:
                            answers0.append(1)
                        else:
                            answers0.append(0)
                        if prediction1 == result1["target"]:
                            answers1.append(1)
                        else:
                            answers1.append(0)

            # calculate the wins, losses, and draws
            as_0 = []
            as_1 = []
            winners = []

            for i in range(len(answers0)):
                # draw
                if answers0[i] == answers1[i]:
                    as_0.append(0)
                    as_1.append(0)
                    winners.append("draw")
                # model 1 won 
                elif answers0[i] > answers1[i]:
                    as_0.append(1)
                    winners.append("model0")
                # model 2 won
                elif answers0[i] < answers1[i]:
                    as_1.append(1)
                    winners.append("model1")

            self.db.record_instance_updates(match=m, 
                                            match_id=match_id, 
                                            samples0=results0["samples"][m.task],
                                            samples1=results1["samples"][m.task],
                                            elo_0=model0_old_elo,
                                            elo_1=model1_old_elo,
                                            winners=winners)
            
            score_diff_0 = min(max(self.score_0 - self.score_1, -MAX_SCORE_DIFF), MAX_SCORE_DIFF)
            expected_score_0 = 1 / (1 + 10**(score_diff_0 / 400))

            score_diff_1 = min(max(self.score_1 - self.score_0, -MAX_SCORE_DIFF), MAX_SCORE_DIFF)
            expected_score_1 = 1 / (1 + 10**(score_diff_1 / 400))            

            # update Elo
            if sum(as_0) > sum(as_1):
                if self.unbounded:
                    self.score_0 = self.score_0 + self.k*(1 - expected_score_0)
                    self.score_1 = self.score_1 + self.k*(0 - expected_score_1)
                else:
                    self.score_0 = max(self._soft_ceiling(self.score_0, self.k*(1 - expected_score_0)), RATING_FLOOR)
                    self.score_1 = max(self._soft_ceiling(self.score_1, self.k*(0 - expected_score_1)), RATING_FLOOR)
            elif sum(as_1) > sum(as_0):
                if self.unbounded:
                    self.score_0 = self.score_0 + self.k*(0 - expected_score_0)
                    self.score_1 = self.score_1 + self.k*(1 - expected_score_1)
                else:
                    self.score_0 = max(self._soft_ceiling(self.score_0, self.k*(0 - expected_score_0)), RATING_FLOOR)
                    self.score_1 = max(self._soft_ceiling(self.score_1, self.k*(1 - expected_score_1)), RATING_FLOOR)
            elif sum(as_0) == sum(as_1):
                if self.unbounded:
                    self.score_0 = self.score_0 + self.k*(0.5 - expected_score_0)
                    self.score_1 = self.score_1 + self.k*(0.5 - expected_score_1)           
                else:
                    self.score_0 = max(self._soft_ceiling(self.score_0, self.k*(0.5 - expected_score_0)), RATING_FLOOR)
                    self.score_1 = max(self._soft_ceiling(self.score_1, self.k*(0.5 - expected_score_1)), RATING_FLOOR)

            model0_old_elo = self.score_0
            model1_old_elo = self.score_1

            index += 1
            print(f"match {index} : score_0, 1 {self.score_0}, {self.score_1}")
            print("----------------------------")

    def offline_elo_update(self, results0, results1, task_names, task_indices):
        # run match
        # sample indices
        # calculate the wins, losses, and draws
        as_1 = []
        as_2 = []
        for i in task_indices:
            # draw
            if results0[i]['acc'] == results1[i]['acc']:
                as_1.append(0)
                as_2.append(0)
            # model 1 won 
            elif results0[i]['acc'] > results1[i]['acc']:
                as_1.append(1)
            # model 2 won
            elif results0[i]['acc'] < results1[i]['acc']:
                as_2.append(1)

        score_diff_0 = min(max(self.score_0 - self.score_1, -MAX_SCORE_DIFF), MAX_SCORE_DIFF)
        expected_score_0 = 1 / (1 + 10**(score_diff_0 / 400))

        score_diff_1 = min(max(self.score_1 - self.score_0, -MAX_SCORE_DIFF), MAX_SCORE_DIFF)
        expected_score_1 = 1 / (1 + 10**(score_diff_1 / 400))
        
        # update Elo
        if sum(as_1) > sum(as_2):
            self.score_0 = self.score_0 + self.k*(1 - expected_score_0)
            self.score_1 = self.score_1 + self.k*(0 - expected_score_1)
        elif sum(as_2) > sum(as_1):
            self.score_0 = self.score_0 + self.k*(0 - expected_score_0)
            self.score_1 = self.score_1 + self.k*(1 - expected_score_1)
        elif sum(as_1) == sum(as_2):
            self.score_0 = self.score_0 + self.k*(0.5 - expected_score_0)
            self.score_1 = self.score_1 + self.k*(0.5 - expected_score_1)           
        print(f"score_0, 1 {self.score_0}, {self.score_1}")
        print("----------------------------")