import numpy as np
from scipy.optimize import minimize
from lm_tournament_eval.score_database import ScoreDatabase
from lm_tournament_eval.api.match import Match
from typing import Dict

class BradleyTerryModel:
    def __init__(self, model0_key, model1_key, db: ScoreDatabase, scale_factor=10):
        self.scale_factor = scale_factor
        self.db = db
        self.soft_ceiling = 3000
        self.hard_floor = 100
        self.model0_key = model0_key
        self.model1_key = model1_key

        if self.db.check_model_exists(*model0_key):
            self.score_0, _, _ = self.db.get_model_score(*model0_key)
        else:
            self.db.insert_model(*model0_key)
            self.score_0 = 1200.0
        
        if self.db.check_model_exists(*model1_key):
            self.score_1, _, _ = self.db.get_model_score(*model1_key)
        else:
            self.db.insert_model(*model1_key)
            self.score_1 = 1200.0

        self.strengths = np.array([self.score_0, self.score_1])
        self.results = []

    def set_results(self, answers0, answers1):
        if sum(answers0) == sum(answers1):
            self.results.append((0, 1, 0.5))
        if sum(answers0) > sum(answers1):
            self.results.append((0, 1, 1))
        elif sum(answers0) < sum(answers1):
            self.results.append((0, 1, 0))

    def nll(self, strengths):
        nll = 0
        for p1, p2, outcome in self.results:
            p = 1 / (1 + 10**((strengths[p2] - strengths[p1]) / self.scale_factor))
            if outcome == 1:
                nll -= np.log(p)
            elif outcome == 0:
                nll -= np.log(1 - p)
            else:  # Draw
                nll -= np.log(0.5)
        return nll

    def apply_constraints(self, strengths):
        # Apply hard floor
        strengths = np.maximum(strengths, self.hard_floor)
        
        # Apply soft ceiling
        over_ceiling = strengths > self.soft_ceiling
        strengths[over_ceiling] = self.soft_ceiling + (strengths[over_ceiling] - self.soft_ceiling) * 0.1
        
        return strengths

    def fit(self):
        if len(self.results) > 0:
            result = minimize(self.nll, self.strengths, method='BFGS', options={'gtol': 1e-6, 'maxiter': 1000})
            if not result.success:
                print(f"Optimization failed: {result.message}")
            new_strengths = self.apply_constraints(result.x)
            # Ensure minimum change
            min_change = 1.0
            diff = new_strengths - self.strengths
            mask = np.abs(diff) < min_change
            diff[mask] = np.sign(diff[mask]) * min_change
            self.strengths = self.strengths + diff
        else:
            print("No results to fit. Keeping initial strengths.")

    def create_results(self, match_id: int, m : Match, results0 : Dict, results1 : Dict):
        index = 0
        print(f"initial values: score_0: {self.score_0} score_1, {self.score_1}")
        print("----------------------------")
        answers0 = []
        answers1 = []

        for i in range(0, len(results0["samples"][m.task]), m.match_size):
            match m.output_type:
                case 'generate_until':
                    for result0, result1 in zip(results0["samples"][m.task][i:i+m.match_size], results1["samples"][m.task][i:i+m.match_size]):
                        if result0['exact_match'] == 1.0:
                            answers0.append(1)
                        else:
                            answers0.append(0)
                        if result1['exact_match'] == 1.0:
                            answers1.append(1)
                        else:
                            answers1.append(0)
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
                # model 0 won 
                elif answers0[i] > answers1[i]:
                    as_0.append(1)
                    as_1.append(0)
                    winners.append("model0")
                # model 1 won
                elif answers0[i] < answers1[i]:
                    as_0.append(0)
                    as_1.append(1)
                    winners.append("model1")

            self.db.record_instance_updates(match=m, 
                                            match_id=match_id, 
                                            samples0=results0["samples"][m.task],
                                            samples1=results1["samples"][m.task],
                                            elo_0=self.score_0,
                                            elo_1=self.score_1,
                                            winners=winners)
                
            # Fit the model after each batch
            self.set_results(as_0, as_1)
            self.fit()    
            self.score_0 = self.strengths[0]
            self.score_1 = self.strengths[1]
            index += 1 
            print(f"match {index} : score_0, 1 {self.score_0}, {self.score_1}")
            print("----------------------------")