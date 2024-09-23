import numpy as np
from scipy.optimize import minimize
from lm_tournament_eval.score_database import ScoreDatabase

def argmax(iterable):
    return max(enumerate(iterable), key=lambda x: x[1])[0]

class BradleyTerryModel:
    def __init__(self, model0_key, model1_key, db : ScoreDatabase, scale_factor=400):
        self.scale_factor = scale_factor
        self.db = db

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

        self.strengths = np.array([self.score_0, self.score_1])

    def set_results(self, answers0, answers1):
        self.results = []
        for i in range(len(answers0)):
            if answers0[i] == answers1[i]:
                self.results.append((0, 1, 0.5))
            if answers0[i] > answers1[i]:
                self.results.append((0, 1, 1))
            elif answers0[i] < answers1[i]:
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
    
    def fit(self):
        result = minimize(self.nll, self.strengths, method='BFGS')
        self.strengths = result.x

    def create_results(self, results0, results1, task_names, match_size):
        index = 0
        print(f"match 0 : score_0, 1 {self.strengths[0]}, {self.strengths[1]}")
        print("----------------------------")
        answers0 = {}
        answers1 = {}
        for task_name in task_names:
            for i in range(0, len(results0["samples"][task_name]), match_size):
                answers0[task_name] = []
                answers1[task_name] = []
                if results0['configs'][task_name]['output_type'] == 'generate_until':
                    for result0, result1 in zip(results0["samples"][task_name][i:i+match_size], results1["samples"][task_name][i:i+match_size]):
                        if result0['exact_match'] == 1.0:
                            answers0[task_name].append(1)
                        else:
                            answers0[task_name].append(0)
                        if result1['exact_match'] == 1.0:
                            answers1[task_name].append(1)
                        else:
                            answers1[task_name].append(0)
                else:    
                    for result0, result1 in zip(results0["samples"][task_name][i:i+match_size], results1["samples"][task_name][i:i+match_size]):
                        if "acc_norm" in result0.keys():
                            if result0["acc_norm"] == 1.0:
                                answers0[task_name].append(1)
                            else:
                                answers0[task_name].append(0)
                        elif "acc" in result0.keys():
                            if result0["acc"] == 1.0:
                                answers0[task_name].append(1)
                            else:
                                answers0[task_name].append(0)
                        if "acc_norm" in result1.keys():
                            if result1["acc_norm"] == 1.0:
                                answers1[task_name].append(1)
                            else:
                                answers1[task_name].append(0)
                        elif "acc" in result1.keys():
                            if result1["acc"] == 1.0:
                                answers1[task_name].append(1)
                            else:
                                answers1[task_name].append(0)
                # calculate the wins, losses, and draws
                as_0 = []
                as_1 = []

                for i in range(len(answers0[task_name])):
                    # draw
                    if answers0[task_name][i] == answers1[task_name][i]:
                        as_0.append(0)
                        as_1.append(0)
                    # model 1 won 
                    elif answers0[task_name][i] > answers1[task_name][i]:
                        as_0.append(1)
                        as_1.append(0)
                    # model 2 won
                    elif answers0[task_name][i] < answers1[task_name][i]:
                        as_0.append(0)
                        as_1.append(1)

                self.set_results(as_0, as_1)
                self.fit()
                index += 1
                print(f"match {index} : score_0, 1 {self.strengths[0]}, {self.strengths[1]}")
                print("----------------------------")
                self.score_0 = self.strengths[0]
                self.score_1 = self.strengths[1]