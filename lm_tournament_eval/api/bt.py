import numpy as np
from scipy.optimize import minimize
from lm_tournament_eval.score_database import ScoreDatabase

class BradleyTerryModel:
    def __init__(self, model0_key, model1_key, db: ScoreDatabase, scale_factor=400):
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

    def add_result(self, outcome):
        self.results.append((0, 1, outcome))

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

    def create_results(self, results0, results1, task_names, match_size):
        print(f"Initial scores: {self.model0_key}: {self.strengths[0]:.2f}, {self.model1_key}: {self.strengths[1]:.2f}")
        print("----------------------------")
        index = 0 
        for task_name in task_names:
            task_results0 = results0["samples"][task_name]
            task_results1 = results1["samples"][task_name]
            
            for i in range(0, len(task_results0), match_size):
                answers0 = []
                answers1 = []
                
                batch = task_results0[i:i+match_size]
                for j, (result0, result1) in enumerate(zip(batch, task_results1[i:i+match_size])):
                    if results0['configs'][task_name]['output_type'] == 'generate_until':
                        answers0.append(1 if result0['exact_match'] == 1.0 else 0)
                        answers1.append(1 if result1['exact_match'] == 1.0 else 0)
                    else:
                        if "acc_norm" in result0:
                            answers0.append(1 if result0["acc_norm"] == 1.0 else 0)
                            answers1.append(1 if result1["acc_norm"] == 1.0 else 0)
                        elif "acc" in result0:
                            answers0.append(1 if result0["acc"] == 1.0 else 0)
                            answers1.append(1 if result1["acc"] == 1.0 else 0)
                
                # Calculate the outcome for this batch
                sum0 = sum(answers0)
                sum1 = sum(answers1)
                if sum0 > sum1:
                    self.add_result(1)  # model0 won
                    outcome = "Model0 won"
                elif sum0 < sum1:
                    self.add_result(0)  # model1 won
                    outcome = "Model1 won"
                else:
                    self.add_result(0.5)  # draw
                    outcome = "Draw"
                
                # Fit the model after each batch
                old_strengths = self.strengths.copy()
                self.fit()
        
                self.score_0 = self.strengths[0]
                self.score_1 = self.strengths[1]
                index += 1 
                print(f"match {index} : score_0, 1 {self.score_0}, {self.score_1}")
                print("----------------------------")