import numpy as np
from scipy.optimize import minimize

def argmax(iterable):
    return max(enumerate(iterable), key=lambda x: x[1])[0]

class BradleyTerryModel:
    def __init__(self, scale_factor=400, initial_rating=1200):
        self.scale_factor = scale_factor
        self.initial_rating = initial_rating
        self.strengths = np.full(2, self.initial_rating)


    def set_results(self, answers0, answers1):
        self.results = []
        for i in range(len(answers0)):
            if answers0[i] == answers1[i]:
                print("draw")
                self.results.append((0, 1, 0.5))
            if answers0[i] > answers1[i]:
                print("model 0 won")
                self.results.append((0, 1, 1))
            elif answers0[i] < answers1[i]:
                print("model 1 won")
                self.results.append((0, 1, 0))
        print(self.results)

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
        #create list of len(num_samples) of correct/incorrect answers from results
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
                        nll0 = [response[0][0] for response in result0["resps"]]
                        nll1 = [response[0][0] for response in result1["resps"]]
                        prediction0 = argmax(nll0)
                        prediction1 = argmax(nll1)
                        if prediction0 == result0["target"]:
                            answers0[task_name].append(1)
                        else:
                            answers0[task_name].append(0)
                        if prediction1 == result1["target"]:
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