from .match import MatchResult, Match
from typing import List, Dict
import logging
import csv

def argmax(iterable):
    return max(enumerate(iterable), key=lambda x: x[1])[0]

class ELO:
    def __init__(self, model0_key, model1_key, initial_elos=None, elo_out=None):

        if initial_elos is not None:
            self.initial_elos = initial_elos
            self.model0_name = model0_key[0]
            self.model0_bpw = model0_key[1]
            self.model1_name = model1_key[0]
            self.model1_bpw = model1_key[1]
            self.elo_out = elo_out
            # set the initial scores
            if model0_key in initial_elos.keys():
                self.score_0 = initial_elos[model0_key]
            else:
                self.score_0 = 1200

            if model1_key in initial_elos.keys():
                self.score_1 = initial_elos[model1_key]
            else:
                self.score_1 = 1200
        else:
            self.initial_elos = {
                model0_key : 1200,
                model1_key : 1200
            }
            self.score_0 = 1200
            self.score_1 = 1200
        self.k = 16

    def online_elo_update(self, results0 : Dict, results1 : Dict, task_names : List, match_size : int, match_results : Dict):
        index = 0
        print(f"match 0 : score_0, 1 {self.score_0}, {self.score_1}")
        print("----------------------------")
        #create list of len(num_samples) of correct/incorrect answers from results
        answers0 = {}
        answers1 = {}
        for task_name in task_names:
            match_results[task_name][0].model0_old_elo = self.score_0
            match_results[task_name][0].model1_old_elo = self.score_1
            for i in range(0, len(results0["samples"][task_name]), match_size):
                breakpoint()
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
                print(f"answers0: {answers0}")
                print(f"answers1: {answers1}")
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
                    # model 2 won
                    elif answers0[task_name][i] < answers1[task_name][i]:
                        as_1.append(1)

                expected_score_0 = 1/(1+10**((self.score_0-self.score_1)/400))
                expected_score_1 = 1/(1+10**((self.score_1-self.score_0)/400))

                # update Elo
                if sum(as_0) > sum(as_1):
                    self.score_0 = self.score_0 + self.k*(1 - expected_score_0)
                    self.score_1 = self.score_1 + self.k*(0 - expected_score_1)
                elif sum(as_1) > sum(as_0):
                    self.score_0 = self.score_0 + self.k*(0 - expected_score_0)
                    self.score_1 = self.score_1 + self.k*(1 - expected_score_1)
                elif sum(as_0) == sum(as_1):
                    self.score_0 = self.score_0 + self.k*(0.5 - expected_score_0)
                    self.score_1 = self.score_1 + self.k*(0.5 - expected_score_1)           
                match_results[task_name][index].model0_old_elo = self.score_0
                match_results[task_name][index].model1_old_elo = self.score_1
                index += 1
                print(f"match {index} : score_0, 1 {self.score_0}, {self.score_1}")
                print("----------------------------")
        
        if self.elo_out is not None:
            print(f"elo_out file = {self.elo_out}")

            logging.info(f"Writing new ELO score for {self.model0_name}.")
            self.initial_elos[(self.model0_name, self.model0_bpw)] = self.score_0
            logging.info(f"Writing new ELO score for {self.model1_name}.")
            self.initial_elos[(self.model1_name, self.model1_bpw)] = self.score_1

            with open(self.elo_out, 'w') as f:
                writer = csv.writer(f, delimiter=',')
                for (k, bpw), v in self.initial_elos.items():
                    writer.writerow([k, bpw, v])

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
        expected_score_0 = 1/(1+10**((self.score_0-self.score_1)/400))
        expected_score_1 = 1/(1+10**((self.score_1-self.score_0)/400))
        
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