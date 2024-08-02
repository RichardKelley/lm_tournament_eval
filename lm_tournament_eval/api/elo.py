from .match import MatchResult, Match
from typing import List

def argmax(iterable):
    return max(enumerate(iterable), key=lambda x: x[1])[0]

class ELO:
    def __init__(self):
        # set the initial scores
        self.score_0 = 1200
        self.score_1 = 1200
        self.k = 16

    def online_elo_update(self, results0 : dict, results1 : dict, task_names : List):
        # self.match_result_list.model0_old_elo = self.score_0
        # self.match_result_list.model1_old_elo = self.score_1
        print(f"score_0, 1 {self.score_0}, {self.score_1}")
        print("----------------------------")
        #create list of len(num_samples) of correct/incorrect answers from results
        answers0 = [] 
        for task_name in task_names:
            for result in results0["samples"][task_name]:
                nll = [response[0][0] for response in result["resps"]]
                prediction = argmax(nll)
                if prediction == result["target"]:
                    answers0.append(1)
                else:
                    answers0.append(0)

        answers1 = [] 
        for task_name in task_names:
            for result in results1["samples"][task_name]:
                nll = [response[0] for response in result["resps"]]
                prediction = argmax(nll)
                if prediction == result["target"]:
                    answers1.append(1)
                else:
                    answers1.append(0)

        # calculate the wins, losses, and draws
        as_0 = []
        as_1 = []

        for i in range(len(answers0)):
            # draw
            if answers0[i] == answers1[i]:
                as_0.append(0)
                as_1.append(0)
            # model 1 won 
            elif answers0[i] > answers1[i]:
                as_0.append(1)
            # model 2 won
            elif answers0[i] < answers1[i]:
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
        print(f"score_0, 1 {self.score_0}, {self.score_1}")
        print("----------------------------")
        # self.match_result_list[n].model0_new_elo = self.score_0
        # self.match_result_list[n].model1_new_elo = self.score_1

    def offline_elo_update(results0, results1, task_names, task_indices):
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