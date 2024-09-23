import math
from lm_tournament_eval.score_database import ScoreDatabase


class GlickoSystem:
    def __init__(self, model0_key, model1_key, db : ScoreDatabase, tau=0.5, initial_rd=350, initial_vol=0.06, floor=100, ceiling=3000, decay_factor=0.1):
        self.tau = tau
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

        self.initial_rating = 1200.0
        self.rd_0 = initial_rd
        self.rd_1 = initial_rd
        self.vol_0 = initial_vol
        self.vol_1 = initial_vol
        self.floor = floor
        self.ceiling = ceiling
        self.decay_factor = decay_factor

    def g(self, rd):
        return 1 / math.sqrt(1 + 3 * rd**2 / (math.pi**2))

    def E(self, rating, opponent_rating, opponent_rd):
        return 1 / (1 + math.exp(-self.g(opponent_rd) * (rating - opponent_rating) / 400))

    def update_player(self, player_rating, player_rd, player_vol, opponent_rating, opponent_rd, score):
        v_inv = 0
        delta = 0

        E = self.E(player_rating, opponent_rating, opponent_rd)
        g = self.g(opponent_rd)
        v_inv += g**2 * E * (1 - E)
        delta += g * (score - E)

        v = 1 / v_inv
        delta *= v

        a = math.log(player_vol**2)
        f = lambda x: (math.exp(x) * (delta**2 - player_rd**2 - v - math.exp(x)) / 
                       (2 * (player_rd**2 + v + math.exp(x))**2)) - \
                      (x - a) / (self.tau**2)

        # Find the root of f(x) using the Illinois algorithm
        A = a
        B = min(a, a - player_rd**2) if delta**2 > player_rd**2 + v else max(a, a - player_rd**2)
        fA, fB = f(A), f(B)
        while abs(B - A) > 1e-6:
            C = A + (A - B) * fA / (fB - fA)
            fC = f(C)
            if fC * fB < 0:
                A, fA = B, fB
            else:
                fA /= 2
            B, fB = C, fC

        new_vol = math.exp(A / 2)
        new_rd = math.sqrt(player_rd**2 + new_vol**2)
        new_rd = 1 / math.sqrt(1 / new_rd**2 + 1 / v)
        raw_change = new_rd**2 * (self.g(opponent_rd) * (score - self.E(player_rating, opponent_rating, opponent_rd)))

        if raw_change > 0:
            decay = math.exp(-self.decay_factor * max(0, player_rating - self.initial_rating) / (self.ceiling - self.initial_rating))
            raw_change *= decay

        new_rating = player_rating + raw_change

        # Apply floor and ceiling
        new_rating = max(self.floor, min(self.ceiling, new_rating))


        return new_rating, new_rd, new_vol
    
    def set_results(self, answers0, answers1):
        # calculate the wins, losses, and draws
        as_0 = []
        as_1 = []
        self.winners = []
        for i in range(len(answers0)):
            # draw
            if answers0[i] == answers1[i]:
                as_0.append(0)
                as_1.append(0)
                self.winners.append("draw")
            # model 0 won 
            elif answers0[i] > answers1[i]:
                as_0.append(1)
                self.winners.append("model0")
            # model 1 won
            elif answers0[i] < answers1[i]:
                as_1.append(1)
                self.winners.append("model1")

        if sum(as_0) > sum(as_1):
            #model0 won
            print("model0 won")
            self.score_0, self.rd_0, self.vol_0 = self.update_player(self.score_0, self.rd_0, self.vol_0, self.score_1, self.rd_1, 1.0)
            self.score_1, self.rd_1, self.vol_1 = self.update_player(self.score_1, self.rd_1, self.vol_1, self.score_0, self.rd_0, 0.0)
        elif sum(as_1) > sum(as_0):
            #model1 won
            print("model1 won")
            self.score_0, self.rd_0, self.vol_0 = self.update_player(self.score_0, self.rd_0, self.vol_0, self.score_1, self.rd_1, 0.0)
            self.score_1, self.rd_1, self.vol_1 = self.update_player(self.score_1, self.rd_1, self.vol_1, self.score_0, self.rd_0, 1.0)
        elif sum(as_0) == sum(as_1):
            #draw
            print("draw")
            self.score_0, self.rd_0, self.vol_0 = self.update_player(self.score_0, self.rd_0, self.vol_0, self.score_1, self.rd_1, 0.5)
            self.score_1, self.rd_1, self.vol_1 = self.update_player(self.score_1, self.rd_1, self.vol_1, self.score_0, self.rd_0, 0.5)


    def create_results(self, results0, results1, task_names, match_size):
        index = 0
        print(f"match {index} : score_0, 1 {self.score_0}, {self.score_1}")
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
                index += 1
                print(f"match {index} : score_0, 1 {self.score_0}, {self.score_1}")
                print("----------------------------")