import os
import math
from typing import Tuple
from lm_tournament_eval.score_database import ScoreDatabase
from lm_tournament_eval.api.match import Match
from typing import Dict

class CustomTrueSkill:
    def __init__(self, model0_key, model1_key, db : ScoreDatabase, mu: float = 1200, sigma: float = 400 / 3, beta: float = 200, tau: float = 5, draw_probability: float = 0.1):
        self.mu = mu
        self.sigma = sigma
        self.beta = beta
        self.tau = tau
        self.hard_floor = 100
        self.soft_ceiling = 3000
        self.decay_factor = 0.1  # Adjust this to control the strength of the soft ceiling
        self.draw_probability = draw_probability
        self.db = db
        self.rank = int(os.environ.get('LOCAL_RANK',-1))

        if self.rank == 0 or self.rank == -1:
            if self.db.check_model_exists(*model0_key):
                self.score_0, self.sigma_0, _ = self.db.get_model_score(*model0_key)
                self.rating_0 = (self.score_0, self.sigma_0)
            else:
                self.db.insert_model(*model0_key)
                self.score_0 = self.mu
                self.sigma_0 = self.sigma
                self.rating_0 = (self.score_0, self.sigma_0)
            
            if self.db.check_model_exists(*model1_key):
                self.score_1, self.sigma_1, _ = self.db.get_model_score(*model1_key)
                self.rating_1 = (self.score_1, self.sigma_1)
            else:
                self.db.insert_model(*model1_key)
                self.score_1 = self.mu
                self.sigma_1 = self.sigma
                self.rating_1 = (self.score_1, self.sigma_1)

    def update_rating_win_loss(self, winner: Tuple[float, float], loser: Tuple[float, float]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        winner_mu, winner_sigma = winner
        loser_mu, loser_sigma = loser

        c = math.sqrt(2 * self.beta * self.beta + winner_sigma * winner_sigma + loser_sigma * loser_sigma)
        v = self.v_win(winner_mu, loser_mu, c)
        w = self.w_win(winner_mu, loser_mu, c)

        winner_mu_new = winner_mu + (winner_sigma * winner_sigma / c) * v
        loser_mu_new = loser_mu - (loser_sigma * loser_sigma / c) * v

        winner_sigma_new = math.sqrt(winner_sigma * winner_sigma * (1 - (winner_sigma * winner_sigma / (c * c)) * w))
        loser_sigma_new = math.sqrt(loser_sigma * loser_sigma * (1 - (loser_sigma * loser_sigma / (c * c)) * w))

        # Apply hard floor and soft ceiling
        winner_mu_new = max(self.hard_floor, min(winner_mu_new, self.apply_soft_ceiling(winner_mu_new)))
        loser_mu_new = max(self.hard_floor, min(loser_mu_new, self.apply_soft_ceiling(loser_mu_new)))

        # Apply dynamic update factor
        winner_mu_new = winner_mu + self.dynamic_update_factor(winner_mu) * (winner_mu_new - winner_mu)
        loser_mu_new = loser_mu + self.dynamic_update_factor(loser_mu) * (loser_mu_new - loser_mu)

        return (winner_mu_new, winner_sigma_new), (loser_mu_new, loser_sigma_new)

    def update_rating_draw(self, player1: Tuple[float, float], player2: Tuple[float, float]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        mu1, sigma1 = player1
        mu2, sigma2 = player2

        c = math.sqrt(2 * self.beta * self.beta + sigma1 * sigma1 + sigma2 * sigma2)
        v = self.v_draw(mu1, mu2, c)
        w = self.w_draw(mu1, mu2, c)

        mu1_new = mu1 + (sigma1 * sigma1 / c) * v
        mu2_new = mu2 - (sigma2 * sigma2 / c) * v

        sigma1_new = math.sqrt(sigma1 * sigma1 * (1 - (sigma1 * sigma1 / (c * c)) * w))
        sigma2_new = math.sqrt(sigma2 * sigma2 * (1 - (sigma2 * sigma2 / (c * c)) * w))

        # Apply hard floor and soft ceiling
        mu1_new = max(self.hard_floor, min(mu1_new, self.apply_soft_ceiling(mu1_new)))
        mu2_new = max(self.hard_floor, min(mu2_new, self.apply_soft_ceiling(mu2_new)))

        # Apply dynamic update factor
        mu1_new = mu1 + self.dynamic_update_factor(mu1) * (mu1_new - mu1)
        mu2_new = mu2 + self.dynamic_update_factor(mu2) * (mu2_new - mu2)

        return (mu1_new, sigma1_new), (mu2_new, sigma2_new)

    def v_win(self, winner_mu: float, loser_mu: float, c: float) -> float:
        return self.v((winner_mu - loser_mu) / c)

    def w_win(self, winner_mu: float, loser_mu: float, c: float) -> float:
        return self.w((winner_mu - loser_mu) / c)

    def v_draw(self, mu1: float, mu2: float, c: float) -> float:
        return self.v((mu1 - mu2) / c) * 2 * self.draw_probability

    def w_draw(self, mu1: float, mu2: float, c: float) -> float:
        return self.w((mu1 - mu2) / c) * 2 * self.draw_probability

    def v(self, x: float) -> float:
        return self.pdf(x) / self.cdf(x)

    def w(self, x: float) -> float:
        v = self.v(x)
        return v * (v + x)

    def pdf(self, x: float) -> float:
        return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

    def cdf(self, x: float) -> float:
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    def apply_soft_ceiling(self, rating: float) -> float:
        if rating > self.soft_ceiling:
            excess = rating - self.soft_ceiling
            decay = 1 - math.exp(-self.decay_factor * excess)
            return self.soft_ceiling + excess * decay
        return rating

    def dynamic_update_factor(self, rating: float) -> float:
        # Slower updates near the ceiling, faster updates near the floor
        return 1 - (rating - self.hard_floor) / (self.soft_ceiling - self.hard_floor)

    def create_results(self, match_id: int, m : Match, results0 : Dict, results1 : Dict):
        index = 0
        print(f"initial values: score_0, sigma_0: {self.score_0},{self.sigma_0} score_1, sigma_1: {self.score_1}, {self.sigma_1}")
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

            if sum(as_0) > sum(as_1):
                print("model0 won")
                self.rating_0, self.rating_1 = self.update_rating_win_loss(self.rating_0, self.rating_1)
            elif sum(as_1) > sum(as_0):
                print("model1 won")
                self.rating_1, self.rating_0 = self.update_rating_win_loss(self.rating_1, self.rating_0)
            elif sum(as_0) == sum(as_1):
                print("draw")
                self.rating_0, self.rating_1 = self.update_rating_draw(self.rating_0, self.rating_1)

            self.score_0, self.sigma_0 = self.rating_0
            self.score_1, self.sigma_1 = self.rating_1
            index += 1
            print(f"match {index} : score_0, sigma_0: {self.score_0},{self.sigma_0} score_1, sigma_1: {self.score_1}, {self.sigma_1}")
            print("----------------------------")
        