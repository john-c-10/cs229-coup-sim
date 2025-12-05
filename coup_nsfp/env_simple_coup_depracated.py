import numpy as np
from config import (
    NUM_CARD_TYPES,
    ACTIONS,
    MAX_TURNS_PER_GAME,
)


class SimpleCoup1v1Env:
    def __init__(self):
        self.reset()

    def reset(self):
        self.p1_cards = self._random_card_counts(total_cards=2)
        self.p2_cards = self._random_card_counts(total_cards=2)

        self.p1_coins = 2
        self.p2_coins = 2
        self.p1_influence = 2
        self.p2_influence = 2

        self.p1_action_hist = np.zeros(NUM_CARD_TYPES, dtype=np.int32)
        self.p2_action_hist = np.zeros(NUM_CARD_TYPES, dtype=np.int32)

        self.turn = 0
        self.current_player = 1  # 1 or 2

        return self._get_obs()

    def _random_card_counts(self, total_cards=2):
        counts = np.zeros(NUM_CARD_TYPES, dtype=np.int32)
        for _ in range(total_cards):
            idx = np.random.randint(0, NUM_CARD_TYPES)
            counts[idx] += 1
        return counts

    def _get_state_for_player(self, player: int):
        if player == 1:
            my_cards = self.p1_cards
            my_coins = self.p1_coins
            opp_coins = self.p2_coins
            my_inf = self.p1_influence
            opp_inf = self.p2_influence
            my_hist = self.p1_action_hist
            opp_hist = self.p2_action_hist
        else:
            my_cards = self.p2_cards
            my_coins = self.p2_coins
            opp_coins = self.p1_coins
            my_inf = self.p2_influence
            opp_inf = self.p1_influence
            my_hist = self.p2_action_hist
            opp_hist = self.p1_action_hist

        state = np.concatenate([
            my_cards.astype(np.float32),
            np.array([my_coins / 10.0], dtype=np.float32),
            np.array([opp_coins / 10.0], dtype=np.float32),
            np.array([float(my_inf)], dtype=np.float32),
            np.array([float(opp_inf)], dtype=np.float32),
            my_hist.astype(np.float32),
            opp_hist.astype(np.float32),
        ])
        return state

    def _get_obs(self):
        return self._get_state_for_player(self.current_player)

    def step(self, action: int):
        reward = 0.0
        done = False
        info = {}

        if self.current_player == 1:
            my_coins = self.p1_coins
            opp_coins = self.p2_coins
            my_inf = self.p1_influence
            opp_inf = self.p2_influence
            my_hist = self.p1_action_hist
            opp_hist = self.p2_action_hist
        else:
            my_coins = self.p2_coins
            opp_coins = self.p1_coins
            my_inf = self.p2_influence
            opp_inf = self.p1_influence
            my_hist = self.p2_action_hist
            opp_hist = self.p1_action_hist

        act_name = ACTIONS[action]

        if act_name == "INCOME":
            my_coins += 1
            reward += 0.05

        elif act_name == "FOREIGN_AID":
            my_coins += 2
            reward += 0.08

        elif act_name == "TAX":  # claim Duke
            my_coins += 3
            reward += 0.1
            my_hist[0] = min(2, my_hist[0] + 1)

        elif act_name == "STEAL":  # claim Captain
            steal_amt = min(2, opp_coins)
            my_coins += steal_amt
            opp_coins -= steal_amt
            reward += 0.1 * steal_amt
            my_hist[2] = min(2, my_hist[2] + 1)

        elif act_name == "EXCHANGE":  # claim Ambassador
            reward += 0.05
            my_hist[3] = min(2, my_hist[3] + 1)

        elif act_name == "ASSASSINATE":  # claim Assassin
            cost = 3
            if my_coins >= cost and opp_inf > 0:
                my_coins -= cost
                opp_inf -= 1
                reward += 1.0
                my_hist[1] = min(2, my_hist[1] + 1)
            else:
                reward -= 0.5

        elif act_name == "COUP":
            cost = 7
            if my_coins >= cost and opp_inf > 0:
                my_coins -= cost
                opp_inf -= 1
                reward += 1.2
            else:
                reward -= 0.5

        if self.current_player == 1:
            self.p1_coins = my_coins
            self.p2_coins = opp_coins
            self.p1_influence = my_inf
            self.p2_influence = opp_inf
            self.p1_action_hist = my_hist
            self.p2_action_hist = opp_hist
        else:
            self.p2_coins = my_coins
            self.p1_coins = opp_coins
            self.p2_influence = my_inf
            self.p1_influence = opp_inf
            self.p2_action_hist = my_hist
            self.p1_action_hist = opp_hist

        if opp_inf <= 0:
            reward += 1.0
            done = True
        elif my_inf <= 0:
            reward -= 1.0
            done = True

        self.turn += 1
        if self.turn >= MAX_TURNS_PER_GAME:
            done = True

        self.current_player = 2 if self.current_player == 1 else 1

        next_state = self._get_obs()
        return next_state, reward, done, info
