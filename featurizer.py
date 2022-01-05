import numpy as np
import torch
from domrl.engine.decision import *
from domrl.engine.state_view import StateView


class Featurizer:
    def __init__(self):
        self.n_features = 6
        high = np.ones(self.n_features, dtype=int)
        high[0] = 8
        high[1] = 20
        high[2] = 8
        high[3] = 4
        high[4] = 4
        high[5] = 3
        self.observation_high = high

    def featurize(self, state_view: StateView, turn):
        deck = state_view.player.domain_obj.all_cards
        all_cards_money = [c.coins for c in deck] or [0]
        hand_money = np.mean(all_cards_money)*5
        hand_actions = np.mean([1 if c.is_type(CardType.ACTION) else 0 for c in deck])*5
        hand_action_producers = np.mean([c.add_actions for c in deck])*5
        hand_buy_producers = np.mean([c.add_buys for c in deck])*5
        features = torch.zeros([self.n_features], dtype=torch.float)
        coins = state_view.player.coins
        features[0] = coins
        features[1] = turn
        features[2] = hand_money
        features[3] = hand_actions
        features[4] = hand_action_producers
        features[5] = hand_buy_producers
        return features
