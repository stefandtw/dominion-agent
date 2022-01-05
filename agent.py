from typing import Union

import torch
from domrl.agents.big_money_agent import get_minimum_coin_card
from domrl.agents.provincial_agent import ProvincialAgent
from domrl.engine.agent import Agent
from domrl.engine.decision import *
from domrl.engine.state_view import StateView
from featurizer import Featurizer

PlayDecision = Union[ActionPhaseDecision, TreasurePhaseDecision]


def find_card_in_decision(decision, card_name):
    """ modified version of big_money_agent.find_card_in_decision() """
    if isinstance(decision, PlayDecision.__args__):
        for idx, move in enumerate(decision.moves):
            if hasattr(move, 'card') and move.card.name == card_name:
                return [idx]
    elif isinstance(decision, BuyPhaseDecision):
        for idx, move in enumerate(decision.moves):
            if hasattr(move, 'card_name') and move.card_name == card_name:
                return [idx]
    if card_name == 'No Buy':
        return [0]
    raise Exception(f"Card not offered in decision object: {card_name}")


class DominionRLAgent(Agent):
    def __init__(self, agent_type, gamma=None, lr=None, fc_dims=None, custom={}):
        self.featurizer = Featurizer()
        self.turn = 0
        self.supply_piles = None
        self.prev_features = None
        self.prev_action = None
        self.learn_at_each_step = ('learn_at_each_step' in custom)
        inputs = self.featurizer.n_features
        self.actions = ("No Buy", "Province", "Duchy", "Estate",
                        "Gold", "Silver", "Smithy",
                        "Witch", "Village", "Market", "Chapel", "Bandit",
                        "Laboratory", "Festival", "Council Room"
                        )
        self.n_actions = len(self.actions)
        self.agent = agent_type(gamma=gamma, lr=lr, n_inputs=inputs, n_actions=self.n_actions,
                                fc_dims=fc_dims, observation_high=self.featurizer.observation_high,
                                custom=custom)
        self.name = f'{"type" in custom and custom["type"].__name__ or agent_type.__name__}_{lr}_{fc_dims}'

    def consider_supply(self, supply_piles):
        self.supply_piles = supply_piles

    def policy(self, decision: Decision, state_view: StateView):
        # modified version of provincial_agent.policy()
        ######################## GATHER STATE DATA #########################
        if state_view.player.phase == TurnPhase.END_PHASE:
            self.turn += 1
        ######################## END GATHER STATE DATA #########################
        ######################## DEFAULT #########################
        if not decision.optional and len(decision.moves) == 1:
            return [0]
        if decision.optional and len(decision.moves) == 0:
            return []
        ###################### END DEFAULT #######################

        ######################## REACTIONS #######################
        # militia/bandit is played
        if decision.prompt == 'Select a card to trash from enemy Bandit.' or \
                decision.prompt == 'Discard down to 3 cards.':
            return get_minimum_coin_card(decision)

        #################### END REACTIONS #######################

        ######################## PHASES ##########################
        if state_view.player.phase == TurnPhase.TREASURE_PHASE:
            return [1]

        if state_view.player.phase == TurnPhase.BUY_PHASE:
            return self.choose_buy(decision, state_view)

        if state_view.player.phase == TurnPhase.ACTION_PHASE:
            provincial_agent = ProvincialAgent()
            return provincial_agent.policy(decision, state_view)
        ###################### END PHASES ########################

        return [0] # default action

    def choose_buy(self, decision, state_view):
        features = self.featurizer.featurize(state_view, self.turn)
        if self.learn_at_each_step and self.prev_features is not None:
            self.agent.learn(self.prev_features, reward=0, state_=features, done=False,
                             action=self.prev_action)
        self.prev_features = features

        # validate the card is available and can be bought
        valid_cards = ['No Buy']
        coins = state_view.player.coins
        for card, pile in state_view.supply_piles.items():
            if pile.buyable and pile.qty > 0 and self.supply_piles[card].card.cost <= coins:
                valid_cards.append(card)
        valid_cards_tensor = torch.zeros(self.n_actions, dtype=torch.float)
        for i, card in enumerate(self.actions):
            if card in valid_cards:
                valid_cards_tensor[i] = 1

        action_i = self.agent.choose_action(features, valid_cards_tensor)
        action = self.actions[action_i]
        if action not in valid_cards:
            raise Exception(f"Chosen action not valid: {action}")
        self.prev_action = action_i
        return find_card_in_decision(decision, action)

    def register_game_result(self, state_view, is_draw, is_win, rl_vp, opp_vp):
        features = self.featurizer.featurize(state_view, self.turn)
        reward = self.calc_reward(is_draw, is_win, opp_vp, rl_vp)
        cards_info = self.get_cards_info(state_view)
        print(f"reward {round(reward, 2)},   {cards_info}")
        self.agent.learn(self.prev_features, reward, features, done=True, action=self.prev_action)
        self.prev_features = None
        self.prev_action = None
        self.turn = 0

    def get_cards_info(self, state_view):
        cards_info_dict = {}
        for c in state_view.player.all_cards:
            if c.name not in cards_info_dict:
                cards_info_dict[c.name] = 0
            cards_info_dict[c.name] += 1
        cards_info = str(cards_info_dict)
        return cards_info

    def calc_reward(self, is_draw, is_win, opp_vp, rl_vp):
        reward_win_0_to_1 = 1 if is_win \
            else 0.5 if is_draw \
            else 0
        # give some incentive for vp to speed up learning
        min_vp = min(rl_vp, opp_vp)
        vp_to_reach_min_1 = 1 - min_vp if min_vp < 1 else 0
        rl_vp_min_1 = rl_vp + vp_to_reach_min_1
        opp_vp_min_1 = opp_vp + vp_to_reach_min_1
        reward_vp_0_to_1 = rl_vp_min_1 / (rl_vp_min_1 + opp_vp_min_1)
        reward = reward_vp_0_to_1 / 2 + reward_win_0_to_1
        return reward

    def save(self, info):
        self.agent.save(f'models/{self.name}_{info}.model')

    def load(self, filename):
        self.agent.load(filename)
