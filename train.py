import sys

import domrl.engine.cards.base as base
import numpy as np
from agent import DominionRLAgent
from algorithms.stb import Stb
from algorithms.stb_dqn import StbDqnAgent
from domrl.agents.big_money_agent import BigMoneyAgent
from domrl.agents.provincial_agent import ProvincialAgent
from domrl.engine.agent import Agent
from domrl.engine.game import Game
from domrl.engine.state_view import StateView
from pandas import DataFrame
from plot import plot_learning_curve


def play_a_game(rl_agent, opponent_agent, scores, test_mode=False):
    game = Game(
        agents=[rl_agent, opponent_agent],
        kingdoms=[base.BaseKingdom],
        verbose=False,
    )
    rl_player = game.state.players[0]
    opponent_player = game.state.players[1]
    rl_agent.consider_supply(game.state.supply_piles)
    end_state = game.run()
    end_state_view = StateView(end_state, rl_player)
    is_draw = len(end_state.get_winners()) != 1
    is_win = not is_draw and end_state.get_winners()[0] is rl_player
    rl_vp = rl_player.total_vp()
    opp_vp = opponent_player.total_vp()
    scores.append(0.5 if is_draw
                  else 1 if is_win
                  else 0)
    if not test_mode:
        rl_agent.register_game_result(end_state_view, is_draw, is_win, rl_vp, opp_vp)


def train(rl_agent, n_games=200, opponent_agent: Agent = BigMoneyAgent(),
          plot=False, test_mode=False):
    scores = []
    avg_score_last_fifty = None

    for i in range(n_games):
        play_a_game(rl_agent, opponent_agent, scores, test_mode)

        if (i+1) % 10 == 0:
            avg_score = np.mean(scores[-10:])
            print('episode ', i+1,
                  'average score  %.2f' % avg_score)
        if (i+1) % 50 == 0:
            avg_score_last_fifty = np.mean(scores[-50:])
            print('episode ', i+1,
                  'average score last fifty  %.2f' % avg_score_last_fifty)
        sys.stdout.flush()
        if avg_score_last_fifty == 0:
            # dead end, no point in continuing
            break

    if plot and avg_score_last_fifty != 0:
        figure_file = f'plots/{rl_agent.name}.png'
        x = [i+1 for i in range(n_games)]
        plot_learning_curve(x, scores, figure_file)
    return avg_score_last_fifty


def train_various(params_list, n_repetitions, n_games):
    results_arr = []
    for params in params_list:
        print(f'starting {params}')
        param_scores = []
        for i in range(n_repetitions):
            rl_agent = DominionRLAgent(*params)
            score = train(rl_agent, n_games=n_games)
            param_scores.append(score)
            if score > 0.6:
                rl_agent.save(score)
        param_avg = np.mean(param_scores)
        param_max = np.max(param_scores)
        param_result = dict(
            agent=params[0].__name__,
            gamma=params[1],
            lr=params[2],
            dims=str(params[3]),
            max=param_max,
            avg=param_avg
        )
        results_arr.append(param_result)
    df = DataFrame(results_arr)
    df = df.sort_values('avg', ascending=False)
    print(df)


def experiment():
    params_list = [
        # [DeepQAgent, 0.95, 4e-8, [128]],
        # [ActorCriticAgent, 0.99, 7e-10, [128]],
        [Stb, 0.99, 12e-11, [1000, 1000], {"type": StbDqnAgent, "learning_starts": 50000}],
        [Stb, 0.99, 7e-9, [1000, 1000], {"type": StbDqnAgent, "learning_starts": 50000}],
    ]
    train_various(params_list, n_repetitions=20, n_games=2000)


def test_saved_model():
    filename = 'models/StbDqnAgent_7e-09_[1000, 1000]_0.75.model'
    agent = DominionRLAgent(Stb, gamma=0.99, lr=7e-9, fc_dims=[1000, 1000],
                            custom={"type": StbDqnAgent, "learning_starts": 50000})
    agent.load(filename)
    score = train(agent, n_games=1000, plot=False,
                  opponent_agent=ProvincialAgent())
    print(score)


if __name__ == '__main__':
    experiment()
    # test_saved_model()
