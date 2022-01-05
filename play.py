import domrl.engine.cards.base as base
from agent import DominionRLAgent
from algorithms.stb import Stb
from algorithms.stb_dqn import StbDqnAgent
from domrl.engine.agent import StdinAgent
from domrl.engine.game import Game


def load_agent():
    filename = 'models/StbDqnAgent_7e-09_[1000, 1000]_0.75.model'
    rl_agent = DominionRLAgent(Stb, gamma=0.99, lr=7e-9, fc_dims=[1000, 1000],
                               custom={"type": StbDqnAgent, "learning_starts": 50000})
    rl_agent.load(filename)
    return rl_agent


if __name__ == '__main__':
    agent = load_agent()
    game = Game(
        agents=[StdinAgent(), agent],
        kingdoms=[base.BaseKingdom],
    )
    agent.consider_supply(game.state.supply_piles)
    game.run()
    game.print_result()
