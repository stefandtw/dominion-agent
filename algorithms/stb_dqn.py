import numpy as np
import torch as th
from algorithms.stb import Stb
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import QNetwork


class StbDqnAgent:
    maxfloat = np.finfo(np.float32).max

    def create_model(self, env, lr, n_inputs, fc_dims, n_actions,
                     gamma, observation_high, custom):
        net_arch = fc_dims or [32, 32]
        return DQN("MlpPolicy",
                   env, verbose=0, learning_rate=lr,
                   learning_starts=custom['learning_starts'],
                   policy_kwargs=dict(net_arch=net_arch)
                   )


def _predict(self, observation: th.Tensor, deterministic: bool = True) -> th.Tensor:
    """ overrides QNetwork._predict to support masking invalid actions """
    q_values = self.forward(observation)
    if Stb.global_mask is not None:
        illegal_action_penalty = (Stb.global_mask - 1) * StbDqnAgent.maxfloat
        q_values += illegal_action_penalty
    ###
    # Greedy action
    action = q_values.argmax(dim=1).reshape(-1)
    return action


QNetwork._predict = _predict
