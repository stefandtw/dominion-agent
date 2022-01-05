import gym
import numpy as np
from gym import spaces
from stable_baselines3.common.logger import configure
from torch import Tensor


class StbGymEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, n_inputs, n_actions, observation_high, custom):
        super(StbGymEnv, self).__init__()
        self.action_space = spaces.Discrete(n_actions)

        low = np.zeros(n_inputs, dtype=int)
        self.observation_space = spaces.Box(low, observation_high, dtype=np.int64)

        self.observation_ = 0
        self.reward = 0
        self.done = False

    def step(self, action):
        return self.observation_, self.reward, self.done, {}

    def reset(self):
        return 0

    def render(self, mode='human'):
        pass

    def close(self):
        pass


class Stb:
    global_mask = None
    logger = configure('/tmp/stb', [])

    def __init__(self, lr, n_inputs, fc_dims, n_actions,
                 gamma, observation_high, custom, deterministic=True):
        self.adapter = custom['type']()
        self.env = StbGymEnv(n_inputs, n_actions, observation_high, custom)
        self.deterministic = deterministic
        self.model = self.adapter.create_model(self.env, lr, n_inputs, fc_dims,
                                               n_actions, gamma, observation_high, custom)
        self.model.set_logger(Stb.logger)
        self.obs = self.env.reset()

    def choose_action(self, observation, action_mask: Tensor):
        self.env.observation = observation
        self.env.action_mask = action_mask
        mask_numeric = []
        mask_0_1 = action_mask.numpy()
        for i in range(len(mask_0_1)):
            if mask_0_1[i] == 1:
                mask_numeric.append(i)
        action = None
        Stb.global_mask = action_mask
        while action not in mask_numeric:
            # repeat in case of invalid action because of random sampling
            action, _states = self.model.predict(observation,
                                                 deterministic=self.deterministic)
        return action

    def learn(self, state, reward, state_, done, action):
        self.env.state = state
        self.env.reward = reward
        self.env.state_ = state_
        self.env.done = done
        self.env.action = action
        self.model.learn(total_timesteps=1, log_interval=10, reset_num_timesteps=False)

    def save(self, filename):
        self.model.save(filename)

    def load(self, filename):
        self.model = self.model.load(filename)
