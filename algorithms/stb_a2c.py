from stable_baselines3 import A2C


class StbA2cAgent:
    def create_model(self, env, lr, n_inputs, fc_dims, n_actions,
                     gamma, observation_high, custom):
        return A2C("MlpPolicy", env, verbose=1, learning_rate=lr)

