from salina import get_arguments, get_class, instantiate_class
from salina.agents.gyma import AutoResetGymAgent, NoAutoResetGymAgent, GymAgent


class EnvAgent(GymAgent):
    # Create the environment agent
    # This agent implements N gym environments with auto-reset
    def __init__(self, cfg):
        super().__init__(get_class(cfg.env), get_arguments(cfg.env), n_envs=cfg.algorithm.n_envs)
        env = instantiate_class(cfg.env)
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        del env

    def get_obs_and_actions_sizes(self):
        return self.observation_space.shape[0], self.action_space.shape[0]

    def get_obs_and_actions_sizes_discrete(self):
        return self.observation_space.shape[0], self.action_space.n


class AutoResetEnvAgent(AutoResetGymAgent):
    # Create the environment agent
    # This agent implements N gym environments with auto-reset
    def __init__(self, cfg, n_envs):
        super().__init__(get_class(cfg.env), get_arguments(cfg.env), n_envs=n_envs)
        env = instantiate_class(cfg.env)
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        del env

    def get_obs_and_actions_sizes(self):
        return self.observation_space.shape[0], self.action_space.shape[0]

    def get_obs_and_actions_sizes_discrete(self):
        return self.observation_space.shape[0], self.action_space.n


class NoAutoResetEnvAgent(NoAutoResetGymAgent):
    # Create the environment agent
    # This agent implements N gym environments with auto-reset
    def __init__(self, cfg, n_envs):
        super().__init__(get_class(cfg.env), get_arguments(cfg.env), n_envs=n_envs)
        env = instantiate_class(cfg.env)
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        del env

    def get_obs_and_actions_sizes(self):
        return self.observation_space.shape[0], self.action_space.shape[0]

    def get_obs_and_actions_sizes_discrete(self):
        return self.observation_space.shape[0], self.action_space.n
