import gym

from salina import get_arguments, get_class, instantiate_class
from salina.agents.gyma import AutoResetGymAgent, NoAutoResetGymAgent, GymAgent


class EnvAgent(GymAgent):
    # Create the environment agent
    # This agent implements N gym environments with auto-reset
    def __init__(self, cfg):
        super().__init__(get_class(cfg.env), get_arguments(cfg.gym_env), n_envs=cfg.algorithm.n_envs)
        env = instantiate_class(cfg.gym_env)
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        del env

    def get_obs_and_actions_sizes(self):
        if isinstance(self.action_space, gym.spaces.Box):
            # Return the size of the observation and action spaces of the environment
            # In the case of a continuous action environment
            return self.observation_space.shape[0], self.action_space.shape[0]
        elif isinstance(self.action_space, gym.spaces.Discrete):
            # Return the size of the observation and action spaces of the environment
            return self.observation_space.shape[0], self.action_space.n
        else:
            print ("unknown type of action space", self.action_space)
            return None


class AutoResetEnvAgent(AutoResetGymAgent):
    # Create the environment agent
    # This agent implements N gym environments with auto-reset
    def __init__(self, cfg, n_envs):
        super().__init__(get_class(cfg.env), get_arguments(cfg.gym_env), n_envs=cfg.algorithm.n_envs)
        env = instantiate_class(cfg.gym_env)
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        del env

    def get_obs_and_actions_sizes(self):
        if isinstance(self.action_space, gym.spaces.Box):
            # Return the size of the observation and action spaces of the environment
            # In the case of a continuous action environment
            return self.observation_space.shape[0], self.action_space.shape[0]
        elif isinstance(self.action_space, gym.spaces.Discrete):
            # Return the size of the observation and action spaces of the environment
            return self.observation_space.shape[0], self.action_space.n
        else:
            print ("unknown type of action space", self.action_space)
            return None


class NoAutoResetEnvAgent(NoAutoResetGymAgent):
    # Create the environment agent
    # This agent implements N gym environments with auto-reset
    def __init__(self, cfg, n_envs):
        super().__init__(get_class(cfg.env), get_arguments(cfg.gym_env), n_envs=cfg.algorithm.n_envs)
        env = instantiate_class(cfg.gym_env)
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        del env

    def get_obs_and_actions_sizes(self):
        if isinstance(self.action_space, gym.spaces.Box):
            # Return the size of the observation and action spaces of the environment
            # In the case of a continuous action environment
            return self.observation_space.shape[0], self.action_space.shape[0]
        elif isinstance(self.action_space, gym.spaces.Discrete):
            # Return the size of the observation and action spaces of the environment
            return self.observation_space.shape[0], self.action_space.n
        else:
            print ("unknown type of action space", self.action_space)
            return None
