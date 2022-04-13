import gym

from salina import get_arguments, get_class, instantiate_class
from salina.agents.gyma import AutoResetGymAgent, NoAutoResetGymAgent, GymAgent


#TODO: can we avoid repeating the same code while inheriting either from GymAgent, AutoResetGymAgent or NoAutoResetGymAgent?
class EnvAgent(GymAgent):
    # Create the environment agent
    # This agent should not be used?
    def __init__(self, cfg):
        super().__init__(get_class(cfg.gym_env), get_arguments(cfg.gym_env), n_envs=cfg.algorithm.n_envs)
        env = instantiate_class(cfg.gym_env)
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        del env

    def is_continuous_action(self):
        return isinstance(self.action_space, gym.spaces.Box)

    def is_discrete_action(self):
        return isinstance(self.action_space, gym.spaces.Discrete)

    def get_obs_and_actions_sizes(self):
        if self.is_continuous_action():
            # Return the size of the observation and action spaces of the environment
            # In the case of a continuous action environment
            return self.observation_space.shape[0], self.action_space.shape[0]
        elif self.is_discrete_action():
            # Return the size of the observation and action spaces of the environment
            return self.observation_space.shape[0], self.action_space.n
        else:
            print("unknown type of action space", self.action_space)
            return None


class AutoResetEnvAgent(AutoResetGymAgent):
    # Create the environment agent
    # This agent implements N gym environments with auto-reset
    def __init__(self, cfg):
        super().__init__(get_class(cfg.gym_env), get_arguments(cfg.gym_env), n_envs=cfg.algorithm.n_envs)
        env = instantiate_class(cfg.gym_env)
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        del env

    def is_continuous_action(self):
        return isinstance(self.action_space, gym.spaces.Box)

    def is_discrete_action(self):
        return isinstance(self.action_space, gym.spaces.Discrete)

    def get_obs_and_actions_sizes(self):
        if isinstance(self.action_space, gym.spaces.Box):
            # Return the size of the observation and action spaces of the environment
            # In the case of a continuous action environment
            return self.observation_space.shape[0], self.action_space.shape[0]
        elif isinstance(self.action_space, gym.spaces.Discrete):
            # Return the size of the observation and action spaces of the environment
            return self.observation_space.shape[0], self.action_space.n
        else:
            print("unknown type of action space", self.action_space)
            return None


class NoAutoResetEnvAgent(NoAutoResetGymAgent):
    # Create the environment agent
    # This agent implements N gym environments without auto-reset
    def __init__(self, cfg):
        super().__init__(get_class(cfg.gym_env), get_arguments(cfg.gym_env), n_envs=cfg.algorithm.n_envs)
        env = instantiate_class(cfg.gym_env)
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        del env

    def is_continuous_action(self):
        return isinstance(self.action_space, gym.spaces.Box)

    def is_discrete_action(self):
        return isinstance(self.action_space, gym.spaces.Discrete)

    def get_obs_and_actions_sizes(self):
        if isinstance(self.action_space, gym.spaces.Box):
            # Return the size of the observation and action spaces of the environment
            # In the case of a continuous action environment
            return self.observation_space.shape[0], self.action_space.shape[0]
        elif isinstance(self.action_space, gym.spaces.Discrete):
            # Return the size of the observation and action spaces of the environment
            return self.observation_space.shape[0], self.action_space.n
        else:
            print("unknown type of action space", self.action_space)
            return None
