import gym

from salina import get_arguments, get_class, instantiate_class
from salina.agents.gyma import AutoResetGymAgent, NoAutoResetGymAgent, GymAgent


#TODO: can we avoid repeating the same code while inheriting either from GymAgent, AutoResetGymAgent or NoAutoResetGymAgent?
class EnvAgent(GymAgent):
    # Create the environment agent
    # This agent should not be used?
    def __init__(self, cfg, n_envs):
        super().__init__(get_class(cfg.gym_env), get_arguments(cfg.gym_env), n_envs)
        env = instantiate_class(cfg.gym_env)
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        del env

    def is_continuous_action(self):
        return isinstance(self.action_space, gym.spaces.Box)

    def is_discrete_action(self):
        return isinstance(self.action_space, gym.spaces.Discrete)

    def is_continuous_state(self):
        return isinstance(self.observation_space, gym.spaces.Box)

    def is_discrete_state(self):
        return isinstance(self.observation_space, gym.spaces.Discrete)

    def get_obs_and_actions_sizes(self):
        action_dim = 0
        state_dim = 0
        if self.is_continuous_action():
            action_dim = self.action_space.shape[0]
        elif self.is_discrete_action():
            action_dim = self.action_space.n
        if self.is_continuous_state():
            state_dim = self.observation_space.shape[0]
        elif self.is_discrete_state():
            state_dim = self.observation_space.n
        return state_dim, action_dim


class AutoResetEnvAgent(AutoResetGymAgent):
    # Create the environment agent
    # This agent implements N gym environments with auto-reset
    def __init__(self, cfg, n_envs):
        super().__init__(get_class(cfg.gym_env), get_arguments(cfg.gym_env), n_envs)
        env = instantiate_class(cfg.gym_env)
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        del env

    def is_continuous_action(self):
        return isinstance(self.action_space, gym.spaces.Box)

    def is_discrete_action(self):
        return isinstance(self.action_space, gym.spaces.Discrete)

    def is_continuous_state(self):
        return isinstance(self.observation_space, gym.spaces.Box)

    def is_discrete_state(self):
        return isinstance(self.observation_space, gym.spaces.Discrete)

    def get_obs_and_actions_sizes(self):
        action_dim = 0
        state_dim = 0
        if self.is_continuous_action():
            action_dim = self.action_space.shape[0]
        elif self.is_discrete_action():
            action_dim = self.action_space.n
        if self.is_continuous_state():
            state_dim = self.observation_space.shape[0]
        elif self.is_discrete_state():
            state_dim = self.observation_space.n
        return state_dim, action_dim


class NoAutoResetEnvAgent(NoAutoResetGymAgent):
    # Create the environment agent
    # This agent implements N gym environments without auto-reset
    def __init__(self, cfg, n_envs):
        super().__init__(get_class(cfg.gym_env), get_arguments(cfg.gym_env), n_envs)
        env = instantiate_class(cfg.gym_env)
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        del env

    def is_continuous_action(self):
        return isinstance(self.action_space, gym.spaces.Box)

    def is_discrete_action(self):
        return isinstance(self.action_space, gym.spaces.Discrete)

    def is_continuous_state(self):
        return isinstance(self.observation_space, gym.spaces.Box)

    def is_discrete_state(self):
        return isinstance(self.observation_space, gym.spaces.Discrete)

    def get_obs_and_actions_sizes(self):
        action_dim = 0
        state_dim = 0
        if self.is_continuous_action():
            action_dim = self.action_space.shape[0]
        elif self.is_discrete_action():
            action_dim = self.action_space.n
        if self.is_continuous_state():
            state_dim = self.observation_space.shape[0]
        elif self.is_discrete_state():
            state_dim = self.observation_space.n
        return state_dim, action_dim
