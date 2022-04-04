from salina import Agent
import torch
import torch.nn as nn 

from salina import Agent, get_arguments, instantiate_class, Workspace, get_class, instantiate_class

class EnvAgent(NoAutoResetGymAgent):
  # Create the environment agent
  # This agent implements N gym environments with auto-reset
  def __init__(self, cfg, n_envs):
    super().__init__(
      get_class(cfg.env),
      get_arguments(cfg.env),
      n_envs=n_envs
    )
    env = instantiate_class(cfg.env)
    self.observation_space=env.observation_space
    self.action_space=env.action_space
    del(env)

  def get_obs_and_actions_sizes(self):
    return self.observation_space.shape[0], self.action_space.shape[0]

  def get_obs_and_actions_sizes_discrete(self):
    return self.observation_space.shape[0], self.action_space.n
