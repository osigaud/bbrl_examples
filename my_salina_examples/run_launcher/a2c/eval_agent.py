import sys
import os
import torch
import torch.nn as nn
import numpy as np

import gym
import my_gym

from omegaconf import OmegaConf
from salina.workspace import Workspace
from salina.agent import Agent
from salina.chrono import Chrono
from torch.distributions.normal import Normal
from salina.agents.gymb import AutoResetGymAgent, NoAutoResetGymAgent
from salina import instantiate_class, get_arguments, get_class

from my_salina_examples.models.actors import TunableVarianceContinuousActor
from my_salina_examples.models.actors import StateDependentVarianceContinuousActor
from my_salina_examples.models.actors import ConstantVarianceContinuousActor
from my_salina_examples.models.actors import DiscreteActor, ProbAgent, ActionAgent
from my_salina_examples.models.critics import VAgent
from my_salina_examples.models.envs import AutoResetEnvAgent, NoAutoResetEnvAgent
from my_salina_examples.models.loggers import Logger


def make_gym_env(max_episode_steps, env_name):
    return gym.make(env_name)


def evaluate_agent(cfg, filename):
    # 3) Create the A2C Agent
    eval_agent = torch.load(filename)
    nb_trials = 900
    means = np.zeros(nb_trials)
    for i in range(900):
        eval_workspace = Workspace()  # Used for evaluation
        eval_agent(eval_workspace, t=0, stop_variable="env/done", stochastic=False)
        rewards = eval_workspace["env/cumulated_reward"][-1]
        means[i] = rewards.mean()
    return means.mean()


params = {
    "gym_env": {"classname": "__main__.make_gym_env",
                "env_name": "CartPoleContinuous-v1"},
}

if __name__ == "__main__":
    chrono = Chrono()
    sys.path.append(os.getcwd())
    config = OmegaConf.create(params)
    folder = "./tmp/policies"
    listdir = os.listdir(folder)
    for policy_file in listdir:
        val = evaluate_agent(config, folder + "/" + policy_file)
        print(f"{policy_file}: {val}")
    chrono.stop()
