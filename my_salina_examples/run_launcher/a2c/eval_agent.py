import sys
import os
import torch

import gym
import my_gym
from gym.wrappers import TimeLimit
from omegaconf import OmegaConf
from salina.workspace import Workspace
from salina.agents import Agents, TemporalAgent


from my_salina_examples.models.actors import TunableVarianceContinuousActor
from my_salina_examples.models.actors import StateDependentVarianceContinuousActor
from my_salina_examples.models.actors import ConstantVarianceContinuousActor
from salina.agents.utils import PrintAgent
from my_salina_examples.models.actors import DiscreteActor, ProbAgent, ActionAgent
from my_salina_examples.models.critics import VAgent
from my_salina_examples.models.envs import AutoResetEnvAgent, NoAutoResetEnvAgent
from my_salina_examples.models.loggers import Logger
from my_salina_examples.chrono import Chrono


def make_gym_env(max_episode_steps, env_name):
    return TimeLimit(gym.make(env_name), max_episode_steps=max_episode_steps)


def evaluate_agent(cfg, filename):
    # 3) Create the A2C Agent
    eval_agent = torch.load(filename)

    eval_workspace = Workspace()  # Used for evaluation
    eval_agent(eval_workspace, t=0, stop_variable="env/done", stochastic=False)
    rewards = eval_workspace["env/cumulated_reward"][-1]
    mean = rewards.mean()
    return mean


params = {
    "algorithm": {
        "seed": 4,
        "n_steps": 200,
        "nb_evals": 20,
        "architecture": {"hidden_size": [25, 25]},
    },
    "gym_env": {"classname": "__main__.make_gym_env",
                "env_name": "CartPoleContinuous-v1",
                "max_episode_steps": 500},
}

if __name__ == "__main__":
    chrono = Chrono()
    sys.path.append(os.getcwd())
    config = OmegaConf.create(params)
    folder = "./data/policies"
    listdir = os.listdir(folder)
    for policy_file in listdir:
        val = evaluate_agent(config, folder + "/" + policy_file)
        print(f"{policy_file}: {val}")
    chrono.stop()
