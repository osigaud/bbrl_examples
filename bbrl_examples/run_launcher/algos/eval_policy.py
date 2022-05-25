import sys
import os

import gym
import my_gym
from gym.wrappers import TimeLimit
from omegaconf import OmegaConf
from bbrl.workspace import Workspace
from bbrl.agents import Agents, TemporalAgent

from bbrl_examples.models.actors import (
    TunableVarianceContinuousActor,
    ProbAgent,
    ActionAgent,
)
from bbrl_examples.models.envs import AutoResetEnvAgent, NoAutoResetEnvAgent
from bbrl.utils.chrono import Chrono


# Create the A2C Agent
def create_a2c_agent(cfg, eval_env_agent, filename):

    if eval_env_agent.is_continuous_action():
        observation_size, action_dim = eval_env_agent.get_obs_and_actions_sizes()
        action_agent = TunableVarianceContinuousActor(
            observation_size, cfg.algorithm.architecture.hidden_size, action_dim
        )
        action_agent.load_model(filename)
        param_agent = action_agent
        ev_agent = Agents(eval_env_agent, action_agent)
    else:
        observation_size, n_actions = eval_env_agent.get_obs_and_actions_sizes()
        param_agent = ProbAgent(
            observation_size, cfg.algorithm.architecture.hidden_size, n_actions
        )
        param_agent.load_model(filename)
        action_agent = ActionAgent()
        ev_agent = Agents(eval_env_agent, param_agent, action_agent)

    # Get an agent that is executed on a complete workspace
    eval_agent = TemporalAgent(ev_agent)
    return eval_agent, param_agent


def make_gym_env(max_episode_steps, env_name):
    return TimeLimit(gym.make(env_name), max_episode_steps=max_episode_steps)


def evaluate_agent(cfg, filename):

    # 2) Create the environment agent
    eval_env_agent = NoAutoResetEnvAgent(cfg, n_envs=cfg.algorithm.nb_evals)

    # 3) Create the A2C Agent
    eval_agent, param_agent = create_a2c_agent(cfg, eval_env_agent, filename)

    eval_workspace = Workspace()  # Used for evaluation
    eval_agent(eval_workspace, t=0, stop_variable="env/done", stochastic=False)
    rewards = eval_workspace["env/cumulated_reward"][-1]
    mean = rewards.mean()
    print(f"reward: {mean}")
    return mean


params = {
    "algorithm": {
        "seed": 4,
        "n_steps": 200,
        "nb_evals": 2000,
        "architecture": {"hidden_size": [25, 25]},
    },
    "gym_env": {
        "classname": "__main__.make_gym_env",
        "env_name": "CartPoleContinuous-v1",
        "max_episode_steps": 500,
    },
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
