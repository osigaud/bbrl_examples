import sys
import os

import gym
import my_gym

from gym.wrappers import TimeLimit
from omegaconf import DictConfig, OmegaConf
from salina import instantiate_class, get_arguments, get_class, Workspace
from salina.agents import Agents, TemporalAgent
from salina.logger import TFLogger
import hydra

import copy
import time

import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import detect_anomaly

from my_salina_examples.models.salina_actors import ProbAgent, ActionAgent
from my_salina_examples.models.salina_critics import VAgent
from my_salina_examples.models.salina_envs import AutoResetEnvAgent
from my_salina_examples.models.salina_loggers import Logger


# Create the A2C Agent
def create_a2c_agent(cfg, env_agent):
    observation_size, n_actions = env_agent.get_obs_and_actions_sizes()
    # print(observation_size, n_actions)
    prob_agent = ProbAgent(observation_size, cfg.algorithm.architecture.hidden_size, n_actions)
    action_agent = ActionAgent()
    critic_agent = VAgent(observation_size, cfg.algorithm.architecture.hidden_size)

    # Combine env and policy agents
    agent = Agents(env_agent, prob_agent, action_agent)
    # Get an agent that is executed on a complete workspace
    agent = TemporalAgent(agent)
    agent.seed(cfg.algorithm.seed)
    return agent, prob_agent, critic_agent


def _index(tensor_3d, tensor_2d):
    """
    This function is used to index a 3d tensors using a 2d tensor
    """
    x, y, z = tensor_3d.size()
    t = tensor_3d.reshape(x * y, z)
    tt = tensor_2d.reshape(x * y)
    v = t[torch.arange(x * y), tt]
    v = v.reshape(x, y)
    return v


def make_gym_env(max_episode_steps, env_name):
    return TimeLimit(gym.make(env_name), max_episode_steps=max_episode_steps)


# Configure the optimizer over the a2c agent
def setup_optimizers(cfg, prob_agent, critic_agent):
    optimizer_args = get_arguments(cfg.optimizer)
    parameters = nn.Sequential(prob_agent, critic_agent).parameters()
    optimizer = get_class(cfg.optimizer)(parameters, **optimizer_args)
    return optimizer


def compute_critic_loss(cfg, reward, done, critic):
    # Compute temporal difference
    target = reward[1:] + cfg.algorithm.discount_factor * critic[1:].detach() * (1 - done[1:].float())
    td = target - critic[:-1]

    # Compute critic loss
    td_error = td ** 2
    critic_loss = td_error.mean()
    return critic_loss, td


def compute_a2c_loss(action_probs, action, td):
    action_logp = _index(action_probs, action).log()
    a2c_loss = action_logp[:-1] * td.detach()
    return a2c_loss.mean()


def run_a2c(cfg, max_grad_norm=0.5):
    # 1)  Build the  logger
    logger = Logger(cfg)

    # 2) Create the environment agent
    env_agent = AutoResetEnvAgent(cfg)

    # 3) Create the A2C Agent
    a2c_agent, prob_agent, critic_agent = create_a2c_agent(cfg, env_agent)

    # 4) Create the temporal critic agent to compute critic values over the workspace
    tcritic_agent = TemporalAgent(critic_agent)

    # 5) Configure the workspace to the right dimension
    # Note that no parameter is needed to create the workspace.
    # In the training loop, calling the agent() and critic_agent()
    # will take the workspace as parameter
    workspace = Workspace()

    # 6) Configure the optimizer over the a2c agent
    optimizer = setup_optimizers(cfg, prob_agent, critic_agent)

    # 7) Training loop
    for epoch in range(cfg.algorithm.max_epochs):
        # Execute the agent in the workspace
        if epoch > 0:
            workspace.zero_grad()
            workspace.copy_n_last_steps(1)
            a2c_agent(workspace, t=1, n_steps=cfg.algorithm.n_timesteps - 1, stochastic=True)
        else:
            a2c_agent(workspace, t=0, n_steps=cfg.algorithm.n_timesteps, stochastic=True)

        # Compute the critic value over the whole workspace
        tcritic_agent(workspace, n_steps=cfg.algorithm.n_timesteps)

        # Get relevant tensors (size are timestep x n_envs x ....)
        critic, done, action_probs, reward, action = workspace["critic", "env/done", "action_probs", "env/reward", "action"]
        # print(action.flatten())
        # print(reward)

        # Compute critic loss
        critic_loss, td = compute_critic_loss(cfg, reward, done, critic)

        # Compute entropy loss
        # entropy_loss = torch.distributions.Categorical(action_probs).entropy().mean()
        entropy_loss = torch.mean(workspace['entropy'])

        # Compute A2C loss
        a2c_loss = compute_a2c_loss(action_probs, action, td)
        # print(a2c_loss.mean())

        # Store the losses for tensorboard display
        logger.log_losses(cfg, epoch, critic_loss, entropy_loss, a2c_loss)

        # Compute the total loss
        loss = (
            -cfg.algorithm.entropy_coef * entropy_loss
            + cfg.algorithm.critic_coef * critic_loss
            - cfg.algorithm.a2c_coef * a2c_loss
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(prob_agent.parameters(), max_grad_norm)
        optimizer.step()

        # Compute the cumulated reward on final_state
        creward = workspace["env/cumulated_reward"]
        creward = creward[done]
        if creward.size()[0] > 0:
            logger.add_log("reward", creward.mean(), epoch)
        print(f"epoch: {epoch}, reward: {creward.mean()}")


params = {
    "logger": {"classname": "salina.logger.TFLogger",
               "log_dir": "./tmp",
               "verbose": False,
               # "cache_size": 10000,
               "every_n_seconds": 10},
    "algorithm": {
        "seed": 4,
        "n_envs": 8,
        "n_timesteps": 200,
        "max_epochs": 1000,
        "discount_factor": 0.95,
        "entropy_coef": 0.001,
        "critic_coef": 1.0,
        "a2c_coef": 0.1,
        "architecture": {"hidden_size": [24, 36]},
    },
    "gym_env": {"classname": "__main__.make_gym_env",
                "env_name": "CartPole-v1",
                "max_episode_steps": 500},
    "optimizer": {"classname": "torch.optim.Adam", "lr": 0.01},
}

if __name__ == "__main__":
     # with autograd.detect_anomaly():
        sys.path.append(os.getcwd())
        config = OmegaConf.create(params)
        run_a2c(config)
