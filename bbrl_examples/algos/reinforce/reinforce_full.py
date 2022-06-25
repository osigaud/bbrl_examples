#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import copy
import time

import gym
import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym.wrappers import TimeLimit
from omegaconf import DictConfig, OmegaConf

import bbrl
import bbrl.utils.functional as RLF
from bbrl import get_arguments, get_class, instantiate_class
from bbrl.agents.agent import Agent
from bbrl.workspace import Workspace
from bbrl.agents import Agents, TemporalAgent

from bbrl_examples.models.actors import TunableVarianceContinuousActor
from bbrl_examples.models.actors import StateDependentVarianceContinuousActor
from bbrl_examples.models.actors import ConstantVarianceContinuousActor
from bbrl_examples.models.actors import DiscreteActor, BernoulliActor
from bbrl_examples.models.critics import VAgent
from bbrl.agents.gymb import AutoResetGymAgent, NoAutoResetGymAgent
from bbrl_examples.models.loggers import Logger
from bbrl.utils.chrono import Chrono

from bbrl.visu.visu_policies import plot_policy
from bbrl.visu.visu_critics import plot_critic


def _index(tensor_3d, tensor_2d):
    """This function is used to index a 3d tensors using a 2d tensor"""
    x, y, z = tensor_3d.size()
    t = tensor_3d.reshape(x * y, z)
    tt = tensor_2d.reshape(x * y)
    v = t[torch.arange(x * y), tt]
    v = v.reshape(x, y)
    return v


# Create the reinforce Agent
def create_reinforce_agent(cfg, env_agent):
    obs_size, act_size = env_agent.get_obs_and_actions_sizes()
    if env_agent.is_continuous_action():
        action_agent = TunableVarianceContinuousActor(
            obs_size, cfg.algorithm.architecture.hidden_size, act_size
        )
        # print_agent = PrintAgent(*{"critic", "env/reward", "env/done", "action", "env/env_obs"})
    else:
        # action_agent = BernoulliActor(obs_size, cfg.algorithm.architecture.hidden_size)
        action_agent = DiscreteActor(
            obs_size, cfg.algorithm.architecture.hidden_size, act_size
        )
    tr_agent = Agents(env_agent, action_agent)

    critic_agent = TemporalAgent(
        VAgent(obs_size, cfg.algorithm.architecture.hidden_size)
    )

    # Get an agent that is executed on a complete workspace
    train_agent = TemporalAgent(tr_agent)
    train_agent.seed(cfg.algorithm.seed)
    return train_agent, critic_agent


def make_gym_env(env_name):
    return gym.make(env_name)


# Configure the optimizer over the a2c agent
def setup_optimizers(cfg, action_agent, critic_agent):
    optimizer_args = get_arguments(cfg.optimizer)
    parameters = nn.Sequential(action_agent, critic_agent).parameters()
    optimizer = get_class(cfg.optimizer)(parameters, **optimizer_args)
    return optimizer


def run_reinforce(cfg):
    logger = instantiate_class(cfg.logger)

    # 2) Create the environment agent
    env_agent = NoAutoResetGymAgent(
        get_class(cfg.gym_env),
        get_arguments(cfg.gym_env),
        cfg.algorithm.n_envs,
        cfg.algorithm.seed,
    )

    reinforce_agent, critic_agent = create_reinforce_agent(env_agent)

    agent = Agents(env_agent, reinforce_agent)

    agent = TemporalAgent(agent)

    # 6) Configure the workspace to the right dimension.
    # The time size is greater than the maximum episode size to be able to store all episode states
    workspace = Workspace()

    # 7) Configure the optimizer over the a2c agent
    optimizer = setup_optimizers(cfg, reinforce_agent, critic_agent)

    # 8) Training loop
    epoch = 0
    for epoch in range(cfg.algorithm.max_epochs):

        # Execute the agent on the workspace to sample complete episodes
        # Since not all the variables of workspace will be overwritten, it is better to clear the workspace
        workspace.clear()
        agent(workspace, stochastic=True, t=0, stop_variable="env/done")

        # Get relevant tensors (size are timestep x n_envs x ....)
        baseline, done, action_probs, reward, action = workspace[
            "v_value", "env/done", "action_probs", "env/reward", "action"
        ]
        r_loss = compute_reinforce_loss(
            reward, action_probs, baseline, action, done, cfg.algorithm.discount_factor
        )

        # Log losses
        [logger.add_scalar(k, v.item(), epoch) for k, v in r_loss.items()]

        loss = (
            -cfg.algorithm.entropy_coef * r_loss["entropy_loss"]
            + cfg.algorithm.baseline_coef * r_loss["baseline_loss"]
            - cfg.algorithm.reinforce_coef * r_loss["reinforce_loss"]
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute the cumulated reward on final_state
        creward = workspace["env/cumulated_reward"]
        tl = done.float().argmax(0)
        creward = creward[tl, torch.arange(creward.size()[1])]
        logger.add_scalar("reward", creward.mean().item(), epoch)


def compute_reinforce_loss(
    reward, action_probabilities, baseline, action, done, discount_factor
):
    """This function computes the reinforce loss, considering that episodes may have different lengths."""
    batch_size = reward.size()[1]

    # Find the first done occurence for each episode
    v_done, trajectories_length = done.float().max(0)
    trajectories_length += 1
    assert v_done.eq(1.0).all()
    max_trajectories_length = trajectories_length.max().item()

    # Shorten trajectories for accelerate computation
    reward = reward[:max_trajectories_length]
    action_probabilities = action_probabilities[:max_trajectories_length]
    baseline = baseline[:max_trajectories_length]
    action = action[:max_trajectories_length]

    # Create a binary mask to mask useless values (of size max_trajectories_length x batch_size)
    arange = (
        torch.arange(max_trajectories_length, device=done.device)
        .unsqueeze(-1)
        .repeat(1, batch_size)
    )
    mask = arange.lt(
        trajectories_length.unsqueeze(0).repeat(max_trajectories_length, 1)
    )
    reward = reward * mask

    # Compute discounted cumulated reward
    cumulated_reward = [torch.zeros_like(reward[-1])]
    for t in range(max_trajectories_length - 1, 0, -1):
        cumulated_reward.append(discount_factor + cumulated_reward[-1] + reward[t])
    cumulated_reward.reverse()
    cumulated_reward = torch.cat([c.unsqueeze(0) for c in cumulated_reward])

    # baseline loss
    g = baseline - cumulated_reward
    baseline_loss = (g) ** 2
    baseline_loss = (baseline_loss * mask).mean()

    # policy loss
    log_probabilities = _index(action_probabilities, action).log()
    policy_loss = log_probabilities * -g.detach()
    policy_loss = policy_loss * mask
    policy_loss = policy_loss.mean()

    # entropy loss
    entropy = torch.distributions.Categorical(action_probabilities).entropy() * mask
    entropy_loss = entropy.mean()

    return {
        "baseline_loss": baseline_loss,
        "reinforce_loss": policy_loss,
        "entropy_loss": entropy_loss,
    }


@hydra.main(config_path=".", config_name="main.yaml")
def main(cfg):
    import torch.multiprocessing as mp

    mp.set_start_method("spawn")
    run_reinforce(cfg)


if __name__ == "__main__":
    main()
