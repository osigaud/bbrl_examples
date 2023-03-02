import sys
import os
import copy

import torch
import torch.nn as nn

import gym
import bbrl_gym
import hydra

from omegaconf import DictConfig

from bbrl import get_arguments, get_class

from bbrl.utils.functionalb import gae

from bbrl_examples.models.loggers import Logger
from bbrl.utils.chrono import Chrono


# The workspace is the main class in BBRL, this is where all data is collected and stored
from bbrl.workspace import Workspace

# Agents(agent1,agent2,agent3,...) executes the different agents the one after the other
# TemporalAgent(agent) executes an agent over multiple timesteps in the workspace,
# or until a given condition is reached
from bbrl.agents import Agents, TemporalAgent

# AutoResetGymAgent is an agent able to execute a batch of gym environments
# with auto-resetting. These agents produce multiple variables in the workspace:
# ’env/env_obs’, ’env/reward’, ’env/timestep’, ’env/done’, ’env/initial_state’, ’env/cumulated_reward’,
# ... When called at timestep t=0, then the environments are automatically reset.
# At timestep t>0, these agents will read the ’action’ variable in the workspace at time t − 1
from bbrl_examples.models.envs import create_env_agents

# Neural network models for actors and critics
from bbrl_examples.models.stochastic_actors import TunableVarianceContinuousActor

# from bbrl_examples.models.stochastic_actors import StateDependentVarianceContinuousActor
from bbrl_examples.models.stochastic_actors import DiscreteActor
from bbrl_examples.models.critics import VAgent

# This one is specific to PPO, it is used to compute the KL divergence between the current and the past policy
from bbrl_examples.models.exploration_agents import KLAgent


# Allow to display a policy and a critic as a 2D map
from bbrl.visu.visu_policies import plot_policy
from bbrl.visu.visu_critics import plot_critic


import matplotlib

matplotlib.use("TkAgg")


def make_gym_env(env_name):
    return gym.make(env_name)


# Create the PPO Agent
def create_ppo_agent(cfg, train_env_agent, eval_env_agent):
    obs_size, act_size = train_env_agent.get_obs_and_actions_sizes()
    if train_env_agent.is_continuous_action():
        policy = TunableVarianceContinuousActor(
            obs_size, cfg.algorithm.architecture.actor_hidden_size, act_size
        )
    else:
        policy = DiscreteActor(
            obs_size, cfg.algorithm.architecture.actor_hidden_size, act_size
        )
    tr_agent = Agents(train_env_agent, policy)

    train_agent = TemporalAgent(tr_agent)
    train_agent.seed(cfg.algorithm.seed)
    return (train_agent,)


def setup_optimizer(cfg, action_agent):
    optimizer_args = get_arguments(cfg.optimizer)
    parameters = action_agent.parameters()
    optimizer = get_class(cfg.optimizer)(parameters, **optimizer_args)
    return optimizer


def run_ppo_v1(cfg):

    train_env_agent, eval_env_agent = create_env_agents(cfg)

    (train_agent,) = create_ppo_agent(cfg, train_env_agent, eval_env_agent)

    train_workspace = Workspace()

    # Configure the optimizer
    optimizer = setup_optimizer(cfg, train_agent)

    # Training loop
    for epoch in range(cfg.algorithm.max_epochs):
        # Execute the training agent in the workspace

        # Handles continuation
        delta_t = 0
        if epoch > 0:
            train_workspace.zero_grad()
            delta_t = 1
            train_workspace.copy_n_last_steps(delta_t)

        # Run the train/old_train agents
        train_agent(
            train_workspace,
            t=delta_t,
            n_steps=cfg.algorithm.n_steps - delta_t,
            stochastic=True,
            predict_proba=False,
            compute_entropy=False,
        )
        transition_workspace = train_workspace.get_transitions()

        policy = train_agent.agent.agents[1]
        cpt = 0

        # We start several optimization epochs on mini_batches
        for opt_epoch in range(cfg.algorithm.opt_epochs):
            if cfg.algorithm.minibatch_size > 0:
                sample_workspace = transition_workspace.select_batch_n(
                    cfg.algorithm.minibatch_size
                )
            else:
                sample_workspace = transition_workspace

            # We only recompute the action of the current agent on the current workspace
            policy(
                sample_workspace,
                t=0,
                n_steps=1,
                stochastic=True,
                compute_entropy=True,
                predict_proba=False,
            )
            print(cpt)
            cpt = cpt + 1

            # The logprob_predict Tensor has been computed on the old_policy outside the loop
            action_logp = sample_workspace["action_logprobs"]
            # print(sample_workspace["env/env_obs"])
            # print(action_logp[0])
            # print(old_action_logp[0])

            act_diff = action_logp[0]

            act_loss = act_diff.mean()
            actor_loss = -cfg.algorithm.actor_coef * act_loss

            optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                train_agent.parameters(), cfg.algorithm.max_grad_norm
            )
            optimizer.step()


@hydra.main(
    config_path="./configs/",
    # config_name="ppo_lunarlander_continuous.yaml",
    # config_name="ppo_lunarlander.yaml",
    # config_name="ppo_swimmer.yaml",
    config_name="ppo_pendulum.yaml",
    # config_name="ppo_cartpole.yaml",
    version_base="1.1",
)
def main(cfg: DictConfig):
    # print(OmegaConf.to_yaml(cfg))
    torch.manual_seed(cfg.algorithm.seed)
    run_ppo_v1(cfg)


if __name__ == "__main__":
    sys.path.append(os.getcwd())
    main()
