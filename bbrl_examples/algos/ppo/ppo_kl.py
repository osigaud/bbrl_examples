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
    ev_agent = Agents(eval_env_agent, policy)

    critic_agent = TemporalAgent(
        VAgent(obs_size, cfg.algorithm.architecture.critic_hidden_size)
    )

    train_agent = TemporalAgent(tr_agent)
    eval_agent = TemporalAgent(ev_agent)
    train_agent.seed(cfg.algorithm.seed)

    old_policy = copy.deepcopy(policy)
    kl_agent = TemporalAgent(KLAgent(old_policy, policy))
    return (
        train_agent,
        eval_agent,
        critic_agent,
        old_policy,
        kl_agent,
    )


def setup_optimizer(cfg, action_agent, critic_agent):
    optimizer_args = get_arguments(cfg.optimizer)
    parameters = nn.Sequential(action_agent, critic_agent).parameters()
    optimizer = get_class(cfg.optimizer)(parameters, **optimizer_args)
    return optimizer


def compute_advantage(cfg, reward, must_bootstrap, v_value):
    # Compute temporal difference with GAE
    advantage = gae(
        v_value,
        reward,
        must_bootstrap,
        cfg.algorithm.discount_factor,
        cfg.algorithm.gae,
    )
    return advantage


def compute_critic_loss(advantage):
    td_error = advantage**2
    critic_loss = td_error.mean()
    return critic_loss


def compute_agent_loss(cfg, advantage, ratio, kl_loss):
    """Computes the PPO loss including KL regularization"""
    actor_loss = (advantage * ratio - cfg.algorithm.beta * kl_loss).mean()
    return actor_loss


def run_ppo_v1(cfg):
    # 1)  Build the  logger
    logger = Logger(cfg)
    best_reward = -10e9
    nb_steps = 0
    tmp_steps = 0

    train_env_agent, eval_env_agent = create_env_agents(cfg)

    (
        train_agent,
        eval_agent,
        critic_agent,
        old_policy,
        kl_agent,
    ) = create_ppo_agent(cfg, train_env_agent, eval_env_agent)

    old_actor = TemporalAgent(old_policy)
    train_workspace = Workspace()

    # Configure the optimizer
    optimizer = setup_optimizer(cfg, train_agent, critic_agent)

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
        with torch.no_grad():
            train_agent(
                train_workspace,
                t=delta_t,
                n_steps=cfg.algorithm.n_steps - delta_t,
                stochastic=True,
                predict_proba=False,
                compute_entropy=False,
            )
            old_actor(
                train_workspace,
                t=delta_t,
                n_steps=cfg.algorithm.n_steps - delta_t,
                # Just computes the probability to get the ratio of probabilities
                predict_proba=True,
                compute_entropy=False,
            )

        # Compute the critic value over the whole workspace
        critic_agent(train_workspace, n_steps=cfg.algorithm.n_steps)

        transition_workspace = train_workspace.get_transitions()

        action = transition_workspace["action"]

        nb_steps += action[0].shape[0]

        done, truncated, reward, action, v_value = transition_workspace[
            "env/done",
            "env/truncated",
            "env/reward",
            "action",
            "v_value",
        ]

        # Determines whether values of the critic should be propagated
        # True if the episode reached a time limit or if the task was not done
        # See https://colab.research.google.com/drive/1W9Y-3fa6LsPeR6cBC1vgwBjKfgMwZvP5?usp=sharing
        must_bootstrap = torch.logical_or(~done[1], truncated[1])

        # then we compute the advantage using the clamped critic values
        advantage = compute_advantage(cfg, reward, must_bootstrap, v_value)
        actor_advantage = advantage.squeeze(0)[0]

        critic_loss = compute_critic_loss(
            advantage
        )  # issue here, can be used only once
        loss_critic = cfg.algorithm.critic_coef * critic_loss
        optimizer.zero_grad()
        loss_critic.backward()
        optimizer.step()

        torch.nn.utils.clip_grad_norm_(
            critic_agent.parameters(), cfg.algorithm.max_grad_norm
        )

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

            with torch.no_grad():
                kl_agent(sample_workspace, t=0, n_steps=1)
                kl = sample_workspace["kl"][0]

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
            action_logp, old_action_logp, entropy = sample_workspace[
                "action_logprobs", "logprob_predict", "entropy"
            ]

            act_diff = action_logp[0] - old_action_logp[0].detach()
            ratios = act_diff.exp()

            act_loss = compute_agent_loss(cfg, actor_advantage.detach(), ratios, kl)
            actor_loss = -cfg.algorithm.actor_coef * act_loss

            optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                train_agent.parameters(), cfg.algorithm.max_grad_norm
            )
            optimizer.step()

            old_policy.copy_parameters(train_agent.agent.agents[1])

            # Entropy loss favors exploration
            entr_loss = entropy[0].mean()
            entropy_loss = -cfg.algorithm.entropy_coef * entr_loss

            # Store the losses for tensorboard display
            logger.log_losses(nb_steps, critic_loss, entropy_loss, actor_loss)

            optimizer.zero_grad()
            entropy_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                train_agent.parameters(), cfg.algorithm.max_grad_norm
            )
            optimizer.step()

            # Evaluate if enough steps have been performed
        if nb_steps - tmp_steps > cfg.algorithm.eval_interval:
            tmp_steps = nb_steps
            eval_workspace = Workspace()  # Used for evaluation
            eval_agent(
                eval_workspace,
                t=0,
                stop_variable="env/done",
                stochastic=True,
                predict_proba=False,
            )
            rewards = eval_workspace["env/cumulated_reward"][-1]
            mean = rewards.mean()
            logger.add_log("reward_mean", mean, nb_steps)
            logger.add_log("reward_max", rewards.max(), nb_steps)
            logger.add_log("reward_min", rewards.min(), nb_steps)
            logger.add_log("reward_std", rewards.std(), nb_steps)
            print(f"nb_steps: {nb_steps}, reward: {mean}")
            if cfg.save_best and mean > best_reward:
                best_reward = mean
                directory = f"./ppo_agent/{cfg.gym_env.env_name}/"
                if not os.path.exists(directory):
                    os.makedirs(directory)
                filename = (
                    directory
                    + cfg.gym_env.env_name
                    + "#ppo_kl#team#"
                    + str(mean.item())
                    + ".agt"
                )
                train_agent.agent.agents[1].save_model(filename)
                if cfg.plot_agents:
                    plot_policy(
                        eval_agent.agent.agents[1],
                        eval_env_agent,
                        "./ppo_plots/",
                        cfg.gym_env.env_name,
                        best_reward,
                        stochastic=False,
                    )
                    plot_critic(
                        critic_agent.agent,
                        eval_env_agent,
                        "./ppo_plots/",
                        cfg.gym_env.env_name,
                        best_reward,
                    )


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
