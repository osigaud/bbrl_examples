"""
This version of PPO works, but it incorrectly samples minibatches randomly from the rollouts
without making sure that each sample is used once and only once
See: https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
for a full description of all the coding tricks that should be integrated
"""


import sys
import os
import copy

import torch
import torch.nn as nn

import gym
import bbrl_gymnasium
import hydra
from tqdm.auto import tqdm

from omegaconf import DictConfig

from bbrl import get_arguments, get_class

from bbrl.utils.functional import gae

from bbrl_examples.models.loggers import Logger
from bbrl.utils.chrono import Chrono

# The workspace is the main class in BBRL, this is where all data is collected and stored
from bbrl.workspace import Workspace

# Agents(agent1,agent2,agent3,...) executes the different agents the one after the other
# TemporalAgent(agent) executes an agent over multiple timesteps in the workspace,
# or until a given condition is reached
from bbrl.agents import Agents, TemporalAgent, PrintAgent

# AutoResetGymAgent is an agent able to execute a batch of gym environments
# with auto-resetting. These agents produce multiple variables in the workspace:
# ’env/env_obs’, ’env/reward’, ’env/timestep’, ’env/done’, ’env/initial_state’, ’env/cumulated_reward’,
# ... When called at timestep t=0, then the environments are automatically reset.
# At timestep t>0, these agents will read the ’action’ variable in the workspace at time t − 1
from bbrl_examples.models.envs import get_env_agents

# Neural network models for actors and critics
from bbrl_examples.models.stochastic_actors import (
    TunableVariancePPOActor,
    TunableVarianceContinuousActor,
    TunableVarianceContinuousActorExp,
    SquashedGaussianActor,
    StateDependentVarianceContinuousActor,
    ConstantVarianceContinuousActor,
    DiscreteActor,
    BernoulliActor,
)
from bbrl_examples.models.critics import VAgent

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
    policy = globals()[cfg.algorithm.policy_type](
        obs_size,
        cfg.algorithm.architecture.policy_hidden_size,
        act_size,
        name="current_policy",
    )
    tr_agent = Agents(train_env_agent, policy)
    ev_agent = Agents(eval_env_agent, policy)

    critic_agent = TemporalAgent(
        VAgent(obs_size, cfg.algorithm.architecture.critic_hidden_size)
    )
    old_critic_agent = copy.deepcopy(critic_agent)

    train_agent = TemporalAgent(tr_agent)
    eval_agent = TemporalAgent(ev_agent)

    old_policy = copy.deepcopy(policy)
    old_policy.set_name("old_policy")

    return (
        train_agent,
        eval_agent,
        critic_agent,
        policy,
        old_policy,
        old_critic_agent,
    )


def setup_optimizer(cfg, actor, critic):
    optimizer_args = get_arguments(cfg.optimizer)
    parameters = nn.Sequential(actor, critic).parameters()
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


def compute_clip_policy_loss(cfg, advantage, ratio):
    """Computes the PPO CLIP loss"""
    clip_range = cfg.algorithm.clip_range

    policy_loss_1 = advantage * ratio
    policy_loss_2 = advantage * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
    policy_loss = torch.minimum(policy_loss_1, policy_loss_2).mean()
    return policy_loss


def run_ppo_clip(cfg):
    # 1)  Build the  logger
    logger = Logger(cfg)
    best_reward = float("-inf")
    nb_steps = 0
    tmp_steps = 0

    train_env_agent, eval_env_agent = get_env_agents(cfg)

    (
        train_agent,
        eval_agent,
        critic_agent,
        policy,
        old_policy_params,
        old_critic_agent,
    ) = create_ppo_agent(cfg, train_env_agent, eval_env_agent)

    # The old_policy params must be wrapped into a TemporalAgent
    old_policy = TemporalAgent(old_policy_params)

    train_workspace = Workspace()

    # Configure the optimizer
    optimizer = setup_optimizer(cfg, train_agent, critic_agent)

    # Training loop
    best_policy = policy
    pbar = tqdm(range(cfg.algorithm.max_epochs))

    for epoch in pbar:
        # Execute the training agent in the workspace

        # Handles continuation
        delta_t = 0
        if epoch > 0:
            train_workspace.zero_grad()
            delta_t = 1
            train_workspace.copy_n_last_steps(1)

        # Run the current policy and evaluate the proba of its action according to the old policy
        # The old_policy can be run after the train_agent on the same workspace
        # because it writes a logprob_predict and not an action.
        # That is, it does not determine the action of the old_policy,
        # it just determines the proba of the action of the current policy given its own probabilities

        with torch.no_grad():
            train_agent(
                train_workspace,
                t=delta_t,
                n_steps=cfg.algorithm.n_steps,
                stochastic=True,
                predict_proba=False,
                compute_entropy=False,
            )
            old_policy(
                train_workspace,
                t=delta_t,
                n_steps=cfg.algorithm.n_steps,
                # Just computes the probability of the old policy's action
                # to get the ratio of probabilities
                predict_proba=True,
                compute_entropy=False,
            )

        # Compute the critic value over the whole workspace
        critic_agent(train_workspace, t=delta_t, n_steps=cfg.algorithm.n_steps)

        transition_workspace = train_workspace.get_transitions()

        terminated, reward, action, v_value = transition_workspace[
            "env/terminated",
            "env/reward",
            "action",
            "v_value",
        ]
        nb_steps += action[0].shape[0]

        # Determines whether values of the critic should be propagated
        must_bootstrap = ~terminated[1]

        # the critic values are clamped to move not too far away from the values of the previous critic
        with torch.no_grad():
            old_critic_agent(train_workspace, n_steps=cfg.algorithm.n_steps)
        old_v_value = transition_workspace["v_value"]

        if cfg.algorithm.clip_range_vf > 0:
            # Clip the difference between old and new values
            # NOTE: this depends on the reward scaling
            v_value = old_v_value + torch.clamp(
                v_value - old_v_value,
                -cfg.algorithm.clip_range_vf,
                cfg.algorithm.clip_range_vf,
            )

        # then we compute the advantage using the clamped critic values
        advantage = compute_advantage(cfg, reward, must_bootstrap, v_value)

        # We store the advantage into the transition_workspace
        transition_workspace.set("advantage", 1, advantage[0])

        critic_loss = compute_critic_loss(advantage)
        loss_critic = cfg.algorithm.critic_coef * critic_loss

        optimizer.zero_grad()
        loss_critic.backward()
        torch.nn.utils.clip_grad_norm_(
            critic_agent.parameters(), cfg.algorithm.max_grad_norm
        )
        optimizer.step()

        # We start several optimization epochs on mini_batches
        for opt_epoch in range(cfg.algorithm.opt_epochs):
            if cfg.algorithm.batch_size > 0:
                sample_workspace = transition_workspace.select_batch_n(
                    cfg.algorithm.batch_size
                )
            else:
                sample_workspace = transition_workspace

            # [[student]] Compute the policy loss

            # Compute the probability of the played actions according to the current policy
            # We do not replay the action: we use the one stored into the dataset
            # Hence predict_proba=True
            # Note that the policy is not wrapped into a TemporalAgent, but we use a single step

            policy(
                sample_workspace,
                t=0,
                compute_entropy=True,
                predict_proba=True,
            )

            # The logprob_predict Tensor has been computed from the old_policy outside the loop
            advantage, action_logp, old_action_logp, entropy = sample_workspace[
                "advantage",
                "current_policy/logprob_predict",
                "old_policy/logprob_predict",
                "entropy",
            ]

            # Compute the ratio of action probabilities
            act_diff = action_logp[0] - old_action_logp[0].detach()
            ratios = act_diff.exp()

            # Compute the policy loss
            # (using compute_clip_policy_loss)
            policy_advantage = advantage.detach()[1]
            policy_loss = compute_clip_policy_loss(cfg, policy_advantage, ratios)
            # [[/student]]

            loss_policy = -cfg.algorithm.policy_coef * policy_loss

            # Entropy loss favors exploration
            assert len(entropy) == 1, f"{entropy.shape}"
            entropy_loss = entropy[0].mean()
            loss_entropy = -cfg.algorithm.entropy_coef * entropy_loss

            # Store the losses for tensorboard display
            logger.log_losses(critic_loss, entropy_loss, policy_loss, nb_steps)
            logger.add_log("advantage", policy_advantage.mean(), nb_steps)

            loss = loss_policy + loss_entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                policy.parameters(), cfg.algorithm.max_grad_norm
            )
            optimizer.step()

        old_policy_params.copy_parameters(policy)
        old_critic_agent = copy.deepcopy(critic_agent)

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
            logger.log_reward_losses(rewards, nb_steps)
            pbar.set_description(f"nb_steps: {nb_steps}, reward: {mean:.3f}")
            if cfg.save_best and mean > best_reward:
                best_reward = mean
                best_policy = copy.deepcopy(policy)
                directory = f"./outputs/{cfg.gym_env.env_name}/ppo_agent_clip"
                if not os.path.exists(directory):
                    os.makedirs(directory)
                filename = (
                    directory
                    + cfg.gym_env.env_name
                    + "#ppo_clip#team#"
                    + str(mean.item())
                    + ".agt"
                )
                best_policy.save_model(filename)
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
    # config_name="ppo_pendulum.yaml",
    # config_name="ppo_cartpole.yaml",
    config_name="ppo_single_state.yaml",
    # config_name="ppo_cartpole_continuous.yaml",
    version_base="1.1",
)
def main(cfg: DictConfig):
    # print(OmegaConf.to_yaml(cfg))
    torch.manual_seed(cfg.algorithm.seed)
    chrono = Chrono()
    run_ppo_clip(cfg)
    chrono.stop()


if __name__ == "__main__":
    sys.path.append(os.getcwd())
    main()
