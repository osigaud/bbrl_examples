import sys
import os
import copy
import torch
import torch.nn as nn
import gym
import my_gym
import hydra
import numpy as np

from omegaconf import DictConfig
from bbrl.utils.chrono import Chrono

from bbrl import get_arguments, get_class
from bbrl.workspace import Workspace
from bbrl.agents import Agents, TemporalAgent, PrintAgent

from bbrl_examples.models.loggers import Logger
from bbrl.utils.replay_buffer import ReplayBuffer

from bbrl_examples.models.actors import SquashedGaussianTQCActor
from bbrl_examples.models.critics import TruncatedQuantileNetwork

from bbrl_examples.models.shared_models import soft_update_params
from bbrl.agents.gymb import AutoResetGymAgent, NoAutoResetGymAgent

from bbrl.visu.visu_policies import plot_policy
from bbrl.visu.visu_critics import plot_critic

# HYDRA_FULL_ERROR = 1

import matplotlib

matplotlib.use("TkAgg")


# Create the TQC Agent
def create_tqc_agent(cfg, train_env_agent, eval_env_agent):
    obs_size, act_size = train_env_agent.get_obs_and_actions_sizes()
    assert (
        train_env_agent.is_continuous_action()
    ), "TQC code dedicated to continuous actions"

    # Actor
    actor = SquashedGaussianTQCActor(
        obs_size, cfg.algorithm.architecture.actor_hidden_size, act_size
    )

    # Train/Test agents
    tr_agent = Agents(train_env_agent, actor)
    ev_agent = Agents(eval_env_agent, actor)

    # Builds the critics
    critic = TruncatedQuantileNetwork(
        obs_size, cfg.algorithm.architecture.critic_hidden_size,
        cfg.algorithm.architecture.n_nets, act_size,
        cfg.algorithm.architecture.n_quantiles
    )
    target_critic = copy.deepcopy(critic)

    train_agent = TemporalAgent(tr_agent)
    eval_agent = TemporalAgent(ev_agent)
    train_agent.seed(cfg.algorithm.seed)
    return (
        train_agent,
        eval_agent,
        actor,
        critic,
        target_critic
    )


def make_gym_env(env_name):
    return gym.make(env_name)


# Configure the optimizer
def setup_optimizers(cfg, actor, critic):
    actor_optimizer_args = get_arguments(cfg.actor_optimizer)
    parameters = actor.parameters()
    actor_optimizer = get_class(cfg.actor_optimizer)(parameters, **actor_optimizer_args)
    critic_optimizer_args = get_arguments(cfg.critic_optimizer)
    parameters = critic.parameters()
    critic_optimizer = get_class(cfg.critic_optimizer)(
        parameters, **critic_optimizer_args
    )
    return actor_optimizer, critic_optimizer


def setup_entropy_optimizers(cfg):
    if cfg.algorithm.target_entropy == "auto":
        entropy_coef_optimizer_args = get_arguments(cfg.entropy_coef_optimizer)
        # Note: we optimize the log of the entropy coef which is slightly different from the paper
        # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
        # Comment and code taken from the SB3 version of SAC
        log_entropy_coef = torch.log(
            torch.ones(1) * cfg.algorithm.entropy_coef
        ).requires_grad_(True)
        entropy_coef_optimizer = get_class(cfg.entropy_coef_optimizer)(
            [log_entropy_coef], **entropy_coef_optimizer_args
        )
    else:
        log_entropy_coef = 0
        entropy_coef_optimizer = None
    return entropy_coef_optimizer, log_entropy_coef


def compute_critic_loss(
        cfg, reward, must_bootstrap,
        t_actor,
        q_agent,
        target_q_agent,
        rb_workspace,
        ent_coef
):
    # Compute quantiles from critic with the actions present in the buffer:
    # at t, we have Qu  ntiles(s,a) from the (s,a) in the RB
    q_agent(rb_workspace, t=0, n_steps=1)
    quantiles = rb_workspace["quantiles"].squeeze()

    with torch.no_grad():
        # Replay the current actor on the replay buffer to get actions of the
        # current policy
        t_actor(rb_workspace, t=1, n_steps=1, stochastic=True)
        action_logprobs_next = rb_workspace["action_logprobs"]

        # Compute target quantiles from the target critic: at t+1, we have
        # Quantiles(s+1,a+1) from the (s+1,a+1) where a+1 has been replaced in the RB

        target_q_agent(rb_workspace, t=1, n_steps=1)
        post_quantiles = rb_workspace["quantiles"][1]

        sorted_quantiles, _ = torch.sort(post_quantiles.reshape(quantiles.shape[0], -1))
        quantiles_to_drop_total = cfg.algorithm.top_quantiles_to_drop * cfg.algorithm.architecture.n_nets
        truncated_sorted_quantiles = sorted_quantiles[:,
                                     :quantiles.size(-1) * quantiles.size(-2) - quantiles_to_drop_total]

        # compute the target
        logprobs = ent_coef * action_logprobs_next[1]
        y = reward[0].unsqueeze(-1) + must_bootstrap.int().unsqueeze(-1) * cfg.algorithm.discount_factor * (
                    truncated_sorted_quantiles - logprobs.unsqueeze(-1))

    # computing the Huber loss
    pairwise_delta = y[:, None, None, :] - quantiles[:, :, :, None]  # batch x nets x quantiles x samples

    abs_pairwise_delta = torch.abs(pairwise_delta)
    huber_loss = torch.where(abs_pairwise_delta > 1,
                             abs_pairwise_delta - 0.5,
                             pairwise_delta ** 2 * 0.5)

    n_quantiles = quantiles.shape[2]
    tau = torch.arange(n_quantiles).float() / n_quantiles + 1 / 2 / n_quantiles
    loss = (torch.abs(tau[None, None, :, None] - (pairwise_delta < 0).float()) * huber_loss).mean()
    return loss


def compute_actor_loss(ent_coef, t_actor, q_agent, rb_workspace):
    """Actor loss computation

    :param ent_coef: The entropy coefficient $\alpha$
    :param t_actor: The actor agent (temporal agent)
    :param q_agent: The critic (temporal agent) (n net of m quantiles)
    :param rb_workspace: The replay buffer (2 time steps, $t$ and $t+1$)
    """
    # Recompute the quantiles from the current policy, not from the actions in the buffer

    t_actor(rb_workspace, t=0, n_steps=1, stochastic=True)
    action_logprobs_new = rb_workspace["action_logprobs"]

    q_agent(rb_workspace, t=0, n_steps=1)
    quantiles = rb_workspace["quantiles"][0]

    actor_loss = (ent_coef * action_logprobs_new[0] - quantiles.mean(2).mean(1))

    return actor_loss.mean()


def run_tqc(cfg):
    # 1)  Build the  logger
    logger = Logger(cfg)
    best_reward = -10e9
    ent_coef = cfg.algorithm.entropy_coef

    # 2) Create the environment agent
    train_env_agent = AutoResetGymAgent(
        get_class(cfg.gym_env),
        get_arguments(cfg.gym_env),
        cfg.algorithm.n_envs,
        cfg.algorithm.seed,
    )
    eval_env_agent = NoAutoResetGymAgent(
        get_class(cfg.gym_env),
        get_arguments(cfg.gym_env),
        cfg.algorithm.nb_evals,
        cfg.algorithm.seed,
    )

    # 3) Create the A2C Agent
    (
        train_agent,
        eval_agent,
        actor,
        critic,
        target_critic
    ) = create_tqc_agent(cfg, train_env_agent, eval_env_agent)

    t_actor = TemporalAgent(actor)
    q_agent = TemporalAgent(critic)
    target_q_agent = TemporalAgent(target_critic)
    train_workspace = Workspace()

    # Creates a replay buffer
    rb = ReplayBuffer(max_size=cfg.algorithm.buffer_size)

    # Configure the optimizer
    actor_optimizer, critic_optimizer = setup_optimizers(cfg, actor, critic)
    entropy_coef_optimizer, log_entropy_coef = setup_entropy_optimizers(cfg)
    nb_steps = 0
    tmp_steps = 0

    # Initial value of the entropy coef alpha. If target_entropy is not auto,
    # will remain fixed
    if cfg.algorithm.target_entropy == "auto":
        target_entropy = -np.prod(train_env_agent.action_space.shape).astype(np.float32)
    else:
        target_entropy = cfg.algorithm.target_entropy

    # Training loop
    for epoch in range(cfg.algorithm.max_epochs):
        # Execute the agent in the workspace
        if epoch > 0:
            train_workspace.zero_grad()
            train_workspace.copy_n_last_steps(1)
            train_agent(
                train_workspace,
                t=1,
                n_steps=cfg.algorithm.n_steps - 1,
                stochastic=True,
            )
        else:
            train_agent(
                train_workspace,
                t=0,
                n_steps=cfg.algorithm.n_steps,
                stochastic=True,
            )

        transition_workspace = train_workspace.get_transitions()
        action = transition_workspace["action"]
        nb_steps += action[0].shape[0]
        rb.put(transition_workspace)

        if nb_steps > cfg.algorithm.learning_starts:
            # Get a sample from the workspace
            rb_workspace = rb.get_shuffled(cfg.algorithm.batch_size)

            done, truncated, reward, action_logprobs_rb = rb_workspace[
                "env/done", "env/truncated", "env/reward", "action_logprobs"
            ]

            # Determines whether values of the critic should be propagated
            # True if the episode reached a time limit or if the task was not done
            # See https://colab.research.google.com/drive/1erLbRKvdkdDy0Zn1X_JhC01s1QAt4BBj?usp=sharing
            must_bootstrap = torch.logical_or(~done[1], truncated[1])

            critic_loss = compute_critic_loss(cfg, reward, must_bootstrap,
                                              t_actor, q_agent, target_q_agent,
                                              rb_workspace, ent_coef)

            logger.add_log("critic_loss", critic_loss, nb_steps)

            actor_loss = compute_actor_loss(
                ent_coef, t_actor, q_agent, rb_workspace
            )
            logger.add_log("actor_loss", actor_loss, nb_steps)

            # Entropy coef update part #####################################################
            if entropy_coef_optimizer is not None:
                # Important: detach the variable from the graph
                # so that we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = torch.exp(log_entropy_coef.detach())
                entropy_coef_loss = -(
                        log_entropy_coef * (action_logprobs_rb + target_entropy)
                ).mean()
                entropy_coef_optimizer.zero_grad()
                # We need to retain the graph because we reuse the
                # action_logprobs are used to compute both the actor loss and
                # the critic loss
                entropy_coef_loss.backward(retain_graph=True)
                entropy_coef_optimizer.step()
                logger.add_log("entropy_coef_loss", entropy_coef_loss, nb_steps)
                logger.add_log("entropy_coef", ent_coef, nb_steps)

            # Actor update part ###############################
            actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                actor.parameters(), cfg.algorithm.max_grad_norm
            )
            actor_optimizer.step()

            # Critic update part ###############################
            critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                critic.parameters(), cfg.algorithm.max_grad_norm
            )
            critic_optimizer.step()
            ####################################################

            # Soft update of target q function
            tau = cfg.algorithm.tau_target
            soft_update_params(critic, target_critic, tau)
            # soft_update_params(actor, target_actor, tau)

        # Evaluate ###########################################
        if nb_steps - tmp_steps > cfg.algorithm.eval_interval:
            tmp_steps = nb_steps
            eval_workspace = Workspace()  # Used for evaluation
            eval_agent(
                eval_workspace,
                t=0,
                stop_variable="env/done",
                stochastic=False,
            )
            rewards = eval_workspace["env/cumulated_reward"][-1]
            mean = rewards.mean()
            logger.add_log("reward/mean", mean, nb_steps)
            logger.add_log("reward/max", rewards.max(), nb_steps)
            logger.add_log("reward/min", rewards.min(), nb_steps)
            logger.add_log("reward/min", rewards.median(), nb_steps)

            print(f"nb_steps: {nb_steps}, reward: {mean}")
            # print("ent_coef", ent_coef)
            if cfg.save_best and mean > best_reward:
                best_reward = mean
                directory = f"./agents/{cfg.gym_env.env_name}/tqc_agent/"
                if not os.path.exists(directory):
                    os.makedirs(directory)
                filename = directory + cfg.gym_env.env_name + "#tqc#team" + str(mean.item()) + ".agt"
                actor.save_model(filename)


@hydra.main(
    config_path="./configs/",
    # config_name="tqc_cartpolecontinuous.yaml",
    # config_name="tqc_pendulum.yaml",
    config_name="tqc_rocket_lander.yaml",
    # config_name="tqc_lunar_lander_continuous.yaml",
    version_base="1.1",
)
def main(cfg: DictConfig):
    # print(OmegaConf.to_yaml(cfg))
    chrono = Chrono()
    torch.manual_seed(cfg.algorithm.seed)
    run_tqc(cfg)
    chrono.stop()


if __name__ == "__main__":
    sys.path.append(os.getcwd())
    main()
