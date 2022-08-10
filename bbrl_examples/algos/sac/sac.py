import sys
import os
import copy
import gym
import my_gym
import numpy as np

from omegaconf import DictConfig
from bbrl import get_arguments, get_class
from bbrl.workspace import Workspace
from bbrl.agents import Agents, TemporalAgent

import hydra

import torch
import torch.nn as nn

from bbrl_examples.models.actors import SquashedGaussianActor
from bbrl_examples.models.actors import DiscreteActor
from bbrl_examples.models.critics import ContinuousQAgent
from bbrl_examples.models.shared_models import soft_update_params
from bbrl.agents.gymb import AutoResetGymAgent, NoAutoResetGymAgent
from bbrl_examples.models.loggers import Logger
from bbrl.utils.chrono import Chrono
from bbrl.utils.replay_buffer import ReplayBuffer

from bbrl.visu.visu_policies import plot_policy
from bbrl.visu.visu_critics import plot_critic

# HYDRA_FULL_ERROR = 1

import matplotlib

matplotlib.use("TkAgg")


# Create the SAC Agent
def create_sac_agent(cfg, train_env_agent, eval_env_agent):
    obs_size, act_size = train_env_agent.get_obs_and_actions_sizes()
    assert (
        train_env_agent.is_continuous_action()
    ), "SAC code dedicated to continuous actions"
    actor = SquashedGaussianActor(
        obs_size, cfg.algorithm.architecture.actor_hidden_size, act_size
    )
    tr_agent = Agents(train_env_agent, actor)
    ev_agent = Agents(eval_env_agent, actor)
    critic_1 = ContinuousQAgent(
        obs_size, cfg.algorithm.architecture.critic_hidden_size, act_size
    )
    target_critic_1 = copy.deepcopy(critic_1)
    critic_2 = ContinuousQAgent(
        obs_size, cfg.algorithm.architecture.critic_hidden_size, act_size
    )
    target_critic_2 = copy.deepcopy(critic_2)
    # Get an agent that is executed on a complete workspace
    train_agent = TemporalAgent(tr_agent)
    eval_agent = TemporalAgent(ev_agent)
    train_agent.seed(cfg.algorithm.seed)
    return (
        train_agent,
        eval_agent,
        actor,
        critic_1,
        target_critic_1,
        critic_2,
        target_critic_2,
    )


def make_gym_env(env_name):
    return gym.make(env_name)


# Configure the optimizer
def setup_optimizers(cfg, actor, critic_1, critic_2):
    actor_optimizer_args = get_arguments(cfg.actor_optimizer)
    parameters = actor.parameters()
    actor_optimizer = get_class(cfg.actor_optimizer)(parameters, **actor_optimizer_args)

    critic_optimizer_args = get_arguments(cfg.critic_optimizer)
    parameters = nn.Sequential(critic_1, critic_2).parameters()
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
    cfg,
    reward,
    action_logprobs,
    must_bootstrap,
    q_values_1,
    q_values_2,
    target_q_values,
    ent_coef,
):
    # Compute temporal difference
    q_next = target_q_values
    v_phi = q_next - ent_coef * action_logprobs[1]
    target = (
        reward[:-1][0] + cfg.algorithm.discount_factor * v_phi * must_bootstrap.int()
    )
    td_1 = target - q_values_1.squeeze(-1)
    td_2 = target - q_values_2.squeeze(-1)
    td_error_1 = td_1**2
    td_error_2 = td_2**2
    critic_loss_1 = td_error_1.mean()
    critic_loss_2 = td_error_2.mean()
    return critic_loss_1, critic_loss_2


def compute_actor_loss(ent_coef, action_logp, current_q_values):
    actor_loss = ent_coef * action_logp - current_q_values
    return actor_loss.mean()


def run_sac(cfg):
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
        cfg.algorithm.n_envs,
        cfg.algorithm.seed,
    )

    # 3) Create the A2C Agent
    (
        train_agent,
        eval_agent,
        actor,
        critic_1,
        target_critic_1,
        critic_2,
        target_critic_2,
    ) = create_sac_agent(cfg, train_env_agent, eval_env_agent)
    ag_actor = TemporalAgent(actor)
    q_agent_1 = TemporalAgent(critic_1)
    target_q_agent_1 = TemporalAgent(target_critic_1)
    q_agent_2 = TemporalAgent(critic_2)
    target_q_agent_2 = TemporalAgent(target_critic_2)
    train_workspace = Workspace()
    rb = ReplayBuffer(max_size=cfg.algorithm.buffer_size)

    # Configure the optimizer
    actor_optimizer, critic_optimizer = setup_optimizers(cfg, actor, critic_1, critic_2)
    entropy_coef_optimizer, log_entropy_coef = setup_entropy_optimizers(cfg)
    nb_steps = 0
    tmp_steps = 0
    # Initial value of the entropy coef alpha. If target_entropy is not auto, will remain fixed
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
                train_workspace, t=1, n_steps=cfg.algorithm.n_steps - 1, stochastic=True
            )
        else:
            train_agent(
                train_workspace, t=0, n_steps=cfg.algorithm.n_steps, stochastic=True
            )

        transition_workspace = train_workspace.get_transitions()
        action = transition_workspace["action"]
        nb_steps += action[0].shape[0]
        rb.put(transition_workspace)
        rb_workspace = rb.get_shuffled(cfg.algorithm.batch_size)

        done, truncated, reward, action, action_logprobs = rb_workspace[
            "env/done",
            "env/truncated",
            "env/reward",
            "action",
            "action_logprobs",
        ]
        # print(f"reward:{reward}, action:{action},  logp:{action_logprobs}")

        if nb_steps > cfg.algorithm.learning_starts:
            # Determines whether values of the critic should be propagated
            # True if the episode reached a time limit or if the task was not done
            # See https://colab.research.google.com/drive/1W9Y-3fa6LsPeR6cBC1vgwBjKfgMwZvP5?usp=sharing
            must_bootstrap = torch.logical_or(~done[1], truncated[1])

            # Compute q_values from both critics: at t, we have Q(s,a) from the (s,a) in the RB
            q_agent_1(rb_workspace, t=0, n_steps=1)
            q_values_1 = rb_workspace["q_value"]
            q_agent_2(rb_workspace, t=0, n_steps=1)
            q_values_2 = rb_workspace["q_value"]

            target_q_agent_1(rb_workspace, t=1, n_steps=1)
            post_q_values_1 = rb_workspace["q_value"]
            target_q_agent_2(rb_workspace, t=1, n_steps=1)
            post_q_values_2 = rb_workspace["q_value"]

            post_q_values = torch.min(post_q_values_1, post_q_values_2).squeeze(-1)

            # Entropy coef update part #####################################################
            if entropy_coef_optimizer is not None:
                # Important: detach the variable from the graph
                # so that we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = torch.exp(log_entropy_coef.detach())
                # See Eq. (17) of the SAC and Applications paper
                entropy_coef_loss = -(
                    log_entropy_coef * (action_logprobs + target_entropy).detach()
                ).mean()
                entropy_coef_optimizer.zero_grad()
                entropy_coef_loss.backward()
                entropy_coef_optimizer.step()
                logger.add_log("entropy_coef_loss", entropy_coef_loss, nb_steps)

            # Critic update part ############################################################
            critic_loss_1, critic_loss_2 = compute_critic_loss(
                cfg,
                reward,
                action_logprobs,
                must_bootstrap,
                q_values_1[0],
                q_values_2[0],
                post_q_values[1],
                ent_coef,
            )
            logger.add_log("critic_loss_1", critic_loss_1, nb_steps)
            logger.add_log("critic_loss_2", critic_loss_2, nb_steps)
            critic_loss = critic_loss_1 + critic_loss_2
            critic_optimizer.zero_grad()
            # We need to retain the graph because we reuse the Q-values to compute the actor loss
            critic_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(
                critic_1.parameters(), cfg.algorithm.max_grad_norm
            )
            torch.nn.utils.clip_grad_norm_(
                critic_2.parameters(), cfg.algorithm.max_grad_norm
            )
            critic_optimizer.step()

            # Compute q_values from both critics: at t, we have Q(s,a) from the (s,a) in the RB
            q_agent_1(rb_workspace, t=0, n_steps=1)
            q_values_1 = rb_workspace["q_value"]
            q_agent_2(rb_workspace, t=0, n_steps=1)
            q_values_2 = rb_workspace["q_value"]
            current_q_values = torch.min(q_values_1, q_values_2).squeeze(-1)
            # Get the log proba of the action the policy takes in the states contained in the replay buffer
            ag_actor(rb_workspace, t=0, n_steps=1, stochastic=True)
            action_logprobs = rb_workspace["action_logprobs"]

            # Actor update part ###################################################################
            actor_loss = compute_actor_loss(
                ent_coef, action_logprobs[0], current_q_values[0]
            )
            logger.add_log("actor_loss", actor_loss, nb_steps)
            actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                actor.parameters(), cfg.algorithm.max_grad_norm
            )
            actor_optimizer.step()
            #########################################################################################

            # Soft update of target q function
            tau = cfg.algorithm.tau_target
            soft_update_params(critic_1, target_critic_1, tau)
            soft_update_params(critic_2, target_critic_2, tau)
            # soft_update_params(actor, target_actor, tau)

        if nb_steps - tmp_steps > cfg.algorithm.eval_interval:
            tmp_steps = nb_steps
            eval_workspace = Workspace()  # Used for evaluation
            eval_agent(eval_workspace, t=0, stop_variable="env/done", stochastic=False)
            rewards = eval_workspace["env/cumulated_reward"][-1]
            mean = rewards.mean()
            logger.add_log("reward", mean, nb_steps)
            print(f"nb_steps: {nb_steps}, reward: {mean}")
            # print("ent_coef", ent_coef)
            if cfg.save_best and mean > best_reward:
                best_reward = mean
                directory = "./sac_agent/"
                if not os.path.exists(directory):
                    os.makedirs(directory)
                filename = directory + "sac_" + str(mean.item()) + ".agt"
                eval_agent.save_model(filename)
                if cfg.plot_agents:
                    plot_policy(
                        actor,
                        eval_env_agent,
                        "./sac_plots/",
                        cfg.gym_env.env_name,
                        best_reward,
                        stochastic=False,
                    )
                    plot_critic(
                        q_agent_1.agent,  # TODO: do we want to plot both critics?
                        eval_env_agent,
                        "./sac_plots/",
                        cfg.gym_env.env_name,
                        best_reward,
                    )


@hydra.main(
    config_path="./configs/",
    config_name="sac_pendulum.yaml",
    version_base="1.1",
)
def main(cfg: DictConfig):
    # print(OmegaConf.to_yaml(cfg))
    chrono = Chrono()
    torch.manual_seed(cfg.algorithm.seed)
    run_sac(cfg)
    chrono.stop()


if __name__ == "__main__":
    sys.path.append(os.getcwd())
    main()
