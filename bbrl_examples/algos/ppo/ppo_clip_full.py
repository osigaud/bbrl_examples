"""
This version of PPO is work in progress to integrate all mechanisms from the official PPO implementation
See: https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
for a full description of all the coding tricks that should be integrated
"""


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
from bbrl.utils.replay_buffer import ReplayBuffer

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
from bbrl_examples.models.envs import create_env_agents

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
    policy = globals()[cfg.algorithm.actor_type](
        obs_size, cfg.algorithm.architecture.actor_hidden_size, act_size
    )
    tr_agent = Agents(train_env_agent, policy)
    ev_agent = Agents(eval_env_agent, policy)

    critic_agent = TemporalAgent(
        VAgent(obs_size, cfg.algorithm.architecture.critic_hidden_size)
    )
    old_critic_agent = copy.deepcopy(critic_agent)

    train_agent = TemporalAgent(tr_agent)
    eval_agent = TemporalAgent(ev_agent)
    train_agent.seed(cfg.algorithm.seed)

    old_policy = copy.deepcopy(policy)

    return train_agent, eval_agent, critic_agent, old_policy, old_critic_agent


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


def compute_clip_actor_loss(cfg, advantage, ratio):
    """Computes the PPO CLIP loss"""
    clip_range = cfg.algorithm.clip_range

    actor_loss_1 = advantage * ratio
    actor_loss_2 = advantage * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
    actor_loss = torch.minimum(actor_loss_1, actor_loss_2).mean()
    return actor_loss


def run_ppo_clip(cfg):
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
        old_critic_agent,
    ) = create_ppo_agent(cfg, train_env_agent, eval_env_agent)

    # We can call the policy instead of a temporal agent because we run on transitions,
    # so we just need one step
    policy = train_agent.agent.agents[1]
    # This is not true of the old_policy, because it works on the train_workspace
    old_actor = TemporalAgent(old_policy)

    train_workspace = Workspace()

    # Configure the optimizer
    optimizer = setup_optimizer(cfg, train_agent, critic_agent)
    if cfg.algorithm.opt_epochs > 0:
        batch_size = int(cfg.algorithm.n_steps / cfg.algorithm.opt_epochs)

    # Training loop
    for epoch in range(cfg.algorithm.max_epochs):
        # Execute the training agent in the workspace

        # Handles continuation
        delta_t = 0
        if epoch > 0:
            train_workspace.zero_grad()
            delta_t = 1
            train_workspace.copy_n_last_steps(1)

        # Run the current actor and evaluate the proba of its action according to the old actor
        # The old_actor can be run after the train_agent on the same workspace
        # because it writes a logprob_predict and not an action.
        # That is, it does not determine the action of the old_actor,
        # it just determines the proba of the action of the current actor given its own probabilities

        with torch.no_grad():
            train_agent(
                train_workspace,
                t=delta_t,
                n_steps=cfg.algorithm.n_steps,
                stochastic=True,
                predict_proba=False,
                compute_entropy=False,
            )
            old_actor(
                train_workspace,
                t=delta_t,
                n_steps=cfg.algorithm.n_steps,
                # Just computes the probability of the current actor's action
                # to get the ratio of probabilities
                predict_proba=True,
                compute_entropy=False,
            )

        # Compute the critic value over the whole workspace
        critic_agent(train_workspace, t=delta_t, n_steps=cfg.algorithm.n_steps)

        done, truncated, reward, action, v_value = train_workspace[
            "env/done",
            "env/truncated",
            "env/reward",
            "action",
            "v_value",
        ]

        # Determines whether values of the critic should be propagated
        # True if the episode reached a time limit or if the task was not done
        # See https://colab.research.google.com/drive/1W9Y-3fa6LsPeR6cBC1vgwBjKfgMwZvP5?usp=sharing
        must_bootstrap = torch.logical_or(~done, truncated)

        # the critic values are clamped to move not too far away from the values of the previous critic

        with torch.no_grad():
            old_critic_agent(train_workspace, n_steps=cfg.algorithm.n_steps)
        old_v_value = train_workspace["v_value"]
        if cfg.algorithm.clip_range_vf > 0:
            # Clip the difference between old and new values
            # NOTE: this depends on the reward scaling
            v_value = old_v_value + torch.clamp(
                v_value - old_v_value,
                -cfg.algorithm.clip_range_vf,
                cfg.algorithm.clip_range_vf,
            )

        # then we compute the advantage using the clamped critic values

        # NB: I had to add must_bootstrap[1:], otherwise the sizes are not correct
        # A weakness of this code is that the advantage tensor is one step shorter than the other variables
        # so it needs to be completed with a 0

        advantage = compute_advantage(cfg, reward, must_bootstrap[1:], v_value)
        last_bit = torch.Tensor([[0]])
        advantage = torch.cat((advantage, last_bit), 0)
        print("adv rollout", advantage)

        # We store the advantage into the transition_workspace
        train_workspace.set_full("advantage", advantage)

        # We rename logprob_predict data into old_action_logprobs
        # We do so because we will rewrite in the logprob_predict variable in mini_batches
        train_workspace.set_full(
            "old_action_logprobs", train_workspace["logprob_predict"].detach()
        )

        train_workspace.clear("logprob_predict")

        rb = ReplayBuffer(train_workspace.batch_size())
        rb.put(train_workspace)
        shuffled = rb.get_shuffled(rb.max_size)

        print_content(shuffled)

        # We start several optimization epochs on mini_batches
        for opt_epoch in range(max(cfg.algorithm.opt_epochs, 1)):
            if cfg.algorithm.opt_epochs > 0:
                from_time = opt_epoch * batch_size
                to_time = (opt_epoch + 1) * batch_size

                print(
                    f"************************************************************** [{from_time}, {to_time}["
                )
                # The +1 is necessary because we need the next step in the transitions
                # sample_workspace = train_workspace.get_time_truncated_workspace(
                sample_workspace = shuffled.subtime(from_time, to_time + 1)
                print("sample workspace:")
                print_ag = TemporalAgent(PrintAgent("env/timestep", "advantage"))
                print_ag(
                    sample_workspace,
                    t=0,
                    n_steps=min(to_time + 1, cfg.algorithm.n_steps) - from_time,
                )
            else:
                sample_workspace = shuffled

            # Compute the probability of the played actions according to the current policy
            # We do not replay the action: we use the one stored into the dataset
            # Hence predict_proba=True
            #
            policy(
                sample_workspace,
                t=0,
                n_steps=1,
                compute_entropy=True,
                predict_proba=True,
            )
            print("sample workspace 2:")
            print_agent = TemporalAgent(PrintAgent("env/timestep", "advantage"))
            print_agent(
                sample_workspace,
                t=0,
                n_steps=1,
            )
            # The old_action_logprobs Tensor has been computed from the old_policy outside the loop
            (
                action,
                advantage,
                action_logp,
                old_action_logp,
                entropy,
            ) = sample_workspace[
                "action",
                "advantage",
                "logprob_predict",
                "old_action_logprobs",
                "entropy",
            ]
            nb_steps += action[0].shape[0]

            print("advantage full", advantage)
            print("advantage [0]", advantage[0])
            critic_loss = compute_critic_loss(advantage[0])
            # critic_loss = action_logp[0][0]
            loss_critic = cfg.algorithm.critic_coef * critic_loss

            act_diff = action_logp[0] - old_action_logp[0].detach()
            ratios = act_diff.exp()

            actor_advantage = advantage.detach().squeeze(0)[0]
            actor_loss = compute_clip_actor_loss(cfg, actor_advantage, ratios)
            loss_actor = -cfg.algorithm.actor_coef * actor_loss

            # Entropy loss favors exploration
            entropy_loss = entropy[0].mean()
            loss_entropy = -cfg.algorithm.entropy_coef * entropy_loss

            # Store the losses for tensorboard display
            logger.log_losses(nb_steps, critic_loss, entropy_loss, actor_loss)
            logger.add_log("advantage", advantage.mean(), nb_steps)

            loss = loss_actor + loss_entropy

            torch.autograd.set_detect_anomaly(True)
            optimizer.zero_grad()
            loss_critic.backward()
            torch.nn.utils.clip_grad_norm_(
                critic_agent.parameters(), cfg.algorithm.max_grad_norm
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                policy.parameters(), cfg.algorithm.max_grad_norm
            )
            optimizer.step()

        old_policy.copy_parameters(policy)
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
            print(f"nb_steps: {nb_steps}, reward: {mean}")
            if cfg.save_best and mean > best_reward:
                best_reward = mean
                directory = f"./ppo_agent/{cfg.gym_env.env_name}/"
                if not os.path.exists(directory):
                    os.makedirs(directory)
                filename = (
                    directory
                    + cfg.gym_env.env_name
                    + "#ppo_clip#team#"
                    + str(mean.item())
                    + ".agt"
                )
                policy.save_model(filename)
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


def print_content(ws):
    for k, v in ws.variables.items():
        print(f" key : {k} : batch : {v.batch_size()} size {v.time_size()}")


def print_time_content(ws):
    for k, v in ws.variables.items():
        if k == "env/timestep":
            print(f" key : {k} : batch : {v}")


def debug_transitions(workspace):
    """ """
    (
        done,
        truncated,
        reward,
    ) = workspace["env/done", "env/truncated", "env/reward"]
    timestep = workspace["env/timestep"]
    assert not done[
        0
    ].max()  # done must be always false in the first timestep of the transition.
    # if not it means we have a transition (step final) => (step initial)

    # timesteps must always follow each other.
    assert (timestep[0] == timestep[1] - 1).all()

    if done[truncated].numel() > 0:
        assert torch.amin(
            done[truncated]
        ), "when truncated is true, done is always true"

    assert reward[1].sum() == len(
        reward[1]
    ), "in cartpole, rewards are always 1"  # only 1 rewards


@hydra.main(
    config_path="./configs/",
    # config_name="ppo_lunarlander_continuous.yaml",
    # config_name="ppo_lunarlander.yaml",
    # config_name="ppo_swimmer.yaml",
    # config_name="ppo_pendulum.yaml",
    config_name="ppo_cartpole.yaml",
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
