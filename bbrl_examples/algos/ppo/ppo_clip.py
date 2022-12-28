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

# Allow to display the behavior of an agent
from bbrl_examples.models.actors import TunableVarianceContinuousActor
from bbrl_examples.models.actors import DiscreteActor
from bbrl_examples.models.critics import VAgent

from bbrl.utils.chrono import Chrono

from bbrl.visu.visu_policies import plot_policy
from bbrl.visu.visu_critics import plot_critic


def make_gym_env(env_name):
    return gym.make(env_name)


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
    old_critic_agent = copy.deepcopy(critic_agent)

    return policy, train_agent, eval_agent, critic_agent, old_policy, old_critic_agent


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


def compute_clip_agent_loss(cfg, advantage, ratio):
    """Computes the PPO CLIP loss"""
    clip_range = cfg.algorithm.clip_range

    actor_loss_1 = advantage * ratio
    actor_loss_2 = advantage * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
    actor_loss = torch.minimum(actor_loss_1, actor_loss_2).mean()
    return actor_loss


def run_ppo_v2(cfg):
    # 1)  Build the  logger
    logger = Logger(cfg)
    best_reward = -10e9
    nb_steps = 0
    tmp_steps = 0

    train_env_agent, eval_env_agent = create_env_agents(cfg)

    (
        policy,
        train_agent,
        eval_agent,
        critic_agent,
        old_policy,
        old_critic_agent,
    ) = create_ppo_agent(cfg, train_env_agent, eval_env_agent)

    actor = TemporalAgent(policy)
    old_actor = TemporalAgent(old_policy)
    train_workspace = Workspace()

    # Configure the optimizer
    optimizer = setup_optimizer(cfg, policy, critic_agent)

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
            compute_entropy=True,
        )
        old_actor(
            train_workspace,
            t=delta_t,
            n_steps=cfg.algorithm.n_steps - delta_t,
            # Just computes the probability to get the ratio of probabilities
            predict_proba=True,
        )

        # Compute the critic value over the whole workspace
        critic_agent(train_workspace, n_steps=cfg.algorithm.n_steps)

        transition_workspace = train_workspace.get_transitions()
        done, truncated, reward, action, action_logp, v_value = transition_workspace[
            "env/done",
            "env/truncated",
            "env/reward",
            "action",
            "action_logprobs",
            "v_value",
        ]

        nb_steps += action[0].shape[0]

        # Determines whether values of the critic should be propagated
        # True if the episode reached a time limit or if the task was not done
        # See https://colab.research.google.com/drive/1W9Y-3fa6LsPeR6cBC1vgwBjKfgMwZvP5?usp=sharing
        must_bootstrap = torch.logical_or(~done[1], truncated[1])

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
        transition_workspace.set("advantage", 0, advantage)
        transition_workspace.set("advantage", 1, torch.zeros_like(advantage))
        # We rename logprob_predict data into old_action_logprobs
        # We do so because we will rewrite in the logprob_predict variable in mini_batches
        transition_workspace.set_full(
            "old_action_logprobs", transition_workspace["logprob_predict"].detach()
        )
        transition_workspace.clear("logprob_predict")

        # We start several optimization epochs on mini_batches
        for opt_epoch in range(cfg.algorithm.opt_epochs):
            if cfg.algorithm.minibatch_size > 0:
                sample_workspace = transition_workspace.sample_subworkspace(
                    1, cfg.algorithm.minibatch_size, 2
                )
            else:
                sample_workspace = transition_workspace

            actor(
                sample_workspace,
                t=0,
                n_steps=1,
                compute_entropy=True,
                predict_proba=True,
            )

            advantage, action_logp, old_action_logp, entropy = sample_workspace[
                "advantage", "logprob_predict", "old_action_logprobs", "entropy"
            ]

            critic_loss = compute_critic_loss(
                advantage
            )  # issue here, can be used only once
            adv_actor = advantage.detach().squeeze(0)[0]
            act_diff = action_logp[0] - old_action_logp[0]
            ratios = act_diff.exp()

            actor_loss = compute_clip_agent_loss(cfg, adv_actor, ratios)

            # Entropy loss favor exploration
            entropy_loss = torch.mean(entropy[0])

            # Store the losses for tensorboard display
            if opt_epoch == 0:
                # Just for the first epoch
                logger.log_losses(nb_steps, critic_loss, entropy_loss, actor_loss)

            loss_critic = cfg.algorithm.critic_coef * critic_loss

            loss = (
                -cfg.algorithm.actor_coef * actor_loss
                - cfg.algorithm.entropy_coef * entropy_loss
            )

            old_policy.copy_parameters(policy)
            old_critic_agent = copy.deepcopy(critic_agent)

            # [[remove]]
            # Compute approximate form of reverse KL Divergence for early stopping
            # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
            # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
            # and Schulman blog: http://joschu.net/blog/kl-approx.html
            """
            with torch.no_grad():
                log_ratio = log_prob - rollout_data.old_log_prob
                approx_kl_div = (
                    torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                )
                approx_kl_divs.append(approx_kl_div)

            if cfg.algorithm.target_kl is not None and approx_kl_div > 1.5 * cfg.algorithm.target_kl:
                continue_training = False
                if cfg.logger.verbose == True:
                    print(
                        f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}"
                    )
                break
            """
            # [[/remove]]

            optimizer.zero_grad()
            loss_critic.backward()  # retain_graph=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                critic_agent.parameters(), cfg.algorithm.max_grad_norm
            )
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


@hydra.main(
    config_path="./configs/",
    config_name="ppo_lunarlander_continuous.yaml",
    # config_name="ppo_lunarlander.yaml",
    # config_name="ppo_swimmer.yaml",
    # config_name="ppo_pendulum.yaml",
    # config_name="ppo_cartpole.yaml",
    version_base="1.1",
)
def main(cfg: DictConfig):
    # print(OmegaConf.to_yaml(cfg))
    torch.manual_seed(cfg.algorithm.seed)
    chrono = Chrono()
    run_ppo_v2(cfg)
    chrono.stop()


if __name__ == "__main__":
    sys.path.append(os.getcwd())
    main()
