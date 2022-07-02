import sys
import os
import copy

import torch
import torch.nn as nn
import gym
import my_gym
import hydra

from omegaconf import DictConfig
from bbrl import get_arguments, get_class
from bbrl.workspace import Workspace
from bbrl.agents import Agents, TemporalAgent

from bbrl.visu.visu_policies import plot_policy
from bbrl.visu.visu_critics import plot_critic

from bbrl_examples.models.actors import ContinuousDeterministicActor
from bbrl_examples.models.critics import ContinuousQAgent
from bbrl.utils.replay_buffer import ReplayBuffer
from bbrl.agents.gymb import AutoResetGymAgent, NoAutoResetGymAgent
from bbrl_examples.models.loggers import Logger, RewardLogger
from bbrl_examples.models.plotters import Plotter
from bbrl.utils.chrono import Chrono

# HYDRA_FULL_ERROR = 1


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


# Create the DDPG Agent
def create_ddpg_agent(cfg, train_env_agent, eval_env_agent):
    obs_size, act_size = train_env_agent.get_obs_and_actions_sizes()
    critic = ContinuousQAgent(
        obs_size, cfg.algorithm.architecture.critic_hidden_size, act_size
    )
    target_critic = copy.deepcopy(critic)
    actor = ContinuousDeterministicActor(
        obs_size, cfg.algorithm.architecture.actor_hidden_size, act_size
    )
    target_actor = copy.deepcopy(actor)
    tr_agent = Agents(train_env_agent, actor)  # TODO : add OU noise
    ev_agent = Agents(eval_env_agent, actor)

    # Get an agent that is executed on a complete workspace
    train_agent = TemporalAgent(tr_agent)
    eval_agent = TemporalAgent(ev_agent)
    train_agent.seed(cfg.algorithm.seed)
    return train_agent, eval_agent, actor, critic, target_actor, target_critic


def make_gym_env(env_name):
    return gym.make(env_name)


# Configure the optimizer
def setup_optimizers(cfg, actor, critic):
    optimizer_args = get_arguments(cfg.optimizer)
    parameters = nn.Sequential(actor, critic).parameters()
    optimizer = get_class(cfg.optimizer)(parameters, **optimizer_args)
    return optimizer


def compute_critic_loss(cfg, reward, must_bootstrap, q_values):
    # Compute temporal difference
    q_next = q_values[1]
    target = reward[:-1] + cfg.algorithm.discount_factor * q_next * must_bootstrap.int()
    td = target - q_values[0]
    # Compute critic loss
    td_error = td**2
    critic_loss = td_error.mean()
    return critic_loss


def compute_actor_loss(reward, must_bootstrap):
    actor_loss = reward.detach() * must_bootstrap.int()
    # print("actor_loss", actor_loss)
    return actor_loss.mean()


def run_ddpg_naked(cfg, reward_logger):
    # 1)  Build the  logger
    logger = Logger(cfg)
    best_reward = -10e9

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

    # 3) Create the DQN-like Agent
    (
        train_agent,
        eval_agent,
        actor,
        critic,
        target_actor,
        target_critic,
    ) = create_ddpg_agent(cfg, train_env_agent, eval_env_agent)
    ag_actor = TemporalAgent(actor)
    # ag_target_actor = TemporalAgent(target_actor)
    q_agent = TemporalAgent(critic)
    target_q_agent = TemporalAgent(target_critic)

    # Note that no parameter is needed to create the workspace.
    train_workspace = Workspace()  # Used for training
    rb = ReplayBuffer(max_size=1e5)

    # 6) Configure the optimizer
    optimizer = setup_optimizers(cfg, train_agent, q_agent)
    nb_steps = 0
    tmp_steps = 0

    # 7) Training loop
    for epoch in range(cfg.algorithm.max_epochs):
        # Execute the agent in the workspace
        if epoch > 0:
            train_workspace.zero_grad()
            train_workspace.copy_n_last_steps(1)
            train_agent(train_workspace, t=1, n_steps=cfg.algorithm.n_steps - 1)
        else:
            train_agent(train_workspace, t=0, n_steps=cfg.algorithm.n_steps)

        transition_workspace = train_workspace.get_transitions()
        action = transition_workspace["action"]
        nb_steps += action[0].shape[0]

        rb.put(transition_workspace)
        rb_workspace = rb.get_shuffled(cfg.algorithm.batch_size)

        done, truncated, reward, action = rb_workspace[
            "env/done", "env/truncated", "env/reward", "action"
        ]
        # replace the action at t+1 in the RB with \pi(s_{t+1})
        ag_actor(rb_workspace, t=1, n_steps=1)
        # action = rb_workspace["action"]
        # print("post", action[0])
        # compute q_values: at t, we have Q(s,a) in the RB, at t+1 we have Q(s_{t+1}, \pi(s_{t+1})
        q_agent(rb_workspace, t=0, n_steps=1)
        target_q_agent(rb_workspace, t=1, n_steps=1)
        q_values = rb_workspace["q_value"]
        # exit(-1)

        # Determines whether values of the critic should be propagated
        # True if the episode reached a time limit or if the task was not done
        # See https://colab.research.google.com/drive/1W9Y-3fa6LsPeR6cBC1vgwBjKfgMwZvP5?usp=sharing
        must_bootstrap = torch.logical_or(~done[1], truncated[1])

        # Compute critic loss
        critic_loss = compute_critic_loss(cfg, reward, must_bootstrap, q_values)

        # Store the loss for tensorboard display
        logger.add_log("critic_loss", critic_loss, nb_steps)

        optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            q_agent.parameters(), cfg.algorithm.max_grad_norm
        )
        optimizer.step()

        # Soft update of target policy and q function
        tau = cfg.algorithm.tau_target
        soft_update_params(critic, target_critic, tau)
        soft_update_params(actor, target_actor, tau)

        if nb_steps - tmp_steps > cfg.algorithm.eval_interval:
            tmp_steps = nb_steps
            eval_workspace = Workspace()  # Used for evaluation
            eval_agent(eval_workspace, t=0, stop_variable="env/done")
            rewards = eval_workspace["env/cumulated_reward"][-1]
            mean = rewards.mean()
            logger.add_log("reward", mean, nb_steps)
            print(f"nb_steps: {nb_steps}, reward: {mean}")
            reward_logger.add(nb_steps, mean)
            if cfg.save_best and mean > best_reward:
                best_reward = mean
                directory = "./ddpg_critic/"
                if not os.path.exists(directory):
                    os.makedirs(directory)
                filename = directory + "ddpg_" + str(mean.item()) + ".agt"
                eval_agent.save_model(filename)
                if cfg.plot_agents:
                    policy = eval_agent.agent.agents[1]
                    plot_policy(
                        policy,
                        eval_env_agent,
                        "./ddpg_plots/",
                        cfg.gym_env.env_name,
                        best_reward,
                        stochastic=False,
                    )
                    plot_critic(
                        q_agent.agent,
                        eval_env_agent,
                        "./ddpg_plots/",
                        cfg.gym_env.env_name,
                        best_reward,
                    )


def main_loop(cfg):
    chrono = Chrono()
    logdir = "./plot/"
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    reward_logger = RewardLogger(logdir + "ddpg.steps", logdir + "ddpg.rwd")
    for seed in range(cfg.algorithm.nb_seeds):
        cfg.algorithm.seed = seed
        torch.manual_seed(cfg.algorithm.seed)
        run_ddpg_naked(cfg, reward_logger)
        if seed < cfg.algorithm.nb_seeds - 1:
            reward_logger.new_episode()
    reward_logger.save()
    chrono.stop()
    plotter = Plotter(logdir + "ddpg.steps", logdir + "ddpg.rwd")
    plotter.plot_reward("ddpg", cfg.gym_env.env_name)


@hydra.main(
    config_path="./configs/",
    config_name="ddpg_pendulum.yaml",
    version_base="1.1",
)
def main(cfg: DictConfig):
    # print(OmegaConf.to_yaml(cfg))
    main_loop(cfg)


if __name__ == "__main__":
    sys.path.append(os.getcwd())
    main()
