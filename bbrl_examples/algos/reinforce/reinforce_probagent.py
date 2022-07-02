import os
import gym
import my_gym
import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig

from bbrl import get_arguments, get_class, instantiate_class
from bbrl.workspace import Workspace
from bbrl.agents import Agents, TemporalAgent, PrintAgent
from bbrl.agents.agent import Agent

from bbrl_examples.models.actors import ProbAgent, ActionAgent
from bbrl_examples.models.actors import StateDependentVarianceContinuousActor
from bbrl_examples.models.actors import ConstantVarianceContinuousActor
from bbrl_examples.models.actors import DiscreteActor, BernoulliActor
from bbrl_examples.models.critics import VAgent
from bbrl.agents.gymb import NoAutoResetGymAgent
from bbrl.utils.functionalb import gae
from bbrl.utils.chrono import Chrono


# This version is useful only to illustrate how one can compute the log probabilities
# of actions a posteriori, rather than online. The online version is clearly more elegant


def apply_sum(reward):
    # print(reward)
    reward_sum = reward.sum(axis=0)
    # print("sum", reward_sum)
    for i in range(len(reward)):
        reward[i] = reward_sum
    # print("final", reward)
    return reward


def make_gym_env(env_name):
    return gym.make(env_name)


def create_reinforce_agent(cfg, env_agent):
    obs_size, act_size = env_agent.get_obs_and_actions_sizes()
    proba_agent = ProbAgent(
        obs_size, cfg.algorithm.architecture.actor_hidden_size, act_size
    )
    action_agent = ActionAgent()
    # print_agent = PrintAgent()
    tr_agent = Agents(env_agent, proba_agent, action_agent)  # , print_agent)

    critic_agent = TemporalAgent(
        VAgent(obs_size, cfg.algorithm.architecture.critic_hidden_size)
    )

    # Get an agent that is executed on a complete workspace
    train_agent = TemporalAgent(tr_agent)
    train_agent.seed(cfg.algorithm.seed)
    return train_agent, proba_agent, critic_agent  # , print_agent


# Configure the optimizer over the a2c agent
def setup_optimizer(cfg, prob_agent, critic_agent):
    optimizer_args = get_arguments(cfg.optimizer)
    parameters = nn.Sequential(prob_agent, critic_agent).parameters()
    optimizer = get_class(cfg.optimizer)(parameters, **optimizer_args)
    return optimizer


def compute_critic_loss(cfg, reward, must_bootstrap, critic):
    # Compute temporal difference
    target = (
        reward[:-1]
        + cfg.algorithm.discount_factor * critic[1:].detach() * must_bootstrap[1:].int()
    )
    td = target - critic[:-1]

    # Compute critic loss
    td_error = td**2
    critic_loss = td_error.mean()
    return critic_loss, td


class Logger:
    def __init__(self, cfg):
        self.logger = instantiate_class(cfg.logger)

    def add_log(self, log_string, loss, epoch):
        self.logger.add_scalar(log_string, loss.item(), epoch)

    # Log losses
    def log_losses(self, epoch, critic_loss, actor_loss):
        self.add_log("critic_loss", critic_loss, epoch)
        self.add_log("actor_loss", actor_loss, epoch)


def run_reinforce(cfg):
    logger = Logger(cfg)

    # 2) Create the environment agent
    env_agent = NoAutoResetGymAgent(
        get_class(cfg.gym_env),
        get_arguments(cfg.gym_env),
        cfg.algorithm.n_envs,
        cfg.algorithm.seed,
    )

    reinforce_agent, proba_agent, critic_agent = create_reinforce_agent(cfg, env_agent)

    # 7) Configure the optimizer over the a2c agent
    optimizer = setup_optimizer(cfg, reinforce_agent, critic_agent)

    # 8) Training loop
    nb_steps = 0

    for episode in range(cfg.algorithm.nb_episodes):
        # print_agent.reset()
        # Execute the agent on the workspace to sample complete episodes
        # Since not all the variables of workspace will be overwritten, it is better to clear the workspace
        # Configure the workspace to the right dimension.
        train_workspace = Workspace()

        reinforce_agent(train_workspace, stochastic=True, t=0, stop_variable="env/done")

        # Get relevant tensors (size are timestep x n_envs x ....)
        obs, done, truncated, action_probs, reward, action = train_workspace[
            "env/env_obs",
            "env/done",
            "env/truncated",
            "action_probs",
            "env/reward",
            "action",
        ]
        critic_agent(train_workspace, stop_variable="env/done")
        v_value = train_workspace["v_value"]
        # print(obs,done,truncated,reward,action)
        # print("val",v_value)

        for i in range(cfg.algorithm.n_envs):
            nb_steps += len(action[:, i])

        # Determines whether values of the critic should be propagated
        # True if the episode reached a time limit or if the task was not done
        # See https://colab.research.google.com/drive/1W9Y-3fa6LsPeR6cBC1vgwBjKfgMwZvP5?usp=sharing
        must_bootstrap = torch.logical_or(~done, truncated)

        critic_loss, td = compute_critic_loss(cfg, reward, must_bootstrap, v_value)

        reward = apply_sum(reward)

        # Take the log probability of the actions performed
        action = action.unsqueeze(-1)
        action_logp = (
            torch.gather(action_probs.squeeze(), dim=2, index=action).squeeze().log()
        )

        # Compute the policy gradient loss based on the log probability of the actions performed
        actor_loss = action_logp * reward.detach() * must_bootstrap.int()
        actor_loss = actor_loss.mean()

        # Log losses
        logger.log_losses(nb_steps, critic_loss, actor_loss)

        loss = (
            cfg.algorithm.critic_coef * critic_loss
            - cfg.algorithm.actor_coef * actor_loss
        )

        # Compute the cumulated reward on final_state
        cumulated_reward = train_workspace["env/cumulated_reward"][-1]
        mean = cumulated_reward.mean()
        print(f"episode: {episode}, reward: {mean}")
        logger.add_log("reward", mean, nb_steps)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


@hydra.main(
    config_path="./configs/",
    config_name="reinforce_cartpole.yaml",  # debugv.yaml",
    version_base="1.1",
)
def main(cfg: DictConfig):
    chrono = Chrono()
    torch.manual_seed(cfg.algorithm.seed)
    run_reinforce(cfg)
    chrono.stop()


if __name__ == "__main__":
    main()
