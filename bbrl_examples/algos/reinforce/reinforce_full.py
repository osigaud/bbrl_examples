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

from bbrl_examples.models.actors import TunableVarianceContinuousActor
from bbrl_examples.models.actors import StateDependentVarianceContinuousActor
from bbrl_examples.models.actors import ConstantVarianceContinuousActor
from bbrl_examples.models.actors import DiscreteActor, BernoulliActor
from bbrl_examples.models.critics import VAgent
from bbrl.agents.gymb import NoAutoResetGymAgent
from bbrl.utils.functionalb import gae
from bbrl_examples.models.loggers import Logger
from bbrl.utils.chrono import Chrono


from bbrl.visu.visu_policies import plot_policy
from bbrl.visu.visu_critics import plot_critic


def apply_sum(reward):
    # print(reward)
    reward_sum = reward.sum(axis=0)
    # print("sum", reward_sum)
    for i in range(len(reward)):
        reward[i] = reward_sum
    # print("final", reward)
    return reward


def apply_discounted_sum(cfg, reward):
    # print(reward)
    tmp = 0
    for i in reversed(range(len(reward))):
        reward[i] = reward[i] + cfg.algorithm.discount_factor * tmp
        tmp = reward[i]
    # print("final", reward)
    return reward


def apply_discounted_sum_minus_baseline(cfg, reward, baseline):
    # print(reward)
    tmp = 0
    for i in reversed(range(len(reward))):
        reward[i] = reward[i] + cfg.algorithm.discount_factor * tmp - baseline[i]
        tmp = reward[i]
    # print("final", reward)
    return reward


# Create the REINFORCE Agent
def create_reinforce_agent(cfg, env_agent):
    obs_size, act_size = env_agent.get_obs_and_actions_sizes()
    if env_agent.is_continuous_action():
        action_agent = TunableVarianceContinuousActor(
            obs_size, cfg.algorithm.architecture.actor_hidden_size, act_size
        )
        # print_agent = PrintAgent(*{"critic", "env/reward", "env/done", "action", "env/env_obs"})
    else:
        # action_agent = BernoulliActor(obs_size, cfg.algorithm.architecture.hidden_size)
        action_agent = DiscreteActor(
            obs_size, cfg.algorithm.architecture.actor_hidden_size, act_size
        )
    # print_agent = PrintAgent()
    tr_agent = Agents(env_agent, action_agent)  # , print_agent)

    critic_agent = TemporalAgent(
        VAgent(obs_size, cfg.algorithm.architecture.critic_hidden_size)
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


def compute_critic_loss(cfg, reward, must_bootstrap, v_value):
    # Compute temporal difference
    # print(f"reward:{reward}, V:{v_value}, MB:{must_bootstrap}")
    target = (
        reward[:-1]
        + cfg.algorithm.discount_factor
        * v_value[1:].detach()
        * must_bootstrap[1:].int()
    )
    td = target - v_value[:-1]
    """
    td = gae(
        v_value,
        reward,
        must_bootstrap[:-1],
        cfg.algorithm.discount_factor,
        cfg.algorithm.gae,
    )
    """

    # Compute critic loss
    td_error = td**2
    critic_loss = td_error.mean()
    # print(f"target:{target}, td:{td}, cl:{critic_loss}")
    return critic_loss


def compute_actor_loss(action_logprob, reward, must_bootstrap):
    a2c_loss = action_logprob * reward.detach() * must_bootstrap.int()
    return a2c_loss.mean()


def run_reinforce(cfg):
    logger = Logger(cfg)
    best_reward = -10e10

    # 2) Create the environment agent
    env_agent = NoAutoResetGymAgent(
        get_class(cfg.gym_env),
        get_arguments(cfg.gym_env),
        cfg.algorithm.n_envs,
        cfg.algorithm.seed,
    )

    reinforce_agent, critic_agent = create_reinforce_agent(cfg, env_agent)

    # 7) Configure the optimizer over the a2c agent
    optimizer = setup_optimizers(cfg, reinforce_agent, critic_agent)

    # 8) Training loop
    nb_steps = 0

    for episode in range(cfg.algorithm.nb_episodes):

        # Execute the agent on the workspace to sample complete episodes
        # Since not all the variables of workspace will be overwritten, it is better to clear the workspace
        # Configure the workspace to the right dimension.
        train_workspace = Workspace()

        reinforce_agent(train_workspace, stochastic=True, t=0, stop_variable="env/done")

        # Get relevant tensors (size are timestep x n_envs x ....)
        obs, done, truncated, action_logprobs, reward, action = train_workspace[
            "env/env_obs",
            "env/done",
            "env/truncated",
            "action_logprobs",
            "env/reward",
            "action",
        ]
        critic_agent(train_workspace, stop_variable="env/done")
        v_value = train_workspace["v_value"]
        # print(f"obs:{obs}")

        for i in range(cfg.algorithm.n_envs):
            nb_steps += len(action[:, i])

        # Determines whether values of the critic should be propagated
        # True if the episode reached a time limit or if the task was not done
        # See https://colab.research.google.com/drive/1W9Y-3fa6LsPeR6cBC1vgwBjKfgMwZvP5?usp=sharing
        must_bootstrap = torch.logical_or(~done, truncated)

        critic_loss = compute_critic_loss(cfg, reward, must_bootstrap, v_value)

        # reward = apply_sum(reward)
        # reward = apply_discounted_sum(cfg, reward)
        reward = apply_discounted_sum_minus_baseline(cfg, reward, v_value)

        actor_loss = compute_actor_loss(action_logprobs, reward, must_bootstrap)

        entropy_loss = torch.mean(train_workspace["entropy"])
        # Log losses
        logger.log_losses(nb_steps, critic_loss, entropy_loss, actor_loss)

        loss = (
            -cfg.algorithm.entropy_coef * entropy_loss
            + cfg.algorithm.critic_coef * critic_loss
            - cfg.algorithm.actor_coef * actor_loss
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute the cumulated reward on final_state
        cumulated_reward = train_workspace["env/cumulated_reward"][-1]
        mean = cumulated_reward.mean()
        logger.add_log("reward", mean, nb_steps)
        print(f"episode: {episode}, reward: {mean}")
        if cfg.save_best and mean > best_reward:
            best_reward = mean
            directory = "./reinforce_critic/"
            if not os.path.exists(directory):
                os.makedirs(directory)
            filename = directory + "reinforce_" + str(mean.item()) + ".agt"
            reinforce_agent.save_model(filename)
            if cfg.plot_agents:
                policy = reinforce_agent.agent.agents[1]
                critic = critic_agent.agent
                plot_policy(
                    policy,
                    env_agent,
                    "./reinforce_plots/",
                    cfg.gym_env.env_name,
                    best_reward,
                    stochastic=False,
                )
                plot_critic(
                    critic,
                    env_agent,
                    "./reinforce_plots/",
                    cfg.gym_env.env_name,
                    best_reward,
                )


@hydra.main(
    config_path="./configs/",
    config_name="reinforce_cartpole.yaml",  # debugv.yaml",
    version_base="1.1",
)
def main(cfg: DictConfig):
    chrono = Chrono()
    run_reinforce(cfg)
    chrono.stop()


if __name__ == "__main__":
    main()
