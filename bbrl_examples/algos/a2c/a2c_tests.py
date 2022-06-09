import sys
import os

import gym
import my_gym

from omegaconf import DictConfig
from bbrl import get_arguments, get_class
from bbrl.workspace import Workspace
from bbrl.agents import Agents, TemporalAgent

from bbrl.utils.functionalb import gae

import hydra

import torch
import torch.nn as nn

from bbrl_examples.models.actors import TunableVarianceContinuousActor
from bbrl_examples.models.actors import StateDependentVarianceContinuousActor
from bbrl_examples.models.actors import ConstantVarianceContinuousActor
from bbrl_examples.models.actors import DiscreteActor
from bbrl_examples.models.critics import VAgent
from bbrl.agents.gymb import AutoResetGymAgent, NoAutoResetGymAgent
from bbrl_examples.models.loggers import Logger
from bbrl.utils.chrono import Chrono

from bbrl.visu.visu_policies import plot_policy
from bbrl.visu.visu_critics import plot_critic

# HYDRA_FULL_ERROR = 1


# Create the A2C Agent
def create_a2c_agent(cfg, train_env_agent, eval_env_agent):
    obs_size, act_size = train_env_agent.get_obs_and_actions_sizes()
    if train_env_agent.is_continuous_action():
        action_agent = TunableVarianceContinuousActor(
            obs_size, cfg.algorithm.architecture.hidden_size, act_size
        )
        # print_agent = PrintAgent(*{"critic", "env/reward", "env/done", "action", "env/env_obs"})
    else:
        action_agent = DiscreteActor(
            obs_size, cfg.algorithm.architecture.hidden_size, act_size
        )
    tr_agent = Agents(train_env_agent, action_agent)
    ev_agent = Agents(eval_env_agent, action_agent)

    critic_agent = TemporalAgent(
        VAgent(obs_size, cfg.algorithm.architecture.hidden_size)
    )

    # Get an agent that is executed on a complete workspace
    train_agent = TemporalAgent(tr_agent)
    eval_agent = TemporalAgent(ev_agent)
    train_agent.seed(cfg.algorithm.seed)
    return train_agent, eval_agent, critic_agent


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
    # target = reward[:-1] + cfg.algorithm.discount_factor * v_value[1:].detach() * must_bootstrap.int()
    # td = target - v_value[:-1]
    td = gae(
        v_value,
        reward,
        must_bootstrap,
        cfg.algorithm.discount_factor,
        cfg.algorithm.gae,
    )
    # Compute critic loss
    td_error = td**2
    critic_loss = td_error.mean()
    return critic_loss, td


def compute_actor_loss(action_logp, td):
    a2c_loss = action_logp[:-1] * td.detach()
    return a2c_loss.mean()


def run_a2c(cfg, max_grad_norm=0.5):
    # 1)  Build the  logger
    chrono = Chrono()
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
        cfg.algorithm.n_envs,
        cfg.algorithm.seed,
    )

    # 3) Create the A2C Agent
    a2c_agent, eval_agent, critic_agent = create_a2c_agent(
        cfg, train_env_agent, eval_env_agent
    )

    # 5) Configure the workspace to the right dimension
    # Note that no parameter is needed to create the workspace.
    # In the training loop, calling the agent() and critic_agent()
    # will take the workspace as parameter
    train_workspace = Workspace()  # Used for training

    # 6) Configure the optimizer over the a2c agent
    optimizer = setup_optimizers(cfg, a2c_agent, critic_agent)
    nb_steps = 0
    tmp_steps = 0

    # 7) Training loop
    for epoch in range(cfg.algorithm.max_epochs):
        # Execute the agent in the workspace
        if epoch > 0:
            train_workspace.zero_grad()
            train_workspace.copy_n_last_steps(1)
            a2c_agent(
                train_workspace, t=1, n_steps=cfg.algorithm.n_steps - 1, stochastic=True
            )
        else:
            a2c_agent(
                train_workspace, t=0, n_steps=cfg.algorithm.n_steps, stochastic=True
            )

        # Compute the critic value over the whole workspace
        critic_agent(train_workspace, n_steps=cfg.algorithm.n_steps)
        nb_steps += cfg.algorithm.n_steps * cfg.algorithm.n_envs

        transition_workspace = train_workspace.get_transitions()

        v_value, done, truncated, reward, action, action_logp = transition_workspace[
            "v_value",
            "env/done",
            "env/truncated",
            "env/reward",
            "action",
            "action_logprobs",
        ]

        # Determines whether values of the critic should be propagated
        # True if the episode reached a time limit or if the task was not done
        # See https://colab.research.google.com/drive/1W9Y-3fa6LsPeR6cBC1vgwBjKfgMwZvP5?usp=sharing
        must_bootstrap = torch.logical_or(~done[1], truncated[1])

        # Compute critic loss
        critic_loss, td = compute_critic_loss(cfg, reward, must_bootstrap, v_value)
        a2c_loss = compute_actor_loss(action_logp, td)

        # Compute entropy loss
        entropy_loss = torch.mean(train_workspace["entropy"])

        # Store the losses for tensorboard display
        logger.log_losses(nb_steps, critic_loss, entropy_loss, a2c_loss)

        # Compute the total loss
        loss = (
            -cfg.algorithm.entropy_coef * entropy_loss
            + cfg.algorithm.critic_coef * critic_loss
            - cfg.algorithm.a2c_coef * a2c_loss
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(a2c_agent.parameters(), max_grad_norm)
        optimizer.step()

        if nb_steps - tmp_steps > cfg.algorithm.eval_interval:
            tmp_steps = nb_steps
            eval_workspace = Workspace()  # Used for evaluation
            eval_agent(eval_workspace, t=0, stop_variable="env/done", stochastic=False)
            rewards = eval_workspace["env/cumulated_reward"][-1]
            mean = rewards.mean()
            logger.add_log("reward", mean, nb_steps)
            print(f"epoch: {epoch}, reward: {mean}")
            if cfg.save_best and mean > best_reward:
                best_reward = mean
                directory = "./a2c_policies/"
                if not os.path.exists(directory):
                    os.makedirs(directory)
                filename = directory + "a2c_" + str(mean.item()) + ".agt"
                policy = eval_agent.agent.agents[1]
                policy.save_model(filename)
                critic = critic_agent.agent
                if cfg.plot_agents:
                    plot_policy(
                        policy,
                        eval_env_agent,
                        "./a2c_plots/",
                        cfg.gym_env.env_name,
                        best_reward,
                        stochastic=False,
                    )
                    plot_critic(
                        critic,
                        eval_env_agent,
                        "./a2c_plots/",
                        cfg.gym_env.env_name,
                        best_reward,
                    )
    chrono.stop()


# @hydra.main(config_path="./configs/", config_name="a2c_pendulum.yaml", version_base="1.1")
# @hydra.main(config_path="./configs/", config_name="a2c_cartpolecontinuous.yaml", version_base="1.1")
@hydra.main(
    config_path="./configs/", config_name="a2c_cartpole.yaml", version_base="1.1"
)
def main(cfg: DictConfig):
    # print(OmegaConf.to_yaml(cfg))
    torch.manual_seed(cfg.algorithm.seed)
    run_a2c(cfg)


if __name__ == "__main__":
    sys.path.append(os.getcwd())
    main()
