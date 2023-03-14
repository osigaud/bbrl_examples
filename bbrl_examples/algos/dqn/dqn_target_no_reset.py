import sys
import os
import copy

import torch
import gym
import my_gym
import hydra

from omegaconf import DictConfig
from bbrl import get_arguments, get_class
from bbrl.workspace import Workspace
from bbrl.agents import Agents, TemporalAgent

from bbrl.visu.visu_policies import plot_policy
from bbrl.visu.visu_critics import plot_critic

from bbrl_examples.models.exploration_agents import EGreedyActionSelector
from bbrl_examples.models.critics import DiscreteQAgent
from bbrl.agents.gymb import NoAutoResetGymAgent
from bbrl_examples.models.loggers import Logger, RewardLogger
from bbrl_examples.models.plotters import Plotter
from bbrl.utils.chrono import Chrono

# HYDRA_FULL_ERROR = 1


# Create the DQN Agent
def create_dqn_agent(cfg, train_env_agent, eval_env_agent):
    obs_size, act_size = train_env_agent.get_obs_and_actions_sizes()
    critic = DiscreteQAgent(obs_size, cfg.algorithm.architecture.hidden_size, act_size)
    target_critic = copy.deepcopy(critic)
    target_q_agent = TemporalAgent(target_critic)
    explorer = EGreedyActionSelector(cfg.algorithm.epsilon_init)
    q_agent = TemporalAgent(critic)
    tr_agent = Agents(train_env_agent, critic, explorer)
    ev_agent = Agents(eval_env_agent, critic)

    # Get an agent that is executed on a complete workspace
    train_agent = TemporalAgent(tr_agent)
    eval_agent = TemporalAgent(ev_agent)
    train_agent.seed(cfg.algorithm.seed)
    return train_agent, eval_agent, q_agent, target_q_agent


def make_gym_env(env_name):
    return gym.make(env_name)


# Configure the optimizer
def setup_optimizers(cfg, q_agent):
    optimizer_args = get_arguments(cfg.optimizer)
    parameters = q_agent.parameters()
    optimizer = get_class(cfg.optimizer)(parameters, **optimizer_args)
    return optimizer


def compute_critic_loss(cfg, reward, must_bootstrap, q_values, target_q_values, action):
    """_summary_

    Args:
        cfg (_type_): _description_
        reward (torch.Tensor): A (T x B) tensor containing the rewards
        must_bootstrap (torch.Tensor): a (T x B) tensor containing 0 if the episode is completed at time $t$
        q_values (torch.Tensor): a (T x B x A) tensor containing Q values
        action (torch.LongTensor): a (T) long tensor containing the chosen action

    Returns:
        torch.Scalar: The DQN loss
    """
    # Compute temporal difference
    max_q = q_values.max(-1)[0].detach()
    target = (
        reward[:-1]
        + cfg.algorithm.discount_factor * max_q[1:] * must_bootstrap[1:].int()
    )

    vals = q_values.squeeze()

    qvals = torch.gather(vals, dim=1, index=action)
    qvals = qvals[:-1]

    td = target - qvals

    # Compute critic loss
    td_error = td**2
    critic_loss = td_error.mean()
    # print(critic_loss)
    return critic_loss


def run_dqn_no_rb_no_target(cfg, reward_logger):
    # 1)  Build the  logger
    logger = Logger(cfg)
    best_reward = -10e9

    # 2) Create the environment agent
    train_env_agent = NoAutoResetGymAgent(
        get_class(cfg.gym_env),
        get_arguments(cfg.gym_env),
        1,
        cfg.algorithm.seed,
    )
    eval_env_agent = NoAutoResetGymAgent(
        get_class(cfg.gym_env),
        get_arguments(cfg.gym_env),
        cfg.algorithm.nb_evals,
        cfg.algorithm.seed,
    )

    # 3) Create the DQN-like Agent
    train_agent, eval_agent, q_agent, target_q_agent = create_dqn_agent(
        cfg, train_env_agent, eval_env_agent
    )

    # Note that no parameter is needed to create the workspace.
    # In the training loop, calling the train_agent
    # will take the workspace as parameter

    # 6) Configure the optimizer
    optimizer = setup_optimizers(cfg, q_agent)
    nb_steps = 0
    tmp_steps = 0
    tmp_steps2 = 0

    for episode in range(cfg.algorithm.nb_episodes):
        train_workspace = Workspace()  # Used for training
        train_agent(train_workspace, t=0, stop_variable="env/done", stochastic=True)

        q_values, done, truncated, reward, action = train_workspace[
            "q_values", "env/done", "env/truncated", "env/reward", "action"
        ]

        with torch.no_grad():
            target_q_agent(train_workspace, t=0, n_steps=2, stochastic=True)

        target_q_values = train_workspace["q_values"]

        if tmp_steps2 == nb_steps:
            with torch.no_grad():
                tmp = torch.clone(q_values)
            assert torch.equal(
                tmp, target_q_values
            ), f"values differ: {tmp} vs {target_q_values}"

        nb_steps += len(q_values)
        # Determines whether values of the critic should be propagated
        # True if the episode reached a time limit or if the task was not done
        # See https://colab.research.google.com/drive/1erLbRKvdkdDy0Zn1X_JhC01s1QAt4BBj?usp=sharing
        must_bootstrap = torch.logical_or(~done, truncated)

        # Compute critic loss
        critic_loss = compute_critic_loss(
            cfg, reward, must_bootstrap, q_values, target_q_values, action
        )

        # Store the loss for tensorboard display
        logger.add_log("critic_loss", critic_loss, nb_steps)

        optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            q_agent.parameters(), cfg.algorithm.max_grad_norm
        )
        optimizer.step()

        if nb_steps - tmp_steps > cfg.algorithm.eval_interval:
            tmp_steps = nb_steps
            eval_workspace = Workspace()  # Used for evaluation
            eval_agent(
                eval_workspace, t=0, stop_variable="env/done", choose_action=True
            )
            rewards = eval_workspace["env/cumulated_reward"][-1]
            mean = rewards.mean()
            logger.add_log("reward", mean, nb_steps)
            print(f"episode: {episode}, reward: {mean}")
            reward_logger.add(nb_steps, mean)
            if cfg.save_best and mean > best_reward:
                best_reward = mean
                directory = "./dqn_critic/"
                if not os.path.exists(directory):
                    os.makedirs(directory)
                filename = directory + "dqn_" + str(mean.item()) + ".agt"
                eval_agent.save_model(filename)
                if cfg.plot_agents:
                    policy = eval_agent.agent.agents[1]
                    plot_policy(
                        policy,
                        eval_env_agent,
                        "./dqn_plots/",
                        cfg.gym_env.env_name,
                        best_reward,
                        stochastic=False,
                    )
                    plot_critic(
                        policy,
                        eval_env_agent,
                        "./dqn_plots/",
                        cfg.gym_env.env_name,
                        best_reward,
                    )


def main_loop(cfg):
    chrono = Chrono()
    logdir = "./plot/"
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    reward_logger = RewardLogger(
        logdir + "naked_no_reset.steps", logdir + "naked_no_reset.rwd"
    )
    for seed in range(cfg.algorithm.nb_seeds):
        cfg.algorithm.seed = seed
        torch.manual_seed(cfg.algorithm.seed)
        run_dqn_no_rb_no_target(cfg, reward_logger)
        if seed < cfg.algorithm.nb_seeds - 1:
            reward_logger.new_episode()
    reward_logger.save()
    chrono.stop()
    plotter = Plotter(logdir + "naked_no_reset.steps", logdir + "naked_no_reset.rwd")
    plotter.plot_reward("naked_no_reset", cfg.gym_env.env_name)


@hydra.main(
    config_path="./configs/",
    config_name="dqn_no_replay_no_target_cartpole.yaml",
    version_base="1.1",
)
def main(cfg: DictConfig):
    # print(OmegaConf.to_yaml(cfg))
    main_loop(cfg)


if __name__ == "__main__":
    sys.path.append(os.getcwd())
    main()
