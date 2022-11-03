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
from bbrl.utils.replay_buffer import ReplayBuffer
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
import matplotlib

matplotlib.use("TkAgg")


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


# Create the DQN Agent
def create_dqn_agent(cfg, train_env_agent, eval_env_agent):
    obs_size, act_size = train_env_agent.get_obs_and_actions_sizes()
    critic = DiscreteQAgent(obs_size, cfg.algorithm.architecture.hidden_size, act_size)
    target_critic = copy.deepcopy(critic)
    explorer = EGreedyActionSelector(cfg.algorithm.epsilon_init)
    q_agent = TemporalAgent(critic)
    target_q_agent = TemporalAgent(target_critic)
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


def compute_critic_loss(cfg, reward, done, q_values, target_q_values, action):
    # Compute temporal difference
    max_q = target_q_values.max(-1)[0].detach()
    # print("maxq", max_q, max_q.shape)
    # print("r", reward[:-1][0], reward[:-1][0].shape)
    target = reward[:-1][0] + cfg.algorithm.discount_factor * max_q * (1 - done.int())
    act = action[0].unsqueeze(-1)
    qvals = torch.gather(q_values, dim=1, index=act).squeeze()
    # Compute critic loss
    # target = target[0]
    # print(target, qvals)
    mse = nn.MSELoss()
    critic_loss = mse(target, qvals)
    return critic_loss


def run_dqn_full(cfg, reward_logger):
    # 1)  Build the  logger
    logger = Logger(cfg)
    best_reward = -10e9

    # 2) Create the environment agent
    train_env_agent = NoAutoResetGymAgent(
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
    # state = train_env_agent._reset(0, False, False)['env_obs']
    # print("seed:", train_env_agent._seed)
    # print("state:", state)

    # 3) Create the DQN-like Agent
    train_agent, eval_agent, q_agent, target_q_agent = create_dqn_agent(
        cfg, train_env_agent, eval_env_agent
    )
    # for name, param in q_agent.named_parameters():
    #    if param.requires_grad:
    #        print(name, param.data)

    # Note that no parameter is needed to create the workspace.
    train_workspace = Workspace()  # Used for training
    rb = ReplayBuffer(max_size=cfg.algorithm.buffer_size)

    # 6) Configure the optimizer over the agent
    optimizer = setup_optimizers(cfg, q_agent)
    nb_steps = 0

    # 7) Training loop
    for episode in range(cfg.algorithm.nb_episodes):
        train_workspace = Workspace()  # Used for training
        train_agent(train_workspace, t=0, stop_variable="env/done", stochastic=True)
        # Execute the agent in the workspace
        train_agent.agent.agents[2].epsilon = max(
            train_agent.agent.agents[2].epsilon * cfg.algorithm.epsilon_decay,
            cfg.algorithm.epsilon_end,
        )

        states, done, truncated, reward, action = train_workspace[
            "env/env_obs", "env/done", "env/truncated", "env/reward", "action"
        ]
        # for i in range(len(states)-1):
        #    print("s ", states[i][0].numpy(), "a ", action[i].numpy(), "ns ", states[i+1][0].numpy(), "r ", reward[i].numpy(), "d ", done[i].numpy())

        transition_workspace = train_workspace.get_transitions()

        action = transition_workspace["action"]
        nb_steps += action[0].shape[0]

        rb.put(transition_workspace)
        # rb.print_obs()

        for i in range(len(states) - 1):
            rb_workspace = rb.get_shuffled(cfg.algorithm.batch_size)

            # The q agent needs to be executed on the rb_workspace workspace (gradients are removed in workspace).
            q_agent(rb_workspace, t=0, n_steps=2, choose_action=False)

            q_values, done, truncated, reward, action = rb_workspace[
                "q_values", "env/done", "env/truncated", "env/reward", "action"
            ]

            with torch.no_grad():
                target_q_agent(rb_workspace, t=0, n_steps=2, stochastic=True)

            target_q_values = rb_workspace["q_values"]
            # assert torch.equal(q_values, target_q_values), "values differ"

            if rb.size() > cfg.algorithm.learning_starts:
                # Compute critic loss
                critic_loss = compute_critic_loss(
                    cfg, reward, done[1], q_values[0], target_q_values[1], action
                )

                # Store the loss for tensorboard display
                logger.add_log("critic_loss", critic_loss, nb_steps)

                optimizer.zero_grad()
                critic_loss.backward()
                # torch.nn.utils.clip_grad_norm_(
                #    q_agent.parameters(), cfg.algorithm.max_grad_norm
                # )
                optimizer.step()
                soft_update_params(q_agent, target_q_agent, cfg.algorithm.tau)

        if episode % 10 == 0:
            eval_workspace = Workspace()  # Used for evaluation
            eval_agent(
                eval_workspace, t=0, stop_variable="env/done", choose_action=True
            )
            rewards = eval_workspace["env/cumulated_reward"][-1]
            mean = rewards.mean()
            logger.add_log("reward", mean, nb_steps)
            print(
                f"episode: {episode}, reward: {mean}, epsilon:{train_agent.agent.agents[2].epsilon}"
            )
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
    reward_logger = RewardLogger(logdir + "dqn_full.steps", logdir + "dqn_full.rwd")
    for seed in range(cfg.algorithm.nb_seeds):
        cfg.algorithm.seed = seed
        torch.manual_seed(1)
        run_dqn_full(cfg, reward_logger)
        if seed < cfg.algorithm.nb_seeds - 1:
            reward_logger.new_episode()
    reward_logger.save()
    chrono.stop()
    plotter = Plotter(logdir + "dqn_full.steps", logdir + "dqn_full.rwd")
    plotter.plot_reward("dqn full", cfg.gym_env.env_name)


@hydra.main(
    config_path="./configs/",
    config_name="dqn_no_reset_lunar_lander.yaml",
    version_base="1.1",
)
def main(cfg: DictConfig):
    # print(OmegaConf.to_yaml(cfg))
    main_loop(cfg)


if __name__ == "__main__":
    sys.path.append(os.getcwd())
    main()
