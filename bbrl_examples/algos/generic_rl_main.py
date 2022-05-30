import sys
import os

import gym
from gym.wrappers import TimeLimit
from omegaconf import DictConfig
from bbrl import instantiate_class
from bbrl.workspace import Workspace
from bbrl.agents import Agents, TemporalAgent, NRemoteAgent
from bbrl.agents.gyma import AutoResetGymAgent
from bbrl.utils.logger import TFLogger
import hydra


# TODO: clean remove this function with better env creation
def make_gym_env(max_episode_steps, env_name):
    return TimeLimit(gym.make(env_name), max_episode_steps=max_episode_steps)


def synchronized_train(cfg):
    # init
    learner = instantiate_class(
        {**cfg.algorithm.learner, **{"cfg": cfg}}
    )  # TODO refactor the way args are given. (**dict1,**dict2) allows to merge dicts.
    logger = instantiate_class(cfg.logger)
    env_agent = AutoResetGymAgent(
        make_gym_env,
        {"max_episode_steps": cfg.env.max_episode_steps, "env_name": cfg.env.env_name},
        n_envs=cfg.algorithm.n_envs // cfg.algorithm.num_processes,
    )

    algorithm_agent = learner.get_acquisition_actor()
    acquisition_agent = Agents(env_agent, algorithm_agent)
    acquisition_agent = TemporalAgent(acquisition_agent)

    n_processes = cfg.algorithm.num_processes
    acquisition_args = learner.get_acquisition_args()
    acquisition_agent, acquisition_workspace = NRemoteAgent.create(
        acquisition_agent,
        num_processes=n_processes,
        t=0,
        n_steps=cfg.algorithm.n_timesteps,
        **acquisition_args,
    )
    acquisition_agent.seed(cfg.algorithm.env_seed)
    n_total_actor_steps = 0
    for epoch in range(cfg.algorithm.max_epochs):
        learner.update_acquisition_agent(acquisition_agent)

        # Get trajectories
        if epoch == 0:
            acquisition_agent(
                acquisition_workspace,
                t=0,
                n_steps=cfg.algorithm.n_timesteps,
                **acquisition_args,
            )
        else:
            acquisition_workspace.copy_n_last_steps(1)
            acquisition_agent(
                acquisition_workspace,
                t=1,
                n_steps=cfg.algorithm.n_timesteps - 1,
                **acquisition_args,
            )

        # Logging cumulated rewards:
        n_actor_steps = (
            acquisition_workspace.time_size() - 1
        ) * acquisition_workspace.batch_size()
        n_total_actor_steps += n_actor_steps
        done = acquisition_workspace["env/done"]
        cumulated_reward = acquisition_workspace["env/cumulated_reward"]
        creward = cumulated_reward[done]
        if len(creward) != 0:
            logger.add_scalar(
                "monitor/reward", creward.mean().item(), n_total_actor_steps
            )
        logger.add_scalar(
            f"monitor/n_interactions: {n_total_actor_steps}", n_total_actor_steps
        )

        learner.train(acquisition_workspace, n_actor_steps, n_total_actor_steps, logger)


@hydra.main(config_path="configs/", config_name="td3.yaml")
def main(cfg: DictConfig):
    import torch.multiprocessing as mp

    mp.set_start_method("spawn")
    synchronized_train(cfg)


if __name__ == "__main__":
    sys.path.append(os.getcwd())
    main()
