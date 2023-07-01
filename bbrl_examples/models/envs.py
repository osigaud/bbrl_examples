from typing import Tuple
from bbrl.agents.gymnasium import make_env, GymAgent, ParallelGymAgent
from functools import partial


def get_env_agents(
    cfg, *, autoreset=True, include_last_state=True
) -> Tuple[GymAgent, GymAgent]:
    # Returns a pair of environments (train / evaluation) based on a configuration `cfg`

    # Train environment
    train_env_agent = ParallelGymAgent(
        partial(make_env, cfg.gym_env.env_name, autoreset=autoreset),
        cfg.algorithm.n_envs,
        include_last_state=include_last_state,
    ).seed(cfg.algorithm.seed)

    # Test environment
    eval_env_agent = ParallelGymAgent(
        partial(make_env, cfg.gym_env.env_name),
        cfg.algorithm.nb_evals,
        include_last_state=include_last_state,
    ).seed(cfg.algorithm.seed)

    return train_env_agent, eval_env_agent
