from bbrl_examples.models.actors import ProbAgent, ActionAgent

# from bbrl_examples.models.critics import VAgent
from bbrl_examples.models.envs import AutoResetEnvAgent

cfg = {
    "logger": {
        "classname": "salina.logger.TFLogger",
        "log_dir": "./tmp",
        "every_n_seconds": 10,
        "verbose": False,
    },
    "algorithm": {
        "seed": 432,
        "n_envs": 8,
        "n_timesteps": 16,
        "max_epochs": 10000,
        "discount_factor": 0.95,
        "entropy_coef": 0.001,
        "critic_coef": 1.0,
        "a2c_coef": 0.1,
        "architecture": {"hidden_size": 32},
        "env": {
            "classname": "__main__.make_env",
            "env_name": "CartPole-v1",
            "max_episode_steps": 100,
        },
        "optimizer": {"classname": "torch.optim.Adam", "lr": 0.01},
    },
}

envAgent = AutoResetEnvAgent(cfg, 1)
pa = ProbAgent(2, [32], 1)
aa = ActionAgent()
