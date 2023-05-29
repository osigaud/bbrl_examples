from abc import ABC

import torch
import torch.nn as nn


from bbrl_examples.models.shared_models import build_mlp, build_backbone

from bbrl.agents.agent import Agent


class BaseActor(Agent, ABC):
    """Generic class to centralize copy_parameters"""

    def copy_parameters(self, other):
        """Copy parameters from other agent"""
        for self_p, other_p in zip(self.parameters(), other.parameters()):
            self_p.data.copy_(other_p)


class DiscreteDeterministicActor(BaseActor):
    def __init__(self, state_dim, hidden_size, n_actions):
        super().__init__()
        self.model = build_mlp(
            [state_dim] + list(hidden_size) + [n_actions], activation=nn.ReLU()
        )

    def forward(self, t, **kwargs):
        """
        Compute the action given either a time step (looking into the workspace)
        or an observation (in kwargs)
        """
        if "observation" in kwargs:
            observation = kwargs["observation"]
        else:
            observation = self.get(("env/env_obs", t))
        action = self.model(observation)
        self.set(("action", t), action)

    def predict_action(self, obs):
        action = self.model(obs)
        return action


class ContinuousDeterministicActor(BaseActor):
    def __init__(self, state_dim, hidden_layers, action_dim):
        super().__init__()
        layers = [state_dim] + list(hidden_layers) + [action_dim]
        self.model = build_mlp(
            layers,
            activation=nn.Tanh(),
            output_activation=nn.Tanh()
            # layers, activation=nn.ReLU(), output_activation=nn.ReLU()
        )

    def forward(self, t, **kwargs):
        obs = self.get(("env/env_obs", t))
        action = self.model(obs)
        self.set(("action", t), action)

    def predict_action(self, obs, stochastic=False):
        """Predict just one action (without using the workspace)"""
        assert (
            not stochastic
        ), "ContinuousDeterministicActor cannot provide stochastic predictions"
        return self.model(obs)
