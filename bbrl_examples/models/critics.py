import torch
import torch.nn as nn

from bbrl.agents.agent import Agent

from bbrl_examples.models.shared_models import build_mlp


class ContinuousQAgent(Agent):
    def __init__(self, state_dim, hidden_layers, action_dim):
        super().__init__()
        self.model = build_mlp(
            [state_dim + action_dim] + list(hidden_layers) + [1], activation=nn.ReLU()
        )

    def forward(self, t, detach_actions=False, **kwargs):
        obs = self.get(("env/env_obs", t))
        action = self.get(("action", t))
        if detach_actions:
            action = action.detach()
        osb_act = torch.cat((obs, action), dim=1)
        q_value = self.model(osb_act)
        self.set(("q_value", t), q_value)


class VAgent(Agent):
    def __init__(self, state_dim, hidden_layers):
        super().__init__()
        self.model = build_mlp(
            [state_dim] + list(hidden_layers) + [1], activation=nn.ReLU()
        )

    def forward(self, t, **kwargs):
        observation = self.get(("env/env_obs", t))
        critic = self.model(observation).squeeze(-1)
        self.set(("v_value", t), critic)


class DiscreteQAgent(Agent):
    def __init__(self, state_dim, hidden_layers, action_dim):
        super().__init__()
        self.model = build_mlp(
            [state_dim] + list(hidden_layers) + [action_dim], activation=nn.ReLU()
        )

    def forward(self, t, choose_action=True, **kwargs):
        obs = self.get(("env/env_obs", t))
        q_values = self.model(obs).squeeze(-1)
        self.set(("q_values", t), q_values)
        if choose_action:
            action = q_values.argmax(1)
            self.set(("action", t), action)
