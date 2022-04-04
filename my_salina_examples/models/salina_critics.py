from salina import Agent
import torch
import torch.nn as nn

from my_salina_examples.models.salina_shared_models import mlp
# from models.salina_shared_models import mlp

class Q_Agent(Agent):
    def __init__(self, state_dim, action_dim, hidden_layers, **kwargs):
        super().__init__()
        self.model = mlp([state_dim + action_dim] + list(hidden_layers) + [1], activation=nn.ReLU)

    def forward(self, t, detach_actions=False, **kwargs):
        obs = self.get(("env/env_obs", t))
        action = self.get(("action", t))
        if detach_actions:
            action = action.detach()
        input = torch.cat((obs, action), dim=1)
        q_value = self.model(input)
        self.set(("q_value", t), q_value)


class V_Agent(Agent):
    def __init__(self, state_dim, hidden_layers, **kwargs):
        super().__init__()
        self.model = mlp([state_dim] + list(hidden_layers) + [1], activation=nn.ReLU)

    def forward(self, t, **kwargs):
        observation = self.get(("env/env_obs", t))
        critic = self.model(observation).squeeze(-1)
        self.set(("critic", t), critic)
