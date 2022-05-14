from salina.agent import Agent
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from my_salina_examples.models.shared_models import build_mlp


class QAgent(Agent):
    def __init__(self, state_dim, hidden_layers, action_dim, **kwargs):
        super().__init__()
        self.model = build_mlp([state_dim + action_dim] + list(hidden_layers) + [1], activation=nn.ReLU())

    def forward(self, t, detach_actions=False, **kwargs):
        obs = self.get(("env/env_obs", t))
        action = self.get(("action", t))
        if detach_actions:
            action = action.detach()
        osb_act = torch.cat((obs, action), dim=1)
        q_value = self.model(osb_act)
        self.set(("q_value", t), q_value)


class VAgent(Agent):
    def __init__(self, state_dim, hidden_layers, **kwargs):
        super().__init__()
        self.model = build_mlp([state_dim] + list(hidden_layers) + [1], activation=nn.ReLU())

    def forward(self, t, **kwargs):
        observation = self.get(("env/env_obs", t))
        critic = self.model(observation).squeeze(-1)
        self.set(("critic", t), critic)

        
class ContinuousCriticAgent(Agent):
    def __init__(self, state_dim, hidden_layers, action_dim, **kwargs):
        super().__init__()
        self.model = build_mlp([state_dim] + list(hidden_layers) + [action_dim], activation=nn.ReLU(), output_activation=nn.Tanh())

        """
        TODO: how to specify the architecture correctly?
        env = instantiate_class(kwargs["env"])
        input_size = env.observation_space.shape[0]
        hs = kwargs["hidden_size"]
        n_layers = kwargs["n_layers"]
        hidden_layers = (
            [nn.Linear(hs, hs), nn.SiLU()] * (n_layers - 1) # TODO: what is SiLu
            if n_layers > 1
            else nn.Identity()
        )
        self.model_critic = nn.Sequential(
            nn.Linear(input_size, hs),
            nn.SiLU(),
            *hidden_layers,
            nn.Linear(hs, 1),
        )
        """

    def forward(self, t, **kwargs):
        obs = self.get(("env/env_obs", t))
        critic = self.model_critic(obs).squeeze(-1)
        self.set(("critic", t), critic)
