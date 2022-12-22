import torch
import torch.nn as nn

from bbrl.agents.agent import Agent

from bbrl_examples.models.shared_models import build_mlp, build_alt_mlp


class ContinuousQAgent(Agent):
    def __init__(self, state_dim, hidden_layers, action_dim):
        super().__init__()
        self.is_q_function = True
        self.model = build_mlp(
            [state_dim + action_dim] + list(hidden_layers) + [1], activation=nn.ReLU()
        )

    def forward(self, t, detach_actions=False):
        obs = self.get(("env/env_obs", t))
        action = self.get(("action", t))
        if detach_actions:
            action = action.detach()
        osb_act = torch.cat((obs, action), dim=1)
        q_value = self.model(osb_act)
        self.set(("q_value", t), q_value)

    def predict_value(self, obs, action):
        obs_act = torch.cat((obs, action), dim=0)
        q_value = self.model(obs_act)
        return q_value


class VAgent(Agent):
    def __init__(self, state_dim, hidden_layers):
        super().__init__()
        self.is_q_function = False
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
        self.is_q_function = True
        self.model = build_alt_mlp(
            [state_dim] + list(hidden_layers) + [action_dim], activation=nn.ReLU()
        )

    def forward(self, t, choose_action=True, **kwargs):
        obs = self.get(("env/env_obs", t))
        q_values = self.model(obs).squeeze(-1)
        self.set(("q_values", t), q_values)
        if choose_action:
            action = q_values.argmax(-1)
            self.set(("action", t), action)

    def predict_action(self, obs, stochastic):
        q_values = self.model(obs).squeeze(-1)
        if stochastic:
            probs = torch.softmax(q_values, dim=-1)
            action = torch.distributions.Categorical(probs).sample()
        else:
            action = q_values.argmax(-1)
        return action

    def predict_value(self, obs, action):
        q_values = self.model(obs).squeeze(-1)
        return q_values[action[0].int()]

class TruncatedQuantileNetwork(Agent):
    def __init__(self, state_dim, hidden_layers, n_nets, action_dim, n_quantiles):
        super().__init__()
        self.is_q_function = True
        self.nets = []
        for i in range(n_nets):
            net = build_mlp([state_dim + action_dim] + list(hidden_layers) + [n_quantiles], activation=nn.ReLU())
            self.add_module(f'qf{i}', net)
            self.nets.append(net)

    def forward(self, t):
        obs = self.get(("env/env_obs", t))
        action = self.get(("action", t))
        obs_act = torch.cat((obs, action), dim=1)
        quantiles = torch.stack(tuple(net(obs_act) for net in self.nets), dim=1)
        self.set(("quantiles", t), quantiles)
        return quantiles

    def predict_value(self, obs, action):
        obs_act = torch.cat((obs, action), dim=0)
        quantiles = torch.stack(tuple(net(obs_act) for net in self.nets), dim=1)
        return quantiles
