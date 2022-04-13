from salina import Agent
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from my_salina_examples.models.salina_shared_models import build_mlp, build_backbone


class DeterministicAgent(Agent):
    def __init__(self, state_dim, hidden_layers, action_dim, **kwargs):
        super().__init__()
        layers = [state_dim] + list(hidden_layers) + [action_dim]
        self.model = build_mlp(layers, activation=nn.ReLU(), output_activation=nn.Tanh())

    def forward(self, t, epsilon=0, **kwargs):
        obs = self.get(("env/env_obs", t))
        action = self.model(obs)

        noise = torch.randn(*action.size(), device=action.device) * epsilon
        action = action + noise
        action = torch.clip(action, min=-1.0, max=1.0)
        self.set(("action", t), action)


class ProbAgent(Agent):
    def __init__(self, state_dim, hidden_layers, n_action, **kwargs):
        super().__init__(name="prob_agent")
        self.model = build_mlp([state_dim] + list(hidden_layers) + [n_action], activation=nn.ReLU())

    def forward(self, t, **kwargs):
        observation = self.get(("env/env_obs", t))
        scores = self.model(observation)
        action_probs = torch.softmax(scores, dim=-1)
        if torch.any(torch.isnan(action_probs)):
            print("Nan Here")
        self.set(("action_probs", t), action_probs)
        entropy = torch.distributions.Categorical(action_probs).entropy()
        self.set(("entropy", t), entropy)


class ActionAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, t, stochastic, **kwargs):
        probs = self.get(("action_probs", t))
        if stochastic:
            action = torch.distributions.Categorical(probs).sample()
        else:
            action = probs.argmax(1)

        self.set(("action", t), action)


# All the actors below use a squashed Gaussian policy, that is the output is the tanh of a Normal distribution


class ContinuousActionTunableVarianceAgent(Agent):
    def __init__(self, state_dim, hidden_layers, action_dim, **kwargs):
        super().__init__()
        layers = [state_dim] + list(hidden_layers) + [action_dim]
        self.model = build_mlp(layers, activation=nn.ReLU(), output_activation=nn.Tanh())
        self.std_param = nn.parameter.Parameter(torch.randn(action_dim, 1))
        self.soft_plus = torch.nn.Softplus()

    def forward(self, t, stochastic, **kwargs):
        obs = self.get(("env/env_obs", t))
        mean = self.model(obs)
        dist = Normal(mean, self.soft_plus(self.std_param))  # std must be positive
        self.set(("entropy", t), dist.entropy())
        if stochastic:
            action = torch.tanh(dist.sample())  # valid actions are supposed to be in [-1,1] range
        else:
            action = torch.tanh(mean)  # valid actions are supposed to be in [-1,1] range
        logp_pi = dist.log_prob(action).sum(axis=-1)
        self.set(("action", t), action)
        self.set(("action_logprobs", t), logp_pi)


class ContinuousActionStateDependentVarianceAgent(Agent):
    def __init__(self, state_dim, hidden_layers, action_dim, **kwargs):
        super().__init__()
        backbone_dim = [state_dim] + list(hidden_layers)
        self.layers = build_backbone(backbone_dim, activation=nn.ReLU())
        self.last_layer = nn.Linear(hidden_layers[-1], action_dim)
        self.mean_layer = nn.Tanh()
        # std must be positive
        self.std_layer = nn.Softplus()
        self.backbone = nn.Sequential(*self.layers)

    def forward(self, t, stochastic, **kwargs):
        obs = self.get(("env/env_obs", t))
        backbone_output = self.backbone(obs)
        mean = self.mean_layer(backbone_output)
        dist = Normal(mean, self.std_layer(backbone_output))
        self.set(("entropy", t), dist.entropy())
        if stochastic:
            action = torch.tanh(dist.sample())  # valid actions are supposed to be in [-1,1] range
        else:
            action = torch.tanh(mean)  # valid actions are supposed to be in [-1,1] range
        logp_pi = dist.log_prob(action).sum(axis=-1)
        self.set(("action", t), action)
        self.set(("action_logprobs", t), logp_pi)


class ContinuousActionConstantVarianceAgent(Agent):
    def __init__(self, state_dim, hidden_layers, action_dim, **kwargs):
        super().__init__()
        layers = [state_dim] + list(hidden_layers) + [action_dim]
        self.model = build_mlp(layers, activation=nn.ReLU(), output_activation=nn.Tanh())
        self.std_param = 2

    def forward(self, t, stochastic, **kwargs):
        obs = self.get(("env/env_obs", t))
        mean = self.model(obs)
        dist = Normal(mean, self.std_param)  # std must be positive
        self.set(("entropy", t), dist.entropy())
        if stochastic:
            action = torch.tanh(dist.sample())  # valid actions are supposed to be in [-1,1] range
        else:
            action = torch.tanh(mean)  # valid actions are supposed to be in [-1,1] range
        logp_pi = dist.log_prob(action).sum(axis=-1)
        self.set(("action", t), action)
        self.set(("action_logprobs", t), logp_pi)
