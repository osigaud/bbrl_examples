import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from bbrl_examples.models.shared_models import build_mlp, build_backbone
from bbrl.agents.agent import Agent


class EGreedyActionSelector(Agent):
    def __init__(self, epsilon):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, t, **kwargs):
        q_values = self.get(("q_values", t))
        nb_actions = q_values.size()[1]
        size = q_values.size()[0]
        is_random = torch.rand(size).lt(self.epsilon).float()
        random_action = torch.randint(low=0, high=nb_actions, size=(size,))
        max_action = q_values.max(1)[1]
        action = is_random * random_action + (1 - is_random) * max_action
        action = action.long()
        self.set(("action", t), action)


class SoftmaxActionSelector(Agent):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature

    def forward(self, t, **kwargs):
        q_values = self.get(("q_values", t))
        probs = torch.softmax(q_values, dim=-1)
        action = torch.distributions.Categorical(probs).sample()
        self.set(("action", t), action)


class RandomDiscreteActor(Agent):
    def __init__(self, nb_actions):
        super().__init__()
        self.nb_actions = nb_actions

    def forward(self, t, **kwargs):
        obs = self.get(("env/env_obs", t))
        size = obs.size()[0]
        action = torch.randint(low=0, high=self.nb_actions, size=(size,))
        self.set(("action", t), action)


class DiscreteActor(Agent):
    def __init__(self, state_dim, hidden_size, n_actions):
        super().__init__()
        self.model = build_mlp(
            [state_dim] + list(hidden_size) + [n_actions], activation=nn.ReLU()
        )

    def forward(self, t, stochastic, replay=False, **kwargs):
        """
        Compute the action given either a time step (looking into the workspace)
        or an observation (in kwargs)
        """
        if "observation" in kwargs:
            observation = kwargs["observation"]
        else:
            observation = self.get(("env/env_obs", t))
        scores = self.model(observation)
        probs = torch.softmax(scores, dim=-1)

        if stochastic:
            action = torch.distributions.Categorical(probs).sample()
        else:
            action = scores.argmax(1)

        entropy = torch.distributions.Categorical(probs).entropy()
        log_probs = probs[torch.arange(probs.size()[0]), action].log()

        if not replay:
            self.set(("action", t), action)
        self.set(("action_logprobs", t), log_probs)
        self.set(("entropy", t), entropy)

    def predict_action(self, obs, stochastic):
        scores = self.model(obs)

        if stochastic:
            probs = torch.softmax(scores, dim=-1)
            action = torch.distributions.Categorical(probs).sample()
        else:
            action = scores.argmax(0)
        return action


class ProbAgent(Agent):
    def __init__(self, state_dim, hidden_layers, n_action):
        super().__init__(name="prob_agent")
        self.model = build_mlp(
            [state_dim] + list(hidden_layers) + [n_action], activation=nn.ReLU()
        )

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
    def __init__(self):
        super().__init__()

    def forward(self, t, stochastic, **kwargs):
        probs = self.get(("action_probs", t))
        if stochastic:
            action = torch.distributions.Categorical(probs).sample()
        else:
            action = probs.argmax(1)

        self.set(("action", t), action)


# All the actors below use a squashed Gaussian policy, that is the output is the tanh of a Normal distribution


class TunableVarianceContinuousActor(Agent):
    def __init__(self, state_dim, hidden_layers, action_dim):
        super().__init__()
        layers = [state_dim] + list(hidden_layers) + [action_dim]
        self.model = build_mlp(layers, activation=nn.ReLU())
        init_variance = torch.randn(action_dim, 1)
        # print("init_variance:", init_variance)
        self.std_param = nn.parameter.Parameter(init_variance)
        self.soft_plus = torch.nn.Softplus()

    def forward(self, t, stochastic, **kwargs):
        obs = self.get(("env/env_obs", t))
        mean = self.model(obs)
        dist = Normal(mean, self.soft_plus(self.std_param))  # std must be positive
        self.set(("entropy", t), dist.entropy())
        if stochastic:
            action = dist.sample()
        else:
            action = mean
        logp_p = dist.log_prob(action).sum(axis=-1)
        self.set(("action", t), action)
        self.set(("action_logprobs", t), logp_p)

    def predict_action(self, obs, stochastic):
        mean = self.model(obs)
        dist = Normal(mean, self.soft_plus(self.std_param))
        if stochastic:
            action = dist.sample()
        else:
            action = mean
        return action


class StateDependentVarianceContinuousActor(Agent):
    def __init__(self, state_dim, hidden_layers, action_dim):
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
        last = self.last_layer(backbone_output)
        mean = self.mean_layer(last)
        assert not torch.any(torch.isnan(mean)), "Nan Here"
        dist = Normal(mean, self.std_layer(last))
        self.set(("entropy", t), dist.entropy())
        if stochastic:
            action = dist.sample()
        else:
            action = mean
        logp_p = dist.log_prob(action).sum(axis=-1)
        self.set(("action", t), action)
        self.set(("action_logprobs", t), logp_p)

    def predict_action(self, obs, stochastic):
        backbone_output = self.backbone(obs)
        last = self.last_layer(backbone_output)
        mean = self.mean_layer(last)
        dist = Normal(mean, self.std_layer(last))
        if stochastic:
            action = dist.sample()
        else:
            action = mean
        return action


class ConstantVarianceContinuousActor(Agent):
    def __init__(self, state_dim, hidden_layers, action_dim):
        super().__init__()
        layers = [state_dim] + list(hidden_layers) + [action_dim]
        self.model = build_mlp(layers, activation=nn.ReLU())
        self.std_param = 2

    def forward(self, t, stochastic, **kwargs):
        obs = self.get(("env/env_obs", t))
        mean = self.model(obs)
        dist = Normal(mean, self.std_param)  # std must be positive
        self.set(("entropy", t), dist.entropy())
        if stochastic:
            action = dist.sample()
        else:
            action = mean
        logp_p = dist.log_prob(action).sum(axis=-1)
        self.set(("action", t), action)
        self.set(("action_logprobs", t), logp_p)

    def predict_action(self, obs, stochastic):
        mean = self.model(obs)
        dist = Normal(mean, self.std_param)
        if stochastic:
            action = dist.sample()
        else:
            action = mean
        return action
