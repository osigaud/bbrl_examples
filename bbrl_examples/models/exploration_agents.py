import math
import torch
from bbrl.agents.agent import Agent
from torch.distributions import Normal


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


class AddGaussianNoise(Agent):
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma

    def forward(self, t, **kwargs):
        act = self.get(("action", t))
        dist = Normal(act, self.sigma)
        action = dist.sample()
        self.set(("action", t), action)


class AddOUNoise(Agent):
    """
    Ornstein Uhlenbeck process noise for actions as suggested by DDPG paper
    """

    def __init__(self, std_dev, theta=0.15, dt=1e-2):
        self.theta = theta
        self.std_dev = std_dev
        self.dt = dt
        self.x_prev = 0

    def forward(self, t, **kwargs):
        act = self.get(("action", t))
        x = (
            self.x_prev
            + self.theta * (act - self.x_prev) * self.dt
            + self.std_dev * math.sqrt(self.dt) * torch.randn(act.shape)
        )
        self.x_prev = x
        self.set(("action", t), x)
