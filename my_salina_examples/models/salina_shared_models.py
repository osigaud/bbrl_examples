import torch
import torch.nn as nn
from salina import Agent


def build_backbone(sizes, activation):
    layers = []
    for j in range(len(sizes) - 2):
        layers += [nn.Linear(sizes[j], sizes[j + 1]), activation]
    return layers


def build_mlp(sizes, activation, output_activation=nn.Identity()):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act]
    return nn.Sequential(*layers)


class GenericAgent(Agent):
    """
    The super class of all policy and critic networks
    Contains general behaviors like loading and saving, and updating from a loss
    The standard loss function used is the Mean Squared Error (MSE)
    """
    def __init__(self, name="Agent"):
        super(GenericAgent, self).__init__(name)
        self.loss_func = torch.nn.MSELoss()

    def update(self, loss) -> None:
        """
        Apply a loss to a network using gradient backpropagation
        :param loss: the applied loss
        :return: nothing
        """
        self.optimizer.zero_grad()
        loss.sum().backward()
        self.optimizer.step()



