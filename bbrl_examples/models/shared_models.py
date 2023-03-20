import numpy as np
import torch.nn as nn


def ortho_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    Function used for orthogonal inialization of the layers
    Taken from here in the cleanRL library: https://github.com/vwxyzjn/ppo-implementation-details/blob/main/ppo.py
    """
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def build_backbone(sizes, activation):
    layers = []
    for j in range(len(sizes) - 1):
        layers += [nn.Linear(sizes[j], sizes[j + 1]), activation]
    return layers


def build_ortho_backbone(sizes, activation):
    layers = []
    for j in range(len(sizes) - 1):
        layers += [ortho_init(nn.Linear(sizes[j], sizes[j + 1])), activation]
    return layers


def build_mlp(sizes, activation, output_activation=nn.Identity()):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act]
    return nn.Sequential(*layers)


def build_ortho_mlp(sizes, activation, output_activation=nn.Identity()):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [ortho_init(nn.Linear(sizes[j], sizes[j + 1])), act]
    return nn.Sequential(*layers)


def build_alt_mlp(sizes, activation):
    layers = []
    for j in range(len(sizes) - 1):
        if j < len(sizes) - 2:
            layers += [nn.Linear(sizes[j], sizes[j + 1]), activation]
        else:
            layers += [nn.Linear(sizes[j], sizes[j + 1])]
    return nn.Sequential(*layers)


def build_ortho_alt_mlp(sizes, activation):
    layers = []
    for j in range(len(sizes) - 1):
        if j < len(sizes) - 2:
            layers += [ortho_init(nn.Linear(sizes[j], sizes[j + 1])), activation]
        else:
            layers += [ortho_init(nn.Linear(sizes[j], sizes[j + 1]))]
    return nn.Sequential(*layers)


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
