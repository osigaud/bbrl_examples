# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import random

import matplotlib.pyplot as plt
import numpy as np
import torch as th

from bbrl.visu.common import final_show


def plot_lunar_lander_true_critic_q(
    agent,
    env,
    env_string,
    directory,
    figure_name,
    plot=True,
    save_figure=True,
    action=None,
):
    """
    Visualization of the critic in a N-dimensional state space
    The N-dimensional state space is projected into its first two dimensions.
    A FeatureInverter wrapper should be used to select which features to put first to plot them
    :param agent: the critic agent to be plotted
    :param env: the environment
    :param env_string: the name of the environment
    :param plot: whether the plot should be interactive
    :param directory: the directory where to save the figure
    :param figure_name: the name of the file where to plot the function
    :param save_figure: whether the plot should be saved into a file
    :param action: the action for which we want to plot the value
    :return: nothing
    """
    if env.observation_space.shape[0] <= 2:
        msg = f"Observation space dim {env.observation_space.shape[0]}, should be > 2"
        raise (ValueError(msg))
    definition = 100
    portrait = np.zeros((definition, definition))
    state_min = [-1.5, -1.5, -5.0, -5.0, -3.14, -5.0, -0.0, -0.0]
    state_max = [1.5, 1.5, 5.0, 5.0, 3.14, 5.0, 1.0, 1.0]

    for index_x, x in enumerate(
        np.linspace(state_min[0], state_max[0], num=definition)
    ):
        for index_y, y in enumerate(
            np.linspace(state_min[1], state_max[1], num=definition)
        ):
            obs = np.array([x])
            obs = np.append(obs, y)
            nb = range(len(state_min) - 2)
            for _ in nb:
                z = random.random() - 0.5
                obs = np.append(obs, z)
            obs = obs.reshape(1, -1)
            obs = th.from_numpy(obs.astype(np.float32))

            if action is None:
                action = th.Tensor([0, 0])
            value = agent.predict_value(obs[0], action)

            portrait[definition - (1 + index_y), index_x] = value.item()

    plt.figure(figsize=(10, 10))
    plt.imshow(
        portrait,
        cmap="inferno",
        extent=[state_min[0], state_max[0], state_min[1], state_max[1]],
        aspect="auto",
    )

    directory += "/" + env_string + "_critics/"
    title = env_string + " Critic"
    plt.colorbar(label="critic value")

    # Add a point at the center
    plt.scatter([0], [0])
    x_label, y_label = getattr(env.observation_space, "names", ["x", "y"])
    final_show(save_figure, plot, directory, figure_name, x_label, y_label, title)
