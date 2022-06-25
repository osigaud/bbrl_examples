import os
import numpy as np

from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
from pathlib import Path

from bbrl.visu.common import final_show
from bbrl_examples.models.loggers import RewardLoader


def format_num(num, pos):
    # Pos is a required parameter, but it is not used
    magnitude = 0
    labels = ["", "K", "M", "G"]
    while abs(num) >= 1e3:
        magnitude += 1
        num /= 1e3

    return f"{num:.1f}{labels[magnitude]}"


def equalize_lengths(steps, rewards):
    min_len = len(steps)
    for i in rewards:
        reward_len = len(i)
        if reward_len < min_len:
            min_len = reward_len
    for i in range(len(rewards)):
        rewards[i] = rewards[i][0:min_len]
    steps = steps[0:min_len]
    return steps, rewards


class Plotter:
    def __init__(self, steps_filename, rewards_filename):
        self.steps_filename = steps_filename
        self.rewards_filename = rewards_filename

    def plot_reward(
        self,
        algo_name,
        env_name,
        mode="mean",
        prefix="",
        suffix=".pdf",
        save_fig=True,
        save_dir="./plots/",
    ):
        _, ax = plt.subplots(figsize=(9, 6))
        formatter = FuncFormatter(format_num)

        colors = ["#09b542", "#008fd5", "#fc4f30", "#e5ae38", "#e5ae38", "#810f7c"]
        color = colors[0]

        loader = RewardLoader(self.steps_filename, self.rewards_filename)
        steps, rewards = loader.load()
        print(steps, rewards)
        # steps, rewards = equalize_lengths(steps, rewards)

        if mode == "best":
            best = rewards.sum(axis=1).argmax()
            mean = rewards[best]
        elif mode == "max":
            mean = np.max(rewards, axis=0)
        else:
            std = rewards.std(axis=0)
            mean = rewards.mean(axis=0)
            ax.fill_between(steps, mean + std, mean - std, alpha=0.1, color=color)
        ax.plot(steps, mean, lw=2, label=f"{algo_name}", color=color)
        ax.xaxis.set_major_formatter(formatter)
        plt.legend()

        save_dir += f"{env_name}/"

        clean_env_name = env_name.split("-")[0]
        figure_name = f"{prefix}{clean_env_name.lower()}_{mode}"
        title = f"{clean_env_name} ({mode})"
        if suffix:
            figure_name += f"{suffix}"
        final_show(save_fig, True, save_dir, figure_name, "timesteps", "rewards", title)

    def plot_histograms(
        self,
        rewards,
        env_name,
        suffix="",
        save_dir="./plots/",
        plot=True,
        save_fig=True,
    ):
        plt.figure(figsize=(9, 6))

        colors = ["#09b542", "#008fd5", "#fc4f30", "#e5ae38", "#e5ae38", "#810f7c"]
        # colors = ["#fc4f30", "#008fd5", "#e5ae38"]

        n_bars = len(rewards)
        x = np.arange(len(list(rewards.values())[0]))
        width = 0.75 / n_bars

        for i, reward in enumerate(rewards.values()):
            plt.bar(x + width * i, np.sort(reward)[::-1], width=width, color=colors[i])

        plt.legend(labels=rewards.keys())
        plt.xticks([], [])

        save_dir += f"{env_name}/"

        clean_env_name = env_name.split("-")[0]
        title = clean_env_name
        figure_name = f"{clean_env_name.lower()}-histograms"

        if suffix:
            title += f" ({suffix})"
            figure_name += f"{suffix}"

        final_show(save_fig, plot, save_dir, figure_name, "", "rewards", title)


class CommonPlotter:
    def __init__(self, logdir, steps_filename):
        self.steps_filename = steps_filename
        self.logdir = logdir

    def plot_rewards(
        self,
        env_name,
        mode="mean",
        prefix="",
        suffix=".pdf",
        save_fig=True,
        save_dir="./plots/",
    ):
        _, ax = plt.subplots(figsize=(9, 6))
        formatter = FuncFormatter(format_num)

        colors = [
            "#09b542",
            "#008fd5",
            "#fc4f30",
            "#e5ae38",
            "#351238",
            "#810f7c",
            "#320f4c",
            "#622f9a",
        ]
        cpt = 0

        listdir = os.listdir(self.logdir)
        for reward_file in listdir:
            print(reward_file)
            algo_name = reward_file.split(".")[0]
            loader = RewardLoader(self.steps_filename, self.logdir + reward_file)
            steps, rewards = loader.load()
            print(steps, rewards)

            if mode == "best":
                best = rewards.sum(axis=1).argmax()
                mean = rewards[best]
            elif mode == "max":
                mean = np.max(rewards, axis=0)
            else:
                std = rewards.std(axis=0)
                mean = rewards.mean(axis=0)
                ax.fill_between(
                    steps, mean + std, mean - std, alpha=0.1, color=colors[cpt]
                )
            ax.plot(steps, mean, lw=2, label=f"{algo_name}", color=colors[cpt])
            cpt += 1 % len(colors)

        ax.xaxis.set_major_formatter(formatter)
        plt.legend()

        save_dir += f"{env_name}/"

        clean_env_name = env_name.split("-")[0]
        figure_name = f"{prefix}{clean_env_name.lower()}_{mode}"
        title = f"{clean_env_name} ({mode})"
        if suffix:
            figure_name += f"{suffix}"
        final_show(save_fig, True, save_dir, figure_name, "timesteps", "rewards", title)
