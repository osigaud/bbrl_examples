import sys
import os

from bbrl_examples.models.plotters import CommonPlotter
import matplotlib

matplotlib.use("TkAgg")


def main():
    logdir = "./plot/"
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    plotter = CommonPlotter(logdir, "./steps/nfq.steps")
    plotter.plot_rewards("CartPole-v1")


if __name__ == "__main__":
    sys.path.append(os.getcwd())
    main()
