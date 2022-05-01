import os

import matplotlib.pyplot as plt
from pathlib import Path

def final_show(save_figure, plot, figure_name, x_label, y_label, title, directory):
    """
    Finalize all plots, adding labels and putting the corresponding file in the
    specified directory
    :param save_figure: boolean stating whether the figure should be saved
    :param plot: whether the plot should be shown interactively
    :param figure_name: the name of the file where to save the figure
    :param x_label: label on the x axis
    :param y_label: label on the y axis
    :param title: title of the figure
    :return: nothing
    """
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    if save_figure:
        if not os.path.exists(directory):
            os.makedirs(directory)
        filename = Path(directory + figure_name)
        plt.savefig(filename)

    if plot:
        plt.show()

    plt.close()

