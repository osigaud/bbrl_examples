import os
import torch
import gym
import bbrl_gym

from bbrl.utils.chrono import Chrono
from bbrl.visu.play import play

path = "/data/policies/"


def make_gym_env(env_name):
    """Create the used environment"""
    return gym.make(env_name)


def read_name(filename):
    """
    :param filename: the file name, including the path
    :return: fields
    """
    fields = filename.split("#")
    tmp = fields[0]
    env_name = tmp.split("/")
    env_name = env_name[-1]
    algo = fields[1]
    end_name = fields[2]
    team_name = end_name.split(".")
    return env_name, algo, team_name[0]


def play_agents(folder) -> None:
    """
    :param: folder : name of the folder containing policies
    Output : none (policies of the folder stored in self.env_dict)
    """
    listdir = os.listdir(folder)
    for policy_file in listdir:
        print(policy_file)
        env_name, algo, team_name = read_name(policy_file)
        print(env_name)
        env = make_gym_env(env_name)
        eval_agent = torch.load(os.getcwd() + path + policy_file)
        eval_agent = eval_agent.agent.agents[1]
        play(env, eval_agent)


if __name__ == "__main__":
    directory = os.getcwd() + path
    c = Chrono()
    play_agents(directory)
    c.stop()
