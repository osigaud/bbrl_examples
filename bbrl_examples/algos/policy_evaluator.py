import os
import numpy as np
import torch
import gym
import bbrl_gym

from bbrl.workspace import Workspace
from bbrl.agents import Agents, PrintAgent, TemporalAgent
from bbrl.agents.gymb import NoAutoResetGymAgent
from bbrl.utils.chrono import Chrono

from bbrl_examples.models.actors import *
from bbrl_examples.models.critics import *

path = "/data/policies/"
nb_trials = 2
seed = 3

from script_ddqn import *

def make_gym_env(env_name):
    """Create the used environment"""
    env = gym.make(env_name)
    # print(env)
    return env


def evaluate_agent(filename, env_name):
    agent = torch.load(os.getcwd() + path + filename)
    if isinstance(agent, TemporalAgent):
        eval_agent = agent
    else:
        eval_env = NoAutoResetGymAgent(
            make_gym_env,
            {"env_name": env_name},
            1,
            seed,
        )
        # pa = PrintAgent()
        agents = Agents(eval_env, agent)
        eval_agent = TemporalAgent(agents)
    print(eval_agent)
    
    means = np.zeros(nb_trials)
    for i in range(nb_trials):
        eval_workspace = Workspace()  # Used for evaluation
        eval_agent(
            eval_workspace,
            t=0,
            stop_variable="env/done",
            stochastic=False,
            # predict_proba=False,
            render=True,
        )
        rewards = eval_workspace["env/cumulated_reward"][-1]
        means[i] = rewards.mean()
    return means


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


class Evaluator:
    """
    A class to evaluate a set of policies stored into the same folder and ranking them according to their scores
    """

    def __init__(self):
        self.score_dic = {}

    def load_policies(self, folder) -> None:
        """
        :param: folder : name of the folder containing policies
        Output : none (policies of the folder stored in self.env_dict)
        """
        listdir = os.listdir(folder)
        for policy_file in listdir:
            print(policy_file)
            env_name, algo, team_name = read_name(policy_file)

            if env_name in self.score_dic:
                scores = evaluate_agent(policy_file, env_name)
                self.score_dic[env_name][scores.mean()] = [
                    team_name,
                    algo,
                    scores.std(),
                ]
            else:
                scores = evaluate_agent(policy_file, env_name)
                tmp_dic = {scores.mean(): [team_name, algo, scores.std()]}
                self.score_dic[env_name] = tmp_dic
            self.display_hall_of_fame()

    def display_hall_of_fame(self) -> None:
        """
        Display the hall of fame of all the evaluated policies
        :return: nothing
        """
        print("Hall of fame")
        for env, dico in self.score_dic.items():
            print("Environment :", env)
            for key, val in sorted(dico.items(), reverse=True):
                print(
                    "team: ",
                    val[0],
                    " \t \t algo:",
                    val[1],
                    " \t \t mean score: ",
                    key,
                    "std: ",
                    val[2],
                )


if __name__ == "__main__":
    directory = os.getcwd() + path
    ev = Evaluator()
    c = Chrono()
    ev.load_policies(directory)
    ev.display_hall_of_fame()
    c.stop()
