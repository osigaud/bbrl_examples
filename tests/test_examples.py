import gym
import my_gym


def test_cartpolecontinuous_v0():
    env = gym.make("CartpoleContinuous-v0")
    env.reset()


def test_cartpolecontinuous_v1():
    env = gym.make("CartpoleContinuous-v1")
    env.reset()
