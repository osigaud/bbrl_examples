import gym
import random
import numpy as np


class RocketLanderWrapper(gym.Wrapper):
    """
    Specific wrapper to shape the reward of the rocket lander environment
    """

    def __init__(self, env):
        super(RocketLanderWrapper, self).__init__(env)
        self.prev_shaping = None

    def reset(self):
        self.prev_shaping = None
        return self.env.reset()

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        # reward shaping
        """
        shaping = -0.5 * (self.env.distance + self.env.speed + abs(self.env.angle) ** 2)
        shaping += 0.1 * (
            self.env.legs[0].ground_contact + self.env.legs[1].ground_contact
        )
        if self.prev_shaping is not None:
            reward += shaping - self.prev_shaping
        self.prev_shaping = shaping
        """
        shaping = 0.02
        # shaping = 0.1 * (self.env.groundcontact - self.env.speed)
        if (
            self.env.legs[0].ground_contact > 0
            and self.env.legs[1].ground_contact > 0
            and self.env.speed < 0.1
        ):
            print("landed !")
            shaping += 3.0
        reward += shaping

        return next_state, reward, done, info


class MazeMDPContinuousWrapper(gym.Wrapper):
    """
    Specific wrapper to turn the Tabular MazeMDP into a continuous state version
    """

    def __init__(self, env):
        super(MazeMDPContinuousWrapper, self).__init__(env)
        # Building a new continuous observation space from the coordinates of each state
        high = np.array(
            [
                env.coord_x.max() + 1,
                env.coord_y.max() + 1,
            ],
            dtype=np.float32,
        )
        low = np.array(
            [
                env.coord_x.min(),
                env.coord_y.min(),
            ],
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(low, high)

    def is_continuous_state(self):
        # By contrast with the wrapped environment where the state space is discrete
        return True

    def reset(self):
        obs = self.env.reset()
        x = self.env.coord_x[obs]
        y = self.env.coord_y[obs]
        xc = x + random.random()
        yc = y + random.random()
        continuous_obs = [xc, yc]
        return continuous_obs

    def step(self, action):
        # Turn the discrete state into a pair of continuous coordinates
        # Take the coordinates of the state and add a random number to x and y to
        # sample anywhere in the [1, 1] cell...
        next_state, reward, done, info = self.env.step(action)
        x = self.env.coord_x[next_state]
        y = self.env.coord_y[next_state]
        xc = x + random.random()
        yc = y + random.random()
        next_continuous = [xc, yc]
        return next_continuous, reward, done, info
