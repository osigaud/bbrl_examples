import gym


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
        shaping = -0.5 * (self.env.distance + self.env.speed + abs(self.env.angle) ** 2)
        shaping += 0.1 * (
            self.env.legs[0].ground_contact + self.env.legs[1].ground_contact
        )
        if self.prev_shaping is not None:
            reward += shaping - self.prev_shaping
        self.prev_shaping = shaping

        return next_state, reward, done, info
