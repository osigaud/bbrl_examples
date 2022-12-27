import numpy as np
from bbrl import instantiate_class


class Logger:
    def __init__(self, cfg):
        self.logger = instantiate_class(cfg.logger)

    def add_log(self, log_string, loss, epoch):
        self.logger.add_scalar(log_string, loss.item(), epoch)

    # Log losses
    def log_losses(self, epoch, critic_loss, entropy_loss, actor_loss):
        self.add_log("critic_loss", critic_loss, epoch)
        self.add_log("entropy_loss", entropy_loss, epoch)
        self.add_log("actor_loss", actor_loss, epoch)


class RewardLogger:
    def __init__(self, steps_filename, rewards_filename):
        self.steps_filename = steps_filename
        self.rewards_filename = rewards_filename
        self.episode = 0
        self.all_rewards = []
        self.all_rewards.append([])
        self.all_steps = []

    def add(self, nb_steps, reward):
        if self.episode == 0:
            self.all_steps.append(nb_steps)
        self.all_rewards[self.episode].append(reward.item())

    def new_episode(self):
        self.episode += 1
        self.all_rewards.append([])

    def save(self):
        # print("reward loader save:", self.all_steps,  self.all_rewards)
        with open(self.steps_filename, "ab") as f:
            np.save(f, self.all_steps)
        with open(self.rewards_filename, "ab") as f:
            np.save(f, self.all_rewards)


class RewardLoader:
    def __init__(self, steps_filename, rewards_filename):
        self.steps_filename = steps_filename
        self.rewards_filename = rewards_filename

    def load(self):
        with open(self.steps_filename, "rb") as f:
            steps = np.load(f, allow_pickle=True)
        with open(self.rewards_filename, "rb") as f:
            rewards = np.load(f, allow_pickle=True)
        return steps, rewards
