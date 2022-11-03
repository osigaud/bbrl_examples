import gym
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

import random
import matplotlib
import matplotlib.pyplot as plt
from collections import deque
from bbrl.utils.chrono import Chrono

matplotlib.use("TkAgg")

env = gym.make("LunarLander-v2")
nb_actions = 4
nb_observations = 8
# hyper parameters
nb_episode = 500
discount_factor = 0.99
learning_rate = 2e-4
test_frequency = 10
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.02
batch_size = 64
size_replay_buffer = int(1e5)
update_frequency = 1
tau = 1e-3
torch.manual_seed(2)
env.seed(2)


# Class pour la creation de la fonction Q action-value
class QNetwork(nn.Module):
    def __init__(
        self, nb_actions, nb_observations  # nombres d'actions
    ):  # nombre d'etas

        super().__init__()
        self.nb_actions = nb_actions
        self.nb_observations = nb_observations

        # reseau de neurones
        self.net = nn.Sequential(
            nn.Linear(nb_observations, 125),
            nn.ReLU(),
            nn.Linear(125, 100),
            nn.ReLU(),
            nn.Linear(100, nb_actions),
        )

    def forward(self, x):
        return self.net(x)


def test(q_network):
    """
    Fonction de test utilisée pour renvoyer la reward moyenne sur un qnetwork
    """
    state = env.reset()
    done = False
    cum_sum = 0
    timestep = 0
    while not done:
        state_t = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        action = torch.argmax(q_network(state_t)).item()
        new_state, reward, done, _ = env.step(action)
        state = new_state
        cum_sum += reward
        timestep += 1

    return cum_sum


chrono = Chrono()
# initialize replay memory D
replay_buffer = deque(maxlen=size_replay_buffer)

# initialize action value function Q
q_network = QNetwork(nb_actions, nb_observations)
# for name, param in q_network.named_parameters():
#    if param.requires_grad:
#        print(name, param.data)

# initialize target action value function Q
q_target_network = QNetwork(nb_actions, nb_observations)
q_target_network.load_state_dict(
    q_network.state_dict()
)  # same weight for target network as the Q function

optimizer = torch.optim.Adam(q_network.parameters(), lr=learning_rate)  # optimizer
list_tests_2 = []
# average_list = deque(maxlen=100)
average_list = []
list_tests_2_std = []
loss_list = []
timestep = 0

# boucle d'apprentissage
for episode in tqdm(range(nb_episode)):

    # initialize sequence
    ghost_params = torch.nn.Parameter(torch.randn(()))
    obs = env.reset()
    done = False
    cumul = 0
    epsilon = max(epsilon * epsilon_decay, epsilon_min)

    while not done:
        state_t = torch.as_tensor(obs, dtype=torch.float32)

        # e-greedy policy
        if random.random() > epsilon:
            action = torch.argmax(q_network(state_t)).item()
        else:
            action = env.action_space.sample()

        # execute action in emulator
        new_state, reward, done, _ = env.step(action)
        # print(f"s {state}, a {action}, ns {new_state}, r {reward}, d {done}")
        cumul += reward

        # store transition in replay_buffer
        transition = (obs, action, done, reward, new_state)
        replay_buffer.append(transition)

        # EXPERIENCE REPLAY
        if len(replay_buffer) >= batch_size and timestep % update_frequency == 0:

            # sample random minibatch of transitions
            batch = random.sample(replay_buffer, batch_size)

            # transformation in tensor because of pytorch
            states = np.asarray([exp[0] for exp in batch], dtype=np.float32)
            actions = np.asarray([exp[1] for exp in batch], dtype=int)
            dones = np.asarray([exp[2] for exp in batch], dtype=int)
            rewards = np.asarray([exp[3] for exp in batch], dtype=np.float32)
            new_states = np.asarray([exp[4] for exp in batch], dtype=np.float32)

            states_t = torch.as_tensor(states, dtype=torch.float32)
            dones_t = torch.as_tensor(dones, dtype=torch.int64)
            new_states_t = torch.as_tensor(new_states, dtype=torch.float32)
            actions_t = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(1)
            rewards_t = torch.as_tensor(rewards, dtype=torch.float32)

            # set y
            y_target = (
                rewards_t
                + discount_factor
                * (1 - dones_t)
                * torch.max(q_target_network(new_states_t), dim=1)[0].detach()
            )

            # perform a gradient descent step
            mse = nn.MSELoss()
            loss = mse(
                torch.gather(q_network(states_t), dim=1, index=actions_t),
                y_target.unsqueeze(1),
            )
            loss_list.append(loss.detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for target_param, local_param in zip(
                q_target_network.parameters(), q_network.parameters()
            ):
                target_param.data.copy_(
                    tau * local_param.data + (1.0 - tau) * target_param.data
                )

        timestep += 1
        state = new_state

    average_list.append(cumul)

    if episode % test_frequency == 0:
        t = []
        for _ in range(10):
            t.append(test(q_network))
        t = np.asarray(t)
        print(
            f"episode {episode} - test reward : {t.mean()} - std : {t.std()}- epsilon {epsilon}"
        )
        list_tests_2.append(t.mean())
        list_tests_2_std.append(t.std())


chrono.stop()
plt.figure()
l1 = np.array(list_tests_2) + np.array(list_tests_2_std)
l2 = np.array(list_tests_2) - np.array(list_tests_2_std)
plt.plot(list_tests_2, c="r", label="DQN with replay buffer et target network mean")
plt.fill_between(
    np.arange(0, len(list_tests_2), 1), l1, l2, color="mistyrose", label="std reward"
)
plt.title("LunarLander-v2 -rewards")
plt.xlabel("episode")
plt.ylabel("reward")
plt.show()

loss_list = np.asarray(loss_list)
print(f"len : {len(loss_list)}, mean {loss_list.mean()}")
print(loss_list)

plt.plot(loss_list, c="g", label="loss")
plt.title("LunarLander-v2 -loss")
plt.show()
