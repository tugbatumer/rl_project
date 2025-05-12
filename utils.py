import numpy as np
import random
import torch.nn as nn

class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = []
        self.size = 0
        self.max_size = max_size

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.stack(states), np.stack(actions), np.stack(rewards),
                np.stack(next_states), np.stack(dones))

def mlp(input_dim):
    return nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
    )