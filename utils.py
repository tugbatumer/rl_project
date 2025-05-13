import numpy as np
import random
import torch
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

def mlp(input_dim, hidden_size):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
    )
    
# Critic Network
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super().__init__()
        self.q1 = nn.Linear(hidden_size, 1)
        self.q2 = nn.Linear(hidden_size, 1)
        self.mlp = mlp(state_dim + action_dim, hidden_size)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.q1(self.mlp(x)), self.q2(self.mlp(x))
