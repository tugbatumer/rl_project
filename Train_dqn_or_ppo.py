import gymnasium as gym
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch.optim import AdamW
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import random
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as plt
from itertools import count
import math 


from DQN import DQNAGENT
from ppo import PPOAGENT


LR = 0.005
GAMMA = 0.95

N_trajectories = 4800
N_updates = 5
N_total = 1e4 # 2e8
CLIP = 0.2
env = gym.make("Pendulum-v1")


model = PPOAGENT(env, LR, GAMMA, N_total, N_trajectories, N_updates, CLIP)
ep_rewards = model.learn()

'''
env = gym.make("CartPole-v1")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
LR = 1e-4
GAMMA = 0.99
TAU = 0.005
BATCH_SIZE = 128
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
REPLAY_SIZE = int(1e6)
N_episode = 600

model = DQNAGENT(env, N_episode, LR, GAMMA, TAU, BATCH_SIZE, EPS_START, EPS_END, EPS_DECAY, REPLAY_SIZE)
ep_rewards = model.learn()
'''

# smooth rewards with a moving average window
def smooth(x, window=20):
    return np.convolve(x, np.ones(window)/window, mode='valid')

plt.figure(figsize=(8,5))
plt.plot( (ep_rewards), label="PPO")
# plt.plot(dqn_steps,   smooth(dqn_rewards),  label="DQN")
plt.xlabel("Environment steps")
plt.ylabel("Episodic return (smoothed)")
plt.legend()
plt.title("Learning curves: PPO vs DQN on Pendulum-v1")
plt.show()





