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

'''
episode_durations = []

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

# plt.ion()

def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

'''


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer   = []
        self.max_size = max_size

    def add(self, state, action, reward, next_state, done):

        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        if isinstance(next_state, torch.Tensor):
            next_state = next_state.cpu().numpy()

        if next_state is None:
            return

        self.buffer.append((state, action, reward, next_state, done))
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.stack(states),
                np.stack(actions),
                np.stack(rewards),
                np.stack(next_states),
                np.stack(dones).astype(np.float32))

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hl1, hl2):
        super(QNetwork, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(state_size, hl1),
            nn.ReLU(),
            nn.Linear(hl1, hl2),
            nn.ReLU(),
            nn.Linear(hl2, action_size),
        )
    
    def forward(self, x):
        x = self.model(x)

        return x


class DQNAGENT:
    def __init__(self, env, N_episode, LR, hidden1, hidden2, GAMMA, TAU, BATCH_SIZE, EPS_START, EPS_END, EPS_DECAY, REPLAY_SIZE):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.LR = LR
        self.steps_done = 0
        self.REPLAY_SIZE = REPLAY_SIZE
        self.EPS_DECAY = EPS_DECAY
        self.EPS_END = EPS_END
        self.EPS_START = EPS_START
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.GAMMA = GAMMA
        self.env = env 
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        # self.max_action = float(env.action_space.high[0])

        self.Q_network = QNetwork(self.state_dim, self.action_dim, hidden1, hidden2).to(device)
        self.Q_target = QNetwork(self.state_dim, self.action_dim, hidden1, hidden2).to(device)

        self.Q_target.load_state_dict(self.Q_network.state_dict())

        self.optimizer = optim.AdamW(self.Q_network.parameters(), lr=self.LR, amsgrad=True)

        self.replay_buffer = ReplayBuffer(self.REPLAY_SIZE)

        self.N_episode = N_episode


    def learn(self):
        ep_rewards = []
        ep_steps = []

        total_env_steps = 0

        for episode in range(N_episode):
            state, info = env.reset()
            '''
            if isinstance(state, tuple):  # Gym >= 0.26 returns (obs, info)
                state = state[0]
            '''
            episode_reward = 0
            rewards_this_episode = []
            for t in count():
                # print(state)
                # state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                #torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                action = self.select_action(state)
                next_state, reward, terminated , truncated, _ = env.step(action)


                done = terminated or truncated
                '''
                if terminated:
                    print("here")
                    next_state = None
                    break
                else:
                    next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
                '''

                # next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
                # mask = 1 if t == env._max_episode_steps else float(not done)
                self.replay_buffer.add(state, action, reward, next_state, done)
                state = next_state

                self.train_dqn()
                rewards_this_episode.append(reward)
                episode_reward += reward
                if done:
                    episode_durations.append(t + 1)
                    # plot_durations()
                    break
            
            ep_reward = sum(rewards_this_episode)
            ep_length = len(rewards_this_episode)

            total_env_steps += ep_length
        
            ep_rewards.append(ep_reward)
            ep_steps.append(total_env_steps)

            print(f"Episode {episode} | Reward: {episode_reward:.2f}")

        return ep_rewards
    

    def select_action(self, state, eps = 0.0):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # state = torch.FloatTensor(state).unsqueeze(0).to(device)
        #state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                    math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if not torch.is_tensor(state):
            state = torch.tensor(state, dtype=torch.float32)
        # 2 Move to device & add batch dim
        state = state.to(device).unsqueeze(0)
        self.Q_network.eval()
        with torch.no_grad():
            action = self.Q_network(state)

        self.Q_network.train()


        if random.random() < eps_threshold:
            random_action = np.random.randint(self.action_dim)
            return random_action
            
        max_action = action.max(1).indices.view(1, 1).squeeze(0)            

        return max_action.item()

    def train_dqn(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if len(self.replay_buffer.buffer) < self.BATCH_SIZE:
            return
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.BATCH_SIZE)
        states = torch.FloatTensor(states).to(device)
        '''
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          next_state)), device=device, dtype=torch.bool)
        
        non_final_next_states = torch.cat([s for s in next_state
                                                if s is not None])
        '''
        
        actions = torch.LongTensor(actions).to(device).unsqueeze(1)

        #actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

        
    


        # print(actions)
        Q_values = self.Q_network(states)
        Q_values = Q_values.gather(1, actions).squeeze(1)

        # Q_targets_next = torch.zeros(BATCH_SIZE, device=device)

        with torch.no_grad():
            Q_targets_next = self.Q_target(next_states).detach().max(1)[0].unsqueeze(1)

        

        expected_q_values = Q_targets_next*self.GAMMA*(1 - dones) + rewards

        loss_fn = nn.SmoothL1Loss()
        loss = loss_fn(Q_values, expected_q_values.squeeze(1))

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_value_(self.Q_network.parameters(), 100)
        self.optimizer.step()
        target_net_state_dict = self.Q_target.state_dict()
        policy_net_state_dict = self.Q_network.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.TAU + target_net_state_dict[key]*(1-self.TAU)



        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.TAU + target_net_state_dict[key]*(1-self.TAU)
        self.Q_target.load_state_dict(target_net_state_dict)


        '''
        self.Q_target.load_state_dict(target_net_state_dict)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
        '''


