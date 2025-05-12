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
from torch.distributions import MultivariateNormal




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




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
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float)
        x = self.model(x)

        return x


class PPOAGENT:

    def __init__(self, env, LR, GAMMA, N_total, N_trajectories, N_updates, CLIP):

        self.N_updates = N_updates
        self.N_total = N_total
        self.N_trajectories = N_trajectories
        self.GAMMA = GAMMA
        self.env = env 
        self.LR = LR
        self.ep_rewards = []

        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        self.CLIP = CLIP
        self.actor = QNetwork(self.state_dim, self.action_dim, 128, 128)
        self.critic = QNetwork(self.state_dim, 1, 128, 128)

        self.opt_actor = optim.AdamW(self.actor.parameters(), lr=self.LR)
        self.opt_critic = optim.AdamW(self.critic.parameters(), lr=self.LR)

        self.cov_var = torch.full(size=(self.action_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)
        self.mse = nn.MSELoss()
    def learn(self):
        t = 0
        while t < self.N_total:
            obs_list, act_list, log_prob_list, rtg_list, passed_time = self.train_ppo_one_iteration()
            t += np.sum(passed_time)
            V, _ = self.estimate_value(obs_list, act_list)

            A_k = rtg_list - V.detach()
            self.update_networks(obs_list, act_list, log_prob_list, A_k)
            print(t)

        return self.ep_rewards

    def update_networks(self, obs, act, log_prob_list, A_k):

        for _ in range(self.N_updates):

            V, log_probs = self.estimate_value(obs, act)
            
            ratio = torch.exp(log_probs - log_prob_list)
            surragate_obj1 = ratio * A_k
            surragate_obj2 = torch.clamp(ratio, 1-self.CLIP, 1+self.CLIP) * A_k

            actor_loss = (-torch.min(surragate_obj1, surragate_obj2)).mean()

            self.opt_actor.zero_grad()
            actor_loss.backward(retain_graph = True)
            self.opt_actor.step()

            critic_loss = self.mse(V, log_prob_list)

            self.opt_critic.zero_grad()
            critic_loss.backward()
            self.opt_critic.step()
        


    def estimate_value(self, obs, act):
        V = self.critic(obs).squeeze()

        mean = self.actor(obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(act)

        return V, log_probs


    def train_ppo_one_iteration(self):


        ep_steps = []
        total_env_steps = 0

        obs_list = []
        act_list = []
        rew_list = []
        log_prob_list = []
        passed_time = []
        trajec = 0
        while trajec < self.N_trajectories:
            episode_reward = []

            obs, _ = self.env.reset()
            done = False

            for episode in count():

                trajec += 1

                obs_list.append(obs)
                action, log_prob = self.select_action(obs)

                obs, rew, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                episode_reward.append(rew)
                act_list.append(action)
                log_prob_list.append(log_prob)
                
                if done:
                    # episode_durations.append(episode + 1)
                    break

            rew_list.append(episode_reward)
            passed_time.append(trajec+1)

            ep_reward = sum(episode_reward)
            ep_length = len(episode_reward)

            total_env_steps += ep_length
        
            self.ep_rewards.append(ep_reward)
            ep_steps.append(ep_length)

            # print(f"Episode {episode} | Reward: {episode_reward:.2f}")

        obs_list = torch.tensor(obs_list, dtype=torch.float)
        act_list = torch.tensor(act_list, dtype=torch.float)
        log_prob_list = torch.tensor(log_prob_list, dtype=torch.float)
        rtg_list = self.rewards_to_go(rew_list)      

        return obs_list, act_list, log_prob_list, rtg_list, passed_time

    def rewards_to_go(self, rew_list):

        rtg_list = []

        for rew_ep in reversed(rew_list):

            discounted_reward = 0

            for rew in reversed(rew_ep):

                discounted_reward = rew + discounted_reward*self.GAMMA
                rtg_list.insert(0, discounted_reward)

        rtg_list = torch.tensor(rtg_list, dtype=torch.float)

        return rtg_list

    def select_action(self, obs):

        mean = self.actor(obs)

        dist = MultivariateNormal(mean, self.cov_mat)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.detach().numpy(), log_prob.detach()
    






        
        
