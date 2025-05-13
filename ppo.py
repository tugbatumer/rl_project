from gymnasium.spaces import Discrete
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
import torch.optim as optim
from time import time
from torch.distributions import Categorical, MultivariateNormal


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
    def __init__(self, 
                 env, 
                 num_trajectories=3, 
                 num_updates=4, 
                 lr=0.005,
                 hidden_size=128, 
                 gamma=0.99, 
                 clip=0.2):

        self.N_updates = num_updates
        self.N_trajectories = num_trajectories
        self.gamma = gamma
        self.env = env 
        self.LR = lr
        self.ep_rewards = []

        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.is_discrete = isinstance(env.action_space, Discrete)
        if self.is_discrete:
            self.action_dim = env.action_space.n
        else:
            self.action_dim = env.action_space.shape[0]

        self.clip = clip
        self.actor = QNetwork(self.state_dim, self.action_dim, hidden_size, hidden_size)
        self.critic = QNetwork(self.state_dim, 1, hidden_size, hidden_size)

        self.opt_actor = optim.AdamW(self.actor.parameters(), lr=self.LR)
        self.opt_critic = optim.AdamW(self.critic.parameters(), lr=self.LR)

        self.cov_var = torch.full(size=(self.action_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)
        self.mse = nn.MSELoss()
        
    def learn(self, num_episodes=100000, max_training_time=float('inf')):
        start_time = time()
        self.ep_rewards = []
        for ep in tqdm(range(0, num_episodes, self.N_trajectories)):
            obs_list, act_list, log_prob_list, rtg_list = self.train_ppo_one_iteration()
            V, _ = self.estimate_value(obs_list, act_list)

            A_k = rtg_list - V.detach()
            self.update_networks(obs_list, act_list, log_prob_list, A_k)
            if time() - start_time > max_training_time:
                break
            
        return self.ep_rewards

    def update_networks(self, obs, act, log_prob_list, A_k):

        for _ in range(self.N_updates):

            V, log_probs = self.estimate_value(obs, act)
            
            ratio = torch.exp(log_probs - log_prob_list)
            surragate_obj1 = ratio * A_k
            surragate_obj2 = torch.clamp(ratio, 1-self.clip, 1+self.clip) * A_k

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
        if self.is_discrete:
            dist = Categorical(logits=mean)
        else:
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
        for _ in range(self.N_trajectories):
            episode_reward = []

            obs, _ = self.env.reset()
            done = False

            while True:
                obs_list.append(obs)
                action, log_prob = self.select_action(obs)

                obs, rew, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                episode_reward.append(rew)
                act_list.append(action)
                log_prob_list.append(log_prob)
                
                if done:
                    break

            rew_list.append(episode_reward)

            ep_reward = sum(episode_reward)
            ep_length = len(episode_reward)

            total_env_steps += ep_length
        
            self.ep_rewards.append(ep_reward)
            ep_steps.append(ep_length)

        obs_list = torch.tensor(obs_list, dtype=torch.float)
        act_list = torch.tensor(act_list, dtype=torch.float)
        log_prob_list = torch.tensor(log_prob_list, dtype=torch.float)
        rtg_list = self.rewards_to_go(rew_list)      

        return obs_list, act_list, log_prob_list, rtg_list

    def rewards_to_go(self, rew_list):

        rtg_list = []

        for rew_ep in reversed(rew_list):

            discounted_reward = 0

            for rew in reversed(rew_ep):

                discounted_reward = rew + discounted_reward*self.gamma
                rtg_list.insert(0, discounted_reward)

        rtg_list = torch.tensor(rtg_list, dtype=torch.float)

        return rtg_list

    def select_action(self, obs):
        mean = self.actor(obs)

        if self.is_discrete:
            dist = Categorical(logits=mean)
        else:
            dist = MultivariateNormal(mean, self.cov_mat)
            
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return (action.item(), log_prob.detach()) if self.is_discrete else \
               (action.detach().numpy(), log_prob.detach())
