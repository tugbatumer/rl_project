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
        
        # 2 Layer MLP
        self.model = nn.Sequential(
            nn.Linear(state_size, hl1),
            nn.ReLU(),
            nn.Linear(hl1, hl2),
            nn.ReLU(),
            nn.Linear(hl2, action_size),
        )
    
    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float).to(device)
        x = x.to(device)

        x = self.model(x)

        return x


class PPOAGENT:
    def __init__(self, 
                 env, 
                 t_timestep=3,
                 num_updates=4,
                 lr=1e-3,
                 hidden_size=128, 
                 gamma=0.99, 
                 clip=0.2):
        '''
        Steps are based on the pseudocode can be found in:
        https://spinningup.openai.com/en/latest/algorithms/ppo.html#background
        '''



        # Step 1: Initialize hyperparameters
        self.N_updates = num_updates
        self.T_timestep = t_timestep
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

        # Step 1: Initialize action value function Q with random weights
        self.actor = QNetwork(self.state_dim, self.action_dim, hidden_size, hidden_size).to(device)
        self.critic = QNetwork(self.state_dim, 1, hidden_size, hidden_size).to(device)

        self.opt_actor = optim.Adam(self.actor.parameters(), lr=self.LR)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=self.LR)

        self.cov_var = torch.full(size=(self.action_dim,), fill_value=0.5).to(device)
        self.cov_mat = torch.diag(self.cov_var).to(device)
        self.mse = nn.MSELoss()
        
    def learn(self, num_episodes=100000, max_training_time=float('inf')):
        start_time = time()
        self.ep_rewards = []
        # Step 2: Loop starts
        for ep in tqdm(range(0, num_episodes)):
            
            # Step 3 and 4: Run old policy in environment for T timestep, and compute Rewards to go
            obs_list, act_list, log_prob_list, rtg_list = self.collect_samples_for_T_timestep()

            # Based on environment estimate the value
            V, _ = self.estimate_value(obs_list, act_list)

            # Step 5: Compute advantage estimates at iteration
            A_k = rtg_list - V.detach()

            # Step 6: Compute advantage estimates, and update the policy
            self.update_networks(obs_list, act_list, log_prob_list, A_k, rtg_list)
            if time() - start_time > max_training_time:
                break
            
        return self.ep_rewards

    def update_networks(self, obs, act, log_prob_list, A_k, rtg_list):
        # Update Networks
        for _ in range(self.N_updates):

            V, log_probs = self.estimate_value(obs, act)
            
            # Compute ratio of pi_theta_current to pi_theta_old
            ratio = torch.exp(log_probs - log_prob_list)

            # Calculate first term of L^CLIP(theta), equation 7 in original paper
            surragate_obj1 = ratio * A_k

            # Calculate second term of L^CLIP(theta), equation 7 in original paper
            surragate_obj2 = torch.clamp(ratio, 1-self.clip, 1+self.clip) * A_k

            # Step 6: We want to find the parameter maximizes the function min(), so we can calculate loss by multiplying with -1 and minimize wrt to it
            actor_loss = (-torch.min(surragate_obj1, surragate_obj2)).mean()

            # Update using stochastic gradient ascent methods (here using descent but we are descenting the negative of the loss)
            self.opt_actor.zero_grad()
            actor_loss.backward()
            self.opt_actor.step()

            # Step 7: Fit value function by regression
            critic_loss = self.mse(V, rtg_list)

            self.opt_critic.zero_grad()
            critic_loss.backward()
            self.opt_critic.step()
        


    def estimate_value(self, obs, act):

        # Calculate Values from models, 
        # if the environment is discrete use Categorical, 
        # if the environment is continous use MultivariateNormal
        V = self.critic(obs).squeeze()

        mean = self.actor(obs)
        if self.is_discrete:
            dist = Categorical(logits=mean)
        else:
            dist = MultivariateNormal(mean, self.cov_mat)
            
        log_probs = dist.log_prob(act)
        return V, log_probs


    def collect_samples_for_T_timestep(self):
        ep_steps = []
        total_env_steps = 0

        obs_list = []
        act_list = []
        rew_list = []
        log_prob_list = []
        for _ in range(self.T_timestep):
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

        obs_list = torch.tensor(obs_list, dtype=torch.float).to(device)
        if self.is_discrete:
            act_list = torch.tensor(act_list, dtype=torch.long).to(device)
        else:
            act_list = torch.tensor(act_list, dtype=torch.float).to(device)
        log_prob_list = torch.tensor(log_prob_list, dtype=torch.float).to(device)

        # Step 4: Compute rewards to go
        rtg_list = self.rewards_to_go(rew_list)      

        return obs_list, act_list, log_prob_list, rtg_list

    def rewards_to_go(self, rew_list):
        
        # Step 4: Compute Rewards to go
        rtg_list = []

        for rew_ep in reversed(rew_list):

            discounted_reward = 0

            for rew in reversed(rew_ep):

                discounted_reward = rew + discounted_reward*self.gamma
                rtg_list.insert(0, discounted_reward)

        rtg_list = torch.tensor(rtg_list, dtype=torch.float).to(device)


        return rtg_list

    def select_action(self, obs):
        # Select action from models, 
        # if the environment is discrete use Categorical, 
        # if the environment is continous use MultivariateNormal

        mean = self.actor(obs)

        if self.is_discrete:
            dist = Categorical(logits=mean)
        else:
            dist = MultivariateNormal(mean, self.cov_mat)
            
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return (action.item(), log_prob.detach()) if self.is_discrete else \
               (action.detach().cpu().numpy(), log_prob.detach())
