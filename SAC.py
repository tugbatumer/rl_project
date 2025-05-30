from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.distributions import Normal
from time import time
from utils import *

# GaussianPolicy: Actor network for SAC that outputs mean and log_std
class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size, action_space=None):
        super().__init__()
        self.mlp = mlp(state_dim, hidden_size)
        self.mu = nn.Linear(hidden_size, action_dim)  # Mean of Gaussian
        self.log_std = nn.Linear(hidden_size, action_dim)  # Log std deviation

        # Rescaling actions to fit the environment's action space
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        mu = self.mu(self.mlp(state))
        log_std = self.log_std(self.mlp(state))
        log_std = torch.clamp(log_std, -20, 2)  # Prevents too high/low std
        std = log_std.exp()

        # Sample from the Gaussian (reparameterization trick)
        normal = Normal(mu, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)   # Squash output to [-1, 1]
        action = y_t * self.action_scale + self.action_bias # Rescale

        # Log-probability of the action (adjusted for tanh transformation)
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob

    def sample(self, state):
        return self.forward(state)

# SAC Agent
class SACAgent:
    def __init__(self,
                 env,
                 lr=3e-4,
                 hidden_size=128,
                 gamma=0.99,  # Discount factor
                 tau=0.005,   # Soft target update factor
                 alpha=0.2,   # Entropy regularization coefficient
                 batch_size=128,
                 start_steps=0,  # Steps initial random policy is used
                 replay_size=int(1e6)):
        self.env = env
        self.lr = lr
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        self.replay_size = replay_size
        self.start_steps = start_steps
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.max_action = float(env.action_space.high[0])

        self.actor = GaussianPolicy(self.state_dim, self.action_dim, hidden_size, env.action_space).to(self.device)
        self.critic = QNetwork(self.state_dim, self.action_dim, hidden_size).to(self.device)
        self.critic_target = QNetwork(self.state_dim, self.action_dim, hidden_size).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

        self.replay_buffer = ReplayBuffer(self.replay_size)

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, _ = self.actor.sample(state)
        return action.cpu().numpy()[0]

    def train(self):

        # Wait until enough samples are collected
        if len(self.replay_buffer.buffer) < max(self.batch_size, self.start_steps):
            return

        # Sample a batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        with torch.no_grad():
            # Next actions and entropy term
            next_action, next_log_prob = self.actor.sample(next_states)
            target_q1, target_q2 = self.critic_target(next_states, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob

            # Bellman target
            target = rewards + (1 - dones) * self.gamma * target_q

        curr_q1, curr_q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(curr_q1, target) + F.mse_loss(curr_q2, target)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        new_actions, log_probs = self.actor.sample(states)
        q1, q2 = self.critic(states, new_actions)
        q = torch.min(q1, q2)

        actor_loss = (self.alpha * log_probs - q).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # Target network soft update
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def learn(self, num_episodes=100000, max_training_time=float('inf')):
        start_time = time()
        episode_rewards = []
        total_steps = 0
        for _ in tqdm(range(num_episodes)):
            state = self.env.reset()
            if isinstance(state, tuple):  # Gym >= 0.26 returns (obs, info)
                state = state[0]
            episode_reward = 0
            done = False
            t = 0

            while not done:
                t += 1
                total_steps += 1
                # Select action randomly during initial exploration phase
                if total_steps < self.start_steps:
                    action = self.env.action_space.sample()
                # Otherwise, select action using the policy
                else:
                    action = self.select_action(state)
                # Take action in the environment
                next_state, reward, done, terminate, _ = self.env.step(action)
                done = done or terminate
                done_bool = float(done) if t < self.env._max_episode_steps else 0
                self.replay_buffer.add(state, action, reward, next_state, done_bool)
                state = next_state
                self.train()
                episode_reward += reward
                
            episode_rewards.append(episode_reward)
            # Stop if training time exceeds limit
            if time() - start_time > max_training_time:
                break

        return episode_rewards

