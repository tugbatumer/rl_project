from tqdm import tqdm
import torch.nn.functional as F
from time import time
from utils import *
import copy

# Actor Network: maps states to actions
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size, action_space=None):
        super().__init__()
        self.mlp = mlp(state_dim, hidden_size)
        self.max_action = action_space.high[0]
        self.linear = nn.Linear(hidden_size, action_dim)

    def forward(self, state):
        a = self.mlp(state)
        return self.max_action * self.linear(torch.tanh(a)) # Output in [-max_action, max_action]

# TD3 Agent
class TD3Agent:
    def __init__(self,
                 env,
                 lr=3e-4,
                 hidden_size=128,
                 gamma=0.99, # Discount factor
                 tau=0.005, # Soft update rate
                 policy_noise=0.2, # Std of noise for target policy smoothing
                 noise_clip=0.5, # Clipping range for policy noise
                 policy_freq=2, # Frequency of policy (actor) update
                 expl_noise=0.1, # Std of exploration noise added to actions
                 batch_size=128,
                 start_steps=0, # Steps initial random policy is used
                 replay_size=int(1e6)):

        self.env = env
        self.lr = lr
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.batch_size = batch_size
        self.replay_size = replay_size
        self.expl_noise = expl_noise
        self.start_steps = start_steps
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.max_action = float(env.action_space.high[0])

        self.actor = Actor(self.state_dim, self.action_dim, hidden_size, env.action_space).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)

        self.critic = QNetwork(self.state_dim, self.action_dim, hidden_size).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

        self.replay_buffer = ReplayBuffer(self.replay_size)

        self.iteration = 0

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(state)
        return action.cpu().numpy().flatten()

    def train(self):
        self.iteration += 1

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
            # Generate target actions with added clipped noise (policy smoothing)
            noise = (torch.randn_like(actions) * float(self.policy_noise)).clamp(-float(self.noise_clip), float(self.noise_clip))
            next_actions = (self.actor_target(next_states) + noise).clamp(-self.max_action, self.max_action)

            # Compute target Q-values using target critic
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target = rewards + (1 - dones) * self.gamma * target_q

        curr_q1, curr_q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(curr_q1, target) + F.mse_loss(curr_q2, target)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # Delayed actor update
        if self.iteration % self.policy_freq == 0:
            actor_loss = -self.critic(states, self.actor(states))[0].mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            # Soft update of target networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
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
                    action = (
                        self.select_action(np.array(state))
                        + np.random.normal(0, self.max_action * self.expl_noise, size=self.action_dim)
                ).clip(-self.max_action, self.max_action)

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