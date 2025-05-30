import numpy as np
from tqdm import tqdm
from time import time
import torch
from torch import nn
import random
import torch.optim as optim
from itertools import count
import math 

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
        
        # 2 Layer MLP instead of CNN since input is not image as in original paper
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
    def __init__(self, 
                 env, # Discrete Environment
                 lr=3e-4, # Learning Rate
                 hidden_size=128,  # Model size
                 gamma=0.99,  # Discount factor
                 tau=0.005, # Polyak averaging value 
                 clip_value = 1, # Gradient Clipping Value  
                 batch_size=128, # Number of sampled transition in every update
                 eps_start=0.9,  # eps greedy start value
                 eps_end=0.05, # eps greedy end value
                 eps_decay=1000, # eps greedy decay parameter
                 replay_size=int(1e6), # Size of the replay buffer
                 learning_start = 50, # Number of episodes to wait before updating the networks
                 train_freq = 4, # Frequency of update of the Q network
                 target_update_freq = 1000, # HARD update frequency as in original paper, if the argument is zero then Polyak averaging is used
                 is_first_update_target_hard = False
                 ): 
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize hyperparameters
        self.LR = lr
        self.steps_done = 0
        self.REPLAY_SIZE = replay_size
        self.EPS_DECAY = eps_decay
        self.EPS_END = eps_end
        self.EPS_START = eps_start
        self.BATCH_SIZE = batch_size
        self.TAU = tau
        self.GAMMA = gamma
        self.env = env 
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.learning_start = learning_start
        self.train_freq = train_freq
        self.target_update_freq = target_update_freq
        self.CLIP_VALUE = clip_value
        self.is_first_update_target_hard = is_first_update_target_hard
        # Initialize replay memory D to capacity N
        self.replay_buffer = ReplayBuffer(self.REPLAY_SIZE)

        # Initialize action-value function Q with random weights
        self.Q_network = QNetwork(self.state_dim, self.action_dim, hidden_size, hidden_size).to(self.device)
        self.Q_target = QNetwork(self.state_dim, self.action_dim, hidden_size, hidden_size).to(self.device)
        self.Q_target.load_state_dict(self.Q_network.state_dict())

        self.optimizer = optim.Adam(self.Q_network.parameters(), lr=self.LR)

    def learn(self, num_episodes=100000, max_training_time=float('inf')):
        self.env_steps   = 0
        self.learn_steps = 0
        start = time()
        ep_rewards = []

        # for episode = 1, M do
        for ep in tqdm(range(num_episodes)):

            # Initialize environment
            state, _ = self.env.reset()
            total_r = 0


            while True:

                # Select action based on eps-greedy policy
                action = self.select_action(state)

                # Execute action in environment and observe reward and next state
                next_state, reward, term, trunc, _ = self.env.step(action)
                done = term or trunc
    
                # Store transition in replay memory
                self.replay_buffer.add(state, action, reward, next_state, done)

                # Set state as next state
                state = next_state
                total_r += reward
                '''
                To have more samples from the environment, 
                wait to fill the replay memory a bit more than BATCH_SIZE 
                and sample from the environment, we added a learning_start parameter.

                And also tried to update the network not in every step but in some frequency.
                '''

                self.env_steps += 1
                if self.env_steps > self.learning_start and self.env_steps % self.train_freq == 0:
                    self.train_dqn()            
                    self.learn_steps += 1
    
                    # HARD target update as in original paper 
                    if self.target_update_freq==0:
                        self.soft_update_target()
                    elif self.learn_steps % self.target_update_freq == 0:
                        self.hard_update_target()
    
                if done or (time() - start) > max_training_time:
                    ep_rewards.append(total_r)
                    break
    
            if (time() - start) > max_training_time:
                break
    
        return ep_rewards

    

    def select_action(self, state):

        # Decrease epsilon to have control over exploration and exploitation
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(- self.steps_done / self.EPS_DECAY)
        self.steps_done += 1

        # Check if state is Torch Tensor
        if not torch.is_tensor(state):
            state = torch.tensor(state, dtype=torch.float32)

        state = state.to(self.device).unsqueeze(0)

        # With probability epsilon select a random action
        # Otherwise select action from action value function 
        if random.random() < eps_threshold:
            random_action = np.random.randint(self.action_dim)
            return random_action
        

        # Action value function to evaluation mode to freezing the gradient computation
        self.Q_network.eval()
        with torch.no_grad():
            action = self.Q_network(state)
            max_action = action.max(1).indices.view(1, 1).squeeze(0)   

        self.Q_network.train()     

        return max_action.item()

    def hard_update_target(self):
        # HARD target network update as in original paper
        self.Q_target.load_state_dict(self.Q_network.state_dict())
        
    def soft_update_target(self):
        # Polyak averaging for target network update
        if self.is_first_update_target_hard:
            self.Q_target.load_state_dict(self.Q_network.state_dict())
            self.is_first_update_target_hard = False
            return
        target_net_state_dict = self.Q_target.state_dict()
        policy_net_state_dict = self.Q_network.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.TAU + target_net_state_dict[key]*(1-self.TAU)
        self.Q_target.load_state_dict(target_net_state_dict)

    def train_dqn(self):
        
        # Wait until replay memory is filled up to BATCH_SIZE to train the networks
        if len(self.replay_buffer.buffer) < self.BATCH_SIZE:
            return
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.BATCH_SIZE)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        Q_values = self.Q_network(states)
        Q_values = Q_values.gather(1, actions).squeeze(1)


        with torch.no_grad():
            Q_targets_next = self.Q_target(next_states).detach().max(1)[0].unsqueeze(1)

        
        # Compute y_j
        expected_q_values = Q_targets_next*self.GAMMA*(1 - dones) + rewards

        # Calculate Loss
        loss_fn = nn.MSELoss()
        loss = loss_fn(Q_values, expected_q_values.squeeze(1))

        # Update Q function
        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_value_(self.Q_network.parameters(), self.CLIP_VALUE)
        self.optimizer.step()
        