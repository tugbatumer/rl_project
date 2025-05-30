# Comparing Four Deep RL Algorithms: DQN, PPO, SAC, TD3

This repository contains implementations of several popular deep reinforcement learning (DRL) algorithms from scratch using PyTorch, including:

* **DQN (Deep Q-Network)**
* **PPO (Proximal Policy Optimization)**
* **SAC (Soft Actor-Critic)**
* **TD3 (Twin Delayed DDPG)**

It supports both **discrete** and **continuous** control environments from OpenAI Gym.

## Repository Structure

```
├── DQN.py                          # Deep Q-Network (Discrete)
├── PPO.py                          # Proximal Policy Optimization (Discrete & Continuous)
├── SAC.py                          # Soft Actor-Critic (Continuous)
├── TD3.py                          # Twin Delayed DDPG (Continuous)
├── discrete_experiments.ipynb      # Visualization notebook for discrete action spaces (PPO & DQN)
├── continuous_experiments.ipynb    # Visualization notebook for continuous action spaces (PPO & DQN)
├── utils.py                        # Shared utilities (ReplayBuffer, MLP, QNetwork)
```
---

## Dependencies

* Python ≥ 3.7
* PyTorch ≥ 1.10
* NumPy
* Matplotlib
* OpenAI Gym

Install with:

```bash
pip install torch numpy matplotlib gym tqdm
```

---

## Authors

Tuğba Tümer, Hasan S. Ünal, Frédéric Khayat, Yağız Gençer
