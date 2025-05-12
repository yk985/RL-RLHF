import gym
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (16, 10)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import torch.optim as optim
from torch.distributions import Categorical
torch.manual_seed(0)

import base64, io

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Policy(nn.Module):
    def __init__(self, state_size=4, action_size=2, hidden_size=32):
        super().__init__()
        # shared layers
        self.fc1 = nn.Linear(state_size, hidden_size)
        # policy head
        self.fc2 = nn.Linear(hidden_size, action_size)
        # value head
        self.v  = nn.Linear(hidden_size, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        # action probabilities
        pi = F.softmax(self.fc2(x), dim=-1)
        # state value
        v  = self.v(x).squeeze(-1)
        return pi, v

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs, value = self.forward(state)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value
    
class Policy_v2(nn.Module): # 2 hidden layers
    def __init__(self, state_size=4, action_size=2, hidden_size1=64, hidden_size2=64):
        super().__init__()
        # shared layer
        self.fc1 = nn.Linear(state_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        # policy head
        self.fc3 = nn.Linear(hidden_size2, action_size)
        # value head
        self.v  = nn.Linear(hidden_size2, 1)
        # activation function
        self.act1 = nn.Tanh()
        self.act2 = F.softmax

    def forward(self, state):
        x = self.act1(self.fc1(state))
        x2 = self.act1(self.fc2(x))
        # action probabilities
        pi = self.act2(self.fc3(x2), dim=-1)
        # state value
        v  = self.v(x2).squeeze(-1)
        return pi, v

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs, value = self.forward(state)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value
    


class MLPActorCritic(nn.Module):    # Policy taken from Schulman et al. (2017)
    def __init__(self, state_dim, action_dim, hidden_size=64):
        super().__init__()
        # Actor: two hidden layers
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_dim),
        )
        # A free log‐std parameter (one per action dimension)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic: its own two hidden layers
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, state):
        """
        Given a batch of states (tensor [B×state_dim]), returns:
          actions [B×action_dim], logprobs [B], values [B], entropy [B]
        """
        μ = self.actor(state)                          # [B×action_dim]
        std = self.log_std.exp().expand_as(μ)          # [B×action_dim]
        dist = Normal(μ, std)
        a = dist.rsample()                             # reparameterized sample
        logp = dist.log_prob(a).sum(-1)                # [B]
        entropy = dist.entropy().sum(-1)               # [B]
        v = self.critic(state).squeeze(-1)             # [B]
        return a, logp, v, entropy

    def act(self, state):
        """
        Single‐step for rollout. Accepts a NumPy state [state_dim]
        and returns (action, logp, value). 
        """
        st = torch.from_numpy(state).float().unsqueeze(0).to(device)
        a, logp, v, _ = self.forward(st)
        return a.cpu().numpy()[0], logp.cpu().item(), v.cpu().item()

    def value(self, state):
        """Quick value estimate for a raw NumPy state."""
        st = torch.from_numpy(state).float().unsqueeze(0).to(device)
        return self.critic(st).item()

    
