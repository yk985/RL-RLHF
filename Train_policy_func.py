import gym
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (16, 10)

import torch
import torch.nn as nn
import torch.nn.functional as F
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
    
