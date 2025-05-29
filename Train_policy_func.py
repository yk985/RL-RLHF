import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions import Categorical



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
        probs = F.softmax(self.fc2(x), dim=-1)
        # state value
        value  = self.v(x).squeeze(-1)
        return probs, value

    def act(self, state):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        elif isinstance(state, torch.Tensor):
            state = state.float().unsqueeze(0).to(device)
        else:
            raise TypeError(f"Unsupported state type: {type(state)}")

        probs, value = self.forward(state)
        dist = Categorical(probs=probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob, value
    
def load_policy(file_path, state_dim, action_dim, device="cpu"):
    policy = Policy(state_size=state_dim, action_size=action_dim).to(device)
    policy.load_state_dict(torch.load(file_path, map_location=device))
    return policy


