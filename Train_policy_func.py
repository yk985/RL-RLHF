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


from PPO import evaluate_policy 
import os
def evaluate_all_policies(env, seeds, env_name, num_episodes=10, device="cpu"):
    results = {}

    # === PI1 and PI2 (vary by seed) ===
    for label in ["pi1", "pi2"]:
        returns = []
        rewards_list=np.zeros(num_episodes)
        for seed in seeds:
            file_path = f"{label}_oppo_{env_name}_seed_{seed}.pth"
            if not os.path.exists(file_path):
                print(f"Missing file: {file_path}")
                continue

            policy = load_policy(file_path, env.observation_space.shape[0], env.action_space.n, device)
            mean_return, rewards_per_ep = evaluate_policy(policy, env, n_episodes=num_episodes)
            returns.append(mean_return)
            rewards_list+=rewards_per_ep
        results[label] = {
            "mean": np.mean(returns),
            "std": np.std(returns),
            "per_seed": returns,
            "graph":rewards_list/len(seeds)
        }


    # === PI_DPO variants (filename includes other params) ===
    pi_DPO_files = [
        
        f"pi_RLHF_{env_name}_seed_{seed}_beta0.5_K200.pth"
    ]

    for file_template in pi_DPO_files:
        label = os.path.splitext(file_template)[0].replace("pi_RLHF_oppo_", "").replace("_seed_{seed}", "")
        returns = []
        rewards_list=np.zeros(num_episodes)
        for seed in seeds:
            file_path = file_template#.format(seed=seed)
            if not os.path.exists(file_path):
                print(f"Missing file: {file_path}")
                continue

            policy = load_policy(file_path, env.observation_space.shape[0], env.action_space.n, device)
            mean_return, rewards_per_ep = evaluate_policy(policy, env, n_episodes=num_episodes)
            returns.append(mean_return)
            rewards_list+=rewards_per_ep
        results[f"pi_RLHF_{label}"] = {
            "mean": np.mean(returns),
            "std": np.std(returns),
            "per_seed": returns,
            "graph":rewards_list/len(seeds)
        }

    return results


# class Policy_v2(nn.Module): # 2 hidden layers
#     def __init__(self, state_size=4, action_size=2, hidden_size1=64, hidden_size2=64):
#         super().__init__()
#         # shared layer
#         self.fc1 = nn.Linear(state_size, hidden_size1)
#         self.fc2 = nn.Linear(hidden_size1, hidden_size2)
#         # policy head
#         self.fc3 = nn.Linear(hidden_size2, action_size)
#         # value head
#         self.v  = nn.Linear(hidden_size2, 1)
#         # activation function
#         self.act1 = nn.Tanh()
#         self.act2 = F.softmax

#     def forward(self, state):
#         x = self.act1(self.fc1(state))
#         x2 = self.act1(self.fc2(x))
#         # action probabilities
#         pi = self.act2(self.fc3(x2), dim=-1)
#         # state value
#         v  = self.v(x2).squeeze(-1)
#         return pi, v

#     def act(self, state):
#         state = torch.from_numpy(state).float().unsqueeze(0).to(device)
#         probs, value = self.forward(state)
#         dist = Categorical(probs=probs)
#         action = dist.sample()
#         return action.item(), dist.log_prob(action), value
    
# class Policy_v3(nn.Module):
#     def __init__(self, state_size=4, action_size=2, hidden_size=32):
#         super().__init__()
#         # shared layers
#         self.fc1 = nn.Linear(state_size, hidden_size)
#         # policy head
#         self.fc2 = nn.Linear(hidden_size, action_size)
#         # value head
#         self.v  = nn.Linear(hidden_size, 1)

#     def forward(self, state):
#         x = F.relu(self.fc1(state))
#         logits = self.fc2(x)
#         # Clip logits to a safe range (e.g., between -10 and +10)
#         logits = torch.clamp(logits, -10, 10)
#         v = self.v(x).squeeze(-1)
#         return logits, v

#     def act(self, state):
#         state = torch.from_numpy(state).float().unsqueeze(0).to(device)
#         logits, value = self.forward(state)
#         dist = Categorical(logits=logits)
#         action = dist.sample()
#         return action.item(), dist.log_prob(action), value
    
class PolicyContinuous(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64):
        super().__init__()
        # shared trunk
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # actor: outputs mean of each action dim
        self.mu_head = nn.Linear(hidden_size, action_dim)
        # we’ll learn a free log‐std for each action dim
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        # critic: shares trunk or own layers
        self.v_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        mu  = self.mu_head(x)
        std = self.log_std.exp().expand_as(mu)
        v   = self.v_head(x).squeeze(-1)
        return mu, std, v

    def act(self, state, clip_low=None, clip_high=None):
        st = torch.from_numpy(state).float().unsqueeze(0).to(device)
        mu, std, v = self.forward(st)
        dist = Normal(mu, std)

        a    = dist.rsample()             # [1×action_dim], requires_grad=True
        logp = dist.log_prob(a).sum(-1)   # Tensor
        a_np = a.detach().cpu().numpy()[0]  # detach first!

        # optional: clip to action bounds
        if clip_low is not None:
            a_np = np.clip(a_np, clip_low, clip_high)

        return a_np, logp, v.item()




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

    
