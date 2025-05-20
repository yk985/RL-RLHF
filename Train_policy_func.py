import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions import Categorical

from PPO import evaluate_policy 


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
