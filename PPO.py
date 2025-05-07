import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam

import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Class to store data from rollouts batches; Uses GAE(Generalized Advantage Estimation) to compute advantages and returns.
class RolloutBuffer:
    def __init__(self):
        self.states, self.actions, self.logprobs = [], [], []
        self.rewards, self.values, self.dones   = [], [], []

    def store(self, s, a, lp, r, v, done):
        # convert env state (NumPy) into torch.Tensor 
        s = torch.from_numpy(s).float().to(device)

         # DETACH the old log‐prob and value 
        lp = lp.detach()
        v  = torch.tensor(v).float().to(device)

        self.states.append(s)
        self.actions.append(a)
        self.logprobs.append(lp)
        self.rewards.append(r)
        self.values.append(v)
        self.dones.append(done)

    def compute_returns_and_advantages(self, last_value, gamma=0.99, lam=0.95):
        returns, advs = [], []
        gae = 0
        for i in reversed(range(len(self.rewards))):
            mask  = 1.0 - float(self.dones[i])
            delta = self.rewards[i] + gamma * last_value * mask - self.values[i]
            gae   = delta + gamma * lam * mask * gae
            advs.insert(0, gae)
            last_value = self.values[i]
        for idx, v in enumerate(self.values):
            returns.append(advs[idx] + v)
        # normalize advantages
        advs = torch.tensor(advs, dtype=torch.float32)
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        returns = torch.tensor(returns, dtype=torch.float32)
        return returns, advs

    def clear(self):
        for lst in (self.states, self.actions, self.logprobs,
                    self.rewards, self.values, self.dones):
            lst.clear()

def ppo_update(policy, optimizer, buffer, clip_eps=0.2, epochs=4, batch_size=64):
    # convert lists to tensors
    states = torch.stack(buffer.states).to(device)
    actions = torch.tensor(buffer.actions).to(device)
    old_lps = torch.stack(buffer.logprobs).to(device)
    # compute returns & advantages
    # you must pass the last value calculated after rollout
    last_state = buffer.states[-1]              # assuming you stored tensors
    _, last_value = policy(last_state.unsqueeze(0))
    returns, advs = buffer.compute_returns_and_advantages(last_value)
    returns, advs = returns.to(device), advs.to(device)

    dataset = TensorDataset(states, actions, old_lps, returns, advs)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for _ in range(epochs):
        for S, A, LP_old, R, ADV in loader:
            pi, V = policy(S)
            dist = Categorical(pi)
            LP_new = dist.log_prob(A)
            ratio = (LP_new - LP_old).exp()

            # clipped policy loss; exactly slide 39 lecture 5
            surr1 = ratio * ADV
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * ADV
            policy_loss = -torch.min(surr1, surr2).mean()

            # value loss
            value_loss = F.mse_loss(V, R)

            # entropy bonus
            entropy = dist.entropy().mean()

            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy # see PPO paper Schulman et al., 2017

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    buffer.clear()
