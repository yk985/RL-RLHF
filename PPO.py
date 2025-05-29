import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data import TensorDataset, DataLoader

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
        v = v.detach()          # instead of torch.tensor(v)
        # v  = torch.tensor(v).float().to(device)


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


def evaluate_policy(policy, env, n_episodes=10, seed=2000): # different seed from training
    ''' 
    Evaluate the policy on the environment
    Args:
        policy: the policy to evaluate
        env: the environment to evaluate on
        n_episodes: number of episodes to evaluate
        seed: random seed for reproducibility
        Returns:
            mean return: average return over n_episodes
            returns: list of returns for each episode (complete episode)
    '''
    is_vec = hasattr(env, "num_envs") 
    returns = []
    state = env.reset(seed=seed) # reset the environment with a given seed
    for _ in range(n_episodes):
        state = env.reset()
        done = False
        ep_ret = 0.0

        while not done:
            a, _, _ = policy.act(state)        # shape (action_dim,) or scalar

            if is_vec:                         # -------- vector env branch -------
                if np.isscalar(a):
                    a_batch = np.array([[a]], dtype=np.float32)
                elif a.ndim == 1:
                    a_batch = a.reshape(1, -1)
                else:
                    a_batch = a                # already (n_envs, action_dim)

                state_b, r_b, done_b, _ = env.step(a_batch)
                state, r, done = state_b[0], r_b[0], done_b[0]

            else:  # -------- single env branch --------
                action_int = a.item() if isinstance(a, torch.Tensor) else a
                state, r, done, _ = env.step(action_int)

            ep_ret += r

        returns.append(ep_ret)
    return np.mean(returns), returns
