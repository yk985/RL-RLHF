import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torch.distributions import Categorical
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm
# ---- 1. Policy Network (assume small discrete envs like CartPole) ----
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Policy_v3(nn.Module):
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
        logits = self.fc2(x)
        # Clip logits to a safe range (e.g., between -10 and +10)
        logits = torch.clamp(logits, -10, 10)
        v = self.v(x).squeeze(-1)
        return logits, v

    # changed s.t. it can take numpy arrays or tensors
    def act(self, state):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        elif isinstance(state, torch.Tensor):
            state = state.float().unsqueeze(0).to(device)
        else:
            raise TypeError(f"Unsupported state type: {type(state)}")

        logits, value = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value
    

  
class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim)
        )

    def forward(self, x):
        return self.net(x)

    def act(self, state):
        logits = self.forward(torch.FloatTensor(state))
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

# ---- 2. Reward Model (trained to mimic preferences) ----
class RewardModel_fla(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        return self.net(state).squeeze(-1)

# ---- 3. Generate Trajectories ----
def generate_trajectories(policy, env_name, n_episodes=20, seed=32):
    env = gym.make(env_name)
    state = env.reset(seed=seed) # Reset the environment and set the seed
    trajectories = []
    for _ in range(n_episodes):
        state = env.reset()
        traj = []
        done = False
        while not done:
            action, log_prob, value = policy.act(state)
            action_int = action.item() if isinstance(action, torch.Tensor) else action
            next_state, reward, done, _ = env.step(action_int)
            traj.append((state, action, reward, log_prob, value))
            state = next_state
        trajectories.append(traj)
    return trajectories

# ---- 4. Generate Preference Pairs ----
def generate_trajectory_pairs(trajectories):
    pairs = []
    n = len(trajectories)
    for i in range(n):
        for j in range(i+1, n):
            traj_i = trajectories[i]
            traj_j = trajectories[j]
            len_i = len(traj_i)
            len_j = len(traj_j)
            # simulate preference: longer trajectory = better
            preference = 1 if len_i >= len_j else 0
            pairs.append((traj_i, traj_j, preference))
    return pairs

# ---- 5. Train Reward Model ----
def train_reward_model(pairs, reward_model, epochs=200, lr=1e-4):
    optimizer = optim.Adam(reward_model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        random.shuffle(pairs)
        for traj_a, traj_b, pref in pairs:
            # Sum predicted rewards over each trajectory
            states_a = torch.from_numpy(np.array([s for (s, _, _, _, _) in traj_a])).float()
            states_b = torch.from_numpy(np.array([s for (s, _, _, _, _) in traj_b])).float()
            r_a = reward_model(states_a).sum()
            r_b = reward_model(states_b).sum()
            logits = torch.stack([r_a, r_b])
            probs = torch.softmax(logits, dim=0)
            target = torch.tensor([1.0, 0.0]) if pref == 1 else torch.tensor([0.0, 1.0])
            loss = - (target * torch.log(probs)).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

# ---- 6. PPO with Learned Reward ----


def ppo_with_learned_reward(env_name, policy, reward_model, epochs=20, gamma=0.99, eps_clip=0.23, device="cpu"):
    env = gym.make(env_name)
    optimizer = optim.Adam(policy.parameters(), lr=6e-4)

    for _ in tqdm(range(epochs), desc="PPO Training", leave=False):
        states, actions, old_log_probs, rewards, dones = [], [], [], [], []

        state = env.reset()
        done = False

        while not done:
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)  # [1, state_dim]
            
            # Act using the current policy
            action, log_prob, _ = policy.act(state_tensor)
            action_int = action.item() if isinstance(action, torch.Tensor) else action
            next_state, _, done, _ = env.step(action_int)

            # === Use reward model instead of env reward ===
            reward = reward_model(state_tensor, torch.tensor([action_int]).to(device)).item()

            # Store everything
            states.append(state)
            actions.append(action_int)
            old_log_probs.append(log_prob)
            rewards.append(reward)
            dones.append(done)

            state = next_state

        # === Compute discounted returns ===
        returns = []
        G = 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            G = r + gamma * G * (1 - d)
            returns.insert(0, G)
        returns = torch.FloatTensor(returns).to(device)

        # === Convert to tensors ===
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(actions).to(device)
        old_log_probs = torch.stack(old_log_probs).to(device)

        # === PPO update ===
        for _ in range(100):  # PPO epochs
            logits, _ = policy(states)  # logits: [batch_size, action_dim]
            logits = torch.clamp(logits, -10, 10)  # Defensive clipping

            if torch.isnan(logits).any():
                print("NaNs found in logits!")
                return

            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions)

            ratio = torch.exp(new_log_probs - old_log_probs.detach())
            advantage = returns - returns.mean()

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantage

            loss = -torch.min(surr1, surr2).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
            optimizer.step()
            

# def ppo_with_learned_reward(env_name, policy, reward_model, epochs=20, gamma=0.99, eps_clip=0.23):
#     env = gym.make(env_name)
#     optimizer = optim.Adam(policy.parameters(), lr=2e-4)

#     for _ in range(epochs):
#         states, actions, old_log_probs, rewards, dones = [], [], [], [], []

#         state = env.reset()
#         done = False
#         while not done:
#             action, log_prob, _ = policy.act(state)
#             next_state, _, done, _ = env.step(action)
#             # Replace env reward with learned reward
#             r = reward_model(torch.FloatTensor(state)).item()
#             states.append(state)
#             actions.append(action)
#             old_log_probs.append(log_prob)
#             rewards.append(r)
#             dones.append(done)
#             state = next_state

#         # compute returns
#         returns = []
#         G = 0
#         for r, d in zip(reversed(rewards), reversed(dones)):
#             G = r + gamma * G * (1 - d)
#             returns.insert(0, G)
#         returns = torch.FloatTensor(returns)

#         # convert to tensors
#         states = torch.FloatTensor(states)
#         actions = torch.LongTensor(actions)
#         old_log_probs = torch.stack(old_log_probs)

#         # PPO update
#         for _ in range(5):  # multiple epochs
#             logits, values = policy(states)
#             logits = torch.clamp(logits, -10, 10)  # Clip logits here
#             # print("logits:", logits)
#             if torch.isnan(logits).any():
#                 print("NaNs found in logits!")
#                 return
#             dist = Categorical(logits=logits)
#             new_log_probs = dist.log_prob(actions)
#             ratio = torch.exp(new_log_probs - old_log_probs.detach())
#             advantage = returns - returns.mean()
#             surr1 = ratio * advantage
#             surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantage
#             loss = -torch.min(surr1, surr2).mean()

#             optimizer.zero_grad()
#             loss.backward()
#             # Exploding gradients can push your model weights into extreme values — leading to NaNs in logits on the next forward pass.
#             # This is a common issue in reinforcement learning, especially with PPO.
#             # Gradient clipping can help mitigate this issue.
#             # Gradient clipping is a technique used to prevent exploding gradients by capping the gradients during backpropagation.
#             # It helps to stabilize training and avoid NaN values in the model's parameters.
#             # It is especially useful in reinforcement learning, where the gradients can sometimes become very large.
#             torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)  # Gradient clipping
#             optimizer.step()







def ppo_update_fla(policy, optimizer, buffer,gamma, lam, c1 = 0.5, c2= 0.01, clip_eps=0.2, epochs=4, batch_size=64):
    # convert lists to tensors
    states = torch.stack(buffer.states).to(device)
    actions = torch.tensor(buffer.actions).to(device)
    old_lps = torch.stack(buffer.logprobs).to(device)

    # compute returns & advantages
    # you must pass the last value calculated after rollout
    last_state = buffer.states[-1]
    with torch.no_grad():
        _, last_value = policy(last_state.unsqueeze(0))

    if buffer.dones[-1]:
        last_value = torch.zeros_like(last_value)

    # last_state = buffer.states[-1]              # assuming you stored tensors
    returns, advs = buffer.compute_returns_and_advantages(last_value,gamma, lam)
    returns, advs = returns.to(device), advs.to(device)

    dataset = TensorDataset(states, actions, old_lps, returns, advs)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for _ in range(epochs):
        for S, A, LP_old, R, ADV in loader:
            logits, V = policy(S)
            dist = Categorical(logits=logits)
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

            loss = policy_loss + c1 * value_loss - c2 * entropy # see PPO paper Schulman et al., 2017

            optimizer.zero_grad()
            loss.backward()

            # Clip all gradients to norm ≤ 0.5
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.2)

            optimizer.step()

    buffer.clear()


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





def compare_policies(env_name, pi_ref, pi_rlhf, n_episodes=100, seed=42, success_threshold = 195):
    print(f"Evaluating policies over {n_episodes} episodes with seed {seed}...")
    
    ref_rewards = evaluate_policy_fla(env_name, pi_ref, n_episodes, seed)
    rlhf_rewards = evaluate_policy_fla(env_name, pi_rlhf, n_episodes, seed)

    avg_ref = np.mean(ref_rewards)
    avg_rlhf = np.mean(rlhf_rewards)

    # success_threshold = 195  # For CartPole-v0 (max = 200)

    ref_success = sum(r >= success_threshold for r in ref_rewards)
    rlhf_success = sum(r >= success_threshold for r in rlhf_rewards)

    print(f"\n📊 Results:")
    print(f"OPPO policy (pi_ref):   avg reward = {avg_ref:.2f}, success rate = {ref_success}/{n_episodes}")
    print(f"PPO-RLHF (pi_rlhf):     avg reward = {avg_rlhf:.2f}, success rate = {rlhf_success}/{n_episodes}")

    # Plotting
    import matplotlib.pyplot as plt
    plt.hist(ref_rewards, bins=20, alpha=0.6, label="OPPO (pi_ref)")
    plt.hist(rlhf_rewards, bins=20, alpha=0.6, label="PPO-RLHF (pi_rlhf)")
    plt.axvline(success_threshold, color='red', linestyle='--', label="Success Threshold")
    plt.xlabel("Total Reward")
    plt.ylabel("Episode Count")
    plt.title("Policy Reward Distribution")
    plt.legend()
    plt.show()

def evaluate_policy_fla(env_name, policy, n_episodes=50, seed=42):
    env = gym.make(env_name)
    set_seed(seed, env)
    total_rewards = []

    for _ in range(n_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _, _= policy.act(state)
            action_int = action.item() if isinstance(action, torch.Tensor) else action
            state, reward, done, _ = env.step(action_int)
            total_reward += reward
        total_rewards.append(total_reward)

    env.close()
    return total_rewards



def set_seed(seed, env=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if env is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
