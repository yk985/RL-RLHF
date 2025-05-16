import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pairs_generator import *
from Generate_traj_func import generate_trajectory


#A voir si necessaire
class RewardModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        # One-hot encode action
        action_onehot = F.one_hot(action, num_classes=2).float()
        x = torch.cat([state, action_onehot], dim=-1)
        return self.fc(x).squeeze(-1)

def train_reward_model(reward_model, dataset, optimizer, epochs=50):
  for epoch in range(epochs):
    total_loss = 0
    for pair in dataset:
      r_pos = compute_reward_from_traj(reward_model,pair[0])
      r_neg = compute_reward_from_traj(reward_model,pair[1])
      preference_score = r_pos - r_neg
      loss = -F.logsigmoid(preference_score).mean()
      
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      total_loss += loss.item()
    
    if epoch % 10 == 0:
      print(f"Epoch {epoch}: Reward model loss = {total_loss:.4f}")


def compute_kl_divergence(p_probs, q_probs):
    return torch.sum(p_probs * (torch.log(p_probs + 1e-8) - torch.log(q_probs + 1e-8)), dim=-1)

def train_policy(policy, ref_policy, reward_model, prompts, optimizer, beta=0.1):
    for state in prompts:
        pi, _ = policy(state)
        with torch.no_grad():
            pi_ref, _ = ref_policy(state)
        
        dist = torch.distributions.Categorical(pi)
        action = dist.sample()

        reward = reward_model(state, action.unsqueeze(0))
        kl = compute_kl_divergence(pi, pi_ref)

        rl_loss = -reward + beta * kl.mean()

        optimizer.zero_grad()
        rl_loss.backward()
        optimizer.step()

def compute_gae(trajectory, gamma=0.99, lam=0.95):
    rewards = [step["reward"] for step in trajectory]
    values  = [step["value"].item() for step in trajectory]
    values.append(0)  # bootstrap value after terminal

    gae = 0
    returns = []
    advantages = []
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t+1] - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
        returns.insert(0, gae + values[t])
    return returns, advantages

def compute_policy_kl(policy_new, policy_old, states, actions):
    """
    Computes average KL divergence between two policies over a trajectory.

    Args:
        policy_new: current (updated) policy
        policy_old: reference (frozen) policy
        states: tensor of shape [T, state_dim]
        actions: tensor of shape [T]

    Returns:
        mean_kl: scalar tensor, average KL divergence over the states
    """
    with torch.no_grad():
        pi_new, _ = policy_new(states)  # shape: [T, action_dim]
        pi_old, _ = policy_old(states)

    dist_new = torch.distributions.Categorical(probs=pi_new)
    dist_old = torch.distributions.Categorical(probs=pi_old)

    kl_div = torch.distributions.kl.kl_divergence(dist_new, dist_old)  # shape: [T]

    return kl_div.mean()


def train_policy_v2(policy, ref_policy, reward_model, env, optimizer, total_updates=100,
                    beta=0.1, ppo_epochs=4, clip_eps=0.2, gamma=0.99, lam=0.95,
                    num_trajectories=10, device="cpu"):
    
    for update_iter in range(total_updates):
        # === Regenerate trajectories using the CURRENT policy ===
        
        all_DKL=[]
        all_rewards=[]

        for _ in range(num_trajectories):
            traj = generate_trajectory(policy, env, device=device)
            states, actions = extract_states_actions(traj, device)
            all_DKL.append(compute_kl_divergence(policy,ref_policy,states,actions))
            # Use reward model to assign rewards
            with torch.no_grad():
                rewards = reward_model(states, actions.unsqueeze(-1)).squeeze(-1)
            all_rewards.append(rewards)

            
        # === Prepare tensors for PPO optimization ===
        
        DKL = torch.tensor(all_DKL, dtype=torch.float32).to(device)
        all_rewards = torch.tensor(all_rewards, dtype=torch.float32).to(device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # === Optimize policy using PPO loss (multiple epochs on same batch) ===
        for _ in range(ppo_epochs):
            

            total_loss = -all_rewards.mean()+beta*DKL.mean()

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        print(f"[Update {update_iter}] total Loss: {total_loss.item():.4f}")

