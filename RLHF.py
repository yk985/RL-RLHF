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

def train_reward_model(reward_model, dataset, optimizer, epochs=1000):
    for epoch in range(epochs):
        total_loss = 0
        for pair in dataset:
            r_pos = compute_reward_from_traj(reward_model,pair["traj_acc"])
            r_neg = compute_reward_from_traj(reward_model,pair["traj_rej"])
            preference_score = r_pos - r_neg
            loss = -F.logsigmoid(preference_score).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}: Reward model loss = {total_loss:.4f}")


def compute_kl_divergence(p_probs, q_probs):
    return torch.sum(p_probs * (torch.log(p_probs + 1e-8) - torch.log(q_probs + 1e-8)), dim=-1)

def train_policy_from_rollouts_n_updates(policy, ref_policy, reward_model, env, optimizer,
                                         N=10, K=10, max_steps=500, beta=0.1, device="cpu"):
    """
    Train the policy using RLHF loss with KL regularization over N update steps.
    Each update step is based on K newly generated trajectories.

    Args:
        policy: trainable policy
        ref_policy: frozen reference policy
        reward_model: learned reward model r_phi(s, a)
        env: environment
        optimizer: optimizer for policy parameters
        N: number of update steps
        K: number of rollouts per update step
        max_steps: max steps per rollout
        beta: KL penalty coefficient
        device: torch device
    """
    for update_step in range(N):
        total_loss = 0.0

        for k in range(K):
            traj = generate_trajectory(policy, env, max_steps=max_steps)
            states, actions = extract_states_actions(traj,device=device)

            pi, _ = policy(states)
            with torch.no_grad():
                pi_ref, _ = ref_policy(states)

            dist = torch.distributions.Categorical(pi)
            reward = compute_reward_from_traj(reward_model,traj)
            kl = torch.distributions.kl.kl_divergence(
                torch.distributions.Categorical(pi_ref),
                torch.distributions.Categorical(pi)
            )

            loss = (-reward + beta * kl).mean()
            total_loss += loss

        avg_loss = total_loss / K

        optimizer.zero_grad()
        avg_loss.backward()
        optimizer.step()

        print(f"[Update {update_step+1}/{N}] Avg Loss: {avg_loss.item():.4f}")

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

