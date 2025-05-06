import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim




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

def train_reward_model(reward_model, dataset, optimizer, epochs=5):
    for epoch in range(epochs):
        total_loss = 0
        for state, a_pos, a_neg in dataset:
            r_pos = reward_model(state, torch.tensor([a_pos]))
            r_neg = reward_model(state, torch.tensor([a_neg]))
            preference_score = r_pos - r_neg
            loss = -F.logsigmoid(preference_score).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
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
