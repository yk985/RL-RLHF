import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

# === Policy ===
class Policy(nn.Module):
    def __init__(self, state_dim=4, action_dim=2, hidden_dim=64):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        probs = F.softmax(self.actor(state), dim=-1)
        value = self.critic(state).squeeze(-1)
        return probs, value

# === GAE ===
def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0
    values = values + [0]  # V(s_T)
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t+1] - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    returns = [a + v for a, v in zip(advantages, values[:-1])]
    return advantages, returns

# === PPO loss ===
def ppo_loss(new_log_probs, old_log_probs, advantages, clip_eps=0.2):
    ratio = torch.exp(new_log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    return -torch.min(surr1, surr2).mean()

# === Train one PPO update ===
def ppo_update(env, policy, optimizer, rollout_length=2048, clip_eps=0.2, gamma=0.99, lam=0.95,
               ppo_epochs=10, minibatch_size=64, device="cpu"):

    states, actions, rewards, values, log_probs = [], [], [], [], []
    state = env.reset()
    total_reward = 0
    steps = 0

    while steps < rollout_length:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            probs, value = policy(state_tensor)
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        next_state, reward, done, _ = env.step(action.item())
        states.append(state)
        actions.append(action.item())
        rewards.append(reward)
        values.append(value.item())
        log_probs.append(log_prob.item())
        total_reward += reward
        steps += 1
        state = next_state if not done else env.reset()

    # Add bootstrap value
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        _, final_value = policy(state_tensor)
    values.append(final_value.item())

    # GAE + returns
    advantages, returns = compute_gae(rewards, values, gamma, lam)
    advantages = torch.tensor(advantages, dtype=torch.float32).to(device)
    returns = torch.tensor(returns, dtype=torch.float32).to(device)
    states = torch.tensor(states, dtype=torch.float32).to(device)
    actions = torch.tensor(actions, dtype=torch.long).to(device)
    old_log_probs = torch.tensor(log_probs, dtype=torch.float32).to(device)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # PPO epochs
    for _ in range(ppo_epochs):
        indices = torch.randperm(len(states))
        for start in range(0, len(states), minibatch_size):
            end = start + minibatch_size
            mb_idx = indices[start:end]
            mb_states = states[mb_idx]
            mb_actions = actions[mb_idx]
            mb_old_log_probs = old_log_probs[mb_idx]
            mb_advantages = advantages[mb_idx]
            mb_returns = returns[mb_idx]

            probs, values = policy(mb_states)
            dist = Categorical(probs)
            new_log_probs = dist.log_prob(mb_actions)

            pg_loss = ppo_loss(new_log_probs, mb_old_log_probs, mb_advantages, clip_eps)
            v_loss = F.mse_loss(values, mb_returns)
            entropy = dist.entropy().mean()
            loss = pg_loss + 0.5 * v_loss - 0.01 * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return total_reward / (rollout_length / 500)  # approx reward per episode

# === Full training loop ===
def train_ppo(env_name="CartPole-v1", updates=1000, device="cpu"):
    env = gym.make(env_name)
    policy = Policy().to(device)
    optimizer = optim.Adam(policy.parameters(), lr=3e-4)
    reward_log = []

    for update in range(1, updates + 1):
        avg_reward = ppo_update(env, policy, optimizer, device=device)
        reward_log.append(avg_reward)
        if update % 10 == 0:
            print(f"[{update}] Avg Reward: {avg_reward:.2f}")
        if avg_reward >= 475:
            print("Solved!")
            break

    return reward_log, policy

# # Run it
# if __name__ == "__main__":
#     train_ppo()
