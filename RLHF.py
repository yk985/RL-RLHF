import torch
import torch.nn as nn
import torch.nn.functional as F
from pairs_generator import compute_reward_from_traj, extract_states_actions
from Generate_traj_func import generate_trajectory

from torch.distributions import Categorical
from torch.utils.data import TensorDataset, DataLoader
from PPO import RolloutBuffer                         # already in your repo



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


def train_policy_from_rollouts_n_updates_v2( # lr should be1e-3
        policy, ref_policy, reward_model, env, optimizer,
        N               = 100,     # how many PPO updates
        K               = 10,     # how many roll-outs per update
        max_steps       = 200,    # horizon per roll-out
        beta            = 0.1,    # KL penalty coefficient
        device          = "cpu",
        gamma           = 0.9,   # discount factor
        lam             = 0.95,   # GAE(λ)
        clip_eps        = 0.2,    # PPO clip
        ppo_epochs      = 10,      # gradient epochs per update
        batch_size      = 64,
        c1              = 0.5,    # value-loss coefficient
        c2              = 0.0):  # entropy-bonus coefficient
    """
    PPO with an RLHF reward:
        r_t = r_φ(s_t,a_t) − β ( log π_θ(a_t|s_t) − log π_ref(a_t|s_t) )

    After collecting K trajectories (θ is frozen while collecting),
    we optimise the clipped PPO surrogate for `ppo_epochs` epochs.
    """
    policy.to(device)
    ref_policy.to(device)
    reward_model.to(device)
    ref_policy.eval()            # must stay frozen

    for update_idx in range(1, N + 1):
        buffer = RolloutBuffer()                     # stores one big batch
        steps_collected = 0

        # 1) ───── Roll-out K episodes with π_θ  ─────────────────────────
        for _ in range(K):
            state, _done = env.reset(), False
            while not _done and steps_collected < max_steps:
                state_t   = torch.as_tensor(state, dtype=torch.float32,
                                            device=device)
                with torch.no_grad():               # θ_old
                    probs, value = policy(state_t.unsqueeze(0))
                dist        = Categorical(probs)
                action_t    = dist.sample()                 # tensor(…)
                logp_old    = dist.log_prob(action_t)       # detach later
                a_int       = action_t.item()

                # ref-policy KL term (log-ratio, not full KL across π)
                with torch.no_grad():
                    probs_ref, _ = ref_policy(state_t.unsqueeze(0))
                logp_ref = Categorical(probs_ref).log_prob(action_t)

                # RLHF reward for this timestep
                with torch.no_grad():
                    r_model = reward_model(state_t, action_t.unsqueeze(0))
                shaped_r   = (r_model
                              - beta * (logp_old - logp_ref)).item()

                # step the env
                next_state, _env_r, _done, _ = env.step(a_int)

                buffer.store(state, a_int, logp_old, shaped_r, value, _done)
                state = next_state
                steps_collected += 1

                if _done:
                    break

        # 2) ───── GAE returns & advantages ──────────────────────────────
        last_state = torch.as_tensor(state, dtype=torch.float32, device=device)
        with torch.no_grad():
            _, last_val = policy(last_state.unsqueeze(0))
        if _done:                                           # trajectory ended
            last_val = torch.zeros_like(last_val)
        returns, advs = buffer.compute_returns_and_advantages(
                            last_val, gamma=gamma, lam=lam)

        # put everything on the correct device
        states      = torch.stack(buffer.states).to(device)
        actions     = torch.tensor(buffer.actions, device=device)
        logp_old    = torch.stack(buffer.logprobs).to(device)
        returns     = returns.to(device)
        advs        = advs.to(device)

        # 3) ───── PPO surrogate optimisation ────────────────────────────
        dataset = TensorDataset(states, actions, logp_old, returns, advs)
        loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for _ in range(ppo_epochs):
            for S, A, LP_old, R, ADV in loader:
                probs, V = policy(S)
                dist     = Categorical(probs)
                LP_new   = dist.log_prob(A)

                ratio    = torch.exp(LP_new - LP_old)
                surr1    = ratio * ADV
                surr2    = torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * ADV
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss  = F.mse_loss(V, R)
                entropy_bns = dist.entropy().mean()

                loss = policy_loss + c1 * value_loss - c2 * entropy_bns

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optimizer.step()

        print(f"[{update_idx}/{N}] "
              f"batch steps={len(buffer.rewards):4d}  "
              f"loss={loss.item():.4f}  "
              f"policy-loss={policy_loss.item():.4f}  "
              f"value-loss={value_loss.item():.4f}")

        buffer.clear()           # ready for the next iteration
