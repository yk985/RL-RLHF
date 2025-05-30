import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data import TensorDataset, DataLoader

from pairs_generator import compute_reward_from_traj
from PPO import RolloutBuffer  ,evaluate_policy             
from tqdm.auto import tqdm

class RewardModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        if state.dim() == 1:
            state = state.unsqueeze(0)  # (1, d)
        if action.dim() == 0:
            action = action.unsqueeze(0)  # (1,)
        elif action.dim() == 2 and action.size(1) == 1:
            action = action.squeeze(1)  # (batch_size,)

        action_onehot = F.one_hot(action, num_classes=2).float()  # (1, 2)
        
        if action_onehot.dim() == 3:
            action_onehot = action_onehot.squeeze(1)  # remove spurious middle dim

        x = torch.cat([state, action_onehot], dim=-1)  # (1, d + 2)
        return self.fc(x).squeeze(-1)


class RewardModel_Acro(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state, action):
        if state.dim() == 1:
            state = state.unsqueeze(0)  # (1, d)
        if action.dim() == 0:
            action = action.unsqueeze(0)  # (1,)
        elif action.dim() == 2 and action.size(1) == 1:
            action = action.squeeze(1)  # (batch_size,)

        action_onehot = F.one_hot(action, num_classes=3).float()  # (1, 2)
        
        if action_onehot.dim() == 3:
            action_onehot = action_onehot.squeeze(1)  # remove spurious middle dim

        x = torch.cat([state, action_onehot], dim=-1)  # (1, d + 2)
        return self.fc(x).squeeze(-1)
    


def train_reward_model(reward_model, dataset, optimizer, epochs=1000):
    for epoch in tqdm(range(epochs), desc="Training Reward Model", leave=False, colour="#14C2C7"):
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
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Reward model loss = {total_loss:.4f}")


def compute_kl_divergence(p_probs, q_probs):
    return torch.sum(p_probs * (torch.log(p_probs + 1e-8) - torch.log(q_probs + 1e-8)), dim=-1)


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


def train_policy_from_rollouts_n_updates_v2( # lr should be1e-3
        policy, ref_policy, reward_model, env, optimizer,
        N               = 100,     # how many PPO updates
        K               = 10,     # how many roll-outs per update. Here we use only max_steps
        max_steps       = 2000,    # horizon per roll-out
        beta            = 0.1,    # KL penalty coefficient
        device          = "cpu",
        gamma           = 0.9,   # discount factor
        lam             = 0.95,   # GAE(λ)
        clip_eps        = 0.2,    # PPO clip
        ppo_epochs      = 4,      # gradient epochs per update
        batch_size      = 64,
        c1              = 0.5,    # value-loss coefficient
        c2              = 0.0):  # entropy-bonus coefficient
    """
    PPO with an RLHF reward:
        r_t = r_φ(s_t,a_t) − β ( log π_θ(a_t|s_t) − log π_ref(a_t|s_t) )

    After collecting K trajectories (θ is frozen while collecting),
    we optimise the clipped PPO surrogate for `ppo_epochs` epochs.
    """
    traj_reward_hist=[]
    policy.to(device)
    ref_policy.to(device)
    reward_model.to(device)
    ref_policy.eval()            # must stay frozen
    loss_hist = []

    for update_idx in range(1, N + 1):
        buffer = RolloutBuffer()                     # stores one big batch
        steps_collected = 0

        # 1) ───── Roll-out K episodes with π_θ  ─────────────────────────
        # for _ in range(K):
        state, done = env.reset(), False
        while not done and steps_collected < max_steps:
        # for _ in range(max_steps):                     # fixed total steps
            state_t = torch.as_tensor(state, dtype=torch.float32, device=device)

            with torch.no_grad():
                probs, value = policy(state_t.unsqueeze(0))
            dist       = Categorical(probs=probs)
            action     = dist.sample()
            logp_old   = dist.log_prob(action).detach()   #  ← detach
            a_int      = action.item()

            # log π_ref(a|s)
            with torch.no_grad():
                probs_ref, _ = ref_policy(state_t.unsqueeze(0))
            logp_ref = Categorical(probs=probs_ref).log_prob(action)

            # # RLHF reward (still raw for now)
            with torch.no_grad():
                r_model = reward_model(state_t, action.unsqueeze(0))
                r_model = r_model.item()  # convert to scalar

            # step the env
            next_state, _, done, _ = env.step(a_int)

            buffer.store(state, a_int, logp_old, r_model, value.detach(), done)
            state = next_state

            if done:          
                state, done = env.reset(), False # start a new episode
            
            steps_collected += 1

        # use the real 'done' flag from the buffer’s last element
        last_done = buffer.dones[-1]
        last_state = torch.as_tensor(state, dtype=torch.float32, device=device)
        with torch.no_grad():
            _, last_val = policy(last_state.unsqueeze(0))
        if last_done:
            last_val = torch.zeros_like(last_val)
        returns, advs = buffer.compute_returns_and_advantages(last_val, gamma, lam)
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)  # normalise
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # normalise

        # put everything on the correct device
        states      = torch.stack(buffer.states).to(device)
        # rewards       = torch.stack(buffer.rewards).to(device)    
        actions     = torch.tensor(buffer.actions, device=device)
        logp_old    = torch.stack(buffer.logprobs).to(device)
        returns     = returns.to(device)
        advs        = advs.to(device)

        # 3) ───── PPO surrogate optimisation ────────────────────────────
        dataset = TensorDataset(states, actions, logp_old, returns, advs)
        loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for _ in range(ppo_epochs):
            for S, A, LP_old, R, ADV in loader:
            # to get KL in the loss:
                pi, _     = policy(S)           # S: [T,obs_dim]
                with torch.no_grad():
                    pi_ref, _ = ref_policy(S)
                dist     = torch.distributions.Categorical(probs=pi)
                dist_ref = torch.distributions.Categorical(probs=pi_ref)
                kl_per_step = torch.distributions.kl.kl_divergence(dist_ref, dist)
                kl2 = kl_per_step.mean()

                probs, V = policy(S)
                v_clip = V + (R - V).clamp(-clip_eps, clip_eps)  # PPO-2 style

                dist     = Categorical(probs=probs)
                LP_new   = dist.log_prob(A)

                ratio    = torch.exp(LP_new - LP_old)
                surr1    = ratio * ADV
                surr2    = torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * ADV
                policy_loss = -torch.min(surr1, surr2).mean()

                # value_loss  = F.mse_loss(V, R)
                value_loss =  torch.max(
                 (V - R) ** 2,
                 (v_clip.detach() - R) ** 2).mean()
                entropy_bns = dist.entropy().mean()

                loss = policy_loss + c1 * value_loss - c2 * entropy_bns + beta * kl2

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optimizer.step()
        loss_hist.append(loss.item())

        if update_idx % 10 == 0:
            # KL divergence between the new and old policy
            kl_div = compute_policy_kl(policy, ref_policy, states, actions)
            target_kl = 0.1
            # if kl_div > target_kl * 1.5:
            #     beta *= 1.5
            # elif kl_div < target_kl / 1.5:
            #     beta /= 1.5
            print(f"[{update_idx}/{N}] "
                #   f"KL-divergence: {kl_div.item():.2f}  "
                  f"KL-divergence: {kl2.item():e}  "
                  f"policy-loss: {policy_loss.item():e}  "
                  f"loss={loss.item():e} "
                  f"value-loss: {value_loss.item():e}  "
                  f"entropy: {entropy_bns.item():e}")
            mean_reward,_=evaluate_policy(policy,env,n_episodes=5)
            traj_reward_hist.append(mean_reward)
        # print(f"[{update_idx}/{N}] "
        #       f"batch steps={len(buffer.rewards):4d}  "
        #       f"loss={loss.item():.4f}  "
        #       f"policy-loss={policy_loss.item():.4f}  "
        #       f"value-loss={value_loss.item():.4f}")

        buffer.clear()           # ready for the next iteration
    return loss_hist, traj_reward_hist

