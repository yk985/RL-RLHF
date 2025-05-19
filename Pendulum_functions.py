# need to redefine the functions to work with VecEnv

import math
import numpy as np
import torch

from stable_baselines3.sac.policies import SACPolicy

if not hasattr(SACPolicy, "action_dist"):          # SB3 < 1.7
    @property
    def _action_dist(self):                        # expose actor.action_dist
        return self.actor.action_dist
    SACPolicy.action_dist = _action_dist



def evaluate_policy_SAC(policy, env, n_episodes: int, seed: int = None, device="cpu"):
    if seed is not None:
        env.seed(seed)

    returns = []
    obs = env.reset()               # shape: (1, obs_dim)

    for _ in range(n_episodes):
        done = [False]
        ep_ret = 0.0

        while not done[0]:
            # 1) Turn the batch of observations into a tensor
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
            # 2) Get the batch of actions (shape (1,1))
            action_tensor = policy(obs_tensor, deterministic=True)
            # 3) Detach & convert the entire batch back to NumPy
            action_batch = action_tensor.detach().cpu().numpy()
            #     --> shape is (1, 1), exactly what DummyVecEnv.step expects

            # 4) Step the VecEnv with the batch
            obs, reward, done, _ = env.step(action_batch)
            ep_ret += reward[0]

        returns.append(ep_ret)
        obs = env.reset()

    return float(np.mean(returns)), returns


def generate_sac_trajectory(model, env, max_steps=500, seed=None):
    if seed is not None:
        env.seed(seed)

    # 1) reset returns an array of shape (1, obs_dim)
    obs_batch = env.reset()  
    trajectory = []

    for _ in range(max_steps):
        # 2) get action batch of shape (1, action_dim)
        action, _ = model.predict(obs_batch, deterministic=True)
        # ensure it's (1,1) not (1,)
        if action.ndim == 1:
            action_batch = action.reshape(1, -1)
        else:
            action_batch = action

        # 3) step with the full batch
        next_obs_batch, reward_batch, done_batch, _ = env.step(action_batch)

        # 4) record the first (and only) env’s transition
        trajectory.append({
            "state":  obs_batch[0],
            "action": action_batch[0],
            "reward": reward_batch[0]
        })

        if done_batch[0]:
            break

        obs_batch = next_obs_batch

    return trajectory

def sample_preference_pairs(pi1, pi2, env, K=100):
    pairs = []
    for _ in range(K):
        t1 = generate_sac_trajectory(pi1, env)
        t2 = generate_sac_trajectory(pi2, env)
        R1, R2 = sum(s['reward'] for s in t1), sum(s['reward'] for s in t2)
        p1 = math.exp(R1) / (math.exp(R1) + math.exp(R2))
        if np.random.uniform()<p1:
            pairs.append({
                "traj_acc": t1, "traj_rej": t2,
                "R_acc": R1, "R_rej": R2, "pref_prob": p1
            })
        else:
            pairs.append({
                "traj_acc": t2, "traj_rej": t1,
                "R_acc": R2, "R_rej": R1, "pref_prob": pi2 # is it not p2 the preference prob?
            })
    return pairs

# def compute_logprob_trajectory_sac(policy, trajectory, device="cpu"):
#     """
#     π is continuous:  π(a|s) ~ N(mean(s), std(s))
#     We sum over t:  log π(τ) = ∑_t log π(a_t | s_t).
#     """
#     # 1) build state-batch
#     # states_np = np.stack([step["state"] for step in trajectory], axis=0).astype(np.float32)
#     states_np = np.stack(
#         [np.asarray(step["state"]).squeeze()  # ← remove extra (1, …) dims
#         for step in trajectory],
#         axis=0, dtype=np.float32
#         )  

#     states    = torch.from_numpy(states_np).to(device)   # shape [T, obs_dim]

#     # 2) build action-batch as floats
#     acts_np   = np.stack([step["action"] for step in trajectory], axis=0).astype(np.float32)
#     actions   = torch.from_numpy(acts_np).to(device)     # shape [T, action_dim]

#     # 3) get Gaussian params from the policy
#     #    (SB3 SACPolicy.get_action_dist_params returns: mean, log_std, kwargs)
#     mean, log_std, dist_kwargs = policy.actor.get_action_dist_params(states)
#     dist = policy.actor.action_dist.proba_distribution(mean, log_std, **dist_kwargs)


#     # 4) log_prob has shape [T, action_dim] for multi-dim actions
#     log_prob_per_dim = dist.log_prob(actions)
#     #    so sum across action dims → [T]
#     log_prob_per_step = log_prob_per_dim.sum(dim=-1)
#     # 5) sum over time → scalar
#     return log_prob_per_step.sum()
#     # scalar = log π(τ)

def compute_logprob_trajectory_sac(policy, trajectory, device="cpu"):
    """
    π is continuous:  π(a|s) ~ N(mean(s), std(s))
    We sum over t:  log π(τ) = ∑_t log π(a_t | s_t).
    """
    # 1) build state-batch
    # states_np = np.stack([step["state"] for step in trajectory], axis=0).astype(np.float32)
    states_np = np.stack(
        [np.asarray(step["state"]).squeeze()  # ← remove extra (1, …) dims
        for step in trajectory],
        axis=0, dtype=np.float32
        )  

    states    = torch.from_numpy(states_np).to(device)   # shape [T, obs_dim]

    # 2) build action-batch as floats
    acts_np   = np.stack([step["action"] for step in trajectory], axis=0).astype(np.float32)
    actions   = torch.from_numpy(acts_np).to(device)     # shape [T, action_dim]

    high = policy.action_space.high           # array([2.]) for Pendulum
    acts_squashed = np.clip(acts_np / high, -1 + 1e-6, 1 - 1e-6)  # scale to [-1,1]

    actions = torch.from_numpy(acts_squashed).to(device)           # [T, act_dim]

    mean, log_std, dist_kwargs = policy.get_action_dist_params(states)
    dist = policy.action_dist.proba_distribution(mean, log_std, **dist_kwargs)

    logp = dist.log_prob(actions).sum(-1)      # [T]  sum across action dims
    return logp.sum()                          # scalar = log π(τ)

