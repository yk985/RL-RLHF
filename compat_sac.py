# ─────────────────────────  compat_sac.py  ─────────────────────────
"""
Drop-in compat layer so that a Stable-Baselines3 SAC policy looks like
the discrete MlpPolicy your old helpers expect, and so that
DummyVecEnv.reset(seed=…) no longer crashes.
Import this ONCE (early) in every notebook/script before you call any
of your helpers (pairs_generator, DPO, …):

    import compat_sac          # nothing else to change

That’s it – no edits required in the rest of your code-base.
"""
import numpy as np
import torch
import gym

class TwoTensorReset(gym.Wrapper):
    """
    Ensure env.reset() → (obs, info)  (length-2 tuple)
    Ensure env.step()  → (obs, reward, terminated, truncated, info)
    so SB3’s DummyVecEnv sees the exact shape it expects.
    """
    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        if isinstance(out, tuple):
            # already (obs, info)
            return out
        return out, {}               # <-- add empty info dict

    def step(self, action):
        out = self.env.step(action)
        if len(out) == 4:
            # old gym 4-tuple → add truncated=False
            obs, reward, done, info = out
            return obs, reward, done, False, info
        return out                   # already 5-tuple


# ── 1. Make DummyVecEnv.reset ignore the unknown `seed=` kwarg ──────────
from stable_baselines3.common.vec_env import DummyVecEnv
_orig_reset = DummyVecEnv.reset                      # keep a handle

def _reset_ignore_seed(self, *args, **kw):
    kw.pop("seed", None)                             # discard if present
    return _orig_reset(self, *args, **kw)

DummyVecEnv.reset = _reset_ignore_seed               # monkey-patch


# ── 2. Give SACPolicy the interface your old code wants ─────────────────
from stable_baselines3.sac.policies import SACPolicy

@torch.no_grad()
def _sac_act(self: SACPolicy, state, deterministic=True):
    """
    Old helpers expect:  a, log_prob, value = policy.act(state)
    * state : 1-D np array  (obs_dim,)
    * returns
        action    : 1-D np array (action_dim,)
        log_prob  : float
        value     : float   (Q-value via qf1, good enough for advantage calc)
    """
    obs_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
    action_t = self(obs_t, deterministic=deterministic)          # (1, act_dim)
    mean, log_std, kw = self.actor.get_action_dist_params(obs_t) # actor = μ,σ network
    dist = self.actor.action_dist.proba_distribution(mean, log_std, **kw)
    logp_t = dist.log_prob(action_t).sum(-1)                      # sum over dims
        # ---- inside _sac_act ------------------------------------------
    # ... we already have obs_t, action_t, mean, log_std, dist, logp_t
    # Replace the old line:
    #     value_t = self.qf1(obs_t, action_t).mean(1)

    # New, version-agnostic critic call
    try:                         # SB3 ≤ 1.6 had qf1
        value_t = self.qf1(obs_t, action_t).mean(1)
    except AttributeError:       # SB3 ≥ 1.7 groups critics in self.critic
        q1, q2 = self.critic(obs_t, action_t)   # returns two tensors
        value_t = torch.min(q1, q2).squeeze(1)  # shape (batch,)

    return (action_t.cpu().numpy()[0],
            logp_t.item(),
            value_t.item())

    # value_t = self.qf1(obs_t, action_t).mean(1)                  # critic 1
    # return action_t.cpu().numpy()[0], logp_t.item(), value_t.item()

def _get_action_dist_params(self: SACPolicy, obs):
    """Expose actor.get_action_dist_params through the policy itself."""
    return self.actor.get_action_dist_params(obs)

# attach the two methods
SACPolicy.act                  = _sac_act
SACPolicy.get_action_dist_params = _get_action_dist_params
# ─────────────────────────────────────────────────────────────────────────
