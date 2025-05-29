import numpy as np
import torch


# Updated for compatibility with SB3
def generate_trajectory(policy, env, max_steps=None, seed = 0): 
    # seed s.t. pi1 and pi2 have same initial state
    """
    Roll out one episode with `policy` in `env` up to max_steps.
    Returns a list of dicts: [{"state": s, "action": a, "reward": r, "log_prob": lp, "value": v}, ...].
    """
    state = env.reset(seed=seed) # Reset the environment and set the seed
    trajectory = []
    done = False
    len_trajectory = 0
    while not done: 
        action, log_prob, value = policy.act(state)

        # ── wrap the action for DummyVecEnv ──────────────────────────
        if hasattr(env, "num_envs"):                 # vector env branch
            if np.isscalar(action):
                a_batch = np.array([[action]], dtype=np.float32)  # (1,1)
            elif isinstance(action, np.ndarray) and action.ndim == 1:
                a_batch = action.reshape(1, -1)                   # (1, act_dim)
            else:
                a_batch = action                                  # already batched

            next_obs_b, r_b, done_b, _ = env.step(a_batch)
            next_state, reward, done = next_obs_b[0], r_b[0], done_b[0]
        else:          
            action_int = action.item() if isinstance(action, torch.Tensor) else action
            # single-env branch
            next_state, reward, done, _ = env.step(action_int)
        # ────────────────────────────────────────────────────────────

        trajectory.append(
            dict(state=state, action=action, reward=reward,
                log_prob=log_prob, value=value)
        )
        state = next_state
        len_trajectory += 1
        
        # Check if the maximum number of steps has been reached
        # in the case were we want trajectory of fixed length
        if max_steps is not None and len_trajectory >= max_steps:
            #print(f"Episode finished after {max_steps} timesteps; the trajectory is truncated")
            break
    return trajectory
