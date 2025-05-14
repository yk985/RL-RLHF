import gym
import numpy as np
import matplotlib.pyplot as plt

def generate_trajectory(policy, env, max_steps=500, seed = 0): # seed s.t. pi1 and pi2 have same initial state
    """
    Roll out one episode with `policy` in `env` up to max_steps.
    Returns a list of dicts: [{"state": s, "action": a, "reward": r, "log_prob": lp, "value": v}, ...].
    """
    state = env.reset(seed=seed) # Reset the environment and set the seed
    trajectory = []
    for step in range(max_steps):
        # policy.act now returns action, log_prob, and value estimate
        action, log_prob, value = policy.act(state)
        next_state, reward, done, _ = env.step(action)
        trajectory.append({
            "state": state,
            "action": action,
            "reward": reward,
            "log_prob": log_prob,
            "value": value
        })
        state = next_state
        if done:
            # print(f"Episode finished after {step+1} timesteps; the trajectory leads to a lost of control")
            break
    return trajectory


