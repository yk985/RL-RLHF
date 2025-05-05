import gym

def generate_trajectory(policy, env, max_steps=500):
    """
    Roll out one episode with `policy` in `env` up to max_steps.
    Returns a list of dicts: [{"state": s, "action": a, "reward": r, "log_prob": lp, "value": v}, ...].
    """
    state = env.reset()
    trajectory = []
    for _ in range(max_steps):
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
            break
    return trajectory
