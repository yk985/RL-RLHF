def generate_trajectory(policy, env, max_steps=500):
    """
    Roll out one episode with `policy` in `env` up to max_steps.
    Returns a list of dicts: [{"state": s, "action": a, "reward": r, "log_prob": lp}, ...].
    """
    state = env.reset()
    trajectory = []
    for t in range(max_steps):
        action, log_prob = policy.act(state)
        next_state, reward, done, _ = env.step(action)
        trajectory.append({
            "state":     state,
            "action":    action,
            "reward":    reward,
            "log_prob":  log_prob
        })
        state = next_state
        if done:
            break
    return trajectory
