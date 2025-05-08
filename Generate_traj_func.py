import gym
import numpy as np
import matplotlib.pyplot as plt

def generate_trajectory(policy, env, max_steps=500): # Add seed s.t. pi1 and pi2 have same initial state
    """
    Roll out one episode with `policy` in `env` up to max_steps.
    Returns a list of dicts: [{"state": s, "action": a, "reward": r, "log_prob": lp, "value": v}, ...].
    """
    state = env.reset()
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


def plot_suboptimality(reward_hist_pi1, reward_hist_pi2, max_reward=None): # in cartpole, 200 is the max reward
    """
    Plot suboptimality = (max_reward - reward) versus update index
    for two policies π₁ and π₂.

    Args:
      reward_hist_pi1: list of average episode rewards per update for π₁
      reward_hist_pi2: same for π₂ (must be same length)
      max_reward:      optional scalar, the optimal/target return
                       (defaults to the max seen in either history)
    """
    # Infer optimal return if not given
    if max_reward is None:
        max_reward = max(max(reward_hist_pi1), max(reward_hist_pi2))

    updates1 = np.arange(1, len(reward_hist_pi1) + 1)
    updates2 = np.arange(1, len(reward_hist_pi2) + 1)
    sub1 = max_reward - np.array(reward_hist_pi1)
    sub2 = max_reward - np.array(reward_hist_pi2)

    plt.figure()
    plt.plot(updates1, sub1, label="π₁ suboptimality")
    plt.plot(updates2, sub2, label="π₂ suboptimality")
    plt.xlabel("Update #")
    plt.ylabel("Suboptimality\n(max_return − achieved)")
    plt.title("Policy Suboptimality Over Training")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_trajectory_performance(traj1, traj2):
    """
    Plot cumulative reward versus timestep for two trajectories.

    Args:
      traj1, traj2: lists of dicts with key 'reward' (as from generate_trajectory)
    """
    # Extract reward sequences
    rew1 = [step["reward"] for step in traj1]
    rew2 = [step["reward"] for step in traj2]

    # Cumulative sums
    cum1 = np.cumsum(rew1)
    cum2 = np.cumsum(rew2)
    steps1 = np.arange(len(cum1))
    steps2 = np.arange(len(cum2))

    plt.figure()
    plt.plot(steps1, cum1, label="Trajectory π₁")
    plt.plot(steps2, cum2, label="Trajectory π₂")
    plt.xlabel("Timestep")
    plt.ylabel("Cumulative Reward")
    plt.title("Single‐Episode Performance Comparison")
    plt.legend()
    plt.grid(True)
    plt.show()

