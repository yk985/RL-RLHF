# At the top of your PPO.py (or a new file like OPPO.py), add:

import numpy as np
import torch
from collections import deque
from tqdm import tqdm

def baseline_1(state):
    """
    A simple hand-coded baseline: uses the pole angle theta to estimate value.
    """
    theta = state[2]   # cartpole’s pole angle
    return 25 * np.cos(theta)


def OPPO_update(policy,
                optimizer,
                env,
                baseline=baseline_1,
                n_episodes=1000,
                max_t=1000,
                gamma=1.0,
                print_every=100,
                early_stop=False):
    """
    A REINFORCE-with-baseline trainer (“OPPO”) for episodic tasks.

    Args:
      policy       – your Policy instance (with .act returning action, log_prob, _)
      optimizer    – torch optimizer for policy.parameters()
      env          – a Gym environment
      baseline     – fn(state)->float giving a baseline value
      n_episodes   – how many episodes to train
      max_t        – max timesteps per episode
      gamma        – discount factor
      print_every  – print stats every this many episodes
      early_stop   - if True, stop once you hit an average ≥195 over last 100 eps

    Returns:
      scores       – list of total episode rewards
    """
    scores_deque = deque(maxlen=100)
    scores = []

    for e in tqdm(range(1, n_episodes + 1)):
        saved_log_probs = []
        rewards         = []
        baseline_vals   = []

        state = env.reset()
        for t in range(max_t):
            # action, log_prob, _ = policy.act(state)
            action, log_prob, _ = policy.act(state)
            saved_log_probs.append(log_prob)

            state, reward, done, _ = env.step(action)
            rewards.append(reward)
            baseline_vals.append(baseline(state))

            if done:
                break

        # record episode score
        total_R = sum(rewards)
        scores_deque.append(total_R)
        scores.append(total_R)

        # compute rewards-to-go
        discounts = [gamma ** i for i in range(len(rewards))]
        # G_t = sum_{k=0 to T-t-1} gamma^k * r_{t+k}
        rewards_to_go = [
            sum(discounts[k] * rewards[k + t] for k in range(len(rewards) - t))
            for t in range(len(rewards))
        ]

        # build policy loss
        policy_loss_terms = []
        for log_prob, G, b in zip(saved_log_probs, rewards_to_go, baseline_vals):
            advantage = G - b
            policy_loss_terms.append(-log_prob * advantage)

        policy_loss = torch.stack(policy_loss_terms).sum()

        # gradient step
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        # logging
        if e % print_every == 0:
            avg_score = float(np.mean(scores_deque))
            print(f"Episode {e}\tAverage Score: {avg_score:.2f}")

        if early_stop and np.mean(scores_deque) >= 195.0:
            print(
                f"Environment solved in {e-100} episodes! "
                + f"Average Score: {np.mean(scores_deque):.2f}"
            )
            break

    return scores
