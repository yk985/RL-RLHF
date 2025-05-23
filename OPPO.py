import numpy as np
import torch
import gym
from collections import deque
import random

# for the display of the environment
import time



def baseline_CartPole_v0_Fla(state, w_angle=0.8, w_ang_vel=0.8): 
    """
    Baseline_Fla
    A simple hand-coded baseline: uses the pole angle theta and angular velocity to estimate value.
    """
    # w_angle = 0.8
    # w_ang_vel = 0.8
    # Linear function of the state variables (e.g. angle and angular velocity)
    # We want to avoid big angles and big angular velocities
    # --> so we want to predict a higher value if the angle is small and angular velocity is low
    # w_angle = 0.8
    # w_ang_vel = 0.8
    # Linear function of the state variables (e.g. angle and angular velocity)
    # We want to avoid big angles and big angular velocities
    # --> so we want to predict a higher value if the angle is small and angular velocity is low
    _, _, angle, ang_velocity = state
    value = w_angle * (0.2 - angle**2) - w_ang_vel * ang_velocity**2
    return value


def baseline_Acrobot_v0(state, w_height=1.0, w_vel=0.05):
    """
    A simple hand-coded baseline for Acrobot-v1:
      - Uses the end-effector height (normalized) as a proxy for how close we are to swinging up.
      - Penalizes large angular velocities to encourage smoother swings.
    
    Args:
      state       - 6-vector: [cos(θ1), sin(θ1), cos(θ2), sin(θ2), θ̇1, θ̇2]
      w_height    - weight on normalized height term (default 1.0)
      w_vel       - weight on velocity penalty (default 0.05)
    Returns:
      scalar value estimate
    """
    c1, s1, c2, s2, d1, d2 = state

    # cos(θ1 + θ2) = cos θ1 cos θ2 − sin θ1 sin θ2
    c12 = c1*c2 - s1*s2

    # Tip height relative to pivot:  h = −(cos θ1 + cos(θ1+θ2))
    # ranges ∈ [−2, +2], where +2 is “both links straight up”
    height = -(c1 + c12)

    # normalize to [−1, +1]
    h_norm = height / 2.0

    # value: higher when tip is up & velocities are small
    value = w_height * h_norm - w_vel * (d1**2 + d2**2)
    return float(value)

def select_action(policy, obs):
    obs_tensor = torch.tensor(obs, dtype=torch.float32)
    probs = policy(obs_tensor)
    dist = torch.distributions.Categorical(probs)
    action = dist.sample()
    return action.item(), dist.log_prob(action)


def OPPO_update(policy,
                optimizer,
                env,
                baseline=None,
                n_episodes=1000,
                max_t=1000,
                gamma=1.0,
                print_every=100,
                early_stop=False,
                seed=42,
                target_score=None,
                env_name="CartPole-v0",
                display_every=False):
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
      seed         – random seed
      targer_score – if not None, stop once you hit this score,
      display_every – if True, display the environment every print_every episodes
    Returns:
      scores       – list of total episode rewards
    """
    # Set random seed
    # not sur about this
    set_seed(seed, env)
    # checkpoint for saving pi2
    checkpoint_reached = False
    
    # stores the last 50 scores
    scores_deque = deque(maxlen=print_every)
    
    # stores the scores of all episodes
    scores = []

    for e in range(1, n_episodes + 1): #, desc="Training", unit="episode", leave=True):
        saved_log_probs = []
        rewards         = []
        baseline_vals   = []

        state = env.reset()
        for step_in_episode in range(max_t):
            action, log_prob, _ = policy.act(state)
            saved_log_probs.append(log_prob)

            state, reward, done, _ = env.step(action)
            rewards.append(reward)
            if baseline is not None:
                baseline_vals.append(baseline(state))      # does it goes after or before the action?
            elif baseline is None:
                baseline_vals.append(0)                    # if no baseline, use 0
            
            if done:
                break

        # record episode score
        total_R = sum(rewards)
        scores_deque.append(total_R)
        scores.append(total_R)

        # compute rewards-to-go
        discounts = [gamma ** i for i in range(len(rewards))]

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
            print(f"Episode {e} \t Average Score over the last {print_every:.0f} episodes: {avg_score:.1f}")
            
            if display_every:
                # Render what the agent does every X episodes
                render_env = gym.make(env_name, render_mode='human')
                obs = render_env.reset()
                done = False
                while not done:
                    render_env.render()
                    action, _, _ = policy.act(state)

                    # action, _ = select_action(policy, obs)
                    obs, _, done, _ = render_env.step(action)
                    # time.sleep(0.03)  # inside the while loop to slow down frames
                render_env.close()

        # Criteria To save current policy as Pi2 with half the target_score
        if not checkpoint_reached and target_score is not None and np.mean(scores_deque) >= target_score/2:
            print(
                f"Environment reached the half target score in {e} episodes! "
                + f"Average Score: {np.mean(scores_deque):.2f}"
            )
            print(f"Saving the policy in ./Policies/pi2_ref_{env_name}_seed_{seed}.pth")
            torch.save(policy.state_dict(), f"./Policies/pi2_ref_{env_name}_seed_{seed}.pth")
            checkpoint_reached = True
        
        # Stopping criteria with target score
        if target_score is not None and np.mean(scores_deque) >= target_score:
            print(
                f"Environment reached the target score (cumulative rewards) in {e} episodes! "
                + f"Average Score: {np.mean(scores_deque):.2f}"
            )
            print(f"Saving the policy in ./Policies/pi1_ref_{env_name}_seed_{seed}.pth...")
            torch.save(policy.state_dict(), f"./Policies/pi1_ref_{env_name}_seed_{seed}.pth")
            break
        
    return scores


def set_seed(seed, env=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if env is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)


def OPPO_update_Acrobot(policy,
                optimizer,
                env,
                baseline=None,
                n_episodes=1000,
                max_t=1000,
                gamma=1.0,
                print_every=100,
                early_stop=False,
                seed=42,
                target_score=None,
                env_name="CartPole-v0",
                display_every=False):
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
      seed         – random seed
      targer_score – if not None, stop once you hit this score,
      display_every – if True, display the environment every print_every episodes
    Returns:
      scores       – list of total episode rewards
    """
    # Set random seed
    # not sur about this
    set_seed(seed, env)
    # checkpoint for saving pi2
    checkpoint_reached = False
    
    # stores the last 50 scores
    scores_deque = deque(maxlen=print_every)
    
    # stores the scores of all episodes
    scores = []

    for e in range(1, n_episodes + 1): #, desc="Training", unit="episode", leave=True):
        saved_log_probs = []
        rewards         = []
        baseline_vals   = []

        state = env.reset()
        for step_in_episode in range(max_t):
            action, log_prob, _ = policy.act(state)
            saved_log_probs.append(log_prob)

            state, reward, done, _ = env.step(action)
            rewards.append(reward)
            if baseline is not None:
                baseline_vals.append(baseline(state))      # does it goes after or before the action?
            elif baseline is None:
                baseline_vals.append(0)                    # if no baseline, use 0
            
            if done:
                break

        # record episode score
        total_R = sum(rewards)
        scores_deque.append(total_R)
        scores.append(total_R)

        # compute rewards-to-go
        discounts = [gamma ** i for i in range(len(rewards))]

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
            print(f"Episode {e} \t Average Score over the last {print_every:.0f} episodes: {avg_score:.1f}")
            
            if display_every:
                # Render what the agent does every X episodes
                render_env = gym.make(env_name, render_mode='human')
                obs = render_env.reset()
                done = False
                while not done:
                    render_env.render()
                    action, _, _ = policy.act(state)

                    # action, _ = select_action(policy, obs)
                    obs, _, done, _ = render_env.step(action)
                    # time.sleep(0.03)  # inside the while loop to slow down frames
                render_env.close()

        # Criteria To save current policy as Pi2 with half the target_score
        if not checkpoint_reached and target_score is not None and np.mean(scores_deque) >= -250:
            print(
                f"Environment reached the half target score in {e} episodes! "
                + f"Average Score: {np.mean(scores_deque):.2f}"
            )
            print(f"Saving the policy in ./Policies/pi2_ref_{env_name}_seed_{seed}.pth")
            torch.save(policy.state_dict(), f"./Policies/pi2_ref_{env_name}_seed_{seed}.pth")
            checkpoint_reached = True
        
        # Stopping criteria with target score
        if target_score is not None and np.mean(scores_deque) >= target_score:
            print(
                f"Environment reached the target score (cumulative rewards) in {e} episodes! "
                + f"Average Score: {np.mean(scores_deque):.2f}"
            )
            print(f"Saving the policy in ./Policies/pi1_ref_{env_name}_seed_{seed}.pth...")
            torch.save(policy.state_dict(), f"./Policies/pi1_ref_{env_name}_seed_{seed}.pth")
            break
        
    return scores

    
    
    
    