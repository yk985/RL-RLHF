import numpy as np
import torch
import gym
import time
from collections import deque
from tqdm.auto import tqdm

def baseline_1(state):
    """
    A simple hand-coded baseline: uses the pole angle theta to estimate value.
    """
    theta = state[2]   # cartpole’s pole angle
    return 25 * np.cos(theta)

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

def baseline_CartPole_V0(state, w_angle=0.8, w_ang_vel=0.8):
    """
    A simple hand-coded baseline: uses the pole angle theta to estimate value.
    """
    theta = state[2]   # cartpole’s pole angle
    return 25 * np.cos(theta) + 0.5 * state[3]  # add velocity term




def baseline_MountainCar(state):
    """
    A simple hand-coded baseline: uses the car's position to estimate value.
    """
    position = state[0]  # car's position
    velocity = state[1]  # car's velocity
    if np.abs(position) >= 0.2:
        return -10*( position * velocity) 
    else:
        return 100 * np.abs(position) + 10000 * velocity**2



def select_action(policy, obs):
    obs_tensor = torch.tensor(obs, dtype=torch.float32)
    probs = policy(obs_tensor)
    dist = torch.distributions.Categorical(probs)
    action = dist.sample()
    return action.item(), dist.log_prob(action)


def OPPO_update(policy,
                optimizer,
                env,
                baseline=baseline_1,
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
    env.reset(seed=seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # checkpoint for saving pi2
    checkpoint_reached = False
    
    # stores the last 50 scores
    scores_deque = deque(maxlen=print_every)
    
    # stores the scores of all episodes
    scores = []

    for e in tqdm(range(1, n_episodes + 1)):
        saved_log_probs = []
        rewards         = []
        baseline_vals   = []

        state = env.reset()
        for step_in_episode in range(max_t):
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
            print(f"Saving the policy in pi2_oppo_{env_name}.pth...")
            torch.save(policy.state_dict(), f"pi2_oppo_{env_name}.pth")
            checkpoint_reached = True
        
        # Stopping criteria with target score
        if target_score is not None and np.mean(scores_deque) >= target_score:
            print(
                f"Environment reached the target score (cumulative rewards) in {e} episodes! "
                + f"Average Score: {np.mean(scores_deque):.2f}"
            )
            print(f"Saving the policy in pi1_oppo_{env_name}.pth...")
            torch.save(policy.state_dict(), f"pi1_oppo_{env_name}.pth")
            break
        
        # Early stopping criteria
        # but still problem with the dependence on the environment
        # if early_stop and np.mean(scores_deque) >= 195.0:
        #     print(
        #         f"Environment solved in {e-100} episodes! "
        #         + f"Average Score: {np.mean(scores_deque):.2f}"
        #     )
        #     break

    return scores
