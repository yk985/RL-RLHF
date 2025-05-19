# need to redefine the functions to work with VecEnv
import warnings
from typing import Any, Callable, Optional, Union

import gymnasium as gym
import numpy as np

from stable_baselines3.common import type_aliases
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped

import math
import numpy as np
import torch

from stable_baselines3.sac.policies import SACPolicy

if not hasattr(SACPolicy, "action_dist"):          # SB3 < 1.7
    @property
    def _action_dist(self):                        # expose actor.action_dist
        return self.actor.action_dist
    SACPolicy.action_dist = _action_dist



def evaluate_policy(
    model: "type_aliases.PolicyPredictor",
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[dict[str, Any], dict[str, Any]], None]] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
    warn: bool = True,
) -> Union[tuple[float, float], tuple[list[float], list[int]]]:
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    If a vector env is passed in, this divides the episodes to evaluate onto the
    different elements of the vector env. This static division of work is done to
    remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
    details and discussion.

    .. note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.

    :param model: The RL agent you want to evaluate. This can be any object
        that implements a `predict` method, such as an RL algorithm (``BaseAlgorithm``)
        or policy (``BasePolicy``).
    :param env: The gym environment or ``VecEnv`` environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to do additional checks,
        called after each step. Gets locals() and globals() passed as parameters.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of rewards and episode lengths
        per episode will be returned instead of the mean.
    :param warn: If True (default), warns user about lack of a Monitor wrapper in the
        evaluation environment.
    :return: Mean reward per episode, std of reward per episode.
        Returns ([float], [int]) when ``return_episode_rewards`` is True, first
        list containing per-episode rewards and second containing per-episode lengths
        (in number of steps).
    """
    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]

    is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    observations = env.reset()
    states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    while (episode_counts < episode_count_targets).any():
        actions, states = model.predict(
            observations,  # type: ignore[arg-type]
            state=states,
            episode_start=episode_starts,
            deterministic=deterministic,
        )
        new_observations, rewards, dones, infos = env.step(actions)
        current_rewards += rewards
        current_lengths += 1
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:
                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                done = dones[i]
                info = infos[i]
                episode_starts[i] = done

                if callback is not None:
                    callback(locals(), globals())

                if dones[i]:
                    if is_monitor_wrapped:
                        # Atari wrapper can send a "done" signal when
                        # the agent loses a life, but it does not correspond
                        # to the true end of episode
                        if "episode" in info.keys():
                            # Do not trust "done" with episode endings.
                            # Monitor wrapper includes "episode" key in info if environment
                            # has been wrapped with it. Use those rewards instead.
                            episode_rewards.append(info["episode"]["r"])
                            episode_lengths.append(info["episode"]["l"])
                            # Only increment at the real end of an episode
                            episode_counts[i] += 1
                    else:
                        episode_rewards.append(current_rewards[i])
                        episode_lengths.append(current_lengths[i])
                        episode_counts[i] += 1
                    current_rewards[i] = 0
                    current_lengths[i] = 0

        observations = new_observations

        if render:
            env.render()

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return mean_reward, std_reward, episode_rewards


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

