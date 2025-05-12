import numpy as np
import torch
import gym
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

# Import your PPO implementation
from PPO import RolloutBuffer, ppo_update, evaluate_policy, device  # citeturn3file0
from Train_policy_func import Policy  # citeturn2file3

# Fix seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Create the environment
ENV_ID = "CartPole-v1"

def make_env():
    env = gym.make(ENV_ID)
    return env

# Define the search space for PPO hyperparameters
dim_space = [
    Real(1e-5, 1e-2, "log-uniform", name="learning_rate"),
    Real(0.9, 0.999, name="gamma"),
    Real(0.8, 0.99, name="gae_lambda"),
    Real(0.1, 0.3, name="clip_eps"),
    Real (0.0, 1, name="value_coef"),
    Real(0.0, 0.05, name="entropy_coef"),
    Integer(64, 512, name="n_steps"),
    Integer(2, 10, name="epochs"),
    Integer(32, 256, name="batch_size"),
]

@use_named_args(dim_space)
def objective(learning_rate, gamma, gae_lambda, clip_eps, value_coef, entropy_coef,
              n_steps, epochs, batch_size):
    """
    Train a PPO agent with given hyperparameters and return negative mean reward.
    """
    # New policy and optimizer per trial
    policy = Policy().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    buffer = RolloutBuffer()

    env = make_env()
    obs = env.reset(seed=42)
    total_timesteps = 2000
    steps = 0

    # Collect rollouts and update until budget exhausted
    while steps < total_timesteps:
        for _ in range(n_steps):
            action, logp, value = policy.act(obs)
            next_obs, reward, done, _ = env.step(action)
            buffer.store(obs, action, logp, reward, torch.tensor(value), done)
            obs = next_obs
            steps += 1
            if done:
                obs = env.reset()
        # Perform PPO update
        ppo_update(
            policy, optimizer, buffer,
            gamma=gamma,
            lam=gae_lambda,
            c1=value_coef,
            c2=entropy_coef,
            clip_eps=clip_eps,
            epochs=epochs,
            batch_size=batch_size
        )

    # Evaluate performance
    mean_reward, _ = evaluate_policy(policy, env, n_episodes=5)
    # We minimize the negative of performance
    return -mean_reward

if __name__ == "__main__":
    # Run Bayesian optimization
    result = gp_minimize(
        func=objective,
        dimensions=dim_space,
        n_calls=20,
        random_state=42
    )

    # Best hyperparameters
    best_params = {dim.name: val for dim, val in zip(dim_space, result.x)}
    print("Best hyperparameters found:")
    for key, val in best_params.items():
        print(f"  {key}: {val}")

    # # Optionally, retrain a final model on full budget
    # print("Retraining final model with best hyperparameters...")
    # policy = Policy().to(device)
    # optimizer = torch.optim.Adam(policy.parameters(), lr=best_params['learning_rate'])
    # buffer = RolloutBuffer()
    # env = make_env()
    # obs = env.reset(seed=42)
    # steps = 0
    # total_timesteps = 5000  # more timesteps for final training
    # while steps < total_timesteps:
    #     for _ in range(best_params['n_steps']):
    #         action, logp, value = policy.act(obs)
    #         next_obs, reward, done, _ = env.step(action)
    #         buffer.store(obs, action, logp, reward, torch.tensor(value), done)
    #         obs = next_obs
    #         steps += 1
    #         if done:
    #             obs = env.reset()
    #     ppo_update(
    #         policy, optimizer, buffer,
    #         gamma=best_params['gamma'],
    #         lam=best_params['gae_lambda'],
    #         c1=0.5,
    #         c2=best_params['entropy_coef'],
    #         clip_eps=best_params['clip_eps'],
    #         epochs=best_params['epochs'],
    #         batch_size=best_params['batch_size']
    #     )

    # # Final evaluation
    # mean_reward, rewards = evaluate_policy(policy, env, n_episodes=10)
    # print(f"Final mean reward: {mean_reward}")
