"""This file is for defining a function that will determine a seed, train a policy, generate trajectories, 
 and apply DPO and RLHF for a certain environment"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
from Train_policy_func import Policy, device
from pairs_generator import sample_preference_pairs, generate_trajectory
from PPO import ppo_update, RolloutBuffer

def ppo_training(env,total_steps  = 1000,     # per update
                updates      = 500,        # how many times to run rollout+update
                clip_eps     = 0.2,      # PPO clipping parameter
                gamma   = 0.99, 
                lam =0.95, # GAE parameters
                lr           = 0.0001,
                batch_size   = 64,
                epochs       = 4):
    policy    = Policy(state_size=env.observation_space.shape[0],
                     action_size=env.action_space.n).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    buffer    = RolloutBuffer()
    avg_returns = []
    for update in range(updates):
        state = env.reset()
        steps = 0

        # for tracking episode returns within this batch
        episode_rewards     = []
        current_ep_reward   = 0.0

        # 1) Roll out until we have total_steps
        while steps < total_steps:
            action, logp, value = policy.act(state)
            next_state, reward, done, _ = env.step(action)
            buffer.store(state, action, logp, reward, value, done)
            state = next_state
            steps += 1

            # accumulate for this episode
            current_ep_reward += reward

            if done:
                episode_rewards.append(current_ep_reward)
                current_ep_reward = 0.0
                state = env.reset()

        # in case the last episode didn’t terminate exactly on total_steps
        if current_ep_reward > 0.0:
            episode_rewards.append(current_ep_reward)

        # compute average return for this batch
        avg_ret = np.mean(episode_rewards)
        avg_returns.append(avg_ret)

        # 2) Compute last value (for GAE)
        _, last_value = policy(torch.from_numpy(state).float().unsqueeze(0).to(device))

        # 3) PPO update
        ppo_update(policy, optimizer, buffer, clip_eps, epochs, batch_size)

        # 4) Saving checkpoint of the policy to get pi2
        if (update ) % 100 == 0:
            print(f"Update {update}/{updates} completed.")
            torch.save(policy.state_dict(), f"pi2_checkpoint_{update}.pth")

    # 5) at the end, save pi1
    torch.save(policy.state_dict(), "pi1_final.pth")
    print("Saved final policy as pi1_final.pth")
    return policy, avg_returns




def run_train_gen_loop(env,train_func,seeds_list,max_steps_gen_traj=500,K=200):
    avg_returns_list_1=[]
    avg_returns_list_2=[]
    traj1_list=[]
    traj2_list=[]
    for seed in seeds_list:
        torch.manual_seed(seed)
        env.seed(seed)
        state_size,action_size=env.shape
        policy1=Policy(state_size=state_size,action_size=action_size)
        policy2=Policy(state_size=state_size,action_size=action_size)
        policy1,avg_return_1=train_func()#add assignements of average reward, loss ...
        policy2,avg_return_2=train_func()
        avg_returns_list_1.append(avg_return_1.copy())
        avg_returns_list_2.append(avg_return_2.copy())
        policy1.eval()
        policy2.eval()
        traj1 = generate_trajectory(policy1, env, max_steps=max_steps_gen_traj)
        traj2 = generate_trajectory(policy2, env, max_steps=max_steps_gen_traj)
        traj1_list.append(traj1.copy())
        traj2_list.append(traj2.copy())
        print(f"π₂ → length {len(traj2)}, total reward {sum(s['reward'] for s in traj2):.1f}")
        print(f"π₁ → length {len(traj1)}, total reward {sum(s['reward'] for s in traj1):.1f}")

        prefs = sample_preference_pairs(policy1, policy2, env, K=200) # Need to define K elsewhere as hyperparameter
        print(f"Collected {len(prefs)} preference pairs.")
