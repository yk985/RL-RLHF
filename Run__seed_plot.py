"""This file is for defining a function that will determine a seed, train a policy, generate trajectories, 
 and apply DPO and RLHF for a certain environment"""

import torch
from Train_policy_func import Policy, device
from pairs_generator import sample_preference_pairs, generate_trajectory

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
