import numpy as np
from tqdm.auto import tqdm
from Generate_traj_func import *




def reward_from_trajectory(trajectory):
    tot_reward=0
    for dic in trajectory:
        tot_reward+=dic["reward"]
    return tot_reward

def generate_preference_pair(policy1,policy2,env,max_steps=500):
    """
    returns the trajectories from both policies, the first one 
    is the accepted policy and the second is the rejected one
    """
    trajectory1=generate_trajectory(policy1,env,max_steps=max_steps)
    trajectory2=generate_trajectory(policy2,env,max_steps=max_steps)
    reward_1=reward_from_trajectory(trajectory1)
    reward_2=reward_from_trajectory(trajectory2)
    prob_traj_1=1/(1+np.exp(reward_2-reward_1))
    if np.random.uniform()<prob_traj_1:
        return [trajectory1,trajectory2]
    else:
        return [trajectory2,trajectory1]
    

def generate_preference_datasets(policy1,policy2,env,nb_traj_pairs=100,max_steps=500):
    traj_pair_list=[]
    for i in tqdm(range(nb_traj_pairs), desc="Generating preference pairs", leave=False):
        traj_pair_list.append(generate_preference_pair(policy1,policy2,env,max_steps=max_steps))
    return traj_pair_list
