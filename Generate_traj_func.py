import gym
import numpy as np
from tqdm import tqdm
def generate_trajectory(policy, env, max_steps=500):
    """
    Roll out one episode with `policy` in `env` up to max_steps.
    Returns a list of dicts: [{"state": s, "action": a, "reward": r, "log_prob": lp, "value": v}, ...].
    """
    state = env.reset()
    trajectory = []
    for _ in range(max_steps):
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
            break
    return trajectory


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
    for i in tqdm(range(nb_traj_pairs)):
        traj_pair_list.append(np.array(generate_preference_pair(policy1,policy2,env,max_steps=max_steps)))
    return np.array(traj_pair_list)
