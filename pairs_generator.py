import math
from Generate_traj_func import generate_trajectory

def sample_preference_pairs(pi1, pi2, env, K=100):
    pairs = []
    for _ in range(K):
        t1 = generate_trajectory(pi1, env)
        t2 = generate_trajectory(pi2, env)
        R1, R2 = sum(s['reward'] for s in t1), sum(s['reward'] for s in t2)
        p1 = math.exp(R1) / (math.exp(R1) + math.exp(R2))
        pairs.append({
            "traj1": t1, "traj2": t2,
            "R1": R1, "R2": R2, "pref_prob": p1
        })
    return pairs


