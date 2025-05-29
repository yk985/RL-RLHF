import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pairs_generator import compute_logprob_trajectory
from PPO import evaluate_policy
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def dpo_loss(policy, ref_policy, dataset, beta):
    losses = []
    for pair in dataset:
        

        # Use log probabilities, add small epsilon for stability
        log_pi_pos = compute_logprob_trajectory(policy,pair["traj_acc"])+ 1e-8
        log_pi_neg = compute_logprob_trajectory(policy,pair["traj_rej"])+ 1e-8
        log_ref_pos = compute_logprob_trajectory(ref_policy,pair["traj_acc"])+ 1e-8
        log_ref_neg = compute_logprob_trajectory(ref_policy,pair["traj_rej"])+ 1e-8

        advantage = beta * ((log_pi_pos - log_ref_pos) - (log_pi_neg - log_ref_neg))
        loss = -F.logsigmoid(advantage)
        losses.append(loss)
    
    return torch.stack(losses).mean()


def DPO_training(policy, ref_policy, preference_dataset, beta,env,optimizer,nb_epochs=500):
    loss_hist = []
    traj_reward_hist=[]
    for epoch in range(nb_epochs):
        optimizer.zero_grad()
        loss = dpo_loss(policy, ref_policy, preference_dataset, beta)
        loss.backward()
        optimizer.step()

        loss_hist.append(loss.item())
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: DPO Loss = {loss.item():e}")
            mean_reward,_=evaluate_policy(policy,env,n_episodes=5)
            traj_reward_hist.append(mean_reward)

    return loss_hist, traj_reward_hist



