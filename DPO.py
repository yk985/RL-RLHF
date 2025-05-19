import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pairs_generator import extract_states_actions, compute_logprob_trajectory, log_policy_of_traj
from Pendulum_functions import compute_logprob_trajectory_sac
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


def DPO_training(policy, ref_policy, preference_dataset, beta,optimizer,nb_epochs=500):
    for epoch in range(nb_epochs):
        optimizer.zero_grad()
        loss = dpo_loss(policy, ref_policy, preference_dataset, beta)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: DPO Loss = {loss.item():.4f}")


#As SAC use in Pendulum use a different policy:
# def dpo_loss_sac(policy, ref_policy, dataset, beta):
#     losses = []
#     for pair in dataset:
        

#         # Use log probabilities, add small epsilon for stability
#         log_pi_pos = compute_logprob_trajectory_sac(policy, pair["traj_acc"], device=device)+ 1e-8
#         log_pi_neg = compute_logprob_trajectory_sac(policy, pair["traj_rej"], device=device)+ 1e-8
#         log_ref_pos = compute_logprob_trajectory_sac(ref_policy, pair["traj_acc"], device=device)+ 1e-8
#         log_ref_neg = compute_logprob_trajectory_sac(ref_policy, pair["traj_rej"], device=device)+ 1e-8


#         advantage = beta * ((log_pi_pos - log_ref_pos) - (log_pi_neg - log_ref_neg))
#         loss = -F.logsigmoid(advantage)
#         losses.append(loss)
    
#     return torch.stack(losses).mean()


def dpo_loss_sac(policy, ref_policy, dataset, beta=1e-3, device="cpu"):
    losses = []
    for pair in dataset:
        log_pi_pos  = compute_logprob_trajectory_sac(policy,     pair["traj_acc"], device)
        log_pi_neg  = compute_logprob_trajectory_sac(policy,     pair["traj_rej"], device)
        log_ref_pos = compute_logprob_trajectory_sac(ref_policy, pair["traj_acc"], device)
        log_ref_neg = compute_logprob_trajectory_sac(ref_policy, pair["traj_rej"], device)

        diff  = (log_pi_pos - log_pi_neg) - (log_ref_pos - log_ref_neg)
        #v1:
        # beta_diff = beta * diff
        # diff_clamped = beta_diff.clamp(-10.0, 10.0)  
        losses.append(torch.nn.functional.softplus(-beta*diff))   # stable
        # v2: 
        # diff_clamped = diff.clamp(-10.0, 10.0)
        # losses.append(torch.nn.functional.softplus(-beta * diff))   # stable
        #v3:
        # margin = 1.0
        # losses.append( torch.relu(margin - beta*diff) )

    return torch.stack(losses).mean()

def dpo_loss_sac(policy, ref_policy, dataset, beta=1e-3, device="cpu"):
    """
    DPO loss with adaptive beta scaling based on the std of Δ:
      Δ = (log π_pos – log π_neg) – (log π_ref_pos – log π_ref_neg)
    We first collect all Δ, compute their stddev, then rescale β ← β / (std+ε)
    so that β·Δ has unit variance. Finally apply the stable softplus loss.
    """
    # 1) collect all raw diffs
    diffs = []
    for pair in dataset:
        lp  = compute_logprob_trajectory_sac(policy,     pair["traj_acc"], device)
        ln  = compute_logprob_trajectory_sac(policy,     pair["traj_rej"], device)
        lrp = compute_logprob_trajectory_sac(ref_policy, pair["traj_acc"], device)
        lrn = compute_logprob_trajectory_sac(ref_policy, pair["traj_rej"], device)
        diffs.append((lp - ln) - (lrp - lrn))   # raw Δ

    diffs = torch.stack(diffs)  # shape [K]
    # 2) rescale beta so that std(β·Δ) ≈ 1
    std = diffs.std(unbiased=False) + 1e-8
    beta_scaled = beta / std
    

    # 3) compute the stable softplus loss with scaled beta
    losses = torch.nn.functional.softplus(-beta_scaled * diffs)
    return losses.mean()



def DPO_training_sac(policy, ref_policy, preference_dataset, beta,optimizer,nb_epochs=500):
    for epoch in range(nb_epochs):
        optimizer.zero_grad()
        loss = dpo_loss_sac(policy, ref_policy, preference_dataset, beta)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.actor.parameters(), 1.0)
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: DPO Loss = {loss.item():e}")