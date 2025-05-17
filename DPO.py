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
def dpo_loss_sac(policy, ref_policy, dataset, beta):
    losses = []
    for pair in dataset:
        

        # Use log probabilities, add small epsilon for stability
        log_pi_pos = compute_logprob_trajectory_sac(policy, pair["traj_acc"], device=device)
        log_pi_neg = compute_logprob_trajectory_sac(policy, pair["traj_rej"], device=device)
        log_ref_pos = compute_logprob_trajectory_sac(ref_policy, pair["traj_acc"], device=device)
        log_ref_neg = compute_logprob_trajectory_sac(ref_policy, pair["traj_rej"], device=device)


        advantage = beta * ((log_pi_pos - log_ref_pos) - (log_pi_neg - log_ref_neg))
        loss = -F.logsigmoid(advantage)
        losses.append(loss)
    
    return torch.stack(losses).mean()

def DPO_training_sac(policy, ref_policy, preference_dataset, beta,optimizer,nb_epochs=500):
    for epoch in range(nb_epochs):
        optimizer.zero_grad()
        loss = dpo_loss_sac(policy, ref_policy, preference_dataset, beta)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: DPO Loss = {loss.item():.4f}")