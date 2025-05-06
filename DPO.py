import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def dpo_loss(policy, ref_policy, dataset, beta):
    losses = []
    for state, a_pos, a_neg in dataset:
        pi, _ = policy(state)
        pi_ref, _ = ref_policy(state)

        # Use log probabilities, add small epsilon for stability
        log_pi_pos = torch.log(pi[a_pos] + 1e-8)
        log_pi_neg = torch.log(pi[a_neg] + 1e-8)
        log_ref_pos = torch.log(pi_ref[a_pos] + 1e-8)
        log_ref_neg = torch.log(pi_ref[a_neg] + 1e-8)

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