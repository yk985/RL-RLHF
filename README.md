# RL-RLHF

## Main objective
In this repository, there's the code related to the RLHF project in the context of Reinforcement Learning. Mainly, the scripts allow the training of two policies using REINFORCE algorithm on two different OpenAI environments : Cartpole v-0 and Acrobot. One of the policies produces a good reward while the other one reaches less good reward. These policies are used to generate trajectories and build a preference dataset. This dataset will then be used to train a new policy (here initialized as the less good policy) using DPO and PPO-RLHF

## Main Files
There's 4 main files: Two methods and two environments. The main file are jupyter notebooks of the names: DPO_Final_Acrobot.ipynb , DPO_Final_CartPole_v0.ipynb , RLHF_Final_CartPole_v0.ipynb and RLHF_Final_Acrobot_v2.ipynb. 

## How to run and implementation
The four files start by importing the necessary modules and coded classes and functions. Then, a gym environment is created. We train the two policies (instances of the class Policy define in Train_polocy_func.py) and store them, or we can load them if they already exist. to train the policies we use REINFORCE algorithm implemented in the file OPPO.py (confusion in the name of the algorithm in the beginning and we kept it that way). The file contains some baselines we tested with REINFORCE updates functions for Cartpole OPPO_update() and another one for Acrobot OPPO_update_Acrobo().
The main scripts allow then to test the trained/loaded policies (there are booleans to control the tests and the loading/training of policies). The evaluation of the policy is done through the function evaluate_policy() that calculate the cumulative reward of the environment over a full trajectory and over a number of trajectories. The function returns the mean and the array of rewards over the trajectories (the function is defined in the file PPO.py).

The rest of the files are dedicated to train policies for the different environments either using DPO or PPO-RLHF. 

For the DPO notebooks, we generate a number of trajectory pairs (using the function sample_preference_pairs() defined in pairs_generator.py) and then initialize a new policy as the less good one trained before. The training is done by calling the function DPO_training defined in the file DPO.py. To do this a function dpo_loss() was defined to calculate the loss via the logprob of a trajectory given a policy. The policy is then upgraded using a normal backward propagation step. The trained policy is then evaluated and compared to the initial version and the reference policy passed to it during the training.

For the RLHF notebooks, we generated as well the trajectory pairs but this time we also train a reward model (instance of the class RewardModel defined in RLHF.py). The reward model is trained in the scripts Train_RewardModel_{env_name}.ipynb and then loaded to the main files. Again a new policy is initialized and trained using train_policy_from_rollouts_n_updates_v2() defined in RLHF.py. The policy is then evaluated and compared to the initial version and the reference policy passed to it during the training.

The evaluations and plots are averaged over three seeds and the plot functions are defined in Plot_Functions.py.

