import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('xtick', labelsize=14) 
matplotlib.rc('ytick', labelsize=14)
font = {'size'   : 16}

matplotlib.rc('font', **font)


def plot_Scores(reward_hist_pi1, reward_hist_pi2): # in cartpole, 200 is the max reward
  """
  Plot average episode reward versus update index
  for two policies π₁ and π₂.
  Args:
    reward_hist_pi1: list of average episode rewards per update for π₁
    reward_hist_pi2: same for π₂ (must be same length)
  """

  runs = np.arange(1, len(reward_hist_pi1) + 1)
  runs = np.arange(1, len(reward_hist_pi2) + 1)

  plt.figure(figsize=(7, 5))
  plt.plot(runs, reward_hist_pi1, label=fr"$\pi_1$ Scores \n mean = {np.mean(reward_hist_pi1):.1f}")
  plt.plot(runs, reward_hist_pi2, label=fr"$\pi_2$ Scores \n mean = {np.mean(reward_hist_pi2):.1f}")
  plt.xlabel("Episodes (complete Runs)")
  plt.ylabel("Score at each episode $=$ $R_{run}$ ")
  plt.legend(loc="lower right")
  plt.grid(alpha=0.5)
  plt.show()


def plot_suboptimality(reward_hist_pi1, reward_hist_pi2, max_reward=None): # in cartpole, 200 is the max reward
  """
  Plot suboptimality = (max_reward - reward) versus update index
  for two policies π₁ and π₂.

  Args:
    reward_hist_pi1: list of average episode rewards per update for π₁
    reward_hist_pi2: same for π₂ (must be same length)
    max_reward:      optional scalar, the optimal/target return
                      (defaults to the max seen in either history)
  """
  # Infer optimal return if not given
  if max_reward is None:
      max_reward = max(max(reward_hist_pi1), max(reward_hist_pi2))

  updates1 = np.arange(1, len(reward_hist_pi1) + 1)
  updates2 = np.arange(1, len(reward_hist_pi2) + 1)
  sub1 = max_reward - np.array(reward_hist_pi1)
  sub2 = max_reward - np.array(reward_hist_pi2)

  plt.figure(figsize=(7, 5))
  plt.plot(updates1, sub1, label=f"$\pi_1$ suboptimality \n mean = {np.mean(sub1):.1f}")
  plt.plot(updates2, sub2, label=f"$\pi_2$ suboptimality \n mean = {np.mean(sub2):.1f}")
  plt.xlabel("Episodes (complete Runs)")
  plt.ylabel("Suboptimality $=$ $R_{run}-R_{max}$ ")
  plt.title("Policy Suboptimality Over Training")
  plt.legend(loc="upper right")
  plt.grid(alpha=0.5)
  plt.show()


def plot_scores_RLHF(rewards_init, rewards_ref, rewards_rlhf, algo="DPO", save=False, n_pair=None, env_name=None):
  """
  Plot average episode reward versus update index
  for two policies: DPO and Reference policy.

  Args:
    reward_hist_dpo: list of average episode rewards per update for DPO
    reward_hist_ref: same for Reference policy
    max_reward: optional scalar, optimal return (defaults to max seen in any history)
  """
  if len(rewards_init) != len(rewards_ref) or len(rewards_ref) != len(rewards_rlhf):
      raise ValueError("All reward histories must be of the same length.")
  
  runs = np.arange(1, len(rewards_init) + 1)

  plt.figure(figsize=(8, 5))
  plt.plot(runs, rewards_init, label=f"Initial policy = $\pi_2$  \n mean = {np.mean(rewards_init):.1f}", linewidth=2,    color= '#1a4aff')
  plt.plot(runs, rewards_ref, label=f"Reference policy = $\pi_1$ \n mean = {np.mean(rewards_ref):.1f}", lw=2, linestyle='--', color= 'black')
  plt.plot(runs, rewards_rlhf, label=algo+f" policy \n mean = {np.mean(rewards_rlhf):.1f}",   linewidth=2,    color= '#DB281C')
  plt.xlabel("Episodes (complete runs)")
  plt.ylabel("Cumulative Reward over the run")
  plt.xlim(0, len(runs)+1)
  plt.title(f"{env_name} | $n_{{pairs}}={n_pair}$ |averaged over 3 seeds")
  plt.legend(loc="lower right")
  plt.grid(alpha=0.3)
  plt.tight_layout()
  
  if save:
    plt.savefig(f"{env_name}_{algo}_performance_{n_pair}.png", dpi=300)
  else:
    plt.show()


def plot_suboptimality_three_policies(reward_hist_dpo, reward_hist_init, reward_hist_ref, max_reward=None, algo="DPO"):
  """
  Plot suboptimality = (max_reward - reward) for three policies:
  - DPO policy
  - Init policy
  - Reference policy

  Args:
    reward_hist_dpo: list of average rewards per episode (DPO)
    reward_hist_init: same for Init policy
    reward_hist_ref: same for Reference policy
    max_reward: optional scalar, optimal return (defaults to max seen in any history)
  """
  # Infer optimal return if not provided
  if max_reward is None:
      max_reward = max(
          max(reward_hist_dpo),
          max(reward_hist_init),
          max(reward_hist_ref)
      )

  runs = np.arange(1, len(reward_hist_dpo) + 1)
  sub_dpo = max_reward - np.array(reward_hist_dpo)
  sub_init = max_reward - np.array(reward_hist_init)
  sub_ref = max_reward - np.array(reward_hist_ref)

  plt.figure(figsize=(8, 5))
  plt.plot(runs, sub_init, label= rf"Init policy = \pi_2 \n mean = {np.mean(sub_init):.1f}", linestyle='--')
  plt.plot(runs, sub_ref, label= rf"Reference policy = \pi_1; mean = {np.mean(sub_ref):.1f}", linestyle=':')
  plt.plot(runs, sub_dpo, label= algo+f" policy \n mean = {np.mean(sub_dpo):.1f}", linewidth=2)
  
  plt.xlabel("Episodes (complete runs)")
  plt.ylabel("Suboptimality $= R_{max} - R_{run}$")
  plt.title("Policy Suboptimality Over Evaluation Episodes")
  plt.legend(loc="upper right")
  plt.grid(alpha=0.3)
  plt.tight_layout()
  plt.show()


def plot_trajectory_performance(traj1, traj2):
  """
  Plot cumulative reward versus timestep for two trajectories.

  Args:
    traj1, traj2: lists of dicts with key 'reward' (as from generate_trajectory)
  """
  # Extract reward sequences
  rew1 = [step["reward"] for step in traj1]
  rew2 = [step["reward"] for step in traj2]

  # Cumulative sums
  cum1 = np.cumsum(rew1)
  cum2 = np.cumsum(rew2)
  steps1 = np.arange(len(cum1))
  steps2 = np.arange(len(cum2))

  plt.figure()
  plt.plot(steps1, cum1, label="Trajectory π₁")
  plt.plot(steps2, cum2, label="Trajectory π₂")
  plt.xlabel("Timestep")
  plt.ylabel("Cumulative Reward")
  plt.title("Single‐Episode Performance Comparison for Two Given Trajectories")
  plt.legend(loc="upper left")
  plt.grid(alpha=0.6)
  plt.show()


def plot_loss_curve(loss_history1, loss_history2, loss_history3, n_pair, algo="DPO"):
  epochs1 = list(range(1, len(loss_history1)+1))
  epochs2 = list(range(1, len(loss_history2)+1))
  epochs3 = list(range(1, len(loss_history3)+1))
  plt.figure(figsize=(8,5))
  plt.plot(epochs1, loss_history1, label="seed 35", color='blue')
  plt.plot(epochs2, loss_history2, label="seed 42", color='orange')
  plt.plot(epochs3, loss_history3, label="seed 100", color='green')
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.title(algo+f" Loss over Epochs with $n_{{pairs}}$ = {n_pair} |  for 3 different seeds")
  plt.grid(alpha=0.3)
  plt.legend(loc="upper right")
  plt.tight_layout()
  plt.show()


def plot_avg_loss_curves(loss_hist_list, pair_list, algo="DPO",legend=True, save = True, env_name=None):
    plt.figure(figsize=(7,5))
    for n_pairs, seed_hists in zip(pair_list, loss_hist_list):
        arr = np.vstack(seed_hists)                  # shape (n_seeds, epochs)
        mean = arr.mean(axis=0)
        std  = arr.std(axis=0)
        epochs = np.arange(1, arr.shape[1]+1)

        plt.plot(epochs, mean, label=f"{n_pairs} pairs")
        plt.fill_between(epochs,
                         mean - std,
                         mean + std,
                         alpha=0.2)
    plt.xlabel(f"Update step $t$")
    plt.ylabel("Loss")
    # plt.title(f"{algo} Loss curve ±1 $\\theta$ | averaged over 3 seeds")
    plt.title(f"{algo} | {env_name} ")
    if legend:
        plt.legend()
    else:
        plt.legend().set_visible(False)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save:
      plt.savefig(f"{env_name}_{algo}_loss_curve.png", dpi=300)
    plt.show()


def plot_avg_reward_curves(loss_hist_list, pair_list, algo="DPO", legend=True, save = True, env_name=None):
    plt.figure(figsize=(7,5))
    for n_pairs, seed_hists in zip(pair_list, loss_hist_list):
        arr = np.vstack(seed_hists)                  # shape (n_seeds, epochs)
        mean = arr.mean(axis=0)
        std  = arr.std(axis=0)
        epochs = 10*np.arange(1, arr.shape[1]+1)

        plt.plot(epochs, mean, label=f"{n_pairs} pairs")
        plt.fill_between(epochs,
                         mean - std,
                         mean + std,
                         alpha=0.2)
    plt.xlabel(f"Update step $t$")
    plt.ylabel("Reward")
    plt.title(f"{algo} | {env_name}")
    # plt.title(f"{algo} Loss curve ±1 $\\theta$ | averaged over 3 seeds")
    if legend:
        plt.legend()
    else:
        plt.legend().set_visible(False)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save:
      plt.savefig(f"{env_name}_{algo}_reward_curve.png", dpi=300)
    plt.show()