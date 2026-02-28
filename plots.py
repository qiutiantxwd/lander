import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

# FIGURES_DIR = "./figures"
# os.makedirs(FIGURES_DIR, exist_ok=True)


def rolling_average(data, window=100):
    """Compute rolling average with given window size."""
    return np.convolve(data, np.ones(window) / window, mode="valid")


def plot_training_graphs(args):
    """Generate reward and duration vs. episode training plots (Report Item 2)."""
    EXP_DIR = f"./exp/{args.EXP}"
    FIGURES_DIR = f"./exp/{args.EXP}/figures"
    rewards = np.load(f"{EXP_DIR}/episode_rewards.npy")
    durations = np.load(f"{EXP_DIR}/episode_durations.npy")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    episodes = np.arange(len(rewards))

    # ── Reward plot ──
    ax1.plot(episodes, rewards, alpha=0.25, color="steelblue", label="Per-episode reward")
    avg_rewards = rolling_average(rewards)
    ax1.plot(np.arange(99, 99 + len(avg_rewards)), avg_rewards, color="navy", linewidth=2, label="100-ep rolling avg")
    ax1.axhline(y=200, color="green", linestyle="--", linewidth=1, label="Passing threshold (200)")
    ax1.set_ylabel("Reward")
    ax1.set_title("Training Reward per Episode")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    # ── Duration plot ──
    ax2.plot(episodes, durations, alpha=0.25, color="coral", label="Per-episode duration")
    avg_durations = rolling_average(durations)
    ax2.plot(np.arange(99, 99 + len(avg_durations)), avg_durations, color="darkred", linewidth=2, label="100-ep rolling avg")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Duration (steps)")
    ax2.set_title("Training Duration per Episode")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "training_graphs.png")
    plt.savefig(path, dpi=150)
    print(f"Saved training graphs to {path}")
    plt.show()


def plot_eval_histograms(args):
    """Generate evaluation reward and duration histograms (Report Item 4)."""
    EXP_DIR = f"./exp/{args.EXP}"
    FIGURES_DIR = f"{EXP_DIR}/figures"
    eval_rewards = np.load(f"{EXP_DIR}/eval_rewards.npy")
    eval_durations = np.load(f"{EXP_DIR}/eval_durations.npy")

    mean_r = np.mean(eval_rewards)
    mean_d = np.mean(eval_durations)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ── Reward histogram ──
    ax1.hist(eval_rewards, bins=20, edgecolor="black", color="steelblue", alpha=0.8)
    ax1.axvline(mean_r, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_r:.1f}")
    ax1.set_xlabel("Reward")
    ax1.set_ylabel("Count")
    ax1.set_title("Evaluation Reward Distribution (100 episodes)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ── Duration histogram ──
    ax2.hist(eval_durations, bins=20, edgecolor="black", color="coral", alpha=0.8)
    ax2.axvline(mean_d, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_d:.1f}")
    ax2.set_xlabel("Duration (steps)")
    ax2.set_ylabel("Count")
    ax2.set_title("Evaluation Duration Distribution (100 episodes)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "eval_histograms.png")
    plt.savefig(path, dpi=150)
    print(f"Saved evaluation histograms to {path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--EXP", type=str)
    args = parser.parse_args()
    print("Generating training graphs...")
    plot_training_graphs(args)

    # print("\nGenerating evaluation histograms...")
    # plot_eval_histograms(args)
