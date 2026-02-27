import argparse
import numpy as np
import gymnasium as gym
import torch

from train import QNetwork, SEED, device
from plots import plot_eval_histograms

NUM_EVAL_EPISODES = 100


def main(args):
    exp_dir = f"./exp/{args.EXP}"
    model_path = f"{exp_dir}/dqn_lunar_lander.pth"

    # Load trained model
    policy_net = QNetwork().to(device)
    policy_net.load_state_dict(torch.load(model_path, map_location=device))
    policy_net.eval()

    env = gym.make("LunarLander-v3")

    eval_rewards = []
    eval_durations = []

    for ep in range(NUM_EVAL_EPISODES):
        state, _ = env.reset(seed=SEED + ep)
        total_reward = 0.0
        steps = 0

        while True:
            with torch.no_grad():
                state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                action = policy_net(state_t).argmax(dim=1).item()

            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1

            if terminated or truncated:
                break

        eval_rewards.append(total_reward)
        eval_durations.append(steps)
        print(f"Episode {ep + 1:3d}/{NUM_EVAL_EPISODES} | Reward: {total_reward:7.1f} | Steps: {steps}")

    env.close()

    eval_rewards = np.array(eval_rewards)
    eval_durations = np.array(eval_durations)

    np.save(f"{exp_dir}/eval_rewards.npy", eval_rewards)
    np.save(f"{exp_dir}/eval_durations.npy", eval_durations)

    print(f"\nMean reward:   {eval_rewards.mean():.1f} +/- {eval_rewards.std():.1f}")
    print(f"Mean duration: {eval_durations.mean():.1f} +/- {eval_durations.std():.1f}")

    plot_eval_histograms(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--EXP", type=str, required=True)
    args = parser.parse_args()
    main(args)
