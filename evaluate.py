import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from train import QNetwork

EVAL_EPISODES = 100
MODEL_PATH = "dqn_lunar_lander.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    # Load trained model
    policy_net = QNetwork().to(device)
    policy_net.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    policy_net.eval()

    env = gym.make("LunarLander-v3")

    eval_rewards = []
    eval_durations = []

    for ep in range(EVAL_EPISODES):
        state, _ = env.reset()
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

        if (ep + 1) % 25 == 0:
            print(f"  Evaluated {ep + 1}/{EVAL_EPISODES} episodes...")

    env.close()

    mean_reward = np.mean(eval_rewards)
    mean_duration = np.mean(eval_durations)

    np.save("eval_rewards.npy", np.array(eval_rewards))
    np.save("eval_durations.npy", np.array(eval_durations))

    print(f"\n{'='*50}")
    print(f"Evaluation Results ({EVAL_EPISODES} episodes)")
    print(f"{'='*50}")
    print(f"  Mean reward:   {mean_reward:.1f}")
    print(f"  Std reward:    {np.std(eval_rewards):.1f}")
    print(f"  Min reward:    {np.min(eval_rewards):.1f}")
    print(f"  Max reward:    {np.max(eval_rewards):.1f}")
    print(f"  Mean duration: {mean_duration:.1f} steps")
    print(f"{'='*50}")
    print(f"  Result: {'PASS' if mean_reward > 200 else 'FAIL — retrain needed'}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
