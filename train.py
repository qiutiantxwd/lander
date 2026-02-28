import os
import random
from collections import deque, namedtuple
from datetime import datetime

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse

from plots import plot_training_graphs, plot_eval_histograms
# ── Hyperparameters ──────────────────────────────────────────────────────────
GAMMA = 0.99
LR = 1e-4
REPLAY_BUFFER_SIZE = 100_000
MIN_BUFFER_SIZE = 1_000
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995
TAU = 0.005                    # soft target update rate
TRAIN_EVERY = 4                # env steps between gradient updates
GRAD_STEPS_PER_UPDATE = 2      # gradient steps each time we train
HIDDEN_SIZE = 256
SEED = 42
# debug = False
# if debug:
#     EXP_FOLDER = "./exp" 
# else:
#     EXP_FOLDER = f"./exp/{datetime.now().strftime('%m%d_%H%M')}" 
# VIDEO_DIR = f"{EXP_FOLDER}/videos"
# FIGURES_DIR = f"{EXP_FOLDER}/figures"

# ── Reproducibility ──────────────────────────────────────────────────────────
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Q-Network ────────────────────────────────────────────────────────────────
class QNetwork(nn.Module):
    def __init__(self, state_dim=8, action_dim=4, hidden=HIDDEN_SIZE):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# ── Replay Buffer ────────────────────────────────────────────────────────────
Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states = torch.tensor(np.array([t.state for t in batch]), dtype=torch.float32, device=device)
        actions = torch.tensor([t.action for t in batch], dtype=torch.long, device=device)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=device)
        next_states = torch.tensor(np.array([t.next_state for t in batch]), dtype=torch.float32, device=device)
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32, device=device)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

# ── Agent ────────────────────────────────────────────────────────────────────
class DQNAgent:
    def __init__(self, bs):
        self.policy_net = QNetwork().to(device)
        self.target_net = QNetwork().to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.bs = bs

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
        self.epsilon = EPS_START

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(4)
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            return self.policy_net(state_t).argmax(dim=1).item()

    def train_step(self):
        if len(self.replay_buffer) < MIN_BUFFER_SIZE:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.bs)

        # Current Q-values for chosen actions
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(dim=1)[0]
            targets = rewards + GAMMA * next_q_values * (1.0 - dones)

        # loss = F.mse_loss(q_values, targets)
        loss = F.smooth_l1_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def update_target(self):
        for tp, pp in zip(self.target_net.parameters(), self.policy_net.parameters()):
            tp.data.copy_(TAU * pp.data + (1.0 - TAU) * tp.data)

    def decay_epsilon(self):
        self.epsilon = max(EPS_END, self.epsilon * EPS_DECAY)

# ── Training Loop ────────────────────────────────────────────────────────────
def main(args):
    if args.exp == None:
        EXP_FOLDER = "./exp/debug"
    else:
        EXP_FOLDER = f"./exp/{args.exp}" 

    VIDEO_DIR = f"{EXP_FOLDER}/videos"
    FIGURES_DIR = f"{EXP_FOLDER}/figures"

    os.makedirs(VIDEO_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    env = RecordVideo(
        env,
        video_folder=VIDEO_DIR,
        episode_trigger=lambda ep: ep % 25 == 0 or ep == args.EPISODES - 1,
        name_prefix="lunar-lander",
    )

    agent = DQNAgent(args.bs)

    episode_rewards = []
    episode_durations = []

    log_file = open(f"{EXP_FOLDER}/train.log", "w")

    step_count = 0

    for ep in range(args.EPISODES):
        state, _ = env.reset(seed=SEED + ep)
        total_reward = 0.0
        steps = 0

        while True:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.replay_buffer.push(state, action, reward, next_state, float(done))
            step_count += 1
            if step_count % TRAIN_EVERY == 0:
                for _ in range(GRAD_STEPS_PER_UPDATE):
                    agent.train_step()
                    agent.update_target()

            state = next_state
            total_reward += reward
            steps += 1

            if done:
                break

        agent.decay_epsilon()

        episode_rewards.append(total_reward)
        episode_durations.append(steps)

        # Rolling average of last 100 episodes
        avg_reward = np.mean(episode_rewards[-100:])

        if (ep>0 and ep % 25 == 0) or ep == args.EPISODES - 1:
            msg = (
                f"Episode {ep:4d}/{args.EPISODES} | "
                f"Reward: {total_reward:7.1f} | "
                f"Avg100: {avg_reward:7.1f} | "
                f"Eps: {agent.epsilon:.3f}"
            )
            print(msg)
            log_file.write(msg + "\n")
            log_file.flush()

    log_file.close()
    env.close()

    # Save model and metrics
    torch.save(agent.policy_net.state_dict(), f"{EXP_FOLDER}/dqn_lunar_lander.pth")
    np.save(f"{EXP_FOLDER}/episode_rewards.npy", np.array(episode_rewards))
    np.save(f"{EXP_FOLDER}/episode_durations.npy", np.array(episode_durations))

    print("\n✓ Training complete.")
    print(f"  Model saved to dqn_lunar_lander.pth")
    print(f"  Final 100-episode avg reward: {np.mean(episode_rewards[-100:]):.1f}")
    print(f"  Videos saved to {VIDEO_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--EPISODES", type=int, default=600)
    parser.add_argument("--bs", type=int, default=512)
    parser.add_argument("--exp", type=str, default=None)
    args = parser.parse_args()
    main(args)