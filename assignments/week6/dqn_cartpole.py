"""
DQN for the Classic Inverted Pendulum (CartPole) Problem
==========================================================

Problem: Balance a pole on a moving cart by choosing one of TWO discrete
actions at every timestep -> push cart LEFT or push cart RIGHT.

Environment: Gymnasium's CartPole-v1
- State (4 continuous values): [cart position, cart velocity,
                                  pole angle, pole angular velocity]
- Action (discrete, 2 values): 0 = push left, 1 = push right
- Reward: +1 for every timestep the pole stays upright
- Episode ends if: pole falls past ~12 degrees, cart moves off screen,
  or 500 timesteps are reached (solved!)
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import gymnasium as gym

# Directory this script lives in (e.g. .../week6). Using this instead of
# a plain filename means the weights always save next to this file,
# regardless of which folder you happen to run the script from.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(SCRIPT_DIR, "dqn_cartpole_weights.pth")


# ----------------------------------------------------------------------
# 1. Q-NETWORK
#    Maps a state (4 numbers) -> Q-values for each action (2 numbers)
# ----------------------------------------------------------------------
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return self.net(x)


# ----------------------------------------------------------------------
# 2. REPLAY BUFFER
#    Stores past experiences (s, a, r, s', done) so we can train on
#    random mini-batches instead of only the most recent transition.
#    This breaks correlation between consecutive samples and stabilizes
#    training.
# ----------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ----------------------------------------------------------------------
# 3. DQN AGENT
# ----------------------------------------------------------------------
class DQNAgent:
    def __init__(self, state_dim, action_dim, device):
        self.device = device
        self.action_dim = action_dim

        # Two networks: one we actively train (policy_net) and one that
        # stays frozen for a while to provide stable Q-value targets
        # (target_net). Without a separate target network, DQN training
        # is notoriously unstable because the target moves every step.
        self.policy_net = QNetwork(state_dim, action_dim).to(device)
        self.target_net = QNetwork(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=5e-4)
        self.gamma = 0.99          # discount factor for future rewards
        self.batch_size = 64
        self.buffer = ReplayBuffer()
        self.min_buffer_size = 1000  # don't train until we have some real experience
        self.tau = 0.005             # soft update rate for target network

        # Epsilon-greedy exploration: start fully random, decay towards
        # mostly-greedy as the agent learns.
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)  # explore
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.policy_net(state_t)
            return int(torch.argmax(q_values, dim=1).item())  # exploit

    def train_step(self):
        if len(self.buffer) < self.min_buffer_size:
            return None  # not enough real experience yet

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states = torch.tensor(states, device=self.device)
        actions = torch.tensor(actions, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, device=self.device)
        next_states = torch.tensor(next_states, device=self.device)
        dones = torch.tensor(dones, device=self.device)

        # Q(s, a) for the actions we actually took
        q_values = self.policy_net(states).gather(1, actions).squeeze(1)

        # Double DQN target: use policy_net to CHOOSE the best next action,
        # but target_net to EVALUATE it. Plain DQN uses target_net for both,
        # which systematically overestimates Q-values and is a big cause
        # of the instability / forgetting you saw.
        with torch.no_grad():
            best_next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
            max_next_q = self.target_net(next_states).gather(1, best_next_actions).squeeze(1)
            target = rewards + self.gamma * max_next_q * (1 - dones)

        loss = nn.functional.smooth_l1_loss(q_values, target)  # Huber loss: more robust
                                                                # to reward-scale/outliers than MSE

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10)
        self.optimizer.step()

        self.soft_update_target()  # smooth, continuous target update every step

        return loss.item()

    def soft_update_target(self):
        # Polyak averaging: target network slowly tracks the policy network
        # instead of being hard-copied all at once every N episodes. This
        # removes the sudden jumps that were causing performance to crash.
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1 - self.tau) * target_param.data)

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def evaluate_greedy(agent, device, num_eval_episodes=5):
    """
    Run the agent with NO exploration (epsilon=0) to see its true, current
    policy performance. Training-time rewards are noisy because of random
    exploration actions; this gives a cleaner signal of real progress.
    """
    env = gym.make("CartPole-v1")
    scores = []
    for _ in range(num_eval_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            with torch.no_grad():
                state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                action = int(torch.argmax(agent.policy_net(state_t), dim=1).item())
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        scores.append(total_reward)
    env.close()
    return np.mean(scores)


# ----------------------------------------------------------------------
# 4. TRAINING LOOP
# ----------------------------------------------------------------------
def train(num_episodes=600, target_update_freq=10, solved_score=475):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make("CartPole-v1")

    state_dim = env.observation_space.shape[0]   # 4
    action_dim = env.action_space.n               # 2 (left / right)

    agent = DQNAgent(state_dim, action_dim, device)
    scores = deque(maxlen=100)

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # --- Reward shaping ---
            # Default CartPole reward is +1 per timestep no matter where
            # the cart is, so the agent has no incentive to stay centered
            # (it only avoids letting the POLE fall). That's why it can
            # look like it "solved" balancing but still drifts off-screen.
            # We add a small penalty proportional to distance from the
            # center (cart_position is next_state[0], range is [-2.4, 2.4])
            # so staying centered is also rewarded, not just staying upright.
            cart_position = next_state[0]
            shaped_reward = reward - 0.05 * abs(cart_position)

            agent.buffer.push(state, action, shaped_reward, next_state, float(done))
            state = next_state
            episode_reward += reward  # track TRUE reward for solved-check, not shaped

            agent.train_step()

        agent.decay_epsilon()
        scores.append(episode_reward)
        avg_score = np.mean(scores)

        if episode % 10 == 0:
            print(f"Episode {episode:4d} | Reward: {episode_reward:6.1f} | "
                  f"Avg(last 100): {avg_score:6.1f} | Epsilon: {agent.epsilon:.3f}")

        if episode % 20 == 0:
            eval_score = evaluate_greedy(agent, device)
            print(f"  -> Greedy eval (no exploration): {eval_score:.1f} avg over 5 episodes")

        # CartPole-v1 is considered "solved" at avg reward >= 475 over 100 eps
        if len(scores) == 100 and avg_score >= solved_score:
            print(f"\nSolved in {episode} episodes! Avg score: {avg_score:.1f}")
            break

    env.close()

    # Save the trained weights so we can load them later for visualization
    # without having to retrain from scratch.
    torch.save(agent.policy_net.state_dict(), WEIGHTS_PATH)
    print(f"Saved trained weights to {WEIGHTS_PATH}")

    return agent


if __name__ == "__main__":
    trained_agent = train()