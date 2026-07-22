"""
Watch the Trained DQN Agent Balance the Pole

This loads the weights saved by dqn_cartpole.py and runs the agent with
Gymnasium's built-in pygame renderer (render_mode="human") so you can
SEE it actually pushing the cart left/right to keep the pole up.

Run dqn_cartpole.py first (it saves dqn_cartpole_weights.pth), then run
this script:

    python dqn_cartpole.py          # trains and saves weights
    python visualize_agent.py       # opens a pygame window and shows it live

"""

import torch
import gymnasium as gym
from dqn_cartpole import QNetwork


def watch_agent(weights_path="dqn_cartpole_weights.pth", num_episodes=5):
    device = torch.device("cpu")  # rendering is fast enough on CPU
    env = gym.make("CartPole-v1", render_mode="human")  # pygame window opens here

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = QNetwork(state_dim, action_dim).to(device)
    policy_net.load_state_dict(torch.load(weights_path, map_location=device))
    policy_net.eval()  # turn off any training-only behavior

    for ep in range(1, num_episodes + 1):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            with torch.no_grad():
                state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                action = int(torch.argmax(policy_net(state_t), dim=1).item())

            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            env.render()  # draws the current frame with pygame

        print(f"Episode {ep}: pole balanced for {total_reward:.0f} timesteps "
              f"({'SOLVED - hit the 500 step cap' if total_reward >= 500 else 'pole fell'})")

    env.close()


if __name__ == "__main__":
    watch_agent()