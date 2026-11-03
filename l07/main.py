import gymnasium as gym
import torch
from model_rl import ModelRL
from collections import deque
import sys

D_TYPE = torch.float32
D_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test(model):
    env = gym.make("CartPole-v1", render_mode="human", max_episode_steps=1000)
    obs, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = model.act_greedy(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        print(f"{obs} : {action}")
        done = terminated or truncated

    print(f"total_reward: {total_reward}")
    env.close()

def train(model, episodes=1000):
    model.train()
    env = gym.make("CartPole-v1")
    recent = deque(maxlen=20)
    best = float("-inf")

    for episode in range(1, episodes + 1):
        total = model.train_step(env)
        recent.append(total)

        if len(recent) == 20:
            avg = sum(recent) / 20
            if avg > best:
                best = avg
                model.save()
            if episode % 50 == 0:
                print(f"Episode {episode} | last: {total:.0f} | avg20: {avg:.1f} | best: {best:.1f}")

    env.close()

def main():
    model = ModelRL(dtype=D_TYPE, device=D_DEVICE)
    if "train" in sys.argv:
        train(model)
        model.load()
    model.eval()
    test(model)


if __name__ == "__main__":
    main()
# todo : Option 1: Run many environments in parallel and batch them. This is the biggest win.