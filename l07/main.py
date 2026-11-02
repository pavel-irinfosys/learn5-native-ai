import gymnasium as gym
import torch
from model_rl import ModelRL

D_TYPE = torch.float32
D_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test(model):
    env = gym.make("CartPole-v1", render_mode="human")
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

def train(model):
    best = 0
    env = gym.make("CartPole-v1")
    model = ModelRL(dtype=D_TYPE, device=D_DEVICE)

    for episode in range(1, 1001):
        total = model.train_step(env)
        if episode % 50 == 0:
            print(f"Episode {episode} | reward: {total:.0f}")
        if total >= best:
            best = total
            model.save()

def main():
    model = ModelRL(dtype=D_TYPE, device=D_DEVICE)
    model.eval()
    train(model)
    test(model) # type: ignore


if __name__ == "__main__":
    main()
