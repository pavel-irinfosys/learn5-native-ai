import os
import numpy as np
import gymnasium as gym
import torch

env = gym.make("CartPole-v1")
D_TYPE = np.float32


def generate_data_sim(n_steps=1000, seed=0):
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)

    x = np.empty((n_steps, 5), dtype=D_TYPE)
    y = np.empty((n_steps, 4), dtype=D_TYPE)
    keep = np.ones(n_steps, dtype=bool)

    for i in range(n_steps):
        action = int(rng.integers(0, 2))
        x[i, :4], x[i, 4] = obs, action
        next_obs, reward, terminated, truncated, _ = env.step(action)
        y[i, :4] = next_obs - obs  # state AFTER the action
        if terminated or truncated:
            keep[i] = False
            obs, _ = env.reset()
        else:
            obs = next_obs

    return x[keep], y[keep]

def prep(a, m, s): 
    return torch.from_numpy((a - m) / s)

def prepare_data(n_steps=1000, seed=0):
    x, y = generate_data_sim(n_steps=n_steps, seed=seed)
    
    x_mean, x_std = x.mean(0), x.std(0) + 1e-8
    y_mean, y_std = y.mean(0), y.std(0) + 1e-8
    x_all, y_all = prep(x, x_mean, x_std), prep(y, y_mean, y_std)
    return x_all, y_all

def main():
    N_STEPS = 50_000
    x, y = prepare_data(n_steps=N_STEPS, seed=0)
    for i in range(5):
        print(f"{i}: x={x[i]}, y={y[i]}")


if __name__ == "__main__":
    main()
