import os
import numpy as np
import gymnasium as gym

env = gym.make("CartPole-v1")
D_TYPE = np.float32


def generate_data(n_steps=1000, seed=0):
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)

    x = np.empty((n_steps, 5), dtype=D_TYPE)
    y = np.empty((n_steps, 4), dtype=D_TYPE)
    keep = np.ones(n_steps, dtype=bool)

    for i in range(n_steps):
        action = int(rng.integers(0, 2))
        x[i, :4] , x[i, 4]= obs, action
        next_obs, reward, terminated, truncated, _ = env.step(action)
        y[i, :4] = next_obs - obs  # state AFTER the action
        if terminated or truncated:
            keep[i] = False
            obs, _ = env.reset()

    return x[keep], y[keep]


def main():
    N_STEPS = 1_000
    x, y = generate_data(n_steps=N_STEPS, seed=42)
    for i in range(x.shape[0]):
        print(f"{i}: x={x[i]}, y={y[i]}")


if __name__ == "__main__":
    main()