import os
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

env = gym.make("CartPole-v1")
NP_D_TYPE = np.float32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
D_TYPE = torch.float16


def generate_data_sim(n_steps=1000, seed=0):
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)

    x = np.empty((n_steps, 5), dtype=NP_D_TYPE)
    y = np.empty((n_steps, 4), dtype=NP_D_TYPE)
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

def get_model(in_dim=5, out_dim=4, hidden=128, n_layers=3, act=nn.Tanh):
    layers, d = [], in_dim
    for _ in range(n_layers):
        layers += [nn.Linear(d, hidden), act()]
        d = hidden
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)

def train_epoch(model, Xtr, Ytr, Xva, Yva, epochs=50, lr=1e-3, batch_size=256):
    loader = DataLoader(TensorDataset(Xtr, Ytr), batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)

        model.eval()
        with torch.no_grad():
            val = loss_fn(model(Xva), Yva).item()
        print(f"epoch {epoch+1:02d}  train {total/len(Xtr):.5f}  val {val:.5f}")
    return model

def train(x_all, y_all, train_frac=0.8, **kwargs):
    n_train = int(len(x_all) * train_frac)
    
    Xtr, Ytr = x_all[:n_train], y_all[:n_train]
    Xva, Yva = x_all[n_train:], y_all[n_train:]
    return train_epoch(get_model(), Xtr, Ytr, Xva, Yva, **kwargs)


def main():
    N_STEPS = 50_000
    model = get_model()
    print(model)
    x_all, y_all = prepare_data(n_steps=N_STEPS, seed=0)
    train(x_all, y_all, epochs=50, lr=1e-3, batch_size=256)
    # for i in range(5):
    #     print(f"{i}: x={x[i]}, y={y[i]}")

if __name__ == "__main__":
    main()
