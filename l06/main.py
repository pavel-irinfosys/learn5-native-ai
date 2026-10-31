import os
import time
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

NP_D_TYPE = np.float32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "models/cartpole_dynamics.pt"

def generate_data_sim(n_steps=50_000, seed=0):
    env = gym.make("CartPole-v1")
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)

    x = np.empty((n_steps, 5), dtype=NP_D_TYPE)
    y = np.empty((n_steps, 4), dtype=NP_D_TYPE)
    keep = np.ones(n_steps, dtype=bool)

    for i in range(n_steps):
        action = int(rng.integers(0, 2))
        x[i, :4], x[i, 4] = obs, action
        next_obs, _, terminated, truncated, _ = env.step(action)
        y[i] = next_obs - obs  # change in state
        if terminated or truncated:
            keep[i] = False
            obs, _ = env.reset()
        else:
            obs = next_obs
    return x[keep], y[keep]


def prep(a, m, s):
    return torch.from_numpy((a - m) / s)


def prepare_data(n_steps=50_000, seed=0, val_frac=0.2):
    x, y = generate_data_sim(n_steps=n_steps, seed=seed)

    idx = np.random.default_rng(seed).permutation(len(x))
    n_tr = int((1 - val_frac) * len(x))
    tr, va = idx[:n_tr], idx[n_tr:]

    stats = {
        "x_mean": x[tr].mean(0),
        "x_std": x[tr].std(0) + 1e-8,
        "y_mean": y[tr].mean(0),
        "y_std": y[tr].std(0) + 1e-8,
    }
    Xtr = prep(x[tr], stats["x_mean"], stats["x_std"])
    Ytr = prep(y[tr], stats["y_mean"], stats["y_std"])
    Xva = prep(x[va], stats["x_mean"], stats["x_std"])
    Yva = prep(y[va], stats["y_mean"], stats["y_std"])
    return Xtr, Ytr, Xva, Yva, stats


def get_model(in_dim=5, out_dim=4, hidden=128, n_layers=3, act=nn.Tanh):
    layers, d = [], in_dim
    for _ in range(n_layers):
        layers += [nn.Linear(d, hidden), act()]
        d = hidden
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


def train(model, Xtr, Ytr, Xva, Yva, epochs=50, lr=1e-3, batch_size=256):
    model.to(DEVICE)
    Xtr, Ytr, Xva, Yva = (t.to(DEVICE) for t in (Xtr, Ytr, Xva, Yva))
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

def save_checkpoint(model, stats, path=MODEL_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)   # torch.save fails if the folder is missing
    torch.save({"model": model.state_dict(), "stats": stats}, path)


def load_checkpoint(path=MODEL_PATH):
    # weights_only=False because stats holds NumPy arrays (it's your own file, so that's safe)
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    model = get_model()
    model.load_state_dict(ckpt["model"])
    model.to(DEVICE).eval()
    return model, ckpt["stats"]


def evaluate(model, stats, n_steps=10_000, seed=123):
    """Check the model on fresh data, normalized with the SAVED stats."""
    x, y = generate_data_sim(n_steps=n_steps, seed=seed)
    X = prep(x, stats["x_mean"], stats["x_std"]).to(DEVICE)
    Y = prep(y, stats["y_mean"], stats["y_std"]).to(DEVICE)
    with torch.no_grad():
        per_dim = ((model(X) - Y) ** 2).mean(0)
    print("per-dim MSE on fresh data:", per_dim.cpu().numpy())


# ---------------------------------------------------------------------------
# NEW: use the trained model to pick actions, and show frames
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_next_both(model, stats, obs):
    """Predict the next state for BOTH actions (0 = left, 1 = right) in one pass."""
    inp = np.stack([np.append(obs, a) for a in (0, 1)]).astype(NP_D_TYPE)   # (2, 5)
    X = prep(inp, stats["x_mean"], stats["x_std"]).to(DEVICE)
    delta = model(X).cpu().numpy() * stats["y_std"] + stats["y_mean"]       # un-normalize
    return obs + delta                                                       # (2, 4)


def action_from_model(model, stats, obs):
    """action = f(model, obs): choose the action whose predicted next state is best.

    The model maps (obs, action) -> change in state, so it cannot output an action
    by itself. We ask it "what happens if I push left? right?" and pick the action
    that leaves the pole most upright and slow.
    """
    nxt = predict_next_both(model, stats, obs)
    cost = np.abs(nxt[:, 2]) + 0.3 * np.abs(nxt[:, 3]) + 0.05 * np.abs(nxt[:, 0])
    return int(np.argmin(cost))


def run_episode(model, stats, use_model=True, seed=0, max_steps=500, record_every=10):
    """Run one episode. Returns (steps survived, list of (step, action, frame))."""
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    frames = []
    for t in range(max_steps):
        if use_model:
            action = action_from_model(model, stats, obs)
        else:
            action = int(rng.integers(0, 2))
        if t % record_every == 0:
            frames.append((t, action, env.render()))
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    env.close()
    return t + 1, frames


def show_frames(frames, n_show=8, cols=4, path="frames.png"):
    idx = np.linspace(0, len(frames) - 1, min(n_show, len(frames))).astype(int)
    rows = int(np.ceil(len(idx) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.8 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, i in zip(axes, idx):
        t, action, img = frames[i]
        ax.imshow(img)
        ax.set_title(f"step {t}   action = {'right' if action else 'left'}")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    print("saved", path)
    plt.show()


def main(retrain=False):
    if os.path.exists(MODEL_PATH) and not retrain:
        print(f"Loading model from {MODEL_PATH}")
        model, stats = load_checkpoint()
    else:
        Xtr, Ytr, Xva, Yva, stats = prepare_data(n_steps=50_000, seed=0)
        model = train(get_model(), Xtr, Ytr, Xva, Yva, epochs=50, lr=1e-3, batch_size=256)
        save_checkpoint(model, stats)

    model.eval()
    evaluate(model, stats)

    # play with actions chosen by the model, and compare with random actions
    steps_model, frames = run_episode(model, stats, use_model=True, seed=0)
    steps_random, _ = run_episode(model, stats, use_model=False, seed=0)
    print(f"survived  model-chosen actions: {steps_model} steps | random actions: {steps_random} steps")
    show_frames(frames)

if __name__ == "__main__":
    main()