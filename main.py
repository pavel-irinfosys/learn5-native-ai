

import torch
import torch.nn as nn
import time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
D_TYPE = torch.float32

def get_model(device, dtype):
    model = nn.Sequential(
        nn.Linear(1, 5),
        nn.Tanh(),
        # nn.Linear(32, 32),
        # nn.Tanh(),
        nn.Linear(5, 1),
    ).to(device=device, dtype=dtype)
    model = torch.compile(model, mode="reduce-overhead")
    return model

def gen_out(x_seq, device, dtype):
    # x_seq = torch.linspace(0, 1, 100, device=DEVICE, dtype=D_TYPE).unsqueeze(1)
    y = (3 * (x_seq ** 2)) - (2 * x_seq) + 0
    return y

def gen_data(n, device, dtype):
    return torch.rand(n, 1, device=DEVICE, dtype=D_TYPE)

def gen_fold(n, k, device):
    idx = torch.randperm(n, device=device)
    folds = idx.chunk(k)
    return folds

def get_optimizer(model, lr=1e-2):
    return torch.optim.Adam(model.parameters(), lr=lr, fused=True)

def get_loss():
    return nn.MSELoss()

def learn(model, x, target, loss_fn, optimizer, epochs=2000):
    for epoch in range(epochs):
        optimizer.zero_grad()          # clear old gradients
        pred = model(x)                # forward pass
        loss = loss_fn(pred, target)   # compute loss
        loss.backward()                # compute gradients
        optimizer.step()               # update weights
 
        # if epoch % 200 == 0:
        #     print(f"epoch {epoch:5d}  loss {loss.item():.3e}")
 
    return loss.item()

def main():
    n = 100_000
    k = 50
    print(DEVICE)
    model = get_model(DEVICE, D_TYPE)
    x_all = gen_data(n, device=DEVICE, dtype=D_TYPE)
    y_all = gen_out(x_all, device=DEVICE, dtype=D_TYPE)
    folds = gen_fold(n, k, DEVICE)
    val_losses = []

    loss_fn = get_loss()
    optimizer = get_optimizer(model)
    start = time.perf_counter()
    for i in range(k):
        val_idx = folds[i]
        train_idx = torch.cat([f for j, f in enumerate(folds) if j != i])
        x_train, y_train = x_all[train_idx], y_all[train_idx]
        x_val,   y_val   = x_all[val_idx],   y_all[val_idx]

        train_loss = learn(model=model, x=x_train, target=y_train, loss_fn=loss_fn, optimizer=optimizer, epochs=100)
        if i % 100:
            with torch.no_grad():
                elapsed = time.perf_counter() - start
                print(f"elapsed: {elapsed} :: train_loss: {train_loss}")

        #     val_loss = loss_fn(model(x_val), y_val).item()
        #     val_losses.append(val_loss)
        #     print(f"fold {i:2d}  train {train_loss:.3e}  val {val_loss:.3e}")
        #     print(f"   pred  :{model(x_val).squeeze().tolist()}")
        #     print(f"   target:{y_val.squeeze().tolist()}")
    eta = time.perf_counter() - start
    with torch.no_grad():
        val_loss = loss_fn(model(x_all[folds[0]]), y_all[folds[0]]).item()
        print(f"eta: {eta} mean val loss:", val_loss)


if __name__ == "__main__":
    main()
