
import torch
import torch.nn as nn
import time
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
D_TYPE = torch.float16

def gen_data(n, dim, device, dType):
    high = 1
    low = -1
    return torch.rand(n, dim, device=device, dtype=dType) * (high - low) + low

def gen_out(x, device, dType):
    x = x.to(device=device, dtype=dType)
    # sign(x) * |x|^0.33 keeps the term real for negative x
    root_term = torch.sign(x) * torch.abs(x).pow(0.33)
    return 7.3 * root_term + 0.3 * x.pow(2) + 0.3


def get_model(dim, device, dtype):
    hidden_net = 32
    
    model = nn.Sequential(
        nn.Linear(dim, hidden_net),
        nn.Tanh(),
        nn.Linear(hidden_net, hidden_net),
        nn.Tanh(),
        nn.Linear(hidden_net, hidden_net),
        nn.Tanh(),
        nn.Linear(hidden_net, hidden_net),
        nn.Tanh(),
        nn.Linear(hidden_net, hidden_net),
        nn.Tanh(),
        nn.Linear(hidden_net, dim),
    ).to(device=device, dtype=dtype)
    model = torch.compile(model, mode="reduce-overhead")
    return model

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

def train(x_all, y_all, k, folds, model, loss_fn, optimizer):

    for i in range(k):
        val_idx = folds[i]
        train_idx = torch.cat([f for j, f in enumerate(folds) if j != i])
        x_train, y_train = x_all[train_idx], y_all[train_idx]
        x_val,   y_val   = x_all[val_idx],   y_all[val_idx]

        train_loss = learn(
            model=model, 
            x=x_train, target=y_train, 
            loss_fn=loss_fn, optimizer=optimizer, 
            epochs=100
        )
        if i % 100:
            with torch.no_grad():
                print(f"elapsed: {i} :: train_loss: {train_loss}")

                
def save_model(model, path, dim, hidden_net=32):
    raw = getattr(model, "_orig_mod", model)   # unwrap torch.compile if present
    torch.save({
        "state_dict": raw.state_dict(),
        "dim": dim,
        "hidden_net": hidden_net,
    }, path)

def load_model(path, device, dtype):
    ckpt = torch.load(path, map_location=device)
    model = get_model(ckpt["dim"], device, dtype)   # rebuilds the architecture
    raw = getattr(model, "_orig_mod", model)        # unwrap torch.compile
    raw.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def get_or_load_model(path, dim, device, dtype):
    if os.path.exists(path):
        print(f"loading {path}")
        return load_model(path, device, dtype)
    print("no checkpoint found, building new model")
    return get_model(dim, device, dtype)

def main():
    n = 100_000
    dim = 7
    k = 20
    x_all = gen_data(n, dim, device=DEVICE, dType=D_TYPE)
    y_all = gen_out(x_all, device=DEVICE, dType=D_TYPE)
    model = get_or_load_model("models/model.pt", dim=dim, device=DEVICE, dtype=D_TYPE)
    folds = gen_fold(n, k, DEVICE)
    loss_fn = get_loss()
    optimizer = get_optimizer(model)
    start = time.perf_counter()
    train(x_all=x_all, y_all=y_all, k=k, folds=folds, loss_fn=loss_fn, optimizer=optimizer, model=model)

    eta = time.perf_counter() - start
    with torch.no_grad():
        val_loss = loss_fn(model(x_all[folds[0]]), y_all[folds[0]]).item()
        print(f"eta: {eta} mean val loss:", val_loss)

    save_model(model, "models/model.pt", dim=dim)


    

if __name__ == "__main__":
    main()

