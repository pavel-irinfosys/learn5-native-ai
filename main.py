

import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
D_TYPE = torch.float64

def get_model(device, dtype):
    model = nn.Sequential(
        nn.Linear(1, 32),
        nn.Tanh(),
        nn.Linear(32, 32),
        nn.Tanh(),
        nn.Linear(32, 1),
    ).to(device=device, dtype=dtype)
    
    return model

def main():
    
    print(DEVICE)
    model = get_model(DEVICE, D_TYPE)
    print(model)



if __name__ == "__main__":
    main()
