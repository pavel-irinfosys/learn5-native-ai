
import torch
import torch.nn as nn

class ModelRL(nn.Module):
    def __init__(self, obs_size=4, action_size=2, dtype=torch.float16, device="cpu"):
        super().__init__()
        self.__gen_model(obs_size, action_size, dtype, device)

    def forward(self, obs):
        return self.__model(obs)

    def __gen_model(self, obs_size, action_size, dtype, device):
        self.__model = nn.Sequential(
            nn.Linear(obs_size, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
        ).to(device=device, dtype=dtype)

    