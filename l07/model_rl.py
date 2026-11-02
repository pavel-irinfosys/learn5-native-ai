import os
import torch
import torch.nn as nn
from torch.distributions import Categorical


class ModelRL(nn.Module):
    def __init__(
        self,
        obs_size=4,
        action_size=2,
        gamma=0.99,
        lr=1e-3,
        dtype=torch.float16,
        device="cpu",
        path="models/model_rl_01.pt",
    ):
        super().__init__()
        self.obs_size = obs_size
        self.action_size = action_size
        self.device = device
        self.dtype = dtype
        self.gamma = gamma
        self.path = path
        
        self.__gen_model()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)  # 2. optimizer
        self.load(self.path)


    def forward(self, obs):
        return self.__model(obs)

    def select_action(self, obs):
        """Sample an action; return (action, log_prob)."""
        dist = Categorical(logits=self(self._to_tensor(obs)))
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def run_episode(self, env):
        obs, _ = env.reset()
        log_probs, rewards = [], []
        done = False

        while not done:
            action, log_prob = self.select_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            log_probs.append(log_prob)
            rewards.append(reward)

        return log_probs, rewards

    def compute_returns(self, rewards):
        returns, G = [], 0.0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=self.dtype, device=self.device)
        return (returns - returns.mean()) / (returns.std() + 1e-8)

    def train_step(self, env):
        """Play one episode, update weights, return total reward."""
        log_probs, rewards = self.run_episode(env)
        returns = self.compute_returns(rewards)

        loss = -(torch.stack(log_probs) * returns).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return sum(rewards)

    @torch.no_grad()
    def act_greedy(self, obs):
        """Best action without sampling (for evaluation/demo)."""
        return self(self._to_tensor(obs)).argmax().item()

    def save(self, path=None):
        path = path or self.path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "model_state": self.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path=None, load_optimizer=True):
        path = path or self.path
        if not os.path.isfile(path):
            print(f"No checkpoint at '{path}', starting fresh.")
            return False

        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.load_state_dict(checkpoint["model_state"])
        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        print(f"Loaded checkpoint from '{path}'.")
        return True

    def __gen_model(self):
        self.__model = nn.Sequential(
            nn.Linear(self.obs_size, 32),
            nn.ReLU(),
            nn.Linear(32, self.action_size),
        ).to(device=self.device, dtype=self.dtype)

    def _to_tensor(self, obs):
        return torch.as_tensor(obs, dtype=self.dtype, device=self.device)
