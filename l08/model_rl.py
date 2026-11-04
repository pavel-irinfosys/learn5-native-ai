import torch
import torch.nn as nn
from torch.distributions import Categorical
from collections import deque
import numpy as np
import os


class ModelRL(nn.Module):
    def __init__(
        self,
        input_dim=4,
        out_dim=2,
        lr=1e-2,
        gamma=0.99,
        device="cpu",
        dtype=torch.float32,
        path="models/model_rl_01.pt",
    ):
        super().__init__()
        self.device, self.dtype = device, dtype
        self.input_dim, self.out_dim = input_dim, out_dim
        self.gamma = gamma
        self.path = path
        self.best = float("-inf") 
        self.__gen_model()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)  # 2. optimizer
        self.load(self.path)
        
    def forward(self, obs):
        return self.__model(obs)

    def select_action(self, obs):
        """Sample an action from the policy and return it with its log-probability."""
        logits = self(self._to_tensor(obs))
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def train_ml(self, envs, epochs=100):
        self.train()
        best = float("-inf")
        recent = deque(maxlen=20)
        for epoch in range(1, epochs + 1):
            totals = self.__train_step(envs=envs)
            mean_total = sum(totals) / len(totals)
            recent.append(mean_total)

            if len(recent) == 20:
                avg = sum(recent) / 20
                if avg > best:
                    best = avg
                    self.save()
                if epoch % 50 == 0:
                    print(
                        f"Episode {epoch} | last: {mean_total:.0f} | avg20: {avg:.1f} | best: {best:.1f}"
                    )

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
                "best": self.best,
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
        self.best = checkpoint.get("best", float("-inf"))
        print(f"Loaded checkpoint from '{path}' (best: {self.best:.1f}).")
        return True

    def __gen_model(self):
        self.__model = nn.Sequential(
            nn.Linear(self.input_dim, 64), nn.ReLU(), nn.Linear(64, self.out_dim)
        ).to(dtype=self.dtype, device=self.device)

    def _to_tensor(self, obs):
        return torch.as_tensor(obs, dtype=self.dtype, device=self.device)

    def __train_step(self, envs):
        """One REINFORCE update from one episode per env. Returns each env's total reward."""
        log_probs, rewards = self._run_episodes(envs)

        # flatten all episodes into one batch
        all_lp = torch.stack([lp for ep in log_probs for lp in ep])
        all_ret = torch.tensor(
            [g for ep in rewards for g in self._discounted_returns(ep)],
            dtype=self.dtype,
            device=self.device,
        )
        all_ret = (all_ret - all_ret.mean()) / (all_ret.std() + 1e-8)

        loss = -(all_lp * all_ret).sum() / len(envs)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return [sum(r) for r in rewards]

    def _discounted_returns(self, rewards):
        returns, g = [], 0.0
        for r in reversed(rewards):
            g = r + self.gamma * g
            returns.append(g)
        returns.reverse()
        return returns

    def _run_episodes(self, envs):
        """Run one full episode in every env at the same time (batched forward pass)."""
        n = len(envs)
        obs = np.stack([env.reset()[0] for env in envs])
        log_probs = [[] for _ in range(n)]
        rewards = [[] for _ in range(n)]
        active = list(range(n))  # indices of envs whose episode is not finished

        while active:
            batch = self._to_tensor(obs[active])  # shape (k, 4)
            dist = Categorical(
                logits=self(batch)
            )  # one forward pass for all active envs
            actions = dist.sample()
            lps = dist.log_prob(actions)

            still_active = []
            for j, i in enumerate(active):
                # print(f"{active} : {j} {i}")
                o, reward, terminated, truncated, _ = envs[i].step(actions[j].item())
                obs[i] = o
                log_probs[i].append(lps[j])
                rewards[i].append(reward)
                if not (terminated or truncated):
                    still_active.append(i)
            active = still_active

        return log_probs, rewards
