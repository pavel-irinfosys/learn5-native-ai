import torch
import torch.nn as nn
from collections import deque


class ModelRL(nn.Module):
    def __init(self, input_dim=4, out_dim=2, lr=1e-3, device="cpu", dtype=torch.float16):
        super().__init()
        self.device, self.dtype = device, dtype
        self.input_dim, self.out_dim = input_dim, out_dim
        self.__gen_model()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)  # 2. optimizer

    def forward(self, obs):
        return self.__model(obs)

    def train_ml(self, env, epochs=100):
        self.train()
        best = float("-inf")
        recent = deque(maxlen=20)
        for epoch in range(1, epochs + 1):
            total = self.__train_step(env=env)
            recent.append(total)

            if len(recent) == 20:
                avg = sum(recent) / 20
                if avg > best:
                    best = avg
                    # model.save()
                if epoch % 50 == 0:
                    print(
                        f"Episode {epoch} | last: {total:.0f} | avg20: {avg:.1f} | best: {best:.1f}"
                    )

    @torch.no_grad()
    def act_greedy(self, obs):
        """Best action without sampling (for evaluation/demo)."""
        return self(self._to_tensor(obs)).argmax().item()

    def __gen_model(self):
        self.__model = nn.Sequential(
            nn.Linear(self.input_dim, 32), nn.ReLU, nn.Linear(32, self.out_dim)
        ).to(dtype=self.dtype, device=self.device)

    def __train_step(self, env):
        return 0
