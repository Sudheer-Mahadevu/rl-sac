import numpy as np
import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RewardMLP(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256, n_layers=3):
        super().__init__()
        layers = [nn.Linear(obs_dim + act_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, obs, act):
        return self.net(torch.cat([obs, act], dim=-1))


class RewardModel:
    def __init__(self, obs_dim, act_dim, ensemble_size=3, hidden_dim=256,
                 n_layers=3, lr=3e-4):
        self.ensemble = [
            RewardMLP(obs_dim, act_dim, hidden_dim, n_layers).to(DEVICE)
            for _ in range(ensemble_size)
        ]
        self.optimizers = [
            torch.optim.Adam(m.parameters(), lr=lr) for m in self.ensemble
        ]

    def predict_batch(self, obs, act):
        obs_t = torch.FloatTensor(obs).to(DEVICE)
        act_t = torch.FloatTensor(act).to(DEVICE)
        with torch.no_grad():
            preds = torch.stack(
                [m(obs_t, act_t).squeeze(-1) for m in self.ensemble], dim=0
            )
        return preds.mean(0).cpu().numpy()

    def update(self, pref_buffer, batch_size=64, n_epochs=10):
        if len(pref_buffer) < batch_size:
            return
        for model, opt in zip(self.ensemble, self.optimizers):
            for _ in range(n_epochs):
                seg1_obs, seg1_act, seg2_obs, seg2_act, labels = pref_buffer.sample(batch_size)
                s1o = torch.FloatTensor(seg1_obs).to(DEVICE)
                s1a = torch.FloatTensor(seg1_act).to(DEVICE)
                s2o = torch.FloatTensor(seg2_obs).to(DEVICE)
                s2a = torch.FloatTensor(seg2_act).to(DEVICE)
                y   = torch.FloatTensor(labels).to(DEVICE)

                B, L, _ = s1o.shape
                r1 = model(s1o.view(B * L, -1), s1a.view(B * L, -1)).view(B, L).sum(1)
                r2 = model(s2o.view(B * L, -1), s2a.view(B * L, -1)).view(B, L).sum(1)

                logits = torch.stack([r1, r2], dim=1)
                probs  = torch.softmax(logits, dim=1)
                target = torch.stack([y, 1.0 - y], dim=1)
                loss   = -(target * torch.log(probs + 1e-8)).sum(1).mean()

                opt.zero_grad()
                loss.backward()
                opt.step()


class PreferenceBuffer:
    def __init__(self, capacity=3000):
        self.capacity = capacity
        self.buf: list = []
        self._ptr = 0

    def add(self, seg1_obs, seg1_act, seg2_obs, seg2_act, label):
        entry = (
            np.array(seg1_obs, dtype=np.float32),
            np.array(seg1_act, dtype=np.float32),
            np.array(seg2_obs, dtype=np.float32),
            np.array(seg2_act, dtype=np.float32),
            float(label),
        )
        if len(self.buf) < self.capacity:
            self.buf.append(entry)
        else:
            self.buf[self._ptr % self.capacity] = entry
        self._ptr += 1

    def sample(self, batch_size):
        idx = np.random.randint(0, len(self.buf), size=batch_size)
        s1o, s1a, s2o, s2a, labels = zip(*[self.buf[i] for i in idx])
        return (
            np.stack(s1o),
            np.stack(s1a),
            np.stack(s2o),
            np.stack(s2a),
            np.array(labels, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buf)
