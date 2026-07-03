import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd


def mlp(input_dim, hidden_dim, output_dim, hidden_depth):
    if hidden_depth == 0:
        return nn.Sequential(nn.Linear(input_dim, output_dim))
    layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
    for _ in range(hidden_depth - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


class TanhTransform(pyd.transforms.Transform):
    domain    = pyd.constraints.real
    codomain  = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign      = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        return 0.5 * (y.log1p() - (-y).log1p())

    def log_abs_det_jacobian(self, x, y):
        return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))


class SquashedNormal(pyd.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc   = loc
        self.scale = scale
        super().__init__(pyd.Normal(loc, scale), [TanhTransform()])

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu


LOG_STD_MIN = -5
LOG_STD_MAX =  2


class GaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256, hidden_depth=2):
        super().__init__()
        self.trunk = mlp(obs_dim, hidden_dim, 2 * act_dim, hidden_depth)
        self.apply(weight_init)

    def forward(self, obs):
        mu, log_std = self.trunk(obs).chunk(2, dim=-1)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return SquashedNormal(mu, log_std.exp())

    def sample(self, obs):
        dist     = self(obs)
        action   = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        return action, log_prob

    def deterministic(self, obs):
        return self(obs).mean


class DoubleQCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256, hidden_depth=2):
        super().__init__()
        self.Q1 = mlp(obs_dim + act_dim, hidden_dim, 1, hidden_depth)
        self.Q2 = mlp(obs_dim + act_dim, hidden_dim, 1, hidden_depth)
        self.apply(weight_init)

    def forward(self, obs, action):
        sa = torch.cat([obs, action], dim=-1)
        return self.Q1(sa), self.Q2(sa)


class CategoricalActor(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_dim=256, hidden_depth=2):
        super().__init__()
        self.net = mlp(obs_dim, hidden_dim, n_actions, hidden_depth)
        self.apply(weight_init)

    def forward(self, obs):
        return F.softmax(self.net(obs), dim=-1)

    def sample(self, obs):
        probs  = self(obs)
        action = pyd.Categorical(probs).sample()
        return action, probs

    def log_probs(self, obs):
        probs = self(obs)
        return probs, torch.log(probs + 1e-8)


class DiscreteQNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_dim=256, hidden_depth=2):
        super().__init__()
        self.net = mlp(obs_dim, hidden_dim, n_actions, hidden_depth)
        self.apply(weight_init)

    def forward(self, obs):
        return self.net(obs)


class DQNNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_dim=256, hidden_depth=2):
        super().__init__()
        self.net = mlp(obs_dim, hidden_dim, n_actions, hidden_depth)
        self.apply(weight_init)

    def forward(self, obs):
        return self.net(obs)
