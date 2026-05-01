import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd
import math

LOG_STD_MIN, LOG_STD_MAX = -5, 2


def weight_init(m):
    """pytorch_sac-style orthogonal init."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


def mlp(in_dim, hidden_dim, out_dim, hidden_depth=2):
    layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    for _ in range(hidden_depth - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    layers.append(nn.Linear(hidden_dim, out_dim))
    net = nn.Sequential(*layers)
    net.apply(weight_init)
    return net


# ── Squashed Normal (pytorch_sac TanhTransform) ──────────────────────────────

class TanhTransform(pyd.transforms.Transform):
    domain   = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def _call(self, x):    return x.tanh()
    def _inverse(self, y): return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # numerically stable version from TFP
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc   = loc
        self.scale = scale
        self.base_dist = pyd.Normal(loc, scale)
        super().__init__(self.base_dist, [TanhTransform()])

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu


# ── Actor ────────────────────────────────────────────────────────────────────

class DiagGaussianActor(nn.Module):
    """Squashed Gaussian actor (pytorch_sac DiagGaussianActor).
    
    Uses soft tanh rescaling for log_std (NOT hard clamp) so gradients
    flow through the boundary — matches the reference implementation.
    """

    def __init__(self, obs_dim, act_dim, hidden_dim=256, hidden_depth=2):
        super().__init__()
        # single trunk outputs [mu | log_std]  (2 * act_dim outputs)
        self.trunk = mlp(obs_dim, hidden_dim, 2 * act_dim, hidden_depth)

    def forward(self, obs) -> SquashedNormal:
        mu, log_std = self.trunk(obs).chunk(2, dim=-1)
        # soft tanh rescaling — gradient-friendly, matches pytorch_sac exactly
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        std = log_std.exp()
        return SquashedNormal(mu, std)


# ── Double-Q Critic ──────────────────────────────────────────────────────────

class DoubleQCritic(nn.Module):
    """Clipped double Q-critic (pytorch_sac DoubleQCritic)."""

    def __init__(self, obs_dim, act_dim, hidden_dim=256, hidden_depth=2):
        super().__init__()
        self.Q1 = mlp(obs_dim + act_dim, hidden_dim, 1, hidden_depth)
        self.Q2 = mlp(obs_dim + act_dim, hidden_dim, 1, hidden_depth)

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.Q1(x), self.Q2(x)