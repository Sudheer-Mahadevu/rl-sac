import torch
import torch.nn.functional as F
import numpy as np
from networks import DiagGaussianActor, DoubleQCritic
from buffer import ReplayBuffer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def soft_update(target, source, tau):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)


class SACAgent:
    """
    SAC with:
      • clipped double Q-learning
      • squashed Gaussian actor (tanh, reparameterization)
      • pytorch_sac-style soft log_std rescaling
      • shared critic optimizer  (matches reference)
      • auto OR manual temperature
      • 10 K random-action warmup
    """

    def __init__(
        self,
        obs_dim,
        act_dim,
        hidden_dim      = 256,
        hidden_depth    = 2,
        actor_lr        = 3e-4,
        critic_lr       = 3e-4,
        alpha_lr        = 3e-4,
        gamma           = 0.99,
        tau             = 0.005,
        batch_size      = 256,
        buffer_size     = int(1e6),
        random_steps    = 10_000,
        # actor updated every step; critic target updated every step
        actor_update_freq          = 1,
        critic_target_update_freq  = 2,
        auto_alpha      = True,
        init_alpha      = 0.2,
    ):
        self.obs_dim     = obs_dim
        self.act_dim     = act_dim
        self.gamma       = gamma
        self.tau         = tau
        self.batch_size  = batch_size
        self.random_steps = random_steps
        self.actor_update_freq         = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        self.total_steps = 0

        # ── Networks ──────────────────────────────────────────────────────────
        self.actor        = DiagGaussianActor(obs_dim, act_dim, hidden_dim, hidden_depth).to(DEVICE)
        self.critic       = DoubleQCritic(obs_dim, act_dim, hidden_dim, hidden_depth).to(DEVICE)
        self.critic_tgt   = DoubleQCritic(obs_dim, act_dim, hidden_dim, hidden_depth).to(DEVICE)
        self.critic_tgt.load_state_dict(self.critic.state_dict())

        # ── Optimisers ────────────────────────────────────────────────────────
        # single critic optimizer (matches pytorch_sac)
        self.actor_opt  = torch.optim.Adam(self.actor.parameters(),  lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        # ── Temperature ───────────────────────────────────────────────────────
        self.auto_alpha = auto_alpha
        self.log_alpha  = torch.tensor(np.log(init_alpha),
                                       requires_grad=True, device=DEVICE, dtype=torch.float32)
        if auto_alpha:
            self.target_entropy = -float(act_dim)   # heuristic: -|A|
            self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

        # ── Buffer ────────────────────────────────────────────────────────────
        self.buffer = ReplayBuffer(obs_dim, act_dim, buffer_size)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    # ── Interaction ───────────────────────────────────────────────────────────

    def select_action(self, obs, evaluate=False):
        """Uniform random during warmup; stochastic during training; deterministic for eval."""
        if not evaluate and self.total_steps < self.random_steps:
            return np.random.uniform(-1, 1, size=(self.act_dim,)).astype(np.float32)
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            dist = self.actor(obs_t)
            action = dist.mean if evaluate else dist.sample()
        return action.cpu().numpy().flatten()

    def store(self, obs, next_obs, action, reward, terminated):
        self.buffer.add(obs, next_obs, action, reward, float(terminated))
        self.total_steps += 1

    # ── Update ────────────────────────────────────────────────────────────────

    def update(self):
        if len(self.buffer) < self.batch_size or self.total_steps < self.random_steps:
            return

        obs, next_obs, actions, rewards, dones = self.buffer.sample(self.batch_size)
        not_done = 1.0 - dones

        # ── Critic update ─────────────────────────────────────────────────────
        with torch.no_grad():
            dist        = self.actor(next_obs)
            next_action = dist.rsample()
            log_prob    = dist.log_prob(next_action).sum(-1, keepdim=True)
            q1_tgt, q2_tgt = self.critic_tgt(next_obs, next_action)
            target_V    = torch.min(q1_tgt, q2_tgt) - self.alpha.detach() * log_prob
            target_Q    = rewards + not_done * self.gamma * target_V

        current_Q1, current_Q2 = self.critic(obs, actions)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # ── Actor update ──────────────────────────────────────────────────────
        if self.total_steps % self.actor_update_freq == 0:
            dist       = self.actor(obs)
            action     = dist.rsample()
            log_prob   = dist.log_prob(action).sum(-1, keepdim=True)
            q1_pi, q2_pi = self.critic(obs, action)
            actor_loss = (self.alpha.detach() * log_prob - torch.min(q1_pi, q2_pi)).mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            # ── Auto-alpha update ─────────────────────────────────────────────
            if self.auto_alpha:
                # pytorch_sac reference form: α·(-log π - H_target)
                alpha_loss = (self.alpha * (-log_prob.detach() - self.target_entropy)).mean()
                self.alpha_opt.zero_grad()
                alpha_loss.backward()
                self.alpha_opt.step()

        # ── Soft target update ────────────────────────────────────────────────
        if self.total_steps % self.critic_target_update_freq == 0:
            soft_update(self.critic_tgt, self.critic, self.tau)