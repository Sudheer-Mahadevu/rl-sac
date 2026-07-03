import torch
import torch.nn.functional as F
import numpy as np
from networks import GaussianActor, DoubleQCritic
from buffer import ReplayBuffer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def soft_update(target, source, tau):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * sp.data + (1 - tau) * tp.data)


class SACContinuous:
    def __init__(
        self,
        obs_dim,
        act_dim,
        hidden_dim                     = 256,
        hidden_depth                   = 2,
        lr                             = 3e-4,
        adam_betas                     = (0.9, 0.999),
        gamma                          = 0.99,
        tau                            = 0.005,
        batch_size                     = 256,
        buffer_size                    = int(1e6),
        random_steps                   = 10_000,
        actor_update_frequency         = 2,
        critic_target_update_frequency = 2,
        auto_alpha                     = True,
        init_alpha                     = 0.1,
    ):
        self.obs_dim      = obs_dim
        self.act_dim      = act_dim
        self.gamma        = gamma
        self.tau          = tau
        self.batch_size   = batch_size
        self.random_steps = random_steps
        self.total_steps  = 0
        self.update_steps = 0

        self.actor_update_frequency         = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency

        self.actor      = GaussianActor(obs_dim, act_dim, hidden_dim, hidden_depth).to(DEVICE)
        self.critic     = DoubleQCritic(obs_dim, act_dim, hidden_dim, hidden_depth).to(DEVICE)
        self.critic_tgt = DoubleQCritic(obs_dim, act_dim, hidden_dim, hidden_depth).to(DEVICE)
        self.critic_tgt.load_state_dict(self.critic.state_dict())

        self.actor_opt  = torch.optim.Adam(self.actor.parameters(),  lr=lr, betas=adam_betas)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr, betas=adam_betas)

        self.auto_alpha = auto_alpha
        if auto_alpha:
            self.target_entropy = -float(act_dim)
            self.log_alpha      = torch.tensor(np.log(init_alpha), requires_grad=True, device=DEVICE)
            self.alpha_opt      = torch.optim.Adam([self.log_alpha], lr=lr, betas=adam_betas)
        self.alpha = init_alpha

        self.buffer = ReplayBuffer(obs_dim, act_dim, buffer_size)

    def select_action(self, obs, evaluate=False):
        if not evaluate and self.total_steps < self.random_steps:
            return np.random.uniform(-1, 1, size=(self.act_dim,)).astype(np.float32)
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            action = self.actor.deterministic(obs_t) if evaluate else self.actor.sample(obs_t)[0]
        return action.cpu().numpy().flatten()

    def store(self, obs, next_obs, action, reward, terminated, truncated):
        self.buffer.add(obs, next_obs, action, reward, terminated, truncated)
        self.total_steps += 1

    def update(self):
        if len(self.buffer) < self.batch_size or self.total_steps < self.random_steps:
            return

        obs, next_obs, actions, rewards, _, not_dones_no_max = self.buffer.sample(self.batch_size)
        obs              = torch.FloatTensor(obs).to(DEVICE)
        next_obs         = torch.FloatTensor(next_obs).to(DEVICE)
        actions          = torch.FloatTensor(actions).to(DEVICE)
        rewards          = torch.FloatTensor(rewards).to(DEVICE)
        not_dones_no_max = torch.FloatTensor(not_dones_no_max).to(DEVICE)

        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_obs)
            q1_next, q2_next = self.critic_tgt(next_obs, next_action)
            q_next   = torch.min(q1_next, q2_next) - self.alpha * next_log_prob
            q_target = rewards + self.gamma * not_dones_no_max * q_next

        q1, q2 = self.critic(obs, actions)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        self.update_steps += 1

        if self.update_steps % self.actor_update_frequency == 0:
            action, log_prob = self.actor.sample(obs)
            q1_pi, q2_pi    = self.critic(obs, action)
            actor_loss = (self.alpha * log_prob - torch.min(q1_pi, q2_pi)).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            if self.auto_alpha:
                alpha_loss = (self.log_alpha.exp() *
                              (-log_prob - self.target_entropy).detach()).mean()
                self.alpha_opt.zero_grad()
                alpha_loss.backward()
                self.alpha_opt.step()
                self.alpha = self.log_alpha.exp().item()

        if self.update_steps % self.critic_target_update_frequency == 0:
            soft_update(self.critic_tgt, self.critic, self.tau)
