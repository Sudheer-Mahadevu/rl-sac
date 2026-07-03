import os
import math
import json
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd
import matplotlib.pyplot as plt

os.makedirs("logs",    exist_ok=True)
os.makedirs("plots",   exist_ok=True)
os.makedirs("weights", exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

REWARDS     = ["Ra", "Rb", "Rc"]
ALL_SEEDS   = list(range(15))
TOTAL_STEPS = 500_000
EVAL_FREQ   = 10_000
EVAL_EPS    = 20
COLORS      = {"Ra": "#1f77b4", "Rb": "#ff7f0e", "Rc": "#2ca02c"}
BAR_EPISODES = 500
BAR_EP_LEN   = 5000

NEAR_ZERO_VEL_THRESH = 0.05
RC_TIMEOUT           = 1000
RC_RESET_PENALTY     = -20
LOG_STD_MIN, LOG_STD_MAX = -5, 2


# ── Networks ──────────────────────────────────────────────────────────────────

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

    def _call(self, x):    return x.tanh()
    def _inverse(self, y): return 0.5 * (y.log1p() - (-y).log1p())

    def log_abs_det_jacobian(self, x, y):
        return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))


class SquashedNormal(pyd.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc, self.scale = loc, scale
        super().__init__(pyd.Normal(loc, scale), [TanhTransform()])

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu


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


# ── Replay Buffer ─────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr  = 0
        self.size = 0
        self.obs              = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.next_obs         = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.actions          = np.zeros((max_size, act_dim), dtype=np.float32)
        self.rewards          = np.zeros((max_size, 1),       dtype=np.float32)
        self.not_dones        = np.zeros((max_size, 1),       dtype=np.float32)
        self.not_dones_no_max = np.zeros((max_size, 1),       dtype=np.float32)

    def add(self, obs, next_obs, action, reward, terminated, truncated):
        self.obs[self.ptr]              = obs
        self.next_obs[self.ptr]         = next_obs
        self.actions[self.ptr]          = action
        self.rewards[self.ptr]          = reward
        self.not_dones[self.ptr]        = float(not (terminated or truncated))
        self.not_dones_no_max[self.ptr] = float(not terminated)
        self.ptr  = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (self.obs[idx], self.next_obs[idx], self.actions[idx],
                self.rewards[idx], self.not_dones[idx], self.not_dones_no_max[idx])

    def __len__(self):
        return self.size


# ── SAC ───────────────────────────────────────────────────────────────────────

def soft_update(target, source, tau):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * sp.data + (1 - tau) * tp.data)


class SACContinuous:
    def __init__(
        self, obs_dim, act_dim,
        hidden_dim=256, hidden_depth=2,
        lr=3e-4, adam_betas=(0.9, 0.999),
        gamma=0.99, tau=0.005,
        batch_size=256, buffer_size=int(1e6),
        random_steps=10_000,
        actor_update_frequency=2,
        critic_target_update_frequency=2,
        auto_alpha=True, init_alpha=0.1,
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
            self.log_alpha = torch.tensor(np.log(init_alpha), requires_grad=True, device=DEVICE)
            self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr, betas=adam_betas)
        self.alpha = init_alpha

        self.buffer = ReplayBuffer(obs_dim, act_dim, buffer_size)

    def select_action(self, obs, evaluate=False):
        if not evaluate and self.total_steps < self.random_steps:
            return np.random.uniform(-1, 1, size=(self.act_dim,)).astype(np.float32)
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            action = (self.actor.deterministic(obs_t) if evaluate
                      else self.actor.sample(obs_t)[0])
        return action.cpu().numpy().flatten()

    def store(self, obs, next_obs, action, reward, terminated, truncated):
        self.buffer.add(obs, next_obs, action, reward, terminated, truncated)
        self.total_steps += 1

    def update(self):
        if len(self.buffer) < self.batch_size or self.total_steps < self.random_steps:
            return

        obs, next_obs, actions, rewards, _, not_dones_no_max = \
            self.buffer.sample(self.batch_size)

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

        q1, q2      = self.critic(obs, actions)
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


# ── Reacher Environments ──────────────────────────────────────────────────────

from dm_control import suite


def _flatten_obs(obs_dict):
    return np.concatenate([v.ravel() for v in obs_dict.values()]).astype(np.float32)


def _in_target(physics):
    radii = physics.named.model.geom_size[["target", "finger"], 0].sum()
    return physics.finger_to_target_dist() <= radii


class ReacherEnvRa:
    MAX_STEPS = 1000

    def __init__(self, seed=0):
        self._reload(seed)

    def _reload(self, seed):
        self._env  = suite.load("reacher", "easy", task_kwargs={"random": seed})
        self._phys = self._env.physics
        ts = self._env.reset()
        self.obs_dim = _flatten_obs(ts.observation).shape[0]
        self.act_dim = self._env.action_spec().shape[0]
        self._step   = 0

    def reset(self, seed=None):
        if seed is not None:
            self._reload(seed)
        ts = self._env.reset()
        self._step = 0
        return _flatten_obs(ts.observation), {}

    def step(self, action):
        ts        = self._env.step(action)
        obs       = _flatten_obs(ts.observation)
        self._step += 1
        reward    = 1.0 if _in_target(self._phys) else \
                    -float(self._phys.finger_to_target_dist()) - float(np.sum(np.square(action)))
        truncated = self._step >= self.MAX_STEPS
        return obs, reward, False, truncated, {}

    def close(self): pass


class ReacherEnvRb:
    MAX_STEPS = 1000

    def __init__(self, seed=0):
        self._reload(seed)

    def _reload(self, seed):
        self._env  = suite.load("reacher", "easy", task_kwargs={"random": seed})
        self._phys = self._env.physics
        ts = self._env.reset()
        self.obs_dim = _flatten_obs(ts.observation).shape[0]
        self.act_dim = self._env.action_spec().shape[0]
        self._step   = 0

    def reset(self, seed=None):
        if seed is not None:
            self._reload(seed)
        ts = self._env.reset()
        self._step = 0
        return _flatten_obs(ts.observation), {}

    def step(self, action):
        ts        = self._env.step(action)
        obs       = _flatten_obs(ts.observation)
        self._step += 1
        reward    = 1.0 if _in_target(self._phys) else 0.0
        truncated = self._step >= self.MAX_STEPS
        return obs, reward, False, truncated, {}

    def close(self): pass


class ReacherEnvRc:
    def __init__(self, seed=0):
        self._reload(seed)

    def _reload(self, seed):
        self._env  = suite.load("reacher", "easy", task_kwargs={"random": seed})
        self._phys = self._env.physics
        ts = self._env.reset()
        self.obs_dim          = _flatten_obs(ts.observation).shape[0]
        self.act_dim          = self._env.action_spec().shape[0]
        self._step_in_segment = 0

    def reset(self, seed=None):
        if seed is not None:
            self._reload(seed)
        ts = self._env.reset()
        self._step_in_segment = 0
        return _flatten_obs(ts.observation), {}

    def reset_arm(self):
        target_pos = self._phys.named.data.geom_xpos["target"].copy()
        self._env.reset()
        try:
            self._phys.named.data.qpos["target_x"] = target_pos[0]
            self._phys.named.data.qpos["target_y"] = target_pos[1]
            self._phys.forward()
        except Exception:
            pass
        self._step_in_segment = 0
        return _flatten_obs(self._env.task.get_observation(self._phys))

    def step(self, action):
        ts   = self._env.step(action)
        obs  = _flatten_obs(ts.observation)
        self._step_in_segment += 1

        in_tgt   = _in_target(self._phys)
        near_vel = float(np.linalg.norm(self._phys.velocity())) < NEAR_ZERO_VEL_THRESH

        if in_tgt and near_vel:
            return obs, -1.0, True, False, {"timeout": False}
        elif self._step_in_segment >= RC_TIMEOUT:
            return obs, -1.0 + RC_RESET_PENALTY, False, False, {"timeout": True}
        return obs, -1.0, False, False, {"timeout": False}

    def close(self): pass


class ReacherEnvRcEval:
    def __init__(self, seed=0):
        self._env  = suite.load("reacher", "easy", task_kwargs={"random": seed})
        self._phys = self._env.physics
        ts = self._env.reset()
        self.obs_dim = _flatten_obs(ts.observation).shape[0]
        self.act_dim = self._env.action_spec().shape[0]
        self._step   = 0

    def reset(self, seed=None):
        if seed is not None:
            self._env  = suite.load("reacher", "easy", task_kwargs={"random": seed})
            self._phys = self._env.physics
        ts = self._env.reset()
        self._step = 0
        return _flatten_obs(ts.observation), {}

    def step(self, action):
        ts   = self._env.step(action)
        obs  = _flatten_obs(ts.observation)
        self._step += 1

        in_tgt   = _in_target(self._phys)
        near_vel = float(np.linalg.norm(self._phys.velocity())) < NEAR_ZERO_VEL_THRESH

        if in_tgt and near_vel:
            return obs, -1.0, True, False, {}
        elif self._step >= RC_TIMEOUT:
            return obs, -1.0 + RC_RESET_PENALTY, False, True, {}
        return obs, -1.0, False, False, {}

    def close(self): pass


def make_reacher(reward_name, seed=0, eval_mode=False):
    if reward_name == "Ra": return ReacherEnvRa(seed=seed)
    if reward_name == "Rb": return ReacherEnvRb(seed=seed)
    if reward_name == "Rc": return ReacherEnvRcEval(seed=seed) if eval_mode else ReacherEnvRc(seed=seed)
    raise ValueError(f"Unknown reward: {reward_name}")


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_single(agent, eval_env, n_episodes=20):
    returns = []
    for _ in range(n_episodes):
        obs, _ = eval_env.reset()
        done   = False
        total  = 0.0
        while not done:
            action = agent.select_action(obs, evaluate=True)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            total += reward
            done   = terminated or truncated
        returns.append(total)
    return returns


def evaluate_all(agent, eval_envs, n_episodes=20):
    results = {}
    for name, env in eval_envs.items():
        rets = evaluate_single(agent, env, n_episodes)
        results[name] = (float(np.mean(rets)), float(np.std(rets)))
    return results


# ── Training Loops ────────────────────────────────────────────────────────────

def train_ra_rb(agent, train_env, eval_envs, total_steps, eval_freq, eval_eps):
    log = []
    results = evaluate_all(agent, eval_envs, eval_eps)
    log.append((0, results))
    print("[Eval t=      0]  " + "  ".join(f"{k}:{v[0]:.2f}" for k, v in results.items()))

    obs, _    = train_env.reset()
    ep_reward = 0.0
    ep_count  = 0

    for t in range(1, total_steps + 1):
        action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, _ = train_env.step(action)
        done = terminated or truncated

        agent.store(obs, next_obs, action, reward, terminated, truncated)
        agent.update()

        obs        = next_obs
        ep_reward += reward

        if done:
            ep_count += 1
            if ep_count % 50 == 0:
                print(f"  step={t:>7}  ep={ep_count}  ep_return={ep_reward:.2f}")
            obs, _    = train_env.reset()
            ep_reward = 0.0

        if t % eval_freq == 0:
            results = evaluate_all(agent, eval_envs, eval_eps)
            log.append((t, results))
            print(f"[Eval t={t:>7}]  " + "  ".join(f"{k}:{v[0]:.2f}" for k, v in results.items()))

    return log


def train_rc(agent, train_env, eval_envs, total_steps, eval_freq, eval_eps):
    log        = []
    ep_lengths = []

    results = evaluate_all(agent, eval_envs, eval_eps)
    log.append((0, results))
    print("[Eval t=      0]  " + "  ".join(f"{k}:{v[0]:.2f}" for k, v in results.items()))

    obs, _              = train_env.reset()
    ep_reward, ep_len   = 0.0, 0
    ep_count            = 0

    for t in range(1, total_steps + 1):
        action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, info = train_env.step(action)

        ep_reward += reward
        ep_len    += 1

        agent.store(obs, next_obs, action, reward, terminated, False)
        agent.update()

        if info.get("timeout", False):
            obs = train_env.reset_arm()
        elif terminated:
            ep_count += 1
            ep_lengths.append(ep_len)
            if ep_count % 20 == 0:
                print(f"  step={t:>7}  ep={ep_count}  ep_return={ep_reward:.0f}  ep_len={ep_len}")
            obs, _    = train_env.reset()
            ep_reward = 0.0
            ep_len    = 0
        else:
            obs = next_obs

        if t % eval_freq == 0:
            results = evaluate_all(agent, eval_envs, eval_eps)
            log.append((t, results))
            print(f"[Eval t={t:>7}]  " + "  ".join(f"{k}:{v[0]:.2f}" for k, v in results.items()))

    last_50 = ep_lengths[-50:] if len(ep_lengths) >= 50 else ep_lengths
    return log, last_50


# ── JSON Utilities ────────────────────────────────────────────────────────────

def save_seed_json(reward_name, seed, log, extra=None):
    records = []
    for t, res in log:
        row = {"t": t}
        for rname, (mean, std) in res.items():
            row[rname] = {"mean": mean, "std": std}
        records.append(row)
    payload = {"reward": reward_name, "seed": seed,
               "total_steps": TOTAL_STEPS, "log": records}
    if extra:
        payload["extra"] = extra
    fname = f"logs/reacher_{reward_name}_seed{seed}.json"
    with open(fname, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Saved {fname}")


def load_all_jsons(reward_name):
    files = sorted(glob.glob(f"logs/reacher_{reward_name}_seed*.json"))
    if not files:
        print(f"  No files found for SAC-{reward_name}")
        return []
    all_logs = []
    for fpath in files:
        with open(fpath) as f:
            data = json.load(f)
        log = []
        for row in data["log"]:
            t   = row["t"]
            res = {k: (v["mean"], v["std"]) for k, v in row.items() if k != "t"}
            log.append((t, res))
        all_logs.append({"seed": data["seed"], "log": log, "extra": data.get("extra", {})})
    all_logs.sort(key=lambda x: x["seed"])
    print(f"  Loaded {len(all_logs)} seeds for SAC-{reward_name}: {[x['seed'] for x in all_logs]}")
    return all_logs


# ── Plotting ──────────────────────────────────────────────────────────────────

def aggregate(seed_logs, reward_key):
    ts    = np.array([t for t, _ in seed_logs[0]["log"]])
    means = np.array([[res[reward_key][0] for _, res in sl["log"]] for sl in seed_logs])
    gm    = means.mean(axis=0)
    ci    = 1.96 * means.std(axis=0) / np.sqrt(len(seed_logs))
    return ts, gm, ci


def plot_own_reward(all_seed_logs):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, rname in zip(axes, ["Ra", "Rb", "Rc"]):
        sls = all_seed_logs[rname]
        if not sls:
            ax.set_title(f"SAC-{rname}: no data"); continue
        ts, gm, ci = aggregate(sls, rname)
        ax.plot(ts, gm, color=COLORS[rname], lw=2)
        ax.fill_between(ts, gm - ci, gm + ci, alpha=0.2, color=COLORS[rname])
        ax.set_title(f"SAC-{rname} on {rname}  (n={len(sls)} seeds)", fontsize=11)
        ax.set_xlabel("Environment Timesteps")
        ax.set_ylabel(f"Avg Undiscounted Return ({rname})")
        ax.grid(alpha=0.3)
    plt.suptitle("Q2.3.2 – SAC-Ri evaluated on Ri  (mean ± 95% CI)", fontsize=13)
    plt.tight_layout()
    plt.savefig("plots/reacher_q232_own_reward.png", dpi=150)
    plt.close()
    print("Saved plots/reacher_q232_own_reward.png")


def plot_cross_eval(all_seed_logs):
    for eval_reward in ["Ra", "Rb", "Rc"]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for agent_name in ["Ra", "Rb", "Rc"]:
            sls = all_seed_logs[agent_name]
            if not sls: continue
            ts, gm, ci = aggregate(sls, eval_reward)
            ax.plot(ts, gm, color=COLORS[agent_name], lw=2, label=f"SAC-{agent_name}")
            ax.fill_between(ts, gm - ci, gm + ci, alpha=0.15, color=COLORS[agent_name])
        ax.set_title(f"All agents evaluated on {eval_reward}", fontsize=11)
        ax.set_xlabel("Environment Timesteps")
        ax.set_ylabel(f"Avg Undiscounted Return ({eval_reward})")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"plots/reacher_cross_eval_{eval_reward}.png", dpi=150)
        plt.close()
        print(f"Saved plots/reacher_cross_eval_{eval_reward}.png")


def plot_bar_chart(bar_metrics):
    labels = ["Ra", "Rb", "Rc"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, key, title in zip(axes,
                               ["stg", "sit"],
                               ["Steps to Goal  (lower is better)",
                                "Steps in Target  (higher is better)"]):
        vals  = [np.array(bar_metrics[r][key]) for r in labels]
        means = [v.mean() if len(v) else 0 for v in vals]
        cis   = [1.96 * v.std() / np.sqrt(len(v)) if len(v) > 1 else 0 for v in vals]
        bars  = ax.bar(np.arange(len(labels)), means, 0.5,
                       yerr=cis, capsize=7,
                       color=[COLORS[r] for r in labels],
                       alpha=0.85, edgecolor="black", lw=0.8)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels([f"SAC-{r}" for r in labels], fontsize=11)
        ax.set_ylabel("Timesteps")
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        for bar_, m, ci in zip(bars, means, cis):
            ax.text(bar_.get_x() + bar_.get_width() / 2,
                    m + ci + max(means) * 0.01,
                    f"{m:.0f}", ha="center", va="bottom", fontsize=9)
    plt.suptitle(f"Q2.3.3a – Goal Metrics  (CI over seeds, {BAR_EPISODES} eps × {BAR_EP_LEN} steps)",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig("plots/reacher_bar_chart.png", dpi=150)
    plt.close()
    print("Saved plots/reacher_bar_chart.png")


# ── Bar Chart Evaluation ──────────────────────────────────────────────────────

def evaluate_goal_metrics_one_seed(reward_name, seed, n_episodes=500, ep_len=5000):
    wpath = f"weights/reacher_{reward_name}_seed{seed}.pt"
    if not os.path.exists(wpath):
        print(f"  Missing weights: {wpath} — skipping")
        return None, None

    agent = SACContinuous(OBS_DIM, ACT_DIM, auto_alpha=True)
    ckpt  = torch.load(wpath, map_location=DEVICE)
    agent.actor.load_state_dict(ckpt["actor"])
    agent.critic.load_state_dict(ckpt["critic"])
    agent.total_steps = 999_999

    stg_list, sit_list = [], []
    for ep in range(n_episodes):
        env = make_reacher("Rb", seed=9000 + ep)
        obs, _ = env.reset()
        reached, goal_step, in_tgt_count = False, ep_len, 0

        for t in range(ep_len):
            action = agent.select_action(obs, evaluate=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            if reward == 1.0:
                if not reached:
                    reached, goal_step = True, t + 1
                in_tgt_count += 1
            if truncated:
                obs, _ = env.reset()
            if terminated:
                break

        stg_list.append(goal_step)
        sit_list.append(in_tgt_count)
        env.close()

    return np.mean(stg_list), np.mean(sit_list)


def compute_bar_metrics(all_seed_logs):
    bar_metrics = {rn: {"stg": [], "sit": []} for rn in ["Ra", "Rb", "Rc"]}
    for rn in ["Ra", "Rb", "Rc"]:
        available_seeds = [sl["seed"] for sl in all_seed_logs[rn]]
        print(f"\nBar-chart eval for SAC-{rn}  (seeds: {available_seeds})")
        for seed in available_seeds:
            stg, sit = evaluate_goal_metrics_one_seed(rn, seed, BAR_EPISODES, BAR_EP_LEN)
            if stg is not None:
                bar_metrics[rn]["stg"].append(stg)
                bar_metrics[rn]["sit"].append(sit)
                print(f"  seed={seed}: steps_to_goal={stg:.1f}  steps_in_target={sit:.1f}")
    return bar_metrics


def print_summary(bar_metrics):
    print(f"\n{'Agent':<10} {'Steps-to-Goal':>20}   {'Steps-in-Target':>20}")
    print("-" * 56)
    for rn in ["Ra", "Rb", "Rc"]:
        stg = np.array(bar_metrics[rn]["stg"])
        sit = np.array(bar_metrics[rn]["sit"])
        if len(stg) == 0:
            print(f"SAC-{rn:<6}  no data"); continue
        ci_stg = 1.96 * stg.std() / np.sqrt(len(stg)) if len(stg) > 1 else 0
        ci_sit = 1.96 * sit.std() / np.sqrt(len(sit)) if len(sit) > 1 else 0
        print(f"SAC-{rn:<6}  {stg.mean():>8.1f} ± {ci_stg:<8.1f}   {sit.mean():>8.1f} ± {ci_sit:.1f}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _tmp = make_reacher("Rb", seed=0)
    _tmp.reset()
    OBS_DIM = _tmp.obs_dim
    ACT_DIM = _tmp.act_dim
    _tmp.close()
    print(f"obs_dim={OBS_DIM}  act_dim={ACT_DIM}  device={DEVICE}")

    for REWARD in REWARDS:
        print("\n" + "=" * 60)
        print(f"  SAC-{REWARD}  |  seeds 0-14  |  {TOTAL_STEPS:,} steps each")
        print("=" * 60)

        for seed in ALL_SEEDS:
            fname = f"logs/reacher_{REWARD}_seed{seed}.json"
            if os.path.exists(fname):
                print(f"-- SAC-{REWARD} seed {seed}: already exists, skipping --")
                continue

            print(f"\n-- SAC-{REWARD}  seed {seed} --")
            np.random.seed(seed)
            torch.manual_seed(seed)

            train_env = make_reacher(REWARD, seed=seed, eval_mode=False)
            eval_envs = {
                "Ra": make_reacher("Ra", seed=seed + 5000),
                "Rb": make_reacher("Rb", seed=seed + 5000),
                "Rc": make_reacher("Rc", seed=seed + 5000, eval_mode=True),
            }
            agent = SACContinuous(OBS_DIM, ACT_DIM, auto_alpha=True)
            extra = {}

            if REWARD == "Rc":
                log, last_50 = train_rc(agent, train_env, eval_envs,
                                        TOTAL_STEPS, EVAL_FREQ, EVAL_EPS)
                extra["last_50_ep_lengths"] = last_50
                if last_50:
                    print(f"  Rc last-50 ep lengths: {np.mean(last_50):.1f} ± {np.std(last_50):.1f}")
                else:
                    print("  Rc: agent never reached goal.")
            else:
                log = train_ra_rb(agent, train_env, eval_envs,
                                  TOTAL_STEPS, EVAL_FREQ, EVAL_EPS)

            torch.save({"actor":  agent.actor.state_dict(),
                        "critic": agent.critic.state_dict()},
                       f"weights/reacher_{REWARD}_seed{seed}.pt")

            save_seed_json(REWARD, seed, log, extra=extra if extra else None)

            train_env.close()
            for e in eval_envs.values():
                e.close()
            del agent, train_env, eval_envs

    print("\n All 3 rewards × 15 seeds complete. Generating plots...")

    all_seed_logs = {rn: load_all_jsons(rn) for rn in ["Ra", "Rb", "Rc"]}
    plot_own_reward(all_seed_logs)
    plot_cross_eval(all_seed_logs)

    bar_metrics = compute_bar_metrics(all_seed_logs)
    plot_bar_chart(bar_metrics)
    print_summary(bar_metrics)

    rc_logs = load_all_jsons("Rc")
    print("\nRc last-50 training episode lengths:")
    all_last50 = []
    for sl in rc_logs:
        ep_lens = sl["extra"].get("last_50_ep_lengths", [])
        if ep_lens:
            all_last50.extend(ep_lens)
            print(f"  seed={sl['seed']}: mean={np.mean(ep_lens):.1f}  std={np.std(ep_lens):.1f}")
        else:
            print(f"  seed={sl['seed']}: agent never reached goal.")
    if all_last50:
        print(f"\nOverall: mean={np.mean(all_last50):.1f}  std={np.std(all_last50):.1f}")