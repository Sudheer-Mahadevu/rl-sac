# ── 12. Q2.1.4 – visualise learned policy for one target ─────────────────────
# Re-train a single agent (seed 0) for θ=90° and record a trajectory.
from main import set_seeds
from new_env import make_env, TAU_FF_SCALE
from sac import SACAgent
import math
import matplotlib.pyplot as plt
import torch
import numpy as np


THETA_TARGETS  = [0, -10, 30, -60, 90, -90, 120, -150]   # degrees
N_SEEDS        = 15
TOTAL_STEPS    = 20_000     # Pendulum is simple; 100K is enough to converge
EVAL_FREQ      = 10_000
EVAL_EPISODES  = 20
HIDDEN_DIM     = 64
SEEDS = [22, 67, 8, 45, 212, 99, 27, 4, 2003, 89, 11, 5, 37, 502, 75]

WEIGHTS_PATH = 'output/weights/a_alpha_theta0_seed22.pt'
set_seeds(0)
THETA = 0
seed = 42
env = make_env(THETA, render_mode='human', trunc_len=200)
obs_dim  = env.observation_space.shape[0]    # 3
act_dim  = env.action_space.shape[0]         # 1
agent = SACAgent(
                obs_dim, act_dim,
                hidden_dim=64,
                auto_alpha=True,
                init_alpha=0.2,
                random_steps=10_000)

ckpt = torch.load(WEIGHTS_PATH, map_location=torch.device('cpu'))
agent.actor.load_state_dict(ckpt['actor'])
agent.critic.load_state_dict(ckpt['critic'])
agent.actor.eval()
agent.critic.eval()
print(f"Weights loaded from {WEIGHTS_PATH}")

# Record trajectory
fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
for i in range(5):
    obs, _ = env.reset()
    thetas, thetas_dot, taus = [], [], []
    done = False
    while not done:
        action = agent.select_action(obs, evaluate=True)
        next_obs, _, terminated, truncated, _ = env.step(action)
        cos_th, sin_th, th_dot = obs
        thetas.append(math.degrees(math.atan2(sin_th, cos_th)))
        thetas_dot.append(math.degrees(th_dot))
        taus.append(float(np.clip(action[0] * 6.0, -6.0, 6.0)))
        obs = next_obs
        done = terminated or truncated


    l1 = f'θ_target={THETA}°' if i==0 else None
    l2 = 'τ applied' if i==0 else None
    t_axis = np.arange(len(thetas)) * 0.05   # dt = 0.05 s
    axes[0].plot(t_axis, thetas, color='steelblue')
    axes[0].axhline(THETA, color='red', linestyle='--', label=l1)
    axes[0].set_ylabel('θ (deg)'); axes[0].legend(); axes[0].grid(alpha=0.25)
    axes[1].plot(t_axis, thetas_dot, color='darkorange')
    axes[1].set_ylabel('θ̇ (deg/s)'); axes[1].grid(alpha=0.25)
    tau_ff = TAU_FF_SCALE * math.sin(math.radians(THETA))
    axes[2].plot(t_axis, taus, color='forestgreen', label=l2)

    l3 = f'τ_ff={tau_ff:.2f}' if i == 0 else None
    axes[2].axhline(-tau_ff, color='red', linestyle='--', label=l3)

env.close()

axes[2].set_ylabel('τ (Nm)'); axes[2].set_xlabel('Time (s)')
axes[2].legend(); axes[2].grid(alpha=0.25)
fig.suptitle(f'Q2.1.4 – Optimal trajectory (θ_target={THETA}°)', fontsize=13)
plt.tight_layout()
plt.savefig('output/plots/q21_optimal_traj.png', dpi=150)
plt.show()
print('\nTwo aspects of optimal behaviour:')
print('  1. ANGLE ACCURACY  : θ → θ_target (angle error → 0)')
print('  2. BALANCED EFFORT : θ̇ → 0  AND  τ → τ_ff (τ_excess → 0)')