import json
import numpy as np
import matplotlib.pyplot as plt

def load_logs(base_path, thetas):
    files = []
    for t in thetas:
        files.append(f'{base_path}{t}.json')
    # print(files)
    print(f"Found {len(files)} log files")
    eval_returns = np.array([json.load(open(f))["eval_logs"] for f in files])
    train_returns = np.array([json.load(open(f))["train_logs"]['r'] for f in files])
    train_alpha = np.array([json.load(open(f))["train_logs"]['a'] for f in files])
    train_ts = np.array([json.load(open(f))["train_logs"]['t'] for f in files])
    return (eval_returns, train_returns, train_alpha, train_ts)
    
def find_min_max_episodes(train_returns):
    min_e = 1e6; max_e = -1; tot = 0
    for theta in train_returns:
        for seed_ret in theta:
            l = len(seed_ret)
            max_e = max(max_e, l)
            min_e = min(min_e, l)
            tot += l
    return (max_e, min_e, tot/(len(train_returns[0])*len(train_returns)))

def smooth(x, window=2):
    return np.convolve(x, np.ones(window)/window, mode='valid')

def plot_mean_ci(returns_array, alphas,label, color, ax, plot_alpha = False):
    smoothed = np.array([smooth(r) for r in returns_array])
    episodes = np.arange(smoothed.shape[1])
    mean = smoothed.mean(axis=0)
    ci   = 1.96 * smoothed.std(axis=0) / np.sqrt(len(smoothed))
    ax.plot(episodes, mean, label=label, color=color, linewidth=2)
    ax.fill_between(episodes, mean - ci, mean + ci, color=color, alpha=0.2)

    if plot_alpha:
        ax_eps = ax.twinx()
        smoothed_alpha = np.array([smooth(r) for r in alphas])
        mean_alpha = smoothed_alpha.mean(axis=0)
        ci_alpha   = 1.96 * smoothed_alpha.std(axis=0) / np.sqrt(len(smoothed_alpha))
        ax_eps.plot(episodes, mean_alpha, color = "lightgrey", label = "Epsilon", alpha = 0.6)
        ax_eps.fill_between(episodes, mean_alpha - ci_alpha, mean_alpha + ci_alpha, color='lightgrey', alpha=0.2)
        ax_eps.set_ylabel("Epsilon", color='grey')
        ax_eps.tick_params(axis='y', labelcolor='grey')


def aggregate(logs):
    """logs: list-of-lists [(step, mean, std)...]  → (steps, grand_mean, 95%CI)"""
    steps = np.array([t for t, _, _ in logs[0]])
    means = np.array([[m for _, m, _ in lg] for lg in logs])  # (seeds, T)
    grand_mean = means.mean(axis=0)
    ci = 1.96 * means.std(axis=0) / np.sqrt(len(logs))
    return steps, grand_mean, ci

#####################################################################################

pattern = 'output/logs/a_alpha_theta'
THETA_TARGETS = [0, 90, -90, -60, -10]
PALETTE = plt.cm.tab10(np.linspace(0, 1, len(THETA_TARGETS)))
er, tr, ta, tt = load_logs(pattern, THETA_TARGETS)
min_e, max_e, mean_e = find_min_max_episodes(tr)

fig, ax = plt.subplots(figsize=(10, 5))


for i, (col, theta) in enumerate(zip(PALETTE, THETA_TARGETS)):
    print(i, col, theta)
    plot_mean_ci(tr[i], ta[i], f'θ={theta:+4d}°',col, ax)

ax.legend()
ax.grid(True, alpha = 0.3)
ax.set_xlabel('Episodes')
ax.set_ylabel('Return')
ax.set_title('Online Episodes vs Return - Mean ± 95% CI \n'
' for SAC on Pendulumv1 (Hidden Size = 64)')
plt.savefig('output/plots/online_sample_hs32_1seed.png', dpi=150)


fig, ax = plt.subplots(figsize=(10, 5))
for i, (col, theta) in enumerate(zip(PALETTE, THETA_TARGETS)):
    steps, gm, ci = aggregate(er[i])
    ax.plot(steps, gm, label=f'θ={theta:+4d}°', color=col, linewidth=1.8)
    ax.fill_between(steps, gm - ci, gm + ci, alpha=0.15, color=col)

ax.set_xlabel('Environment Timesteps', fontsize=12)
ax.set_ylabel('Avg Undiscounted Return', fontsize=12)
ax.set_title('Q2.1 – SAC (auto-α) on Modified Pendulum\n'
             'all θ_target, 15 seeds, 95% CI', fontsize=13)
ax.legend(ncol=2, fontsize=9)
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig('output/plots/offline_sample_hs64_1seed.png', dpi=150)
plt.show()
print('Plot saved.')