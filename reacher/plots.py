import json
import numpy as np
import matplotlib.pyplot as plt
import glob

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

def load_logs2(pattern):
    files = sorted(glob.glob(pattern))
    print(f"Found {len(files)} log files")
    logs = np.array([json.load(open(f)) for f in files])
    # seeds = np.array([json.load(open(f)) for f in files])
    return 



def load_logs3(pattern):
    files = sorted(glob.glob(pattern))
    print(f"Found {len(files)} log files")
    ra = np.array([[ts["Ra"]["mean"] for ts in json.load(open(f))["log"]] for f in files])
    rb = np.array([[ts["Rb"]["mean"] for ts in json.load(open(f))["log"]] for f in files])
    rc = np.array([[ts["Rc"]["mean"] for ts in json.load(open(f))["log"]] for f in files])
    # seeds = np.array([json.load(open(f)) for f in files])
    return (ra, rb, rc)

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
        # ax_eps.set_ylabel("Epsilon", color='grey')
        ax_eps.tick_params(axis='y', labelcolor='grey')


def aggregate(logs):
    """logs: list-of-lists [(step, mean, std)...]  → (steps, grand_mean, 95%CI)"""
    steps = np.array([t for t, _, _ in logs[0]])
    means = np.array([[m for _, m, _ in lg] for lg in logs])  # (seeds, T)
    grand_mean = means.mean(axis=0)
    ci = 1.96 * means.std(axis=0) / np.sqrt(len(logs))
    return steps, grand_mean, ci

def aggregate2(logs):
    """logs: list-of-lists [(step, mean, std)...]  → (steps, grand_mean, 95%CI)"""
    steps = np.array([logs[0]['timesteps']])
    means = np.array([lg['mean_returns'] for lg in logs])  # (seeds, T)
    grand_mean = means.mean(axis=0)
    ci = 1.96 * means.std(axis=0) / np.sqrt(len(logs))
    return steps, grand_mean, ci

def aggregate3(logs):
    """logs: list-of-lists [(step, mean, std)...]  → (steps, grand_mean, 95%CI)"""
    steps = np.arange(0,810_000,10_000)
    grand_mean = logs.mean(axis=0)
    ci = 1.96 * logs.std(axis=0) / np.sqrt(len(logs))
    return steps, grand_mean, ci

def plot_aggregate_performance(angles, returns, pallete,ax,
                               linecolor = 'lightgray', label = None,
                               add_ang_labels = True):

    aggregate_return = {}
    cis = {}
    for theta in angles:
        seed_aggregates = np.mean(returns[theta], axis=1) # mean return per each seed
        mu = np.mean(seed_aggregates)
        aggregate_return[theta] = mu

        sigma = np.std(seed_aggregates, ddof=1)

        n = len(seed_aggregates)
        ci_val = 1.96* (sigma/np.sqrt(n))
        cis[theta] = ci_val

    
    ax.plot(range(len(angles)),aggregate_return.values(), color=linecolor, linewidth=1.5, zorder=1,
            label = label)
    ax.set_xticks(range(len(angles)))
    ax.set_xticklabels(aggregate_return.keys())

    for i, theta in enumerate(angles):
        point_label = theta if add_ang_labels else None
        ax.errorbar(i, aggregate_return[theta], yerr=cis[theta], fmt='o',
                    capsize=6, markersize=9, color=pallete[i], zorder=2, label = point_label)
        ax.text(i+0.1, aggregate_return[theta], f'{aggregate_return[theta]:.2f}',
                va='center',fontweight='bold', color=pallete[i])


#####################################################################################

rewards = ['Ra', 'Rb', 'Rc']
REWARD_IDX = 2  # Use models trained on following reward to plot (0,1,2) for R(a,b,c)
colors = ['steelblue', 'red', 'green']
labels = [r'$R_{a}$', r'$R_{b}$', r'$R_{c}$']

filename = f"3_{rewards[REWARD_IDX]}_{rewards[REWARD_IDX]}_all_new.png"

fig, ax = plt.subplots(figsize=(10, 5))

for i,(rn,c) in enumerate(zip(rewards,colors)):
    pattern = f'output/reacher_outputs_800k/reacher_{rn}_output/reacher_{rn}_seed*'
    logs = load_logs3(pattern)[REWARD_IDX]
    steps, gm, ci = aggregate3(logs[:15])
    ax.plot(steps, gm, color=c, linewidth=1.8, label = f'trained with {rn}')
    ax.fill_between(steps, gm - ci, gm + ci, alpha=0.15, color=c)

ax.set_xlabel('Environment Timesteps', fontsize=12)
ax.set_ylabel('Avg Undiscounted Return', fontsize=12)
ax.set_title(f"Offline Return for SAC with eval reward {rewards[REWARD_IDX]}"
             ' on reacher (15 seeds, 95% CI)', fontsize=13)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(f'output/plots/{filename}', dpi=150)
print('Plot saved.')
plt.show()