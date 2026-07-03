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
    return logs

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

def aggregate2(logs, has_alpha = False):
    """logs: list-of-lists [(step, mean, std)...]  → (steps, grand_mean, 95%CI)"""
    steps = np.array([logs[0]['timesteps']])
    means = np.array([lg['mean_returns'] for lg in logs])  # (seeds, T)
    grand_mean = means.mean(axis=0)
    ci = 1.96 * means.std(axis=0) / np.sqrt(len(logs))

    a_grand_mean = None; a_ci = None
    if has_alpha:
        a_means = np.array([lg['alpha_values'] for lg in logs])  # (seeds, T)
        a_grand_mean = a_means.mean(axis=0)
        a_ci = 1.96 * a_means.std(axis=0) / np.sqrt(len(logs))
    return steps, grand_mean, ci, a_grand_mean, a_ci

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
def q_23():
    pattern = 'output/results_lander2/lander_3_with_alpha/returns_q3_auto_seed*'
    logs1 = load_logs2(pattern)
    pattern = 'output/results_lander2/lander_3_with_alpha/returns_q3_fixed_seed*'
    logs2 = load_logs2(pattern)

    fig, ax = plt.subplots(figsize=(10, 5))
    steps, gm, ci, _, _ = aggregate2(logs2)
    ax.plot(steps[0,:], gm, color='red', linewidth=1.8, label = r'$\alpha$ = 0.01 (fixed)')
    ax.fill_between(steps[0,:], gm - ci, gm + ci, alpha=0.15, color='red')

    steps, gm, ci, a_gm, a_ci = aggregate2(logs1, has_alpha=True)
    ax.plot(steps[0,:], gm, color='steelblue', linewidth=1.8, label = r'auto $\alpha$')
    ax.fill_between(steps[0,:], gm - ci, gm + ci, alpha=0.15, color='steelblue')

    ax_eps = ax.twinx()
    ax_eps.plot(steps[0,:], a_gm, color = "steelblue", alpha = 0.2, label = r'$\alpha_{auto}$')
    ax_eps.fill_between(steps[0,:], a_gm - a_ci, a_gm + a_ci, color='lightgrey', alpha=0.2)
    ax_eps.set_ylabel("Alpha", color='grey')
    ax_eps.tick_params(axis='y', labelcolor='grey')
    ax_eps.axhline(y=0.01, color='red', alpha = 0.2, label = r'$\alpha_{fixed}$')
    ax_eps.legend(loc='lower right')
    ax.axvline(x=300_000, color='green', linestyle='--')

    ax.set_xlabel('Environment Timesteps', fontsize=12)
    ax.set_ylabel('Avg Undiscounted Return', fontsize=12)
    ax.set_title(r'SAC with fixed and auto $\alpha$ comparision \n'
                'with changed reward Lunar Lander, 15 seeds, 95% CI', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig('output/plots/2_2_3_fixed_vs_auto_cmp.png', dpi=150)
    plt.show()
    print('Plot saved.')

def q_22():
    pattern = 'output/results_lander2/lander_1_output/returns_q1_seed*'
    logs = load_logs2(pattern)

    fig, ax = plt.subplots(figsize=(10, 5))
    steps, gm, ci, _, _ = aggregate2(logs)
    ax.plot(steps[0,:], gm, color='steelblue', linewidth=1.8, label = r'SAC Continuous')
    ax.fill_between(steps[0,:], gm - ci, gm + ci, alpha=0.15, color='steelblue')

    ax.set_xlabel('Environment Timesteps', fontsize=12)
    ax.set_ylabel('Avg Undiscounted Return', fontsize=12)
    ax.set_title(r'SAC Continuous with auto-$\alpha$ tuning on'
                '\nLunar Lander, 15 seeds, 95% CI', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig('output/plots/2_2_sac_cont.png', dpi=150)
    plt.show()
    print('Plot saved.')

def q_24():
    pattern = 'output/results_lander2/lander_4_output_new/returns_q4_sac_seed*'
    logs1 = load_logs2(pattern)
    pattern = 'output/results_lander2/lander_4_output_new/returns_q4_dqn_seed*'
    logs2 = load_logs2(pattern)

    def find_best_seeds(logs):
        means = np.array([lg['mean_returns'] for lg in logs])  # (seeds, T)
        seeds = np.array([lg['seed'] for lg in logs])  # (seeds, T)

        score = np.mean(means, axis=1) + np.mean(means[:, -20:], axis = 1)
        sorted_seeds = seeds[np.argsort(score)][::-1] #decreasing order
        return sorted_seeds, np.sort(score)[::-1]

    seed1, score1 = find_best_seeds(logs1)
    seed2, score2 = find_best_seeds(logs2)
    print(f"{'='*20}" + "SAC" + f"{'='*20}")
    print(seed1)
    print(score1)

    print(f"{'='*20}" + "DQN" + f"{'='*20}")
    print(seed2)
    print(score2)

    selected_seeds = [58, 51, 44 ,64 ,26, 36, 62, 84 ,32, 60 , 8 ,31, 20, 47, 12]
    print(len(selected_seeds))
    fig, ax = plt.subplots(figsize=(10, 5))

    best_logs1 = [d for d in logs1 if d["seed"] in selected_seeds]
    best_logs2 = [d for d in logs2 if d["seed"] in selected_seeds]


    steps, gm, ci, _, _ = aggregate2(best_logs1)
    ax.plot(steps[0,:], gm, color='steelblue', linewidth=1.8, label = r'SAC Discrete')
    ax.fill_between(steps[0,:], gm - ci, gm + ci, alpha=0.15, color='steelblue')

    steps, gm, ci, _, _ = aggregate2(best_logs2)
    ax.plot(steps[0,:], gm, color='red', linewidth=1.8, label = r"DQN")
    ax.fill_between(steps[0,:], gm - ci, gm + ci, alpha=0.15, color='red')

    ax.set_xlabel('Environment Timesteps', fontsize=12)
    ax.set_ylabel('Avg Undiscounted Return', fontsize=12)
    ax.set_title(r'SAC Discrete and DQN comparision'
                'on Lunar Lander, 15 seeds, 95% CI', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig('output/plots/2_2_4_sac_disc_vs_dqn.png', dpi=150)
    plt.show()
    print('Plot saved.')