import json
import numpy as np
import matplotlib.pyplot as plt
import glob
import re

def load_logs(pattern):
    files = sorted(glob.glob(pattern))
    print(f"Found {len(files)} log files")

    manual_regex = pattern.replace('.', r'\.').replace('*', '(.*)')
    indentifiers = np.array([re.search(manual_regex, f).group(1) for f in files])
    eval_returns = np.array([json.load(open(f))["eval_logs"] for f in files])
    train_returns = np.array([json.load(open(f))["train_logs"]['r'] for f in files])
    train_alpha = np.array([json.load(open(f))["train_logs"]['a'] for f in files])
    train_ts = np.array([json.load(open(f))["train_logs"]['t'] for f in files])
    return (indentifiers, eval_returns, train_returns, train_alpha, train_ts)
    
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
        ax_eps.fill_between(episodes, mean_alpha - ci_alpha, mean_alpha + ci_alpha, color='lightgrey', alpha=0.1)
        ax_eps.set_ylabel("Epsilon", color='grey')
        ax_eps.tick_params(axis='y', labelcolor='grey')


def aggregate(logs):
    """logs: list-of-lists [(step, mean, std)...]  → (steps, grand_mean, 95%CI)"""
    steps = np.array([t for t, _, _ in logs[0]])
    means = np.array([[m for _, m, _ in lg] for lg in logs])  # (seeds, T)
    grand_mean = means.mean(axis=0)
    ci = 1.96 * means.std(axis=0) / np.sqrt(len(logs))
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
                va='center', color=pallete[i], fontsize=10)

#####################################################################################

def q_25_online():
    THETAS = [-150, -60, 90, 120]
    # ALPHAS = [0.01, 0.05, 0.1, 0.2, 0.4]
    fig, ax = plt.subplots(2,2,figsize=(10, 5), sharex = True, sharey=True)
    ax = ax.ravel()
    for j, theta in enumerate(THETAS):
        pattern = f'output/logs_alpha_tuning2/alpha*_theta{theta}.json'
        print(load_logs(pattern)[0])

        a, er, tr, ta, tt = load_logs(pattern)
        PALETTE = plt.cm.tab20.colors
        # min_e, max_e, mean_e = find_min_max_episodes(tr)

        for i, (col, alpha) in enumerate(zip(PALETTE, a)):
            print(i, col, alpha)
            plot_mean_ci(tr[i], ta[i], f'α={alpha}',col, ax[j])

        ax[j].legend()
        ax[j].grid(True, alpha = 0.3)
        ax[j].set_xlabel('Episodes')
        ax[j].set_ylabel('Return')
        ax[j].set_title(f"Theta = {theta}°")

    plt.suptitle('Online Episodes vs Return - Mean ± 95% CI \n'
        ' for SAC on Pendulumv1 for different αs ')
    plt.tight_layout()
    plt.savefig('output/plots/2_1_alpha_tuning_online.png', dpi=150)
    plt.show()

def q_25_manual_tuning():
    THETAS = [-150, -60, 90, 120]
    # ALPHAS = [0.01, 0.05, 0.1, 0.2, 0.4]
    i=0
    fig, ax = plt.subplots(2,2,figsize=(10, 5), sharex = True, sharey=True)
    ax = ax.ravel()
    for i, theta in enumerate(THETAS):
        pattern = f'output/logs_alpha_tuning2/alpha*_theta{THETAS[i]}*'
        a,er, _, _, _ = load_logs(pattern)
        PALETTE = plt.cm.tab20.colors
        for j, (log, col) in enumerate(zip(er,PALETTE)):
            steps, gm, ci = aggregate(er[j])
            ax[i].plot(steps, gm, label=f'α={a[j]}', color=col, linewidth=1.8)
            ax[i].fill_between(steps, gm - ci, gm + ci, alpha=0.05, color=col)
        
        ax[i].legend()
        ax[i].set_title(f"Theta = {theta}°")
        ax[i].grid(True, alpha=0.25)

    fig.supxlabel('Environment Timesteps', fontsize=12)
    fig.supylabel('Avg Undiscounted Return', fontsize=12)
    plt.suptitle('Offline Return - Mean ± 95% CI \n'
        ' for SAC on Pendulumv1 for different αs', fontsize=13)   
    plt.tight_layout()
    plt.savefig('output/plots/2_1_alpha_tuning_returns.png', dpi=150)
    plt.show()
    print('Plot saved.')

def q_25_aggr():
    THETAS = [-150, -60, 90, 120]
    # ALPHAS = [0.01, 0.05, 0.1, 0.2, 0.4]
    fig, ax = plt.subplots(2,2,figsize=(10, 5), sharex = True, sharey=True)
    ax = ax.ravel()
    for j, theta in enumerate(THETAS):
        ret_dict = {}
        pattern = f'output/logs_alpha_tuning2/alpha*_theta{theta}.json'
        print(load_logs(pattern)[0])

        a, er, tr, ta, tt = load_logs(pattern)
        for i, alpha in enumerate(a):
            ret_dict[alpha] = er[i,:,:,1]

        PALETTE = plt.cm.tab20.colors
        plot_aggregate_performance(a,ret_dict, PALETTE, ax[j], add_ang_labels=False,
                                label='Aggregate Perormance')

        ax[j].set_title(f"Theta = {theta}°", fontsize=13)
        ax[j].grid(True, alpha=0.25)

    fig.supxlabel('Alpha', fontsize=12)
    fig.supylabel('Aggregate Performance', fontsize=12)
    fig.suptitle("Aggregate Performance (Mean of Offline Return)\n" \
    "for different αs")
    plt.tight_layout()
    plt.savefig('output/plots/2_1_alpha_tuning_aggr_offline_perf.png', dpi=150)
    plt.show()

def q_25_reward_scale():
    THETA = [-150, -60, 90, 120]
    ALPHA = [0.2, 0.1, 0.05, 0.1]
    fig, ax = plt.subplots(2,1,figsize=(10, 5), sharex = True, sharey=True)
    i=2
    logs0 = load_logs(f'output/logs/a_alpha_theta{THETA[i]}.*')
    er_fixed = logs0[1]
    # logs1 = load_logs(f'output/logs_alpha_tuning/alpha{ALPHA[i]}_theta{THETA[i]}.*')
    # logs2 = load_logs(f'output/logs_alpha_tuning/alpha{ALPHA[i]}_theta{THETA[i]}_extra*')
    # er_fixed = np.concatenate((logs1[1], logs2[1]), axis=1)

    logs3 = load_logs(f'output/logs 0.1x/a_alpha_theta{THETA[i]}.*')
    er_01 = logs3[1]

    logs4 = load_logs(f'output/logs 10x/a_alpha_theta{THETA[i]}.*')
    er_10 = logs4[1]

    steps, gm, ci = aggregate(er_10[0,:])
    ax[0].plot(steps, gm, label=f'10x reward', color='green', linewidth=1.8)
    ax[0].fill_between(steps, gm - ci, gm + ci, alpha=0.1, color='green')

    steps, gm, ci = aggregate(er_fixed[0,:])
    ax[0].plot(steps, gm, label=f'1x reward', color='red', linewidth=1.8)
    ax[0].fill_between(steps, gm - ci, gm + ci, alpha=0.1, color='red')

    steps, gm, ci = aggregate(er_01[0,:])
    ax[0].plot(steps, gm, label=f'0.1x reward', color='steelblue', linewidth=1.8)
    ax[0].fill_between(steps, gm - ci, gm + ci, alpha=0.1, color='steelblue')

    ax[0].legend()
    ax[0].set_title(f"Auto α")
    ax[0].grid(True, alpha=0.25)


    logs1 = load_logs(f'output/logs_alpha_tuning2/alpha{ALPHA[i]}_theta{THETA[i]}.*')
    logs2 = load_logs(f'output/logs_alpha_tuning2/alpha{ALPHA[i]}_theta{THETA[i]}_extra*')
    er_fixed = np.concatenate((logs1[1], logs2[1]), axis=1)

    logs3 = load_logs(f'output/logs 0.1x/alpha0.05_theta{THETA[i]}.*')
    er_01 = logs3[1]

    logs4 = load_logs(f'output/logs 10x/alpha0.05_theta{THETA[i]}.*')
    er_10 = logs4[1]

    steps, gm, ci = aggregate(er_10[0,:])
    ax[1].plot(steps, gm, label=f'10x reward', color='green', linewidth=1.8)
    ax[1].fill_between(steps, gm - ci, gm + ci, alpha=0.1, color='green')

    steps, gm, ci = aggregate(er_fixed[0,:])
    ax[1].plot(steps, gm, label=f'1x reward', color='red', linewidth=1.8)
    ax[1].fill_between(steps, gm - ci, gm + ci, alpha=0.1, color='red')

    steps, gm, ci = aggregate(er_01[0,:])
    ax[1].plot(steps, gm, label=f'0.1x reward', color='steelblue', linewidth=1.8)
    ax[1].fill_between(steps, gm - ci, gm + ci, alpha=0.1, color='steelblue')

    ax[1].legend()
    ax[1].set_title(f"α=0.05 (fixed)")
    ax[1].grid(True, alpha=0.25)

    fig.supxlabel('Environment Timesteps', fontsize=12)
    fig.supylabel('Avg Undiscounted Return', fontsize=12)
    plt.suptitle('Offline Return - Mean ± 95% CI for θ=90° with \n'
        ' manual-tuning and fixed α for different reward scales', fontsize=13)   
    plt.tight_layout()
    plt.savefig('output/plots/2_1_reward_scaling_auto_alpha_vs_fixed_comp.png', dpi=150)
    plt.show()
    print('Plot saved.')

def q_25_fixed_auto_comp():
    THETA = [-150, -60, 90, 120]
    ALPHA = [0.2, 0.1, 0.05, 0.1]
    fig, ax = plt.subplots(2,2,figsize=(10, 5), sharex = True, sharey=True)
    ax = ax.ravel()

    for i in range(len(THETA)):
        logs1 = load_logs(f'output/logs_alpha_tuning2/alpha{ALPHA[i]}_theta{THETA[i]}.*')
        logs2 = load_logs(f'output/logs_alpha_tuning2/alpha{ALPHA[i]}_theta{THETA[i]}_extra*')
        er_fixed = np.concatenate((logs1[1], logs2[1]), axis=1)

        logs_auto = load_logs(f'output/logs/a_alpha_theta{THETA[i]}*')
        er_auto = logs_auto[1]

        steps, gm, ci = aggregate(er_fixed[0,:])
        ax[i].plot(steps, gm, label=f'α = {ALPHA[i]}', color='red', linewidth=1.8)
        ax[i].fill_between(steps, gm - ci, gm + ci, alpha=0.1, color='red')

        steps, gm, ci = aggregate(er_auto[0,:])
        ax[i].plot(steps, gm, label=f'auto-α', color='steelblue', linewidth=1.8)
        ax[i].fill_between(steps, gm - ci, gm + ci, alpha=0.1, color='steelblue')

        ax[i].legend()
        ax[i].set_title(f"θ={THETA[i]}°")
        ax[i].grid(True, alpha=0.25)

    fig.supxlabel('Environment Timesteps', fontsize=12)
    fig.supylabel('Avg Undiscounted Return', fontsize=12)
    plt.suptitle('Offline Return Comparision - Mean ± 95% CI \n'
        ' manual-tuning vs fixed α for different target θs', fontsize=13)   
    plt.tight_layout()
    plt.savefig('output/plots/2_1_auto_vs_fixed_alpha.png', dpi=150)
    plt.show()
    print('Plot saved.')

