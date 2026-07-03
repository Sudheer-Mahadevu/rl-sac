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

def load_logs4(pattern):
    files = sorted(glob.glob(pattern))
    print(f"Found {len(files)} log files")
    pebble = [np.array(json.load(open(f))['pebble']) for f in files]
    sac = [np.array(json.load(open(f))['sac']) for f in files]
    # seeds = np.array([json.load(open(f)) for f in files])
    return np.concatenate(pebble, axis=0), np.concatenate(sac, axis=0)


def clean_data(data):
    cleaned_data_np = {}
    for key, lists_of_strings in data.items():
        cleaned_data_np[int(key)] = []
        for s in lists_of_strings:
            # Remove brackets and newlines
            s_clean = s.replace('[', '').replace(']', '').replace('\n', ' ')
            
            # Convert string to numpy array using space as a separator
            arr = np.fromstring(s_clean, sep=' ')
            
            cleaned_data_np[int(key)].append(arr)
    
    for key, val in cleaned_data_np.items():
            cleaned_data_np[key] = np.stack(val, axis=1)
    return cleaned_data_np

def load_logs5(pattern):
    files = sorted(glob.glob(pattern))
    print(f"Found {len(files)} log files")
    budgets = [clean_data(json.load(open(f))['budgets']) for f in files]
    sac = [np.array(json.load(open(f))['sac']) for f in files]
    combined_bugets = {}
    for k,_ in budgets[0].items():
        l = [seed[k] for seed in budgets]
        combined_bugets[k] = np.stack(l, axis = 0)
    # seeds = np.array([json.load(open(f)) for f in files])
    return combined_bugets, np.concatenate(sac, axis=0)

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
def q_43(reward = 'Ra'):
    pattern = f'output/pebble results/question3/reacher_{reward}_own*'
    files = sorted(glob.glob(pattern))
    print(f"Found {len(files)} log files")
    logs = [np.array(json.load(open(f))) for f in files]
    logs = np.concatenate(logs, axis=0)
    print(logs.shape)

    fig, ax = plt.subplots(1,1, sharex=True, figsize=(10, 5))
    s1, gm1, ci1 = aggregate(logs)
    ax.plot(s1, gm1, color='steelblue', linewidth=2.8, label = f'SAC-{reward}')
    ax.fill_between(s1, gm1 - ci1, gm1 + ci1, alpha=0.15, color='steelblue')

    pattern = f'output/pebble results/question3/reacher_{reward}_rb*'
    files = sorted(glob.glob(pattern))
    print(f"Found {len(files)} log files")
    logs = [np.array(json.load(open(f))) for f in files]
    logs = np.concatenate(logs, axis=0)
    print(logs.shape)

    s1, gm1, ci1 = aggregate(logs)
    ax.plot(s1, gm1, color='grey', linewidth=1.8, label = f'PEBBLE-{reward}-Rb')
    ax.fill_between(s1, gm1 - ci1, gm1 + ci1, alpha=0.15, color='grey')

    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    fig.supxlabel('Environment Timesteps', fontsize=12)
    fig.supylabel('Avg Undiscounted Return', fontsize=12)
    plt.suptitle(f'Offline Return for Reacher - Mean ± 95% CI \n'
        ' using PEBBLE and SAC', fontsize=13)   
    plt.tight_layout()
    plt.savefig(f'output/plots/4_reacher_pebble_vs_sac_{reward}.png', dpi=150)
    plt.show()
    print('Plot saved.') 

def q_42():
    pattern = f'output/pebble results/question2/budget_sweep*'
    budgets, sac = load_logs5(pattern)
    fig, ax = plt.subplots(1,1, sharex=True, figsize=(10, 5))

    s, gm, ci = aggregate(sac)
    ax.plot(s, gm, color='steelblue', linewidth=1.8, label = 'SAC')
    ax.fill_between(s, gm - ci, gm + ci, alpha=0.15, color='steelblue')

    # cmap = plt.get_cmap('magma')
    # colors = [cmap(i) for i in np.linspace(0.2, 1, 5)]
    colors = ['green', 'red', 'cyan', 'orange', 'pink']
    styles = ['-', '--', ':', '-.', (0, (3, 5, 1, 5))]
    widths = [2,1.75,1.5,1.25,1]

    for i,(c,(k,v)) in enumerate(zip(colors,budgets.items())):
        s, gm, ci = aggregate(v)
        print(k, v.shape)
        ax.plot(s, gm, color=c, ls=styles[i % len(styles)],lw = widths[i], label = f'Budget: {k}')
        ax.fill_between(s, gm - ci, gm + ci, alpha=0.15, color=c)

    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.supxlabel('Environment Timesteps', fontsize=12)
    fig.supylabel('Avg Undiscounted Return', fontsize=12)
    plt.suptitle(f'Offline Return PEBBLE on Pendulum \n'
        'with different budgets, SAC - Mean ± 95% CI ', fontsize=13)   
    plt.tight_layout()
    plt.savefig('output/plots/5_2_pendulum_diff_budgets.png', dpi=150)
    plt.show()
    print('Plot saved.') 

def q_41():
    THETAS = [-150, -60, 0, 90, 120]
    fig, ax = plt.subplots(2,3, sharex=True, figsize=(10, 5))
    ax = ax.ravel()

    for j, theta in enumerate(THETAS):
        pattern = f'output/pebble results/question1/pendulum_theta_{theta}*'
        pebble, sac = load_logs4(pattern)

        s1, gm1, ci1 = aggregate(pebble)
        s2, gm2, ci2 = aggregate(sac)

        ax[j].plot(s1, gm1, color='grey', linewidth=1.8, label = 'PEBBLE')
        ax[j].fill_between(s1, gm1 - ci1, gm1 + ci1, alpha=0.15, color='grey')

        ax[j].plot(s2, gm2, color='steelblue', linewidth=1.8, label = 'SAC')
        ax[j].fill_between(s2, gm2 - ci2, gm2 + ci2, alpha=0.15, color='steelblue')
        ax[j].legend(fontsize=9)
        ax[j].grid(True, alpha=0.25)
        ax[j].set_title(f"Theta = {theta}°")

    ax[5].axis('off')
    fig.supxlabel('Environment Timesteps', fontsize=12)
    fig.supylabel('Avg Undiscounted Return', fontsize=12)
    plt.suptitle(f'Offline Return - Mean ± 95% CI \n'
        ' using PEBBLE and SAC', fontsize=13)   
    plt.tight_layout()
    plt.savefig('output/plots/4_pendulum_pebble_vs_sac.png', dpi=150)
    plt.show()
    print('Plot saved.') 

q_42()
