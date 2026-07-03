import json
import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("plots", exist_ok=True)


def load_seed_logs(log_dir: str, pattern: str,
                   seeds: list) -> list:
    logs = []
    for s in seeds:
        fpath = os.path.join(log_dir, pattern.format(s))
        if not os.path.exists(fpath):
            print(f"  [warn] {fpath} not found – skipping")
            continue
        with open(fpath) as f:
            logs.append(json.load(f))
    if not logs:
        raise FileNotFoundError(
            f"No log files found in '{log_dir}' matching '{pattern}'")
    return logs


def aggregate_logs(logs: list):
    timesteps  = np.array(logs[0]["timesteps"])
    means      = np.array([l["mean_returns"] for l in logs])  # (S, T)
    grand_mean = means.mean(axis=0)
    sem        = means.std(axis=0) / np.sqrt(len(logs))
    ci         = 1.96 * sem
    return timesteps, grand_mean, ci


def plot_curves(
    curves_dict: dict,
    title:  str,
    xlabel: str  = "Environment Timesteps",
    ylabel: str  = "Avg Undiscounted Return",
    fname:  str  = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, (ts, mean, ci) in curves_dict.items():
        (line,) = ax.plot(ts, mean, label=label, linewidth=1.8)
        ax.fill_between(ts, mean - ci, mean + ci,
                        alpha=0.18, color=line.get_color())
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title,   fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if fname:
        path = f"plots/{fname}.png"
        plt.savefig(path, dpi=150)
        print(f"  Saved → {path}")
    plt.show()
    plt.close()


def plot_hover_phases(
    agg_fixed:    tuple,
    agg_auto:     tuple,
    phase1_steps: int,
    fname:        str = "q3_hover",
):

    ts_f, mean_f, ci_f = agg_fixed
    ts_a, mean_a, ci_a = agg_auto

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ts_f, mean_f, label="Fixed α = 0.01", color="steelblue",   linewidth=1.8)
    ax.fill_between(ts_f, mean_f - ci_f, mean_f + ci_f,
                    alpha=0.18, color="steelblue")
    ax.plot(ts_a, mean_a, label="Auto α (SAC)",  color="darkorange",  linewidth=1.8)
    ax.fill_between(ts_a, mean_a - ci_a, mean_a + ci_a,
                    alpha=0.18, color="darkorange")
    ax.axvline(x=phase1_steps, color="crimson", linestyle="--", linewidth=1.5,
               label="Reward switch  (+200 → −100)")
    ax.set_xlabel("Environment Timesteps", fontsize=12)
    ax.set_ylabel("Avg Undiscounted Return", fontsize=12)
    ax.set_title("Q2.2.3  Hover LunarLander: Fixed α vs Auto α", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = f"plots/{fname}.png"
    plt.savefig(path, dpi=150)
    print(f"  Saved → {path}")
    plt.show()
    plt.close()
