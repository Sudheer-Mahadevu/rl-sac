import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("plots_bonus", exist_ok=True)


def aggregate(all_logs):
    """all_logs: list of [(t, mean, std), ...] → (timesteps, grand_mean, 95%-CI)"""
    timesteps  = np.array([t for t, _, _ in all_logs[0]])
    means      = np.array([[m for _, m, _ in log] for log in all_logs])
    grand_mean = means.mean(0)
    ci         = 1.96 * means.std(0) / np.sqrt(len(all_logs))
    return timesteps, grand_mean, ci


def plot_comparison(curves, title, fname,
                    xlabel="Environment Timesteps",
                    ylabel="Avg Undiscounted GT Return"):
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, (ts, mean, ci) in curves.items():
        ax.plot(ts, mean, label=label)
        ax.fill_between(ts, mean - ci, mean + ci, alpha=0.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = f"plots_bonus/{fname}.png"
    plt.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.close()


def plot_budget_sweep(budget_results, title, fname):
    curves = {f"budget={b}": v for b, v in sorted(budget_results.items())}
    plot_comparison(curves, title=title, fname=fname)


def plot_reacher_teachers(teacher_results, fname):
    plot_comparison(
        teacher_results,
        title="PEBBLE on Reacher – GT Return by Teacher Reward Type",
        fname=fname,
        ylabel="Avg Undiscounted GT Return (teacher's own reward)",
    )


def plot_reacher_common(curves, fname):
    plot_comparison(
        curves,
        title="Q3.3 – PEBBLE on Reacher | All teachers evaluated on Rb (common metric)",
        fname=fname,
        ylabel="Avg Undiscounted GT Return (Rb – in-target sparse)",
    )
