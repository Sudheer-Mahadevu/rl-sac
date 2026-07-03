"""
main_bonus.py – DA6400 PA3 Bonus: PEBBLE experiments.

Usage:
  python main_bonus.py --exp pendulum              # Q3.1
  python main_bonus.py --exp budget                # Q3.2
  python main_bonus.py --exp reacher               # Q3.3
  python main_bonus.py --exp all                   # everything

  # For splitting 15 seeds across multiple Kaggle notebooks:
  python main_bonus.py --exp pendulum --seeds 4 --seed_start 0
  python main_bonus.py --exp pendulum --seeds 4 --seed_start 4
  python main_bonus.py --exp pendulum --seeds 4 --seed_start 8
  python main_bonus.py --exp pendulum --seeds 3 --seed_start 12
"""

import argparse
import pickle
import os
import numpy as np
import torch

from pendulum_env import make_pendulum
from reward_model import RewardModel, PreferenceBuffer
from teacher import SimulatedTeacher
from pebble_agent import PEBBLE
from sac_continuous import SACContinuous
from pebble_training import train_pebble_pendulum, train_sac_pendulum, train_pebble_reacher
from pebble_plot import aggregate, plot_comparison, plot_budget_sweep, plot_reacher_teachers, plot_reacher_common

os.makedirs("results_bonus", exist_ok=True)

THETA_TARGETS  = [0, -60, 90, 120, -150]
BUDGET_DEFAULT = 500
EVAL_FREQ      = 10_000
EVAL_EPISODES  = 20
SEG_LEN        = 50
QUERY_FREQ     = 5_000
QUERIES_ROUND  = 10


def _make_pebble(obs_dim, act_dim, budget=BUDGET_DEFAULT):
    return PEBBLE(
        obs_dim=obs_dim,
        act_dim=act_dim,
        reward_model=RewardModel(obs_dim, act_dim, ensemble_size=3, hidden_dim=256),
        teacher=SimulatedTeacher(teacher_noise=0.0),
        seg_len=SEG_LEN,
        query_freq=QUERY_FREQ,
        queries_per_round=QUERIES_ROUND,
        budget=budget,
        reward_update_epochs=10,
        sac_kwargs=dict(auto_alpha=True, random_steps=10_000),
    )


def _make_sac(obs_dim, act_dim):
    return SACContinuous(obs_dim, act_dim, auto_alpha=True, random_steps=10_000)


def run_pendulum(seeds=5, total_steps=100_000, budget=BUDGET_DEFAULT, seed_start=0):
    print("\n===== Bonus Q3.1 – PEBBLE vs SAC on Modified Pendulum =====")
    for theta in THETA_TARGETS:
        print(f"\n  θ_target = {theta}°")
        logs_pebble, logs_sac = [], []

        for seed in range(seed_start, seed_start + seeds):
            print(f"  -- seed {seed} --")
            np.random.seed(seed)
            torch.manual_seed(seed)

            env_p      = make_pendulum(theta, seed=seed)
            eval_env_p = make_pendulum(theta, seed=seed + 1000)
            env_s      = make_pendulum(theta, seed=seed)
            eval_env_s = make_pendulum(theta, seed=seed + 1000)

            obs_dim = env_p.observation_space.shape[0]
            act_dim = env_p.action_space.shape[0]

            pebble = _make_pebble(obs_dim, act_dim, budget=budget)
            log_p  = train_pebble_pendulum(pebble, env_p, eval_env_p,
                                           total_steps=total_steps,
                                           eval_freq=EVAL_FREQ,
                                           eval_episodes=EVAL_EPISODES)
            logs_pebble.append(log_p)
            env_p.close(); eval_env_p.close()

            sac   = _make_sac(obs_dim, act_dim)
            log_s = train_sac_pendulum(sac, env_s, eval_env_s,
                                       total_steps=total_steps,
                                       eval_freq=EVAL_FREQ,
                                       eval_episodes=EVAL_EPISODES)
            logs_sac.append(log_s)
            env_s.close(); eval_env_s.close()

        key = f"theta_{theta}"
        with open(f"results_bonus/pendulum_{key}_s{seed_start}_n{seeds}.pkl", "wb") as f:
            pickle.dump({"pebble": logs_pebble, "sac": logs_sac}, f)

        ts_p, mean_p, ci_p = aggregate(logs_pebble)
        ts_s, mean_s, ci_s = aggregate(logs_sac)
        plot_comparison(
            {f"PEBBLE (budget={budget})": (ts_p, mean_p, ci_p),
             "SAC (GT reward)":           (ts_s, mean_s, ci_s)},
            title=f"Q3.1 – PEBBLE vs SAC | θ_target={theta}°",
            fname=f"q31_pendulum_theta{theta}_s{seed_start}",
        )


def run_budget_sweep(seeds=5, total_steps=100_000,
                     budgets=None, seed_start=0):
    if budgets is None:
        budgets = [50, 100, 250, 500, 1000]
    theta = 90
    print(f"\n===== Bonus Q3.2 – Budget Sweep on Pendulum (θ={theta}°) =====")

    budget_results = {}
    logs_sac = []

    for budget in budgets:
        logs_pebble = []
        for seed in range(seed_start, seed_start + seeds):
            print(f"  budget={budget} seed={seed}")
            np.random.seed(seed)
            torch.manual_seed(seed)

            env_p      = make_pendulum(theta, seed=seed)
            eval_env_p = make_pendulum(theta, seed=seed + 1000)
            obs_dim    = env_p.observation_space.shape[0]
            act_dim    = env_p.action_space.shape[0]

            pebble = _make_pebble(obs_dim, act_dim, budget=budget)
            log_p  = train_pebble_pendulum(pebble, env_p, eval_env_p,
                                           total_steps=total_steps,
                                           eval_freq=EVAL_FREQ,
                                           eval_episodes=EVAL_EPISODES)
            logs_pebble.append(log_p)
            env_p.close(); eval_env_p.close()

        budget_results[budget] = aggregate(logs_pebble)

    for seed in range(seed_start, seed_start + seeds):
        np.random.seed(seed)
        torch.manual_seed(seed)
        env_s      = make_pendulum(theta, seed=seed)
        eval_env_s = make_pendulum(theta, seed=seed + 1000)
        obs_dim    = env_s.observation_space.shape[0]
        act_dim    = env_s.action_space.shape[0]
        sac   = _make_sac(obs_dim, act_dim)
        log_s = train_sac_pendulum(sac, env_s, eval_env_s,
                                   total_steps=total_steps,
                                   eval_freq=EVAL_FREQ,
                                   eval_episodes=EVAL_EPISODES)
        logs_sac.append(log_s)
        env_s.close(); eval_env_s.close()

    with open(f"results_bonus/budget_sweep_s{seed_start}_n{seeds}.pkl", "wb") as f:
        pickle.dump({"budgets": budget_results, "sac": logs_sac}, f)

    plot_budget_sweep(
        {b: v for b, v in budget_results.items() if isinstance(b, int)},
        title=f"Q3.2 – PEBBLE Budget Sweep | θ_target={theta}°",
        fname=f"q32_budget_sweep_s{seed_start}",
    )
    all_curves = {f"PEBBLE budget={b}": v
                  for b, v in budget_results.items() if isinstance(b, int)}
    all_curves["SAC (GT reward)"] = aggregate(logs_sac)
    plot_comparison(all_curves,
                    title=f"Q3.2 – Budget Sweep + SAC Baseline | θ={theta}°",
                    fname=f"q32_budget_sweep_with_sac_s{seed_start}")


def run_reacher(seeds=5, total_steps=300_000, budget=BUDGET_DEFAULT, seed_start=0):
    try:
        from reacher_env import make_reacher, DMC_AVAILABLE
        if not DMC_AVAILABLE:
            print("dm_control not available – skipping Reacher experiment.")
            return
    except ImportError:
        print("reacher_env.py not found – skipping.")
        return

    print("\n===== Bonus Q3.3 – PEBBLE on Reacher (Easy) =====")
    reward_types = ["Ra", "Rb", "Rc"]
    teacher_own_logs = {r: [] for r in reward_types}
    teacher_rb_logs  = {r: [] for r in reward_types}

    for rtype in reward_types:
        print(f"\n  Teacher = {rtype}")
        for seed in range(seed_start, seed_start + seeds):
            print(f"  -- seed {seed} --")
            np.random.seed(seed)
            torch.manual_seed(seed)

            env      = make_reacher(rtype, seed=seed,        max_steps=1000)
            eval_env = make_reacher(rtype, seed=seed + 1000, max_steps=1000)

            if rtype != "Rb":
                rb_eval_env = make_reacher("Rb", seed=seed + 2000, max_steps=1000)
                extra_eval  = {"Rb": rb_eval_env}
            else:
                rb_eval_env = None
                extra_eval  = {}

            pebble = _make_pebble(env.obs_dim, env.act_dim, budget=budget)
            result = train_pebble_reacher(
                pebble, env, eval_env,
                total_steps=total_steps,
                eval_freq=EVAL_FREQ,
                eval_episodes=EVAL_EPISODES,
                extra_eval_envs=extra_eval,
            )

            teacher_own_logs[rtype].append(result["own"])
            teacher_rb_logs[rtype].append(
                result["Rb"] if "Rb" in result else result["own"]
            )

            env.close(); eval_env.close()
            if rb_eval_env is not None:
                rb_eval_env.close()

        with open(f"results_bonus/reacher_{rtype}_own_s{seed_start}_n{seeds}.pkl", "wb") as f:
            pickle.dump(teacher_own_logs[rtype], f)
        with open(f"results_bonus/reacher_{rtype}_rb_s{seed_start}_n{seeds}.pkl", "wb") as f:
            pickle.dump(teacher_rb_logs[rtype], f)

    teacher_own_results = {
        rtype: aggregate(teacher_own_logs[rtype]) for rtype in reward_types
    }
    plot_reacher_teachers(teacher_own_results, fname=f"q33_reacher_teachers_s{seed_start}")

    teacher_rb_results = {
        f"PEBBLE-{rtype} teacher": aggregate(teacher_rb_logs[rtype])
        for rtype in reward_types
    }
    plot_reacher_common(teacher_rb_results, fname=f"q33_reacher_common_rb_s{seed_start}")

    print("\nFinal Rb-metric returns:")
    for label, (ts, mean, ci) in teacher_rb_results.items():
        print(f"  {label}: {mean[-1]:.3f} ± {ci[-1]:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp",        type=str, default="all",
                        choices=["pendulum", "budget", "reacher", "all"])
    parser.add_argument("--seeds",      type=int, default=5)
    parser.add_argument("--seed_start", type=int, default=0,
                        help="First seed index (for splitting across notebooks)")
    parser.add_argument("--steps",      type=int, default=100_000,
                        help="Steps for Pendulum experiments (Reacher uses 3×)")
    parser.add_argument("--budget",     type=int, default=BUDGET_DEFAULT)
    args = parser.parse_args()

    if args.exp in ("pendulum", "all"):
        run_pendulum(seeds=args.seeds, total_steps=args.steps,
                     budget=args.budget, seed_start=args.seed_start)

    if args.exp in ("budget", "all"):
        run_budget_sweep(seeds=args.seeds, total_steps=args.steps,
                         seed_start=args.seed_start)

    if args.exp in ("reacher", "all"):
        run_reacher(seeds=args.seeds, total_steps=args.steps * 3,
                    budget=args.budget, seed_start=args.seed_start)
