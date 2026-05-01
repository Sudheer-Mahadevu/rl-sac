# ── 6. Training utilities ─────────────────────────────────────────────────────
import time
import numpy as np

def evaluate_agent(agent, eval_env, n_episodes=20):
    """Deterministic rollouts; returns list of undiscounted returns."""
    returns = []
    for _ in range(n_episodes):
        obs, _ = eval_env.reset()
        # print(f"Initial Eval Obs: {obs}")
        total, done = 0.0, False
        while not done:
            action = agent.select_action(obs, evaluate=True)
            obs, r, terminated, truncated, _ = eval_env.step(action)
            total += r
            done   = terminated or truncated
        returns.append(total)
    return returns


def train_agent(
    agent,
    env,
    eval_env,
    total_steps   = 100_000,
    eval_freq     = 10_000,
    eval_episodes = 20,
    verbose       = False,
):
    """
    Standard training loop.
    Returns list of (timestep, mean_return, std_return).
    """
    start_t = time.perf_counter()
    log = []
    all_episode_returns = []
    all_episode_ts = []
    all_episode_alpha = []
    obs, _ = env.reset()

    rets   = evaluate_agent(agent, eval_env, eval_episodes)
    mean_r = float(np.mean(rets))
    std_r  = float(np.std(rets))
    log.append((0, mean_r, std_r))
    curr_t = time.perf_counter()
    if verbose:
        print(f'  t=0  eval mean={mean_r:.2f}  std={std_r:.2f} time={(curr_t-start_t):.3f}', flush = True)
    
    episode_returns = 0
    episode_ts = 0
    for t in range(1, total_steps + 1):
        action   = agent.select_action(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done     = terminated or truncated

        episode_ts += 1
        episode_returns += reward

        agent.store(obs, next_obs, action, reward, terminated)
        agent.update()

        obs = next_obs
        if done:
            all_episode_returns.append(episode_returns)
            all_episode_ts.append(episode_ts)
            all_episode_alpha.append(agent.alpha.detach().cpu().item())
            episode_returns = 0
            episode_ts = 0
            obs, _ = env.reset()
            # print(f"Initial obs of a train episode: {obs}")

        if t % eval_freq == 0:
            rets   = evaluate_agent(agent, eval_env, eval_episodes)
            mean_r = float(np.mean(rets))
            std_r  = float(np.std(rets))
            log.append((t, mean_r, std_r))
            curr_t = time.perf_counter()
            if verbose:
                print(f'  t={t:>7}  eval mean={mean_r:.2f}  std={std_r:.2f} time={(curr_t-start_t):.3f}', flush = True)
                start_t = curr_t

    return log, all_episode_returns, all_episode_ts, all_episode_alpha