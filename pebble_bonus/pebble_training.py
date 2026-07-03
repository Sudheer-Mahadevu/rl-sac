import numpy as np


def _reset(env):
    out = env.reset()
    return out[0] if isinstance(out, tuple) else out


def _step(env, action):
    out = env.step(action)
    if len(out) == 5:
        obs, reward, terminated, truncated, _ = out
    elif len(out) == 4:
        obs, reward, terminated, truncated = out
    else:
        raise ValueError(f"Unexpected env.step output length: {len(out)}")
    return obs, reward, terminated, truncated


def evaluate_gt(agent, env, n_episodes=20):
    returns = []
    for _ in range(n_episodes):
        obs   = _reset(env)
        total = 0.0
        done  = False
        while not done:
            action = agent.select_action(obs, evaluate=True)
            obs, reward, terminated, truncated = _step(env, action)
            total += reward
            done   = terminated or truncated
        returns.append(total)
    return returns


def train_pebble_pendulum(
    pebble_agent, env, eval_env,
    total_steps=100_000, eval_freq=10_000, eval_episodes=20, verbose=True,
):
    log = []
    obs = _reset(env)
    seg_obs, seg_act, seg_gt = [], [], []

    for t in range(1, total_steps + 1):
        action = pebble_agent.select_action(obs)
        next_obs, gt_reward, terminated, truncated = _step(env, action)
        done = terminated or truncated

        pebble_agent.store(obs, next_obs, action, gt_reward, terminated, truncated)
        seg_obs.append(obs.copy())
        seg_act.append(action.copy())
        seg_gt.append(gt_reward)
        pebble_agent.update()
        pebble_agent.end_of_step(seg_obs, seg_act, seg_gt, done)

        obs = next_obs

        if done:
            obs = _reset(env)
            seg_obs, seg_act, seg_gt = [], [], []
        elif len(seg_obs) >= pebble_agent.seg_len:
            seg_obs, seg_act, seg_gt = [], [], []

        if t % eval_freq == 0:
            returns = evaluate_gt(pebble_agent, eval_env, n_episodes=eval_episodes)
            mean_r  = float(np.mean(returns))
            std_r   = float(np.std(returns))
            log.append((t, mean_r, std_r))
            if verbose:
                print(f"  [Eval t={t:>7}] GT mean={mean_r:.2f}  std={std_r:.2f}"
                      f"  queries_used={pebble_agent.queries_used}")

    return log


def train_pebble_reacher(
    pebble_agent, env, eval_env,
    total_steps=300_000, eval_freq=10_000, eval_episodes=20,
    verbose=True, extra_eval_envs=None,
):
    extra_eval_envs = extra_eval_envs or {}
    log_own   = []
    log_extra = {k: [] for k in extra_eval_envs}

    obs = env.reset()
    seg_obs, seg_act, seg_gt = [], [], []

    for t in range(1, total_steps + 1):
        action = pebble_agent.select_action(obs)
        next_obs, gt_reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        pebble_agent.store(obs, next_obs, action, gt_reward, terminated, truncated)
        seg_obs.append(obs.copy())
        seg_act.append(action.copy())
        seg_gt.append(gt_reward)
        pebble_agent.update()
        pebble_agent.end_of_step(seg_obs, seg_act, seg_gt, done)

        obs = next_obs

        if done:
            obs = env.reset()
            seg_obs, seg_act, seg_gt = [], [], []
        elif len(seg_obs) >= pebble_agent.seg_len:
            seg_obs, seg_act, seg_gt = [], [], []

        if t % eval_freq == 0:
            returns = evaluate_gt(pebble_agent, eval_env, n_episodes=eval_episodes)
            mean_r  = float(np.mean(returns))
            std_r   = float(np.std(returns))
            log_own.append((t, mean_r, std_r))
            if verbose:
                print(f"  [Eval t={t:>7}] own mean={mean_r:.2f}  std={std_r:.2f}"
                      f"  queries_used={pebble_agent.queries_used}")

            for label, xenv in extra_eval_envs.items():
                xreturns = evaluate_gt(pebble_agent, xenv, n_episodes=eval_episodes)
                xmean    = float(np.mean(xreturns))
                xstd     = float(np.std(xreturns))
                log_extra[label].append((t, xmean, xstd))
                if verbose:
                    print(f"  [Eval t={t:>7}] {label:>4} mean={xmean:.2f}  std={xstd:.2f}")

    result = {"own": log_own}
    result.update(log_extra)
    return result


def train_sac_pendulum(
    sac_agent, env, eval_env,
    total_steps=100_000, eval_freq=10_000, eval_episodes=20, verbose=True,
):
    log = []
    obs = _reset(env)

    for t in range(1, total_steps + 1):
        action = sac_agent.select_action(obs)
        next_obs, gt_reward, terminated, truncated = _step(env, action)
        done = terminated or truncated

        sac_agent.store(obs, next_obs, action, gt_reward, terminated, truncated)
        sac_agent.update()

        obs = next_obs
        if done:
            obs = _reset(env)

        if t % eval_freq == 0:
            returns = evaluate_gt(sac_agent, eval_env, n_episodes=eval_episodes)
            mean_r  = float(np.mean(returns))
            std_r   = float(np.std(returns))
            log.append((t, mean_r, std_r))
            if verbose:
                print(f"  [SAC Eval t={t:>7}] GT mean={mean_r:.2f}  std={std_r:.2f}")

    return log
