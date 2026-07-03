import numpy as np

HOVER_BUFFER_SIZE = 100_000


def evaluate(agent, env, n_episodes=20):
    returns = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done   = False
        total  = 0.0
        while not done:
            action = agent.select_action(obs, evaluate=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done   = terminated or truncated
        returns.append(total)
    return returns


def train(agent, env, eval_env,
          total_steps=300_000, eval_freq=10_000, eval_episodes=20, verbose=True):
    log = []
    obs, _ = env.reset()
    ep_reward = 0.0
    ep_count  = 0

    for t in range(1, total_steps + 1):
        action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.store(obs, next_obs, action, reward, terminated, truncated)
        agent.update()

        obs        = next_obs
        ep_reward += reward

        if done:
            ep_count += 1
            if verbose and ep_count % 50 == 0:
                print(f"  step={t:>7}  ep={ep_count}  ep_return={ep_reward:.1f}")
            obs, _    = env.reset()
            ep_reward = 0.0

        if t % eval_freq == 0:
            returns = evaluate(agent, eval_env, n_episodes=eval_episodes)
            mean_r  = float(np.mean(returns))
            std_r   = float(np.std(returns))
            log.append((t, mean_r, std_r))
            print(f"[Eval t={t}] mean={mean_r:.1f}  std={std_r:.1f}")

    return log


def _run_phase(agent, env, eval_env, steps, t_offset, eval_freq, eval_episodes):
    log = []
    obs, _ = env.reset()
    for t in range(1, steps + 1):
        action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.store(obs, next_obs, action, reward, terminated, truncated)
        agent.update()
        obs = next_obs
        if done:
            obs, _ = env.reset()
        if t % eval_freq == 0:
            returns = evaluate(agent, eval_env, n_episodes=eval_episodes)
            mean_r  = float(np.mean(returns))
            std_r   = float(np.std(returns))
            log.append((t_offset + t, mean_r, std_r))
            print(f"  [Eval t={t_offset + t}] mean={mean_r:.1f}  std={std_r:.1f}")
    return log


def train_hover(
    agent_fixed, agent_auto,
    env_fixed, env_auto,
    eval_env_fixed, eval_env_auto,
    phase1_steps=200_000, phase2_steps=200_000,
    eval_freq=10_000, eval_episodes=20,
):
    print("=== Phase 1: hover +200 ===")
    log_fixed = _run_phase(agent_fixed, env_fixed, eval_env_fixed,
                           phase1_steps, 0, eval_freq, eval_episodes)
    log_auto  = _run_phase(agent_auto,  env_auto,  eval_env_auto,
                           phase1_steps, 0, eval_freq, eval_episodes)

    for e in [env_fixed, env_auto, eval_env_fixed, eval_env_auto]:
        e.set_hover_reward(-100)

    print("=== Phase 2: hover -100 ===")
    log_fixed += _run_phase(agent_fixed, env_fixed, eval_env_fixed,
                            phase2_steps, phase1_steps, eval_freq, eval_episodes)
    log_auto  += _run_phase(agent_auto,  env_auto,  eval_env_auto,
                            phase2_steps, phase1_steps, eval_freq, eval_episodes)

    return log_fixed, log_auto
