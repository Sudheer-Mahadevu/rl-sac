import os
import json
import numpy as np
import torch
from envs import make_continuous_env, make_continuous_hover_env, make_discrete_env
from sac_continuous import SACContinuous
from sac_discrete import SACDiscrete
from dqn import DQN
from training import evaluate

def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def process_log_file(filepath: str, mean_ret: float, std_ret: float):
    if not os.path.exists(filepath):
        print(f"  [Error] File not found: {filepath}")
        return
        
    with open(filepath, 'r') as f:
        data = json.load(f)
        
    if len(data['timesteps']) > 0:
        last_step = data['timesteps'][-1]
        if last_step < 600000:
            print(f"  [WARNING] {filepath} stopped early at {last_step} steps! (Expected 600,000)")
        else:
            print(f"  [OK] {filepath} reached 600,000 steps.")
    else:
        print(f"  [WARNING] {filepath} is completely empty!")

    # 2. INJECT STEP 0 (Avoid duplicates)
    if len(data['timesteps']) > 0 and data['timesteps'][0] == 0:
        print(f"  [Skipped] Step 0 already exists in {filepath}\n")
        return
        
    data['timesteps'].insert(0, 0)
    data['mean_returns'].insert(0, float(mean_ret))
    data['std_returns'].insert(0, float(std_ret))
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  [Updated] Injected Step 0 -> {filepath}\n")


Q1_SEEDS = [7, 8, 14, 22, 23, 26, 28, 45, 67, 74, 76, 77, 82, 85, 87, 88, 95, 97, 212]
Q3_SEEDS = [0, 7, 8, 10, 22, 28, 41, 47, 67, 69, 70, 76, 84, 87, 90, 97]
Q4_SEEDS = [8, 12, 14, 20, 23, 31, 32, 36, 44, 50, 51, 58, 62, 68, 83, 84]

EVAL_EPISODES = 20

print("\n" + "="*50)
print("Evaluating Step 0 for Q2.2.1 (Continuous SAC)")
print("="*50)
for seed in Q1_SEEDS:
    _seed_everything(seed)
    eval_env = make_continuous_env(seed=seed + 10_000)
    obs_dim = eval_env.observation_space.shape[0]
    act_dim = eval_env.action_space.shape[0]
    
    agent = SACContinuous(obs_dim, act_dim, hidden_dim=256, auto_alpha=True)
    rets = evaluate(agent, eval_env, n_episodes=EVAL_EPISODES)
    

    filepath = f"../lander_1_output/returns_q1_seed{seed}.json"
    process_log_file(filepath, np.mean(rets), np.std(rets))     
    eval_env.close()

print("\n" + "="*50)
print("Evaluating Step 0 for Q2.2.3 (Hover - Fixed & Auto)")
print("="*50)
for seed in Q3_SEEDS:
    _seed_everything(seed)
    eval_env = make_continuous_hover_env(seed=seed + 10_000, hover_reward=200)
    obs_dim = eval_env.observation_space.shape[0]
    act_dim = eval_env.action_space.shape[0]
    
    agent_fixed = SACContinuous(obs_dim, act_dim, hidden_dim=256, auto_alpha=False, init_alpha=0.01)
    agent_auto = SACContinuous(obs_dim, act_dim, hidden_dim=256, auto_alpha=True)
    
    rets_fixed = evaluate(agent_fixed, eval_env, n_episodes=EVAL_EPISODES)
    rets_auto = evaluate(agent_auto, eval_env, n_episodes=EVAL_EPISODES)

    path_fixed = f"../lander_3_output/returns_q3_fixed_seed{seed}.json"
    path_auto = f"../lander_3_output/returns_q3_auto_seed{seed}.json"
    
    process_log_file(path_fixed, np.mean(rets_fixed), np.std(rets_fixed))
    process_log_file(path_auto, np.mean(rets_auto), np.std(rets_auto))
    eval_env.close()

print("\n" + "="*50)
print("Evaluating Step 0 for Q2.2.4 (Discrete SAC & DQN)")
print("="*50)
for seed in Q4_SEEDS:
    _seed_everything(seed)
    eval_env = make_discrete_env(seed=seed + 10_000)
    obs_dim = eval_env.observation_space.shape[0]
    n_actions = eval_env.action_space.n
    
    agent_sac = SACDiscrete(obs_dim, n_actions, hidden_dim=256)
    agent_dqn = DQN(obs_dim, n_actions, hidden_dim=256)
    
    rets_sac = evaluate(agent_sac, eval_env, n_episodes=EVAL_EPISODES)
    rets_dqn = evaluate(agent_dqn, eval_env, n_episodes=EVAL_EPISODES)
    
    path_sac = f"../lander_4_output/returns_q4_sac_seed{seed}.json"
    path_dqn = f"../lander_4_output/returns_q4_dqn_seed{seed}.json"
    
    process_log_file(path_sac, np.mean(rets_sac), np.std(rets_sac))
    process_log_file(path_dqn, np.mean(rets_dqn), np.std(rets_dqn))
    eval_env.close()

print("\n All files verified and Step 0 successfully injected.")