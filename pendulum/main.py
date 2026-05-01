import random
import numpy as np
import json
import torch
import os

from environment import make_env
from sac import SACAgent
from train import train_agent

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)
    
def make_run(angles, seeds, save_path, args):
    for theta in angles:
        auto_logs = []
        train_logs = { 'r': [], 't': [], 'a':[]} # list of ret arrays for each seed. similarly for t,a
        print(f"\n{'='*60}")
        print(f'θ_target = {theta:+4d}°   (auto-α SAC)')
        print(f"{'='*60}")
        for seed in seeds:
            set_seeds(seed)

            env      = make_env(theta)
            env.reset(seed = seed)
            eval_env = make_env(theta)
            eval_env.reset(seed = seed+5000)
            obs_dim  = env.observation_space.shape[0]    # 3
            act_dim  = env.action_space.shape[0]         # 1

            agent = SACAgent(
                obs_dim, act_dim,
                hidden_dim=args['hidden_dim'],
                auto_alpha=True,
                init_alpha=0.2,
                random_steps=10_000,
            )
            print(env.theta_target)
            log, ep_rets, ep_ts, ep_a = train_agent(
                agent, env, eval_env,
                total_steps=args['total_steps'],
                eval_freq=args['eval_freq'],
                eval_episodes=args['eval_episodes'],
                verbose=True,
            )
            auto_logs.append(log)
            train_logs['r'].append(ep_rets)
            train_logs['t'].append(ep_ts)
            train_logs['a'].append(ep_a)
            env.close(); eval_env.close()

            os.makedirs(f"{save_path}/weights", exist_ok = True)
            
            torch.save({"actor": agent.actor.state_dict(),
                        "critic": agent.critic.state_dict()},
                        f"{save_path}/weights/a_alpha_theta{theta}_seed{seed}.pt")
            print(f'  seed {seed:2d} done  |  final eval mean = {log[-1][1]:.2f}')

            info = {'eval_logs': auto_logs, 'train_logs': train_logs}
        
            os.makedirs(f"{save_path}/logs", exist_ok = True)
            file_path = f'{save_path}/logs/a_alpha_theta{theta}.json'
            with open(file_path, 'w') as f:
                json.dump(info, f, cls=NumpyEncoder)
            print(f'Logs updated')
    

if __name__ == '__main__':

    THETA_TARGETS  = [0, -10, 30, -60, 90, -90, 120, -150]   # degrees
    N_SEEDS        = 15
    TOTAL_STEPS    = 20_000     # Pendulum is simple; 100K is enough to converge
    EVAL_FREQ      = 10_000
    EVAL_EPISODES  = 20
    HIDDEN_DIM     = 64
    SEEDS = [22, 67, 8, 45, 212, 99, 27, 4, 2003, 89, 11, 5, 37, 502, 75]

    print(f'Targets : {THETA_TARGETS}')
    print(f'Seeds   : {N_SEEDS}   |  Total steps/run : {TOTAL_STEPS:,}')
    print(f'Total runs: {len(THETA_TARGETS) * N_SEEDS}')

    save_path = '.'

    num_seeds = 1

    args = {'hidden_dim': 64,
            'total_steps': 80_000,
            'eval_freq': 10_000,
            'eval_episodes': 20}

    make_run([0, 90], SEEDS[:num_seeds], save_path='output', args=args)



