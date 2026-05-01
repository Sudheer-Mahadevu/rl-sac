import torch

# from environment import make_env
from new_env import make_env
from sac import SACAgent

WEIGHTS_PATH = 'output/weights/a_alpha_theta90_seed22.pt'

THETA = 90
seed = 42
env = make_env(THETA, render_mode='human', trunc_len=200)
obs_dim  = env.observation_space.shape[0]    # 3
act_dim  = env.action_space.shape[0]         # 1
agent = SACAgent(
                obs_dim, act_dim,
                hidden_dim=64,
                auto_alpha=True,
                init_alpha=0.2,
                random_steps=10_000)

ckpt = torch.load(WEIGHTS_PATH, map_location=torch.device('cpu'))
agent.actor.load_state_dict(ckpt['actor'])
agent.critic.load_state_dict(ckpt['critic'])
agent.actor.eval()
agent.critic.eval()
print(f"Weights loaded from {WEIGHTS_PATH}")

NUM_EPISODES = 10

env.reset(seed = seed)

returns = []
for _ in range(NUM_EPISODES):
    obs, _ = env.reset()
    # print(f"Initial Eval Obs: {obs}")
    total, done = 0.0, False
    while not done:
        action = agent.select_action(obs, evaluate=True)
        obs, r, terminated, truncated, _ = env.step(action)
        total += r
        done   = terminated or truncated
    returns.append(total)
    # time.sleep(5)
    print(f"Episode reward: {total}")

env.close()