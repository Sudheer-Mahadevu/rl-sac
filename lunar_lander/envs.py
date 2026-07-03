import gymnasium as gym

class HoverLunarLander(gym.Wrapper):

    def __init__(self, env: gym.Env, hover_reward: float = 200.0):
        super().__init__(env)
        self.hover_reward  = hover_reward
        self._hover_given  = False

    # called mid-training to flip +200 → -100
    def set_hover_reward(self, value: float) -> None:
        self.hover_reward = value

    def reset(self, **kwargs):
        self._hover_given = False
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        x, y = float(obs[0]), float(obs[1])
        # give hover reward at most ONCE per episode
        if not self._hover_given and abs(x) < 0.1 and 0.4 < y < 0.6:
            reward += self.hover_reward
            self._hover_given = True
        return obs, reward, terminated, truncated, info

def make_continuous_env(seed: int = 0) -> gym.Env:
    """Standard continuous LunarLander-v3 (Q2.2.1, Q2.2.2)."""
    env = gym.make("LunarLander-v3", continuous=True)
    env.reset(seed=seed)
    return env


def make_continuous_hover_env(seed: int = 0, hover_reward: float = 200.0) -> HoverLunarLander:
    """Hover-variant continuous LunarLander-v3 (Q2.2.3)."""
    base = gym.make("LunarLander-v3", continuous=True)
    env  = HoverLunarLander(base, hover_reward=hover_reward)
    env.reset(seed=seed)
    return env


def make_discrete_env(seed: int = 0) -> gym.Env:
    """Discrete LunarLander-v3 (Q2.2.4)."""
    env = gym.make("LunarLander-v3", continuous=False)
    env.reset(seed=seed)
    return env
