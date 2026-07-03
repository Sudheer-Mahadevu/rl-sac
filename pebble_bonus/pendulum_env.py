import numpy as np
import gymnasium as gym


class ModifiedPendulum(gym.Wrapper):
    """Pendulum-v1 with a configurable target angle theta_target (degrees)."""

    def __init__(self, theta_target_deg=0.0, seed=0):
        env = gym.make("Pendulum-v1")
        super().__init__(env)
        self.theta_target = np.deg2rad(theta_target_deg)
        self.reset(seed=seed)

    def compute_gt_reward(self, obs, action):
        cos_th, sin_th, omega = obs[0], obs[1], obs[2]
        theta      = np.arctan2(sin_th, cos_th)
        angle_diff = _wrap_angle(theta - self.theta_target)
        u          = np.clip(action[0], -2.0, 2.0)
        return -(angle_diff ** 2) - 0.1 * (omega ** 2) - 0.001 * (u ** 2)

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        return obs, self.compute_gt_reward(obs, action), terminated, truncated, info


def _wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def make_pendulum(theta_target_deg, seed=0):
    return ModifiedPendulum(theta_target_deg=theta_target_deg, seed=seed)
