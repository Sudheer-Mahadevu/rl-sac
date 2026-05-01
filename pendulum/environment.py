import gymnasium as gym
import math
import numpy as np

M, G, L = 1.0, 10.0, 1.0                    # Pendulum-v1 physical constants
TAU_FF_SCALE = M * G * L / 2                 # = 5.0


def angle_normalize(x: float) -> float:
    """Wrap angle to [-π, π]."""
    return ((x + math.pi) % (2 * math.pi)) - math.pi


class TargetPendulum(gym.Wrapper):
    """
    Modifies Pendulum-v1 to target an arbitrary angle θ_target (degrees).

    Observation: unchanged [cos θ, sin θ, θ̇]
    Action:      unchanged [τ] ∈ [-2, 2]  (already normalised by gym)
    Reward:      custom (see module docstring)
    """

    def __init__(self, theta_target_deg: float,render_mode: str = None ):
        env = gym.make('Pendulum-v1', max_episode_steps = 1000, render_mode=render_mode)
        super().__init__(env)
        self.theta_target = math.radians(theta_target_deg)

        # 5.0 is required to stay at 90 degrees
        self.env.unwrapped.max_torque = 6.0
        self.action_space = gym.spaces.Box(-6.0, 6.0, shape=(1,), dtype=np.float32)
        
        # feed-forward torque to balance gravity at target angle
        self.tau_ff = TAU_FF_SCALE * math.sin(self.theta_target)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, _orig_reward, terminated, truncated, info = self.env.step(action*6)
        cos_th, sin_th, th_dot = obs

        # Current angle (from cos/sin, range [-π, π])
        theta = math.atan2(sin_th, cos_th)

        # Angle error (shortest arc to target)
        angle_err = angle_normalize(theta - self.theta_target)

        # Torque: gymnasium clips to [-2, 2] internally but the raw action
        # is what the actor outputs (already squashed by tanh → [-1,1] and
        # scaled to [-2,2] inside gym***NO***).  We read the clipped torque from info
        # if available, otherwise use 2 * action[0] as proxy.
        tau = float(np.clip(action[0] * 6.0, -6.0, 6.0))   # max_torque = 6.0
        tau_excess = tau - self.tau_ff

        reward = -(angle_err ** 2
                   + 0.5 * abs(angle_err)
                   + 0.1  * th_dot ** 2
                   + 0.001 * tau_excess ** 2)

        return obs, reward, terminated, truncated, info


def make_env(theta_deg: float, render_mode: str = None) -> TargetPendulum:
    env = TargetPendulum(theta_deg, render_mode=render_mode)
    return env