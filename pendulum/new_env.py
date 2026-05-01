import pygame
import math
import gymnasium as gym
import numpy as np

M, G, L = 1.0, 10.0, 1.0                    # Pendulum-v1 physical constants
TAU_FF_SCALE = M * G * L / 2                 # = 5.0


def angle_normalize(x: float) -> float:
    """Wrap angle to [-π, π]."""
    return ((x + math.pi) % (2 * math.pi)) - math.pi

class TargetPendulum(gym.Wrapper):
    def __init__(self, theta_target_deg: float, render_mode: str = None,trunc_len = 1000):
        # 1. Store the user's intent in a custom variable
        self.display_mode = render_mode
        
        # 2. Force the internal environment to 'rgb_array' so we can capture frames
        # even if the user wants 'human' (this prevents the blinking).
        internal_render = 'rgb_array' if render_mode == 'human' else render_mode
        
        env = gym.make('Pendulum-v1', max_episode_steps=trunc_len, render_mode=internal_render)
        super().__init__(env)
        
        # 5.0 is required to stay at 90 degrees
        self.env.unwrapped.max_torque = 6.0
        self.action_space = gym.spaces.Box(-6.0, 6.0, shape=(1,), dtype=np.float32)
        self.theta_target = math.radians(theta_target_deg)
        self.tau_ff = TAU_FF_SCALE * math.sin(self.theta_target)
        
        self.last_obs = None
        self.last_action = 0.0
        
        # 3. Setup our custom window variables
        self.window = None
        self.clock = None
        if self.display_mode == "human":
            pygame.font.init()
            self.font = pygame.font.SysFont('Arial', 20, bold=True)

    def step(self, action):
        self.last_action = float(action[0] * 6.0)
        obs, reward, terminated, truncated, info = self.env.step(action*6)
        self.last_obs = obs
        
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
        
        # Use our custom variable to check for rendering
        if self.display_mode == "human":
            self.render()
            
        return obs, reward, terminated, truncated, info

    def render(self):
        # Grab the frame from the internal rgb_array env
        frame = self.env.render()

        if self.display_mode == "human":
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return frame
                
            if self.window is None:
                pygame.init()
                self.window = pygame.display.set_mode((frame.shape[1], frame.shape[0]))
                pygame.display.set_caption("Target Pendulum Overlay")
                self.clock = pygame.time.Clock()

            # Render logic (same as before)
            surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
            self.window.blit(surf, (0, 0))

            if self.last_obs is not None:
                cos_th, sin_th, th_dot = self.last_obs
                theta = math.atan2(sin_th, cos_th)
                deg = '°'
                stats = [
                    f"Target: {self.theta_target:.2f} rad ({math.degrees(self.theta_target):.2f}°)",
                    f"Current: {theta:.2f} rad ({math.degrees(theta):.2f}°)",
                    f"Ang Vel: {th_dot:.2f}",
                    f"Torque: {self.last_action:.2f}",
                    f"Torque Target: {self.tau_ff:.2f}"
                ]

                for i, text in enumerate(stats):
                    text_surf = self.font.render(text, True, (0, 0, 0))
                    self.window.blit(text_surf, (20, 20 + (i * 25)))

            pygame.display.flip()
            self.clock.tick(30) # Maintain 30FPS

        return frame

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
        super().close()
def make_env(theta_deg: float, trunc_len = 1000, render_mode: str = None) -> TargetPendulum:
    env = TargetPendulum(theta_deg, render_mode=render_mode, trunc_len = trunc_len)
    return env