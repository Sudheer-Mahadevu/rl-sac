import numpy as np

try:
    from dm_control import suite as dm_suite
    DMC_AVAILABLE = True
except ImportError:
    DMC_AVAILABLE = False
    print("[reacher_env] dm_control not found – Reacher experiments will be skipped.")


def _extract_obs(time_step):
    return np.concatenate(
        [v.flatten() for v in time_step.observation.values()]
    ).astype(np.float32)


def _in_target(time_step):
    to_target = time_step.observation.get("to_target", None)
    if to_target is None:
        return False
    return float(np.linalg.norm(to_target)) < 0.02


def _near_zero_velocity(time_step):
    velocity = time_step.observation.get("velocity", np.array([1.0, 1.0]))
    return float(np.linalg.norm(velocity)) < 0.05


class ReacherEnv:
    """
    DMControl Reacher-Easy with three reward formulations (Ra, Rb, Rc).

    Ra: r=1 if in target, else -dist - ||action||^2
    Rb: r=1 if in target, else 0  (sparse, fixed-length episodes)
    Rc: r=-1 every step; episode terminates on reaching target with near-zero velocity
    """

    def __init__(self, reward_type="Rb", seed=0, max_steps=1000):
        assert DMC_AVAILABLE, "dm_control is required for Reacher experiments."
        assert reward_type in ("Ra", "Rb", "Rc"), f"Unknown reward_type: {reward_type}"
        self.reward_type = reward_type
        self.max_steps   = max_steps

        self._env = dm_suite.load("reacher", "easy", task_kwargs={"random": seed})

        ts = self._env.reset()
        obs = _extract_obs(ts)
        self._obs_dim = obs.shape[0]
        self._act_dim = self._env.action_spec().shape[0]
        self._act_min = float(self._env.action_spec().minimum[0])
        self._act_max = float(self._env.action_spec().maximum[0])
        self._step_count = 0

    @property
    def obs_dim(self):
        return self._obs_dim

    @property
    def act_dim(self):
        return self._act_dim

    def reset(self):
        self._step_count = 0
        ts = self._env.reset()
        self._last_ts = ts
        return _extract_obs(ts)

    def step(self, action):
        action = np.clip(action, self._act_min, self._act_max)
        ts = self._env.step(action)
        self._last_ts = ts
        self._step_count += 1

        obs       = _extract_obs(ts)
        reward    = self._compute_reward(ts, action)
        in_target = _in_target(ts)

        if self.reward_type == "Rc":
            terminated = in_target and _near_zero_velocity(ts)
            truncated  = (not terminated) and (self._step_count >= self.max_steps)
        else:
            terminated = False
            truncated  = self._step_count >= self.max_steps

        return obs, reward, terminated, truncated, {}

    def close(self):
        pass

    def _compute_reward(self, ts, action):
        in_tgt = _in_target(ts)
        if self.reward_type == "Ra":
            if in_tgt:
                return 1.0
            dist     = float(np.linalg.norm(ts.observation.get("to_target", np.zeros(2))))
            act_cost = float(np.sum(action ** 2))
            return -dist - act_cost
        elif self.reward_type == "Rb":
            return 1.0 if in_tgt else 0.0
        else:  # Rc
            return -1.0


def make_reacher(reward_type, seed=0, max_steps=1000):
    return ReacherEnv(reward_type=reward_type, seed=seed, max_steps=max_steps)
