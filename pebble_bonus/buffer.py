import numpy as np


class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, max_size=int(1e6), discrete=False):
        self.max_size = max_size
        self.ptr      = 0
        self.size     = 0

        self.obs              = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.next_obs         = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.rewards          = np.zeros((max_size, 1),       dtype=np.float32)
        self.not_dones        = np.zeros((max_size, 1),       dtype=np.float32)
        self.not_dones_no_max = np.zeros((max_size, 1),       dtype=np.float32)
        self.actions = (
            np.zeros((max_size,),        dtype=np.int64)   if discrete else
            np.zeros((max_size, act_dim), dtype=np.float32)
        )

    def add(self, obs, next_obs, action, reward, terminated, truncated):
        self.obs[self.ptr]              = obs
        self.next_obs[self.ptr]         = next_obs
        self.actions[self.ptr]          = action
        self.rewards[self.ptr]          = reward
        self.not_dones[self.ptr]        = float(not (terminated or truncated))
        self.not_dones_no_max[self.ptr] = float(not terminated)
        self.ptr  = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            self.obs[idx],
            self.next_obs[idx],
            self.actions[idx],
            self.rewards[idx],
            self.not_dones[idx],
            self.not_dones_no_max[idx],
        )

    def __len__(self):
        return self.size
