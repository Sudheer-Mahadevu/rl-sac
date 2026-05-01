import torch
import numpy as np

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr  = 0
        self.size = 0
        self.obs      = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.actions  = np.zeros((max_size, act_dim), dtype=np.float32)
        self.rewards  = np.zeros((max_size, 1),       dtype=np.float32)
        self.dones    = np.zeros((max_size, 1),       dtype=np.float32)

    def add(self, obs, next_obs, action, reward, done):
        self.obs[self.ptr]      = obs
        self.next_obs[self.ptr] = next_obs
        self.actions[self.ptr]  = action
        self.rewards[self.ptr]  = reward
        self.dones[self.ptr]    = done
        self.ptr  = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.obs[idx]).to(DEVICE),
            torch.FloatTensor(self.next_obs[idx]).to(DEVICE),
            torch.FloatTensor(self.actions[idx]).to(DEVICE),
            torch.FloatTensor(self.rewards[idx]).to(DEVICE),
            torch.FloatTensor(self.dones[idx]).to(DEVICE),
        )

    def __len__(self):
        return self.size