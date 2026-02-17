import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity, obs_shape, action_shape, device):
        self._capacity = capacity
        self._obs_shape = obs_shape
        self._action_shape = action_shape
        self._device = device

        self.obs_buf = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.next_obs_buf = np.zeros((capacity, *obs_shape), np.uint8)
        self.action_buf = np.zeros((capacity, *action_shape), dtype=np.float32)
        self.reward_buf = np.zeros(capacity, dtype=np.float32)
        self.done_buf = np.zeros(capacity, dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.action_buf[self.ptr] = action
        self.reward_buf[self.ptr] = reward
        self.done_buf[self.ptr] = done

        self.ptr = (self.ptr + 1) % self._capacity
        self.size = min(self.ptr + 1, self._capacity)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)

        obs = torch.as_tensor(self.obs_buf[idxs], device=self._device).float() / 255.0
        next_obs = (
            torch.as_tensor(self.next_obs_buf[idxs], device=self._device).float()
            / 255.0
        )
        action = torch.as_tensor(self.action_buf[idxs], device=self._device)
        reward = torch.as_tensor(self.reward_buf[idxs], device=self._device)
        done = torch.as_tensor(self.done_buf[idxs], device=self._device)

        return obs, action, reward, next_obs, done
