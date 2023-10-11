import numpy as np
from typing import Tuple

class TransitionBuffer():
    def __init__(
        self,
        capacity,
        obs_shape: Tuple[int],
        action_size: int,
        seq_len: int, 
        batch_size: int,
        obs_type=np.float32,
        action_type=np.float32,
    ):

        self.capacity = capacity
        self.obs_shape = obs_shape
        self.action_size = action_size
        self.obs_type = obs_type
        self.action_type = action_type
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.idx = 0
        self.full = False
        self.observation = np.empty((capacity, *obs_shape), dtype=obs_type) 
        self.action = np.empty((capacity, action_size), dtype=np.float32)
        self.reward = np.empty((capacity,), dtype=np.float32) 
        self.terminal = np.empty((capacity,), dtype=bool)
        self.truncated = np.empty((capacity,), dtype=bool)

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        truncated : bool
    ):
        self.observation[self.idx] = obs
        self.action[self.idx] = action 
        self.reward[self.idx] = reward
        self.terminal[self.idx] = done
        self.truncated[self.idx] = truncated
        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0


    def _sample_idx(self, L):
        idxs = np.array([])  # Initialize idxs
        valid_idx = False
        while not valid_idx:
            idx = np.random.randint(0, self.capacity if self.full else self.idx - L)
            idxs = np.arange(idx, idx + L) % self.capacity
            valid_idx = not self.idx in idxs[1:]
        return idxs

    def _retrieve_batch(self, idxs, n, l):
        vec_idxs = idxs.transpose().reshape(-1)
        observation = self.observation[vec_idxs]
        return observation.reshape(l, n, *self.obs_shape), self.action[vec_idxs].reshape(l, n, -1), \
            self.reward[vec_idxs].reshape(l, n), self.terminal[vec_idxs].reshape(l, n), self.truncated[vec_idxs].reshape(l, n)

# Because both term and trun appears in our env
    def sample(self):
        n = self.batch_size
        l = self.seq_len+1
        obs,act,rew,term,trun = self._retrieve_batch(np.asarray([self._sample_idx(l) for _ in range(n)]), n, l)
        obs,act,rew,term,trun = self._shift_sequences(obs,act,rew,term,trun)
        done = term | trun
        return obs,act,rew,done
    
    def _shift_sequences(self, obs, actions, rewards, terminals, truncateds):
        obs = obs[1:]
        actions = actions[:-1]
        rewards = rewards[:-1]
        terminals = terminals[:-1]
        truncateds = truncateds[:-1]
        return obs, actions, rewards, terminals, truncateds
