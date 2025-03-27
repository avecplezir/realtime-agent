import numpy as np
from typing import NamedTuple
import torch as th


class ReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    rewards: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor


class ReplayMemory():
    def __init__(self, buffer_limit, obs_size, action_size, obs_dtype, device):
        print('buffer limit is = ', buffer_limit)
        self.obs_size = (obs_size, )
        self.buffer_limit = buffer_limit
        self.observation = np.empty((buffer_limit,) + self.obs_size, dtype=obs_dtype)
        self.next_observation = np.empty((buffer_limit,) + self.obs_size, dtype=obs_dtype)

        self.action = np.empty((buffer_limit, int(action_size)), dtype=np.float32)
        self.reward = np.empty((buffer_limit,), dtype=np.float32) 
        self.terminal = np.empty((buffer_limit,), dtype=bool)
        self.idx = 0
        self.full = False
        self.device = device
        self.indices = None

    def reset_indices(self, batch_size):
        self.indices = np.random.choice(self.buffer_limit if self.full else self.idx, batch_size)

    def sample_next(self, ):
        idxes = self.indices
        obs, act, rew, next_obs, term = self.observation[idxes], self.action[idxes], self.reward[idxes], \
            self.next_observation[idxes], self.terminal[idxes]
        obs = th.tensor(obs, dtype=th.float32).to(self.device)
        act = th.tensor(act, dtype=th.float32).to(self.device)
        rew = th.tensor(rew, dtype=th.float32).to(self.device)
        next_obs = th.tensor(next_obs, dtype=th.float32).to(self.device)
        term = th.tensor(term, dtype=th.float32).to(self.device)
        self.indices = (self.indices + 1) % len(self)
        return ReplayBufferSamples(obs, act, rew, next_obs, term)

    def add(self, transition):
        state, next_state, action, reward, done, *_ = transition
        self.observation[self.idx] = state
        self.next_observation[self.idx] = next_state
        self.action[self.idx] = action 
        self.reward[self.idx] = reward
        self.terminal[self.idx] = done
        self.idx = (self.idx + 1) % self.buffer_limit
        self.full = self.full or self.idx == 0
    
    def sample(self, n):
        idxes = np.random.randint(0, self.buffer_limit if self.full else self.idx, size=n)
        obs, act, rew, next_obs, term = self.observation[idxes], self.action[idxes], self.reward[idxes], \
            self.next_observation[idxes], self.terminal[idxes]
        obs = th.tensor(obs, dtype=th.float32).to(self.device)
        act = th.tensor(act, dtype=th.float32).to(self.device)
        rew = th.tensor(rew, dtype=th.float32).to(self.device)
        next_obs = th.tensor(next_obs, dtype=th.float32).to(self.device)
        term = th.tensor(term, dtype=th.float32).to(self.device)
        return ReplayBufferSamples(obs, act, rew, next_obs, term)

    def sample_seq(self, seq_len, batch_size):
        n = batch_size
        l = seq_len
        obs, act, rew, next_obs, term = self._retrieve_batch(np.asarray([self._sample_idx(l) for _ in range(n)]), n, l)
        obs = th.tensor(obs, dtype=th.float32).to(self.device)
        act = th.tensor(act, dtype=th.float32).to(self.device)
        rew = th.tensor(rew, dtype=th.float32).to(self.device)
        next_obs = th.tensor(next_obs, dtype=th.float32).to(self.device)
        term = th.tensor(term, dtype=th.float32).to(self.device)
        return ReplayBufferSamples(obs, act, rew, next_obs, term)

    def sample_probe_data(self, data_size):
        idxes = np.random.randint(0, self.buffer_limit if self.full else self.idx, size=data_size)
        return self.observation[idxes]

    def _sample_idx(self, L):
        valid_idx = False 
        while not valid_idx:
            idx = np.random.randint(0, self.buffer_limit if self.full else self.idx-L)
            idxs = np.arange(idx, idx+L)%self.buffer_limit
            valid_idx = (not self.idx in idxs[1:]) and (not self.terminal[idxs[:-1]].any())
        return idxs 

    def _retrieve_batch(self, idxs, n, l):
        vec_idxs = idxs.transpose().reshape(-1)
        return self.observation[vec_idxs].reshape((l, n) + self.obs_size), self.action[vec_idxs].reshape(l, n, -1), self.reward[vec_idxs].reshape(l, n), \
            self.next_observation[vec_idxs].reshape((l, n) + self.obs_size), self.terminal[vec_idxs].reshape(l, n)
    
    def __len__(self):
        return self.buffer_limit if self.full else self.idx+1

class ReplayBufferSamplesWM(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    rewards: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    wm_observations: th.Tensor


class ReplayMemoryWM(ReplayMemory):
    def __init__(self, buffer_limit, obs_size, action_size, obs_dtype, initial_obs_size, device, wm_index=2):
        super().__init__(buffer_limit, obs_size, action_size, obs_dtype, device)
        self.wm_index = wm_index
        self.initial_obs_size = initial_obs_size
        self.initial_observation = np.empty((buffer_limit,) + (self.initial_obs_size,), dtype=obs_dtype)

    def add(self, transition):
        state, next_state, action, reward, done, info = transition
        self.observation[self.idx] = state
        self.next_observation[self.idx] = next_state
        self.action[self.idx] = action
        self.reward[self.idx] = reward
        self.terminal[self.idx] = done
        self.initial_observation[self.idx] = info['initial_observation'][0][None, :]
        self.idx = (self.idx + 1) % self.buffer_limit
        self.full = self.full or self.idx == 0

    def sample_seq(self, seq_len, batch_size):
        n = batch_size
        l = seq_len
        obs, act, rew, next_obs, term, wm_obs = self._retrieve_batch(np.asarray([self._sample_idx(l) for _ in range(n)]), n, l)
        obs = th.tensor(obs, dtype=th.float32).to(self.device)
        act = th.tensor(act, dtype=th.float32).to(self.device)
        rew = th.tensor(rew, dtype=th.float32).to(self.device)
        next_obs = th.tensor(next_obs, dtype=th.float32).to(self.device)
        term = th.tensor(term, dtype=th.float32).to(self.device)
        wm_obs = th.tensor(wm_obs, dtype=th.float32).to(self.device)
        return ReplayBufferSamplesWM(obs, act, rew, next_obs, term, wm_obs)

    def _sample_idx(self, L):
        valid_idx = False
        while not valid_idx:
            idx = np.random.randint(0, self.buffer_limit - self.wm_index if self.full else self.idx - L - self.wm_index)
            idxs = np.arange(idx, idx + L) % self.buffer_limit
            valid_idx = (not self.idx in idxs[1:]) and (not self.terminal[idxs[:-1]].any())
        return idxs

    def _retrieve_batch(self, idxs, n, l):
        wm_idxs = idxs[:, -1] + self.wm_index
        vec_idxs = idxs.transpose().reshape(-1)
        return self.observation[vec_idxs].reshape((l, n) + self.obs_size), self.action[vec_idxs].reshape(l, n, -1), \
               self.reward[vec_idxs].reshape(l, n), self.next_observation[vec_idxs].reshape((l, n) + self.obs_size), \
               self.terminal[vec_idxs].reshape(l, n), self.initial_observation[wm_idxs] #self.observation[wm_idxs]


class ReplayMemoryDisc():
    def __init__(self, buffer_limit, obs_shape, action_size, obs_dtype, device):
        print('buffer limit is = ', buffer_limit)
        self.obs_size = obs_shape
        self.buffer_limit = buffer_limit

        print('obs_shape', *obs_shape)
        print('obs_dtype', obs_dtype)
        self.observation = np.empty((buffer_limit, *self.obs_size), dtype=obs_dtype)
        self.next_observation = np.empty((buffer_limit, *self.obs_size), dtype=obs_dtype)

        print('action_size', action_size)
        self.action = np.empty((buffer_limit, int(action_size)), dtype=np.float32)
        self.reward = np.empty((buffer_limit,), dtype=np.float32) 
        self.terminal = np.empty((buffer_limit,), dtype=bool)
        self.idx = 0
        self.full = False
        self.device = device
        self.indices = None

    def reset_indices(self, batch_size):
        self.indices = np.random.choice(self.buffer_limit if self.full else self.idx, batch_size)

    def add(self, transition):
        state, next_state, action, reward, done, *_ = transition
        # print('self.observation[self.idx]', self.observation[self.idx].shape, state.shape)

        if (isinstance(state, th.Tensor)):
            state = state.cpu()

        self.observation[self.idx] = state
        self.next_observation[self.idx] = next_state
        self.action[self.idx] = action 
        self.reward[self.idx] = reward
        self.terminal[self.idx] = done
        self.idx = (self.idx + 1) % self.buffer_limit
        self.full = self.full or self.idx == 0
    
    def sample(self, n):
        idxes = np.random.randint(0, self.buffer_limit if self.full else self.idx, size=n)
        obs, act, rew, next_obs, term = self.observation[idxes], self.action[idxes], self.reward[idxes], \
            self.next_observation[idxes], self.terminal[idxes]
        obs = th.tensor(obs, dtype=th.float32).to(self.device)
        act = th.tensor(act, dtype=th.float32).to(self.device)
        rew = th.tensor(rew, dtype=th.float32).to(self.device)
        next_obs = th.tensor(next_obs, dtype=th.float32).to(self.device)
        term = th.tensor(term, dtype=th.float32).to(self.device)
        return ReplayBufferSamples(obs, act, rew, next_obs, term)

    def sample_next(self, ):
        idxes = self.indices
        obs, act, rew, next_obs, term = self.observation[idxes], self.action[idxes], self.reward[idxes], \
            self.next_observation[idxes], self.terminal[idxes]
        obs = th.tensor(obs, dtype=th.float32).to(self.device)
        act = th.tensor(act, dtype=th.float32).to(self.device)
        rew = th.tensor(rew, dtype=th.float32).to(self.device)
        next_obs = th.tensor(next_obs, dtype=th.float32).to(self.device)
        term = th.tensor(term, dtype=th.float32).to(self.device)
        self.indices = (self.indices + 1) % len(self)
        return ReplayBufferSamples(obs, act, rew, next_obs, term)

    def sample_seq(self, seq_len, batch_size):
        n = batch_size
        l = seq_len
        obs, act, rew, next_obs, term = self._retrieve_batch(np.asarray([self._sample_idx(l) for _ in range(n)]), n, l)
        obs = th.tensor(obs, dtype=th.float32).to(self.device)
        act = th.tensor(act, dtype=th.float32).to(self.device)
        rew = th.tensor(rew, dtype=th.float32).to(self.device)
        next_obs = th.tensor(next_obs, dtype=th.float32).to(self.device)
        term = th.tensor(term, dtype=th.float32).to(self.device)
        return ReplayBufferSamples(obs, act, rew, next_obs, term)

    def sample_probe_data(self, data_size):
        idxes = np.random.randint(0, self.buffer_limit if self.full else self.idx, size=data_size)
        return self.observation[idxes]

    def _sample_idx(self, L):
        valid_idx = False 
        while not valid_idx:
            idx = np.random.randint(0, self.buffer_limit if self.full else self.idx-L)
            idxs = np.arange(idx, idx+L)%self.buffer_limit
            valid_idx = (not self.idx in idxs[1:]) and (not self.terminal[idxs[:-1]].any())
        return idxs 

    def _retrieve_batch(self, idxs, n, l):
        vec_idxs = idxs.transpose().reshape(-1)
        return self.observation[vec_idxs].reshape((l, n) + self.obs_size), self.action[vec_idxs].reshape(l, n, -1), self.reward[vec_idxs].reshape(l, n), \
            self.next_observation[vec_idxs].reshape((l, n) + self.obs_size), self.terminal[vec_idxs].reshape(l, n)
    
    def __len__(self):
        return self.buffer_limit if self.full else self.idx+1