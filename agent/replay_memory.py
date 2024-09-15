# agent/replay_memory.py

import random
from collections import deque
import numpy as np

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        samples = random.sample(self.memory, batch_size)
        state_batch = np.array([s[0] for s in samples])
        action_batch = np.array([s[1] for s in samples])
        reward_batch = np.array([s[2] for s in samples])
        next_state_batch = np.array([s[3] for s in samples])
        done_batch = np.array([s[4] for s in samples])
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.memory)
