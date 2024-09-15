# agent/dqn_agent.py

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from agent.model import LinearQNet
from agent.replay_memory import ReplayMemory
from config import *

class DQNAgent:
    def __init__(self):
        self.n_states = 11
        self.n_actions = 3  # [straight, right, left]
        self.epsilon = EPSILON_START
        self.memory = ReplayMemory(MEMORY_SIZE)
        self.model = LinearQNet(self.n_states, 256, self.n_actions)
        self.target_model = LinearQNet(self.n_states, 256, self.n_actions)
        self.update_target_network()
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)  # Add batch dimension
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
            with torch.no_grad():
                return torch.argmax(self.model(state)).item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def learn(self):
        if len(self.memory) < BATCH_SIZE:
            return

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(BATCH_SIZE)

        state_batch = torch.tensor(state_batch, dtype=torch.float)
        action_batch = torch.tensor(action_batch, dtype=torch.long)
        reward_batch = torch.tensor(reward_batch, dtype=torch.float)
        next_state_batch = torch.tensor(next_state_batch, dtype=torch.float)
        done_batch = torch.tensor(done_batch, dtype=torch.bool)

        q_values = self.model(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_model(next_state_batch).max(1)[0]
        target_q_values = reward_batch + (DISCOUNT_FACTOR * next_q_values * ~done_batch)

        loss = self.criterion(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_epsilon(self):
        if self.epsilon > EPSILON_END:
            self.epsilon *= EPSILON_DECAY

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def save_model(self, filepath):
        torch.save(self.model.state_dict(), filepath)

    def load_model(self, filepath):
        self.model.load_state_dict(torch.load(filepath))
        self.update_target_network()
