import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from model import DQN

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), action, reward, np.array(next_state), done
        
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, n_actions, device):
        self.n_actions = n_actions
        self.device = device
        
        # Hyperparameters
        self.lr = 1e-4
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.99995
        self.epsilon_min = 0.05
        self.batch_size = 32
        
        # Networks
        self.model = DQN(n_actions).to(device)
        self.target_model = DQN(n_actions).to(device)
        self.update_target_network()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.memory = ReplayBuffer(100000)
        
    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())
        
    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
        return q_values.argmax().item()
        
    def learn(self):
        if len(self.memory) < self.batch_size:
            return
            
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        done = torch.BoolTensor(done).to(self.device)
        
        # Q(s, a)
        q_values = self.model(state).gather(1, action.unsqueeze(1)).squeeze(1)
        
        # max Q(s', a')
        with torch.no_grad():
            next_q_values = self.target_model(next_state).max(1)[0]
            
        # Target Q value
        expected_q_values = reward + self.gamma * next_q_values * (~done)
        
        loss = nn.MSELoss()(q_values, expected_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def save(self, filename):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'epsilon': self.epsilon
        }, filename)
        
    def load(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.update_target_network()
