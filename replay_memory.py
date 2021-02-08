"""
Replay Memory Class for DQN Agent for Vector Observation Learning
"""

import torch
import numpy as np
import random
from collections import namedtuple, deque
from model import QNetwork

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
Defines a replay memory buffer for a DQN agent. The buffer holds memories of : 
[state, action, reward, next, next state] tuples
Random batches of replay memories are sampled for learning
"""

class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state"])
        self.seed = random.seed(seed)
    def add(self, state, action, reward, next_state):
        e = self.experience(state, action, reward, next_state)
        self.memory.append(e)
    def sample(self):
        experiences = random.sample(self.memory, k = self.batch_size)
        #print(experiences.state)
        #input('hui')

        states = torch.from_numpy(np.vstack(np.array([np.array([e.state]) for e in experiences if e is not None]))).float().to(device)

        actions = torch.from_numpy(np.vstack(np.array([e.action for e in experiences if e is not None]))).long().to(device)
        rewards = torch.from_numpy(np.vstack(np.array([e.reward for e in experiences if e is not None]))).float().to(device)
        next_states = torch.from_numpy(np.vstack(np.array([np.array([e.next_state]) for e in experiences if e is not None]))).float().to(device)
        #exp = self.experience(*zip(*experiences))

        #states = torch.tensor(exp.state).to(device)
        #actions = torch.tensor(exp.action).unsqueeze(1).to(device)
        #rewards = torch.tensor(exp.reward).unsqueeze(1).to(device)
        #next_states = torch.tensor(exp.next_state).to(device)

        return (states, actions, rewards, next_states)
    def __len__(self):
        return len(self.memory)
    def upd_goal(self, n):
        if(len(self.memory)>n+2):
            for i in range(n):
                el = self.memory[-1-i]
                new_el = self.experience(el.state, el.action, (n-i)*0.2, el.next_state)
                #el.reward = el.reward + (n-i)*0.1
                self.memory[-1-i] = new_el
    def we_goll(self):
        n=35
        if(len(self.memory)>20+3):
            for i in range(n):
                el = self.memory[-3-i]
                new_el = self.experience(el.state, el.action, i*1.2, el.next_state)
                #el.reward = el.reward + (n-i)*0.1
                self.memory[-1-i] = new_el
    def us_goll(self):
        n=35
        if(len(self.memory)>20+3):
            for i in range(n):
                el = self.memory[-3-i]
                new_el = self.experience(el.state, el.action,  -i*1.2, el.next_state)
                #el.reward = el.reward + (n-i)*0.1
                self.memory[-1-i] = new_el

































































































