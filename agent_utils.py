import torch as tr
import torch.nn as nn
import random
import numpy as np

def NNSequential(dimensions, activation, output_activation=nn.Identity):
    layers = []
    for i in range(len(dimensions)-1):
        act = activation if i < len(dimensions)-2 else output_activation
        layers += [nn.Linear(dimensions[i], dimensions[i+1]), act]
    return nn.Sequential(*layers)

class Memory:
    def __init__(self, max_size):
        self.buffer = TensorQueue(max_size)
    
    def push(self, state, action, reward, next_state):
        experience = (state, action, reward, next_state)
        # raise Exception(experience.shape)
        self.buffer.enqueue(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []

        batch = self.buffer.sample(batch_size)

        for experience in batch:
            state, action, reward, next_state = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
        
        return state_batch, action_batch, reward_batch, next_state_batch

    def __len__(self):
        return len(self.buffer)


class TensorQueue:
    def __init__(self, max_size):
        self.queue = []
        self.max_size = max_size

    def is_empty(self):
        return len(self.queue) == 0

    def is_full(self):
        return len(self.queue) == self.max_size

    def enqueue(self, item):
        if self.is_full():
            self.dequeue()
        self.queue.append(item)

    def dequeue(self):
        if self.is_empty():
            raise IndexError("Queue is empty.")
        item = self.queue[0].item()
        self.queue = self.queue[1:]
        return item

    def size(self):
        return len(self.queue)
    
    def sample(self, N_samples):
        indices = random.sample(range(self.size()), min(N_samples, self.size()))
        
        queue_elements = self.queue[indices]
        raise Exception(queue_elements.shape)