import numpy as np
import matplotlib.pyplot as plt
import os

class RLLogger():
    def __init__(self):
        self.states = []
        self.losses = []
        self.episode_states = []
        self.episode_losses = []
        self.episode_steps = []

    def save_state(self, state):
        self.states.append(state)

    def save_step(self, state):
        self.save_state(state)

    def save_episode(self, n_steps):
        self.episode_states.append(self.states[-n_steps:])
        self.episode_losses.append(self.losses[-n_steps:])
        self.episode_steps.append(n_steps)

    def save_loss(self, loss):
        self.losses.append(loss)
    

        