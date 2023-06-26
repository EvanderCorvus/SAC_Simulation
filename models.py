import torch as tr
import torch.nn as nn
from agent_utils import *
from torch.distributions.normal import Normal

#Soft Actor Critic
log_std_max = 2
log_std_min = -20

class Actor(nn.Module):
    #state-space: position, force, orientation
    def __init__(self, mlp_dims, activation, output_activation=nn.Identity()):
        super(Actor, self).__init__()
        self.net = NNSequential(mlp_dims, activation, output_activation)
        self.mu_layer = nn.Linear(mlp_dims[-1], 2)
        self.log_std_layer = nn.Linear(mlp_dims[-1], 1)

    def forward(self, state):
        #Compute Gaussian Parameters        
        net_output = self.net(state)
        mu = self.mu_layer(net_output)
        mu = tr.atan2(mu[:,1], mu[:,0])
        log_std = self.log_std_layer(net_output)
        log_std = tr.clamp(log_std, log_std_min, log_std_max)
        std = log_std.exp()
        policy = Normal(mu, std)

        action = policy.rsample()
        log_policy = policy.log_prob(action)
        
        return action, log_policy

class Critic(nn.Module):
    def __init__(self, mlp_dims, activation, output_activation=nn.Identity()):
        super(Critic,self).__init__()
        self.net = NNSequential(mlp_dims, activation, output_activation)
        
    def forward(self,state,action):
        state_action = tr.cat([state, action], dim=1)
        return self.net(state_action)
    

    


