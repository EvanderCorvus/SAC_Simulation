import torch as tr
import torch.nn as nn
import numpy as np
from agent_utils import *

device = tr.device('cuda' if tr.cuda.is_available() else 'cpu')
tr.autograd.set_detect_anomaly(True)
tr.set_default_tensor_type(tr.FloatTensor)

class NeuralNet(nn.Module):
    def __init__(self, mlp_dims, activation, output_activation):
        super(NeuralNet, self).__init__()
        self.net = NNSequential(mlp_dims, activation, output_activation)

    def forward(self, state):
        return self.net(state)

mlp_dims = [2, 64, 64, 64, 1]
activation = nn.ReLU()
output_activation = nn.Identity()
model = NeuralNet(mlp_dims, activation, output_activation)
state = tr.tensor([1,2])
print(model.forward(state))