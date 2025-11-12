#defines the types of networks used to train AS and BR

import torch
import torch.nn as nn


#simple feedforward network that takes in state and outputs action (num_actions=7) 
#used for AS that trains over supervised data of past actions / rewards
class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims=(128, 128)):
        super().__init__()
        layers = []
        last = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        layers.append(nn.Linear(last, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

#trains to minimize the difference between Q(s,a) the predicted reward and r + decay * Q of best possible action in s'
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=(128, 128)):
        super().__init__()
        layers = []
        last = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        layers.append(nn.Linear(last, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)  # directly outputs Q(s,a) for all actions