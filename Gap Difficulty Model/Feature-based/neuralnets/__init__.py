import torch
from torch import nn
import torch.nn.functional as F

class CTestConfigMLR(nn.Module):
    """ A model for error-rate prediction on C-Test gaps.
    Generate PyTorch MLP from any given configuration """
    def __init__(self, config, input_dim=61):
        super(CTestConfigMLR, self).__init__()
        self.layers = nn.ModuleList()
        self.input_dim = input_dim # Input dimension
        last_dim = self.input_dim
        for _,v in sorted(config.items()):
            self.layers.append(nn.Linear(last_dim, v))
            last_dim = v
        # Output Layer 
        self.layers.append(nn.Linear(last_dim, 1)) 

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = torch.sigmoid(self.layers[-1](x)) # We have 4 classes
        return x

class CTestConfigMLR_Linear(nn.Module):
    """ A model for error-rate prediction on C-Test gaps.
    Pytorch MLP with two layers (32,8) and ReLU activation function """
    def __init__(self, config, input_dim=61):
        super(CTestConfigMLR_Linear, self).__init__()
        self.layers = nn.ModuleList()
        self.input_dim = input_dim # Input dimension
        last_dim = self.input_dim
        for _,v in sorted(config.items()):
            self.layers.append(nn.Linear(last_dim, v))
            last_dim = v
        # Output Layer 
        self.layers.append(nn.Linear(last_dim, 1)) 


    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = self.layers[-1](x) # Wrap to [0,1] 
        return x

class CTestConfigMLR_ReLU(nn.Module):
    """ A model for error-rate prediction on C-Test gaps.
    Pytorch MLP with two layers (32,8) and ReLU activation function """
    def __init__(self, config, input_dim=61):
        super(CTestConfigMLR_ReLU, self).__init__()
        self.layers = nn.ModuleList()
        self.input_dim = input_dim # Input dimension
        last_dim = self.input_dim
        for _,v in sorted(config.items()):
            self.layers.append(nn.Linear(last_dim, v))
            last_dim = v
        # Output Layer 
        self.layers.append(nn.Linear(last_dim, 1)) # Check using relu for final activation


    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        #x = self.layers[-1](x) # Wrap to [0,1] 
        return x




