# Implement my first model: shallownet

# import the necessary packages
import torch
from torch import nn
from torch.utils.data import DataLoader

# implement class by subclassing nn.Module
class ShallowNet(nn.Module):
    def __init__(self):
        super(ShallowNet, self).__init__()

    def forward(self, x):
        pass

# script code for testing my ShallowNet class
if __name__ == "__main__":
    pass
