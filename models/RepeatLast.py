import torch
import torch.nn as nn
import numpy as np


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

    def forward(self, x):
            last = x[:, -1, :]
            output = last.unsqueeze(1).repeat(1, x.size(1), 1)
            
            return output 