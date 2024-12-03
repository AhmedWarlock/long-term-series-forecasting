import torch
import torch.nn as nn
import numpy as np


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

    def forward(self, x):
        return torch.zeros_like(x) 