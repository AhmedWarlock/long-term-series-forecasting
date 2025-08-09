import torch
import torch.nn as nn


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.in_channels = configs.seq_len
        self.out_channels = configs.pred_len
        hidden_dim=configs.hidden_dim

        self.past_proj = nn.Linear(self.in_channels, self.out_channels + 1)

        self.net = nn.Sequential(
            nn.Linear(self.out_channels +1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.out_channels)
        )


    def forward(self, x,t, past):
      x = torch.cat([x, t], dim=1)
      x = x + self.past_proj(past)
      output = self.net(x)
      return output