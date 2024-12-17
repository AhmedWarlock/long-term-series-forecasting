import torch
import torch.nn as nn
import numpy as np

class LinearLayer(nn.Module):
    def __init__(self,in_channels, out_channels, num_feats):
        super().__init__()
        self.num_feats = num_feats
        self.layers = nn.ModuleList()
        self.proj_layers = nn.ModuleList()
        self.out_channels = out_channels
        self.in_channels = in_channels
        for _ in range(self.num_feats) :
                self.layers.append(nn.Linear(self.in_channels, self.out_channels))
                self.proj_layers.append(nn.Linear(self.num_feats, 1))

    def forward(self, x): 
        output = torch.zeros([x.shape[0],self.out_channels,self.num_feats]).to(x.device) 
        for i in range(self.num_feats):
                    out = self.layers[i](x.permute(0,2,1)) # [batch_size, num_feats, out_channels]
                    out = self.proj_layers[i](out.permute(0,2,1)) # [batch_size, out_channels, 1]
                    output[:,:,i] = out.squeeze(-1)
        return output # [batch_size, out_channels, num_feats]


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.in_channels = args.seq_len + 48
        self.out_channels = args.pred_len
        self.num_feats = args.enc_in 
        self.individual = args.individual
        self.time_embed_layer = nn.Embedding(1,48)
        self.num_layers = args.num_lin_layers
        if self.individual:
            self.lin = nn.Sequential(
                  LinearLayer(self.in_channels, self.out_channels, self.num_feats),
                  *[LinearLayer(self.out_channels, self.out_channels,self.num_feats) for _ in range(self.num_layers-1)]
            )
        else : 
            self.lin = nn.Sequential(
                  *[nn.Linear(self.in_channels, self.out_channels) for _ in range(self.num_layers)]
            )

    def forward(self, x, time_stamp):
            batch_size, _, in_feats = x.shape
            temp_enc = self.time_embed_layer(time_stamp.long()) # [16, 336, 2, 48]
            temp_enc =temp_enc.mean(dim=[1,2]) # [16, 48]
            temp_enc = temp_enc.unsqueeze(-1).repeat(1,1,in_feats) # [16, 48, 21]
            input = torch.cat((x,temp_enc), dim = 1) # [16, 336 + 48, 21]
            if self.individual:
                output = self.lin(input)         
            else:
                output = self.lin(input.permute(0,2,1)).permute(0,2,1)
            return output