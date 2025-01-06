import torch
import torch.nn as nn
import numpy as np

class LinearBlock(nn.Module):

    def __init__(self,in_channels, out_channels, num_feats):
        super().__init__()
        self.temporal_layers = nn.ModuleList()
        # self.feature_layers = nn.Linear(num_feats,num_feats)
        self.norm = nn.BatchNorm1d(num_features=num_feats)

        self.out_channels = out_channels
        self.in_channels = in_channels
        self.num_feats = num_feats

        # self.activation = nn.ReLU()

        for _ in range(num_feats) :
                self.temporal_layers.append(nn.Linear(self.in_channels, self.out_channels))


    
    def forward(self, x):
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        output = torch.zeros([x.shape[0],self.out_channels,self.num_feats]).to(x.device)

        for i in range(self.num_feats):
                    output[:,:,i] = self.temporal_layers[i](x[:,:,i])
        return output



class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.in_channels = args.seq_len
        self.out_channels = args.pred_len
        self.num_feats = args.enc_in
        self.individual = args.individual
        self.num_layers = args.num_lin_layers
        if self.individual:
            self.lin = nn.Sequential(
                  LinearBlock(self.in_channels, self.out_channels, self.num_feats),
                  *[LinearBlock(self.out_channels, self.out_channels,self.num_feats) for _ in range(self.num_layers-1)]
            )
        else : 
            self.lin =  nn.Linear(self.in_channels, self.out_channels)

    def forward(self, x):
            seq_last = x[:,-1:,:].detach()
            x = x - seq_last
            if self.individual:
                output = self.lin(x)         
            else:
                output = self.lin(x.permute(0,2,1)).permute(0,2,1)
            output = output + seq_last
            return output 
    
