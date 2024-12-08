import torch
import torch.nn as nn
import numpy as np

class LinearLayer(nn.Module):

    def __init__(self,in_channels, out_channels, num_feats):
        super().__init__()
        self.layers = nn.ModuleList()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.num_feats = num_feats
        for _ in range(num_feats) :
                self.layers.append(nn.Linear(self.in_channels, self.out_channels))

    
    def forward(self, x):
        output = torch.zeros([x.shape[0],self.out_channels,x.shape[2]]).to(x.device)
        for i in range(self.num_feats):
                    output[:,:,i] = self.layers[i](x[:,:,i])
        
        return output



class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.in_channels = args.seq_len 
        self.out_channels = args.pred_len
        self.num_feats = args.enc_in *2
        self.individual = args.individual
        self.temp_enc_layer = nn.Linear(4, args.enc_in)
        self.proj_layer = nn.Linear(args.enc_in *2, args.enc_in)
        if self.individual:
            self.lin = nn.Sequential(
                  LinearLayer(self.in_channels, self.out_channels, self.num_feats),
                  *[LinearLayer(self.out_channels, self.out_channels,self.num_feats) for _ in range(1)]
            )
        else : 
            # self.lin =  nn.Linear(self.in_channels, self.out_channels)
            self.lin = nn.Sequential(
                  nn.Linear(self.in_channels, self.out_channels),
            )

    def forward(self, x, time_stamp):
            temp_enc = self.temp_enc_layer(time_stamp)
            input = torch.cat((x,temp_enc), dim = 2)
            if self.individual:
                output = self.lin(input)         
            else:
                output = self.lin(input.permute(0,2,1)).permute(0,2,1)
                output = self.proj_layer(output)
            
            return output






        
        



     
